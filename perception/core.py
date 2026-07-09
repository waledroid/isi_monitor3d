"""``PerceptionCore`` — capture → detect → pose → emit detection sets.

The producer half of Direction 1. Per tick (paced at ``perception_fps``):

1. pull the newest ``(image, capture_ts)`` per camera from the injected
   frame provider (the dashboard's camera hub in-process, or this package's
   own RTSP sources when standalone — one decode per camera either way);
2. run the zone-scoped object detector on all cameras' crops in ONE batched
   call (the same ``ZoneScopedDetector`` the Backbone used in frames mode;
   no zones ⇒ no object detector — pose-only, identical policy);
3. run the pose model every ``pose_every_n``-th tick (persons coast on the
   metric engine's Kalman between ticks, exactly as before);
4. emit one ``DetectionSetMessage`` per camera with a FRESH frame — explicit
   empty when nothing was detected (the heartbeat that distinguishes "empty
   scene" from "dead producer"), and NOTHING when the frame is stale (true
   silence ⇒ the metric engine degrades that camera, as if it died).

``capture_ts`` policy: every message is stamped with the capture time of the
exact frame the detections were computed on — the single KPI clock survives
the process split untouched.

Zero FastAPI imports, zero monitor_web imports: this module must be able to
run as its own systemd service. The frame provider is the only seam the host
process fills in.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Protocol

import numpy as np

from backbone.comms.schemas import DetectionSetMessage, WireDetection
from backbone.comms.udp_sink import send_json_datagram
from backbone.core.interfaces import detector_registry
from backbone.core.types import Detection, Frame, FramePair
from backbone.shared.camera_rig import CameraRig
from backbone.shared.config_fingerprint import config_fingerprint
from backbone.shared.mask_poly import mask_to_polygon
from backbone.shared.timestamps import now
from backbone.shared.zones import ZoneRegistry

logger = logging.getLogger(__name__)


class FrameProvider(Protocol):
    """The host's frame seam: newest real frame + capture_ts, or None."""

    def __call__(self, camera_id: str) -> tuple[np.ndarray, float] | None: ...


def _build_object_detector(cfg: dict, rig: CameraRig, zones: ZoneRegistry):
    """The Backbone's frames-mode detector recipe, verbatim semantics:
    zone-scoped (crops at ``zone_imgsz``, batched, remapped), masks decoded
    by default under zone scope, no zones ⇒ None (pose-only system)."""
    det_cfg = dict(cfg.get("detection", {}))
    if not det_cfg or not det_cfg.get("plugin"):
        return None
    det_plugin = det_cfg.pop("plugin")
    for pose_key in ("pose_onnx_path", "pose_confidence_threshold", "pose_imgsz",
                     "pose_every_n", "person_pallet_max_distance_m"):
        det_cfg.pop(pose_key, None)
    imgsz = det_cfg.pop("inference_imgsz", None)
    if imgsz:
        det_cfg["input_size"] = (int(imgsz), int(imgsz))
    scope = str(det_cfg.pop("scope", "zones"))
    zone_imgsz = int(det_cfg.pop("zone_imgsz", 384) or 384)
    scope_is_zones = scope == "zones"
    if det_plugin in ("yolo_onnx_seg", "yolo_openvino_seg"):
        det_cfg.setdefault("decode_masks", scope_is_zones)
    if scope_is_zones and len(zones) == 0:
        logger.warning("perception: zone scope with no zones — object detection OFF "
                       "(pose-only). Draw zones to enable it.")
        return None
    if scope_is_zones:
        det_cfg["input_size"] = (zone_imgsz, zone_imgsz)
    detector = detector_registry.create(det_plugin, **det_cfg)
    if scope_is_zones:
        from backbone.detection.zone_scope import ZoneScopedDetector, zone_crop_boxes
        boxes = zone_crop_boxes(rig, zones)
        detector = ZoneScopedDetector(
            detector, boxes,
            {cid: rig[cid].image_size_wh for cid in rig.camera_ids})
    return detector


def _build_pose_detector(cfg: dict):
    det = cfg.get("detection", {})
    pose_path = det.get("pose_onnx_path")
    if not pose_path:
        return None
    try:
        kwargs: dict = {}
        if det.get("pose_imgsz"):
            kwargs["input_size"] = (int(det["pose_imgsz"]), int(det["pose_imgsz"]))
        pose = detector_registry.create(
            "yolo_onnx_pose", onnx_path=pose_path,
            confidence_threshold=float(det.get("pose_confidence_threshold", 0.3)),
            **kwargs)
        logger.info("perception: pose detector enabled (%s)", pose_path)
        return pose
    except Exception as exc:
        logger.warning("perception: pose detector disabled (%s)", exc)
        return None


def _to_wire(d: Detection) -> WireDetection:
    poly = None
    if d.mask is not None:
        poly = mask_to_polygon(d.mask, getattr(d, "mask_offset_xy", None))
    return WireDetection(
        cls=str(d.cls),
        confidence=float(d.confidence),
        bbox_xyxy=tuple(round(float(v), 1) for v in d.bbox_xyxy),
        foot_uv=(round(float(d.foot_uv[0]), 1), round(float(d.foot_uv[1]), 1)),
        keypoints_uv=(tuple((round(float(u), 1), round(float(v), 1), round(float(c), 3))
                            for u, v, c in np.asarray(d.keypoints_uv).reshape(-1, 3))
                      if d.keypoints_uv is not None else None),
        mask_poly=(tuple((float(round(x)), float(round(y))) for x, y in poly)
                   if poly else None),
    )


class PerceptionCore:
    """The perception loop. ``start()``/``stop()`` manage a daemon thread;
    hosts that want their own scheduling can call ``tick()`` directly."""

    def __init__(
        self,
        *,
        camera_ids: list[str],
        frame_provider: FrameProvider,
        object_detector,
        pose_detector,
        ingest_addr: tuple[str, int],
        fingerprint: str | None = None,
        perception_fps: float = 12.0,
        pose_every_n: int = 1,
        producer_id: str = "perception",
    ) -> None:
        self._camera_ids = list(camera_ids)
        self._frames = frame_provider
        self._detector = object_detector
        self._pose = pose_detector
        self._addr = (str(ingest_addr[0]), int(ingest_addr[1]))
        self._fingerprint = fingerprint
        self._interval = 1.0 / max(0.5, float(perception_fps))
        self._pose_every_n = max(1, int(pose_every_n))
        self._producer_id = producer_id

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq: dict[str, int] = dict.fromkeys(self._camera_ids, 0)
        self._last_ts: dict[str, float] = dict.fromkeys(self._camera_ids, -1.0)
        self._tick_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Operator-visible stats (read by the host's status endpoint).
        self.sets_sent: dict[str, int] = dict.fromkeys(self._camera_ids, 0)
        self.last_tick_ms: float = 0.0
        self.stage_ms: dict[str, float] = {}   # per-stage timing of the last tick
        self.last_error: str | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="perception")
        self._thread.start()
        logger.info("perception: started → %s:%d @ %.1f fps (pose 1/%d)",
                    self._addr[0], self._addr[1], 1.0 / self._interval,
                    self._pose_every_n)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        try:
            self._sock.close()
        except OSError:
            pass
        logger.info("perception: stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- the loop ----

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t0 = now()
            try:
                self.tick()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("perception: tick failed", exc_info=True)
            self.last_tick_ms = (now() - t0) * 1000.0
            # Operator heartbeat: one stats line every ~30 s at the target
            # rate, so tick health is visible in the producer's own log.
            if self._tick_count % max(1, int(30.0 / self._interval)) == 0:
                logger.info("perception: tick %.0f ms (stages %s), sets %s",
                            self.last_tick_ms,
                            {k: round(v) for k, v in self.stage_ms.items()},
                            dict(self.sets_sent))
            # Fixed pacing minus work time — detection latency doesn't
            # compound into the tick rate until it exceeds the interval.
            self._stop_event.wait(max(0.0, self._interval - (now() - t0)))

    def tick(self) -> None:
        """One perception pass over every camera with a FRESH frame."""
        self._tick_count += 1
        t = now()
        fresh: dict[str, Frame] = {}
        for cam_id in self._camera_ids:
            got = self._frames(cam_id)
            if got is None:
                continue
            image, ts = got
            if ts <= self._last_ts[cam_id]:
                continue        # stale frame → SILENCE (degradation signal)
            self._last_ts[cam_id] = ts
            fresh[cam_id] = Frame(camera_id=cam_id, capture_ts=ts,
                                  frame_idx=self._seq[cam_id], image=image)
        self.stage_ms["frames"] = (now() - t) * 1000.0
        if not fresh:
            return

        dets_by_cam: dict[str, list[Detection]] = {cid: [] for cid in fresh}
        pair = FramePair(capture_ts=max(f.capture_ts for f in fresh.values()),
                         frame_idx=self._tick_count, frames=fresh)
        t = now()
        if self._detector is not None:
            for cid, dets in self._detector.detect(pair).items():
                dets_by_cam.setdefault(cid, []).extend(dets)
        self.stage_ms["detect"] = (now() - t) * 1000.0
        t = now()
        if self._pose is not None and self._tick_count % self._pose_every_n == 0:
            try:
                for cid, dets in self._pose.detect(pair).items():
                    dets_by_cam.setdefault(cid, []).extend(dets)
            except Exception:
                logger.debug("perception: pose failed this tick", exc_info=True)
        self.stage_ms["pose"] = (now() - t) * 1000.0

        t = now()
        for cam_id, frame in fresh.items():
            h, w = frame.image.shape[:2]
            msg = DetectionSetMessage(
                ts=frame.capture_ts,
                camera_id=cam_id,
                frame_wh=(int(w), int(h)),
                seq=self._seq[cam_id],
                producer_id=self._producer_id,
                config_fingerprint=self._fingerprint,
                dets=tuple(_to_wire(d) for d in dets_by_cam.get(cam_id, [])),
            )
            self._seq[cam_id] += 1
            try:
                send_json_datagram(self._sock, self._addr,
                                   msg.model_dump_json().encode("utf-8"))
                self.sets_sent[cam_id] += 1
            except OSError:
                logger.warning("perception: emit failed", exc_info=True)
        self.stage_ms["emit"] = (now() - t) * 1000.0


def build_perception_core(
    cfg: dict,
    frame_provider: FrameProvider,
    *,
    producer_id: str = "perception",
) -> PerceptionCore:
    """Build a core from a loaded ``backbone.yaml`` dict — the shared recipe
    for the in-process (monitor_web) and standalone (``python -m perception``)
    hosts. Reads the SAME config the metric engine reads, so the fingerprint
    matches by construction."""
    rig = CameraRig.from_file(cfg["calibration_path"])
    zones_path = cfg.get("zones_path")
    zones = ZoneRegistry.load(zones_path) if zones_path else ZoneRegistry.empty()

    # Auto-registration: importing the package fires the @register decorators.
    import backbone.detection  # noqa: F401

    points_cfg = cfg.get("ingestion", {}).get("points", {})
    perception_cfg = cfg.get("perception", {})
    det = cfg.get("detection", {})
    return PerceptionCore(
        camera_ids=list(cfg["cameras"]),
        frame_provider=frame_provider,
        object_detector=_build_object_detector(cfg, rig, zones),
        pose_detector=_build_pose_detector(cfg),
        ingest_addr=(str(points_cfg.get("listen_host", "127.0.0.1")),
                     int(points_cfg.get("listen_port", 9010))),
        fingerprint=config_fingerprint(cfg),
        perception_fps=float(perception_cfg.get("fps", 12.0)),
        pose_every_n=int(det.get("pose_every_n", 1)),
        producer_id=producer_id,
    )
