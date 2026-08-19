"""``IsistreamCore`` — capture → detect → pose → emit detection sets.

The producer half of Direction 1. Per tick (paced at ``isistream.fps``):

1. pull the newest ``(image, capture_ts)`` per camera from the injected
   frame provider (the dashboard's camera hub in-process, or this package's
   own RTSP sources when standalone — one decode per camera either way);
2. run the zone-scoped object detector on all cameras' crops in ONE batched
   call (the same ``ZoneScopedDetector`` the Backbone used in frames mode;
   no zones ⇒ no object detector — pose-only, identical policy);
3. run the pose model every ``pose_every_n``-th tick, One-Euro-smoothing the
   keypoints per camera/person before emission (``pose_smooth.py``; foot
   points stay raw for the metric engine — persons coast on the
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
    # Mirror of pose_enabled: switch OFF object detection while keeping the
    # zones + model config intact (zones stay drawn; no object model is built).
    if not det_cfg.pop("object_enabled", True):
        logger.info("isistream: object detection DISABLED via settings")
        return None
    det_plugin = det_cfg.pop("plugin")
    for pose_key in ("pose_onnx_path", "pose_enabled", "pose_confidence_threshold",
                     "pose_imgsz", "pose_every_n", "person_pallet_max_distance_m",
                     "trt_enabled"):
        det_cfg.pop(pose_key, None)
    if det_plugin == "rfdetr_onnx_seg":
        # RF-DETR is NMS-free and always decodes its masks — these are YOLO-only
        # knobs. A plugin switch in Settings leaves them behind in the YAML, and
        # forwarding them here is a constructor TypeError that kills the whole
        # producer at boot (same whitelist as engines.get_detector).
        for yolo_key in ("iou_threshold", "keep_classes", "decode_masks"):
            det_cfg.pop(yolo_key, None)
    # Global zone-inference options (Settings ▸ Isistream). ONE model, ONE
    # batched call for every zone of every camera — these apply to all zones.
    sahi_cfg = det_cfg.pop("sahi", None)
    enhance_cfg = det_cfg.pop("enhance", None)
    imgsz = det_cfg.pop("inference_imgsz", None)
    if imgsz:
        det_cfg["input_size"] = (int(imgsz), int(imgsz))
    scope = str(det_cfg.pop("scope", "zones"))
    zone_imgsz = int(det_cfg.pop("zone_imgsz", 384) or 384)
    # Extreme-aspect zone crops square-tile themselves (edge-on zones would
    # otherwise letterbox their objects into invisibility); 0 disables.
    from backbone.detection.zone_scope import _MAX_CROP_ASPECT
    max_aspect = float(det_cfg.pop("zone_crop_max_aspect", _MAX_CROP_ASPECT))
    crop_h = float(det_cfg.pop("zone_crop_height_m", 0.0) or 0.0)
    zone_tol = float(det_cfg.pop("zone_membership_tol_m", 0.15))
    fill_on = bool(det_cfg.pop("zone_crop_polygon_fill", True))
    scope_is_zones = scope == "zones"
    if det_plugin in ("yolo_onnx_seg", "yolo_openvino_seg"):
        det_cfg.setdefault("decode_masks", scope_is_zones)
    if scope_is_zones and len(zones) == 0:
        logger.warning("isistream: zone scope with no zones — object detection OFF "
                       "(pose-only). Draw zones to enable it.")
        return None
    if scope_is_zones:
        det_cfg["input_size"] = (zone_imgsz, zone_imgsz)
    if det_plugin == "rfdetr_onnx_seg":
        # RF-DETR's exported graph is STATIC (e.g. 432x432): forcing the
        # slider/zone size onto it fails ORT shape validation on every tick
        # ("Got: 320 Expected: 432") — silently, since the tick loop swallows
        # inference errors into empty sets. The plugin reads its own fixed
        # input from the ONNX; crops are resized to it.
        det_cfg.pop("input_size", None)
    detector = detector_registry.create(det_plugin, **det_cfg)
    if scope_is_zones:
        from backbone.detection.zone_scope import (
            ZoneScopedDetector,
            build_zone_membership_filter,
            zone_crop_boxes,
            zone_fill_polygons,
        )
        boxes = zone_crop_boxes(rig, zones, crop_height_m=crop_h)
        # In-zone guarantee: a zone only reports objects metrically inside a
        # zone polygon (±tol), however far the rectangular crop reaches.
        zfilter = build_zone_membership_filter(rig, zones, tol_m=zone_tol)
        # Polygon fill (default on): pixels outside the dilated zone polygon
        # are blanked to gray before inference — the detector never sees the
        # crop's off-zone corner triangles.
        fill_polys = (zone_fill_polygons(rig, zones, crop_height_m=crop_h)
                      if fill_on else None)
        # No batch bucketing: ONNX paths run plain CUDA EP (any batch), and
        # native .engine files carry a dynamic batch profile (1..32) — the
        # lazy TRT-EP per-shape compiles that bucketing worked around are gone.
        detector = ZoneScopedDetector(
            detector, boxes,
            {cid: rig[cid].image_size_wh for cid in rig.camera_ids},
            sahi=sahi_cfg, enhance=enhance_cfg,
            max_crop_aspect=max_aspect, zone_filter=zfilter,
            fill_polys=fill_polys)
        if sahi_cfg and sahi_cfg.get("enabled"):
            logger.info("isistream: SAHI tiling ON (tile=%s, overlap=%.2f)",
                        sahi_cfg.get("tile") or "model input",
                        float(sahi_cfg.get("overlap", 0.2)))
        if enhance_cfg and enhance_cfg.get("enabled"):
            logger.info("isistream: crop enhancement ON (CLAHE clip=%.1f, gamma=%.2f)",
                        float(enhance_cfg.get("clip_limit", 2.0)),
                        float(enhance_cfg.get("gamma", 1.0)))
    return detector


def _build_pose_detector(cfg: dict):
    det = cfg.get("detection", {})
    if not det.get("pose_enabled", True):
        logger.info("isistream: pose detection DISABLED via settings")
        return None
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
        logger.info("isistream: pose detector enabled (%s)", pose_path)
        return pose
    except Exception as exc:
        logger.warning("isistream: pose detector disabled (%s)", exc)
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


class IsistreamCore:
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
        producer_id: str = "isistream",
        motion_gate=None,
        pose_smoother=None,
        etagere_detector=None,
    ) -> None:
        self._camera_ids = list(camera_ids)
        self._frames = frame_provider
        self._detector = object_detector
        self._pose = pose_detector
        self._etagere = etagere_detector
        self._addr = (str(ingest_addr[0]), int(ingest_addr[1]))
        self._fingerprint = fingerprint
        self._interval = 1.0 / max(0.5, float(perception_fps))
        self._pose_every_n = max(1, int(pose_every_n))
        self._producer_id = producer_id

        self._gate = motion_gate
        self._smoother = pose_smoother   # One Euro keypoint smoothing (display)
        self._wire_obj: dict[str, tuple] = {}      # cam → cached object WireDetections
        self._wire_person: dict[str, tuple] = {}   # cam → cached person WireDetections
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq: dict[str, int] = dict.fromkeys(self._camera_ids, 0)
        self._last_ts: dict[str, float] = dict.fromkeys(self._camera_ids, -1.0)
        self._tick_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Operator-visible stats (read by the host's status endpoint).
        self.sets_sent: dict[str, int] = dict.fromkeys(self._camera_ids, 0)
        self.etagere_sent: dict[str, int] = {}
        self.last_tick_ms: float = 0.0
        self.stage_ms: dict[str, float] = {}   # per-stage timing of the last tick
        self.last_error: str | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="isistream")
        self._thread.start()
        logger.info("isistream: started → %s:%d @ %.1f fps (pose 1/%d)",
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
        logger.info("isistream: stopped")

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
                logger.warning("isistream: tick failed", exc_info=True)
            self.last_tick_ms = (now() - t0) * 1000.0
            # Operator heartbeat: one stats line every ~30 s at the target
            # rate, so tick health is visible in the producer's own log.
            if self._tick_count % max(1, int(30.0 / self._interval)) == 0:
                gate = ("" if self._gate is None else
                        f", gated obj={self._gate.obj_skips} pose={self._gate.pose_skips}")
                logger.info("isistream: tick %.0f ms (stages %s), sets %s%s",
                            self.last_tick_ms,
                            {k: round(v) for k, v in self.stage_ms.items()},
                            dict(self.sets_sent), gate)
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

        now_s = now()
        # Motion gate: run each model only on cameras whose relevant regions
        # visibly changed (or whose forced-refresh interval elapsed). Gated
        # cameras re-emit their CACHED wire detections — same boxes, same
        # mask polygons — under the NEW frame's capture_ts, so downstream
        # sees an uninterrupted observation stream while the GPU idles.
        obj_cams = dict(fresh)
        pose_cams = dict(fresh) if self._tick_count % self._pose_every_n == 0 else {}
        if self._gate is not None:
            obj_cams = {cid: f for cid, f in fresh.items()
                        if self._gate.objects_due(cid, f.image, now_s)}
            pose_cams = {cid: f for cid, f in pose_cams.items()
                         if self._gate.pose_due(cid, f.image, now_s)}

        t = now()
        if self._detector is not None and obj_cams:
            pair = FramePair(capture_ts=max(f.capture_ts for f in obj_cams.values()),
                             frame_idx=self._tick_count, frames=obj_cams)
            for cid, dets in self._detector.detect(pair).items():
                self._wire_obj[cid] = tuple(_to_wire(d) for d in dets)
        self.stage_ms["detect"] = (now() - t) * 1000.0
        t = now()
        if self._pose is not None and pose_cams:
            try:
                pair = FramePair(capture_ts=max(f.capture_ts for f in pose_cams.values()),
                                 frame_idx=self._tick_count, frames=pose_cams)
                for cid, dets in self._pose.detect(pair).items():
                    if self._smoother is not None:
                        # keypoints only — foot_uv stays raw for the metric engine
                        dets = self._smoother.smooth(
                            cid, dets, pose_cams[cid].capture_ts)
                    self._wire_person[cid] = tuple(_to_wire(d) for d in dets)
            except Exception:
                logger.debug("isistream: pose failed this tick", exc_info=True)
        self.stage_ms["pose"] = (now() - t) * 1000.0

        if self._etagere is not None:
            t = now()
            try:
                etagere_msgs = self._etagere.run(fresh, now())
            except Exception:
                etagere_msgs = ()
                self.last_error = "etagere"
                logger.warning("isistream: étagère stage failed", exc_info=True)
            for msg in etagere_msgs:
                try:
                    send_json_datagram(self._sock, self._addr,
                                       msg.model_dump_json().encode("utf-8"))
                    self.etagere_sent[msg.zone_id] = self.etagere_sent.get(msg.zone_id, 0) + 1
                except OSError:
                    logger.warning("isistream: étagère emit failed", exc_info=True)
            self.stage_ms["etagere"] = (now() - t) * 1000.0

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
                dets=(self._wire_obj.get(cam_id, ())
                      + self._wire_person.get(cam_id, ())),
            )
            self._seq[cam_id] += 1
            try:
                send_json_datagram(self._sock, self._addr,
                                   msg.model_dump_json().encode("utf-8"))
                self.sets_sent[cam_id] += 1
            except OSError:
                logger.warning("isistream: emit failed", exc_info=True)
        self.stage_ms["emit"] = (now() - t) * 1000.0


def _build_etagere_stage(cfg: dict, config_path: str | None, *, producer_id: str):
    """Build the étagère detector, or ``None`` if the feature is off.

    Two distinct failure modes here, logged at two distinct levels: no
    config file / an empty (no-model-or-zones) one is a normal DISABLED
    state (silent) — most deployments never touch étagère at all. A
    PRESENT, non-trivial config that fails to load or whose detector fails
    to build is an operator mistake (bad YAML, unreachable onnx_path,
    missing labels) and must be loud (ERROR + traceback), or the feature
    silently no-ops forever with every cell stuck "unknown"."""
    if config_path is None:
        return None
    from backbone.shared.etagere import load_etagere_config, resolve_config_path
    et_path = resolve_config_path(cfg, config_path)
    et_cfg = None
    try:
        et_cfg = load_etagere_config(et_path)
    except Exception:
        logger.error("isistream: étagère config at %s is invalid — feature off",
                     et_path, exc_info=True)
        return None
    if et_cfg is None or not et_cfg.enabled:
        return None
    try:
        from isistream.etagere import build_etagere_detector
        return build_etagere_detector(et_cfg, producer_id=producer_id,
                                      fingerprint=config_fingerprint(cfg))
    except Exception:
        logger.error("isistream: étagère detector failed to build — feature off",
                     exc_info=True)
        return None


def build_isistream_core(
    cfg: dict,
    frame_provider: FrameProvider,
    *,
    producer_id: str = "isistream",
    config_path: str | None = None,
) -> IsistreamCore:
    """Build a core from a loaded ``backbone.yaml`` dict — the shared recipe
    for the in-process (monitor_web) and standalone (``python -m isistream``)
    hosts. Reads the SAME config the metric engine reads, so the fingerprint
    matches by construction.

    ``config_path`` is ``backbone.yaml``'s own path — needed only to resolve
    the sibling ``etagere.yaml`` (``etagere.config_path`` or ``<dir>/etagere.yaml``).
    ``None`` (or a broken étagère config/model) leaves the feature off."""
    rig = CameraRig.from_file(cfg["calibration_path"])
    zones_path = cfg.get("zones_path")
    zones = ZoneRegistry.load(zones_path) if zones_path else ZoneRegistry.empty()

    # Auto-registration: importing the package fires the @register decorators.
    import backbone.detection  # noqa: F401

    points_cfg = cfg.get("ingestion", {}).get("points", {})
    # Config key is `isistream:`; the pre-rename `perception:` still reads.
    perception_cfg = cfg.get("isistream", cfg.get("perception", {}))
    det = cfg.get("detection", {})

    # Motion gate (default ON): zone-crop signatures gate the object
    # detector, a full-frame signature gates pose. `isistream.motion_gate:
    # false` disables; `motion_refresh_s` tunes the self-heal interval.
    gate = None
    if bool(perception_cfg.get("motion_gate", True)):
        from backbone.detection.zone_scope import zone_crop_boxes
        from isistream.motion_gate import MotionGate
        gate = MotionGate(
            # Same crop height as the detector — the gate must watch exactly
            # the pixels the detector will see.
            zone_crop_boxes(
                rig, zones,
                crop_height_m=float(det.get("zone_crop_height_m", 0.0) or 0.0),
            ) if len(zones) else {},
            {cid: rig[cid].image_size_wh for cid in rig.camera_ids},
            refresh_s=float(perception_cfg.get("motion_refresh_s", 2.0)))

    smoother = None
    if bool(det.get("pose_smoothing", True)):
        from isistream.pose_smooth import PoseSmoother
        smoother = PoseSmoother(
            min_cutoff=float(det.get("pose_smooth_min_cutoff", 1.0)),
            beta=float(det.get("pose_smooth_beta", 0.01)))

    etagere = _build_etagere_stage(cfg, config_path, producer_id=producer_id)

    return IsistreamCore(
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
        motion_gate=gate,
        pose_smoother=smoother,
        etagere_detector=etagere,
    )
