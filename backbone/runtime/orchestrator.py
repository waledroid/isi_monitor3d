"""``Orchestrator`` — reads ``backbone.yaml`` and wires the full pipeline.

The orchestrator is the **only** place in the codebase that calls
``registry.create()``. It composes the eight Backbone sub-modules into one
runnable process, owns thread lifecycle, and propagates a graceful shutdown
on signal.

Pipeline (per ``FramePair``, single pipeline thread):

    FrameSource[s] ──► FrameSynchronizer ──► FrameBus  (ingestion threads)
                                                │
                                                ▼  (pipeline thread)
                                            Detector
                                                │
                                                ▼
                                        FootProjector → CrossCamFusion → DisagreementGate
                                                │
                                                ▼
                                            Tracker (bytetrack)
                                                │
                                                ▼
                                       TemporalStabilizer ──► Publisher.publish_track_2d
                                                │
                                                ▼
                                      SubscriptionManager (S5 filter)
                                                │
                                                ▼
                                  KeypointAssociator → Triangulator → ReprojectionGate → Tracker3D
                                                                                            │
                                                                                            ▼
                                                                          Publisher.publish_track_3d

Threading model: one thread per ``FrameSource`` (already managed by each
source's GStreamer mainloop), one thread that pulls frames from the sources
and feeds the ``FrameSynchronizer``, one thread that consumes the bus and
runs the rest of the pipeline. Graceful shutdown joins all of them.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import yaml

# Importing each layer's package triggers `@register` for its plugin(s).
# These are listed explicitly so adding the orchestrator to a YAML config
# is enough to instantiate any registered plugin — the user doesn't need
# to know which package contains which seam.
import backbone.comms
import backbone.detection
import backbone.homography
import backbone.ingestion
import backbone.triangulation  # noqa: F401  — registers opencv_dlt
from backbone.comms import Publisher
from backbone.comms.diagnostics_publisher import DiagnosticsPublisher
from backbone.comms.schemas import (
    CalibrationFactCheck,
    ConfigMessage,
    ObservationDet,
    ObservationsMessage,
    ProximityMessage,
    ProximityPair,
    ZoneSpec,
    ZoneStateMessage,
)
from backbone.core.interfaces import (
    detector_registry,
    frame_source_registry,
    metadata_sink_registry,
    tracker_registry,
    triangulator_registry,
)
from backbone.core.types import Track2D
from backbone.homography import (
    ByteTrackMeters,
    CrossCamFusion,
    DisagreementGate,
    FootProjector,
    PalletOccupancy,
    TemporalStabilizer,
    TrackConfig,
)
from backbone.ingestion import FrameBus, FrameSynchronizer
from backbone.shared.camera_rig import CameraRig
from backbone.shared.mask_poly import mask_to_polygon
from backbone.shared.snapshot_writer import SnapshotWriter
from backbone.shared.timestamps import LatencyMeter, elapsed_ms, now
from backbone.shared.zone_state import ZoneStateTracker
from backbone.shared.zone_transitions import ZoneTransitionDetector
from backbone.shared.zones import ZoneRegistry
from backbone.triangulation import (
    KeypointAssociator,
    ReprojectionGate,
    SubscriptionManager,
    Track3DConfig,
    Tracker3D,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Composes all Backbone sub-modules from ``backbone.yaml``."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._config = yaml.safe_load(self._config_path.read_text())
        self._stop_event = threading.Event()
        self._signals_installed = False   # set by install_signal_handlers()
        self._build()

    # ---- build ----

    def _build(self) -> None:
        cfg = self._config
        self._node_id: str = cfg.get("node_id", "node")
        self._rig = CameraRig.from_file(cfg["calibration_path"])

        # Frame sources.
        self._sources: dict[str, Any] = {}
        for cam_id, cam_cfg in cfg["cameras"].items():
            src_cfg = dict(cam_cfg["source"])
            plugin_name = src_cfg.pop("name")
            self._sources[cam_id] = frame_source_registry.create(
                plugin_name, camera_id=cam_id, **src_cfg,
            )

        # Operational mode — single-cam (Mode 1) skips the triangulation stack
        # entirely. Two cameras → Mode 2 (full pipeline). Architectures are
        # constrained to N <= 2 per S7's CrossCamFusion assertion.
        n_cams = len(self._sources)
        self._mode = "single_cam_homography" if n_cams == 1 else "dual_cam_homography_triangulation"
        logger.info(
            "orchestrator: %s cameras configured → mode=%s",
            n_cams, self._mode,
        )

        # Per-source liveness tracking. The ingestion thread updates this on
        # exit; the latency probe + step() guard read it.
        self._source_status: dict[str, str] = {cam_id: "alive" for cam_id in self._sources}

        # Zones + subscriptions (only meaningful in dual-cam mode).
        zones_path = cfg.get("zones_path")
        self._zones = ZoneRegistry.load(zones_path) if zones_path else ZoneRegistry.empty()
        subs_path = cfg.get("subscriptions_path")
        if subs_path:
            self._subscriptions = SubscriptionManager.load(subs_path, self._zones)
        else:
            self._subscriptions = SubscriptionManager([], self._zones)

        # Ingestion.
        ing_cfg = cfg.get("ingestion", {})
        sync_cfg = dict(ing_cfg.get("frame_sync", {}))
        self._sync = FrameSynchronizer(camera_ids=list(self._sources), **sync_cfg)
        bus_cfg = dict(ing_cfg.get("frame_bus", {}))
        self._bus = FrameBus(**bus_cfg)

        # Detection.
        det_cfg = dict(cfg["detection"])
        det_plugin = det_cfg.pop("plugin")
        # `pose_onnx_path` configures a separate person-pose model (consumed by the
        # dashboard overlay today; the S5.5 `yolo_onnx_pose` plugin later) — it is
        # not a kwarg of the object detector, so drop it before constructing one.
        det_cfg.pop("pose_onnx_path", None)
        det_cfg.pop("pose_confidence_threshold", None)   # pose-engine setting, not an object-detector kwarg
        det_cfg.pop("pose_imgsz", None)                  # pose-engine setting, not an object-detector kwarg
        # Run the pose model on every Nth pair only (1 = every pair). Person
        # tracks coast between pose frames — see step().
        self._pose_every_n = max(1, int(det_cfg.pop("pose_every_n", 1)))
        # `inference_imgsz` is the runtime input size (the dashboard slider). It maps
        # to the detector's square `input_size`; effective only on a dynamic ONNX.
        imgsz = det_cfg.pop("inference_imgsz", None)
        if imgsz:
            det_cfg["input_size"] = (int(imgsz), int(imgsz))
        # The pipeline never reads Detection.mask except as an optional area
        # refinement in pallet_occupancy (bbox fallback) — per-detection
        # full-frame mask assembly is pure CPU cost here. Default it OFF for
        # the YOLO seg plugins; a config `decode_masks: true` re-enables.
        # (Dashboard overlays build their own detectors and keep masks.)
        if det_plugin in ("yolo_onnx_seg", "yolo_openvino_seg"):
            det_cfg.setdefault("decode_masks", False)
        # The system is ZONE-BASED: with `detection.scope: zones` (the default)
        # the object detector sees only the configured floor zones' crops —
        # and with NO zones configured it is not built at all (pose stays
        # global; person tracks continue). `scope: full_frame` restores the
        # everything-visible behaviour.
        scope = str(det_cfg.pop("scope", "zones"))
        # Zone crops are letterboxed to the model input INDIVIDUALLY, so cost
        # scales with CROP COUNT, not crop area — at the full-frame 640 four
        # crops would cost MORE than two full frames. `zone_imgsz` (default
        # 384) sizes the zone-scoped inference instead; needs a dynamic-export
        # model (a static one keeps its baked size, same rule as the slider).
        zone_imgsz = int(det_cfg.pop("zone_imgsz", 384) or 384)
        if scope not in ("zones", "full_frame"):
            raise ValueError(f"detection.scope={scope!r}, expected 'zones' or 'full_frame'")
        if scope == "zones" and len(self._zones) == 0:
            self._detector = None
            logger.warning(
                "orchestrator: detection.scope=zones with no zones configured — "
                "object detection is OFF (pose-only). Draw zones to enable it.")
        else:
            if scope == "zones":
                det_cfg["input_size"] = (zone_imgsz, zone_imgsz)
            self._detector = detector_registry.create(det_plugin, **det_cfg)
            if scope == "zones":
                from backbone.detection.zone_scope import (
                    ZoneScopedDetector,
                    zone_crop_boxes,
                )
                boxes = zone_crop_boxes(self._rig, self._zones)
                self._detector = ZoneScopedDetector(
                    self._detector, boxes,
                    {cid: self._rig[cid].image_size_wh for cid in self._rig.camera_ids})

        # Optional person-pose detector — reuses the SAME config keys the dashboard
        # overlay uses (`detection.pose_onnx_path` / `pose_confidence_threshold`), so
        # configuring a pose model lights up person tracking with no extra config.
        # When set, the orchestrator runs it alongside the object detector and emits
        # person `Track2D` (foot = ankle midpoint) for person↔pallet distance. A
        # missing/unloadable model degrades cleanly to "no persons".
        self._person_detector = None
        pose_path = cfg["detection"].get("pose_onnx_path")
        if pose_path:
            try:
                pose_conf = float(cfg["detection"].get("pose_confidence_threshold", 0.3))
                # `pose_imgsz` shrinks the pose input on a dynamic export (a
                # static export keeps its baked size — same rule as the object
                # detector's `inference_imgsz`). 480 ≈ half the 640 cost.
                pose_kwargs: dict = {}
                pose_imgsz = cfg["detection"].get("pose_imgsz")
                if pose_imgsz:
                    pose_kwargs["input_size"] = (int(pose_imgsz), int(pose_imgsz))
                self._person_detector = detector_registry.create(
                    "yolo_onnx_pose", onnx_path=pose_path, confidence_threshold=pose_conf,
                    **pose_kwargs,
                )
                logger.info("orchestrator: person-pose detector enabled (%s)", pose_path)
            except Exception as exc:
                logger.warning("orchestrator: person-pose detector disabled (%s)", exc)
                self._person_detector = None

        # Homography layer.
        hg_cfg = cfg.get("homography", {})
        self._projector = FootProjector(self._rig)
        self._fusion = CrossCamFusion(**hg_cfg.get("cross_cam_fusion", {}))
        self._disagreement_gate = DisagreementGate(**hg_cfg.get("disagreement_gate", {}))
        tracker_cfg = dict(hg_cfg.get("tracker", {"plugin": "bytetrack"}))
        tracker_plugin = tracker_cfg.pop("plugin")
        track_config_cfg = hg_cfg.get("track_config", {})
        if track_config_cfg:
            tracker_cfg.setdefault("track_config", TrackConfig(**track_config_cfg))
        self._tracker = tracker_registry.create(tracker_plugin, **tracker_cfg)
        if not isinstance(self._tracker, ByteTrackMeters):
            # The stabilizer reads internal-track state; tracker plugins other
            # than ByteTrackMeters would need their own stabilizer adapter.
            raise NotImplementedError(
                f"Stabilizer only supports ByteTrackMeters; got {type(self._tracker).__name__}"
            )
        self._stabilizer = TemporalStabilizer(self._tracker, **hg_cfg.get("stabilizer", {}))

        # Pallet occupancy (empty/full KPI): enriches pallet tracks with a voted
        # occupancy state from the same per-frame detections (A image overlap + B
        # metric margin fusion). Tunables live under `homography.occupancy`.
        self._occupancy = PalletOccupancy(self._projector, **hg_cfg.get("occupancy", {}))

        # Triangulation layer — Mode 2 only. Mode 1 has no second camera, so
        # the entire 3D stack is meaningless and we skip its instantiation.
        if self._mode == "dual_cam_homography_triangulation":
            tri_cfg = cfg.get("triangulation", {})
            triangulator_plugin = tri_cfg.get("plugin", "opencv_dlt")
            self._triangulator = triangulator_registry.create(triangulator_plugin, rig=self._rig)
            self._associator = KeypointAssociator(self._rig, self._projector)
            gate_kwargs = tri_cfg.get("reprojection_gate", {})
            self._reproj_gate = ReprojectionGate(self._rig, **gate_kwargs)
            track_3d_cfg = tri_cfg.get("tracker_3d", {})
            self._tracker_3d = (
                Tracker3D(Track3DConfig(**track_3d_cfg)) if track_3d_cfg else Tracker3D()
            )
            # Confidence stamped on a single-view (Z=0 floor fallback) Track3D —
            # lower than a real 2-view triangulation. Tunable; default 0.5.
            self._single_view_confidence = float(tri_cfg.get("single_view_confidence", 0.5))
        else:
            self._triangulator = None
            self._associator = None
            self._reproj_gate = None
            self._tracker_3d = None
            self._single_view_confidence = 0.5

        # Metadata.
        meta_cfg = cfg.get("metadata", {})
        self._area: str = meta_cfg.get("area", "")
        sinks = []
        for sink_cfg in meta_cfg.get("sinks", []):
            sink_cfg = dict(sink_cfg)
            plugin = sink_cfg.pop("plugin")
            sinks.append(metadata_sink_registry.create(plugin, **sink_cfg))
        if not sinks:
            raise ValueError(
                "backbone.yaml has no metadata.sinks — at least one sink (e.g. 'udp') is required."
            )
        self._publisher = Publisher(sinks)

        # Zone transitions (enter/leave events). Enabled by default; opt-out via
        # ``metadata.passings.enabled: false`` in backbone.yaml.
        self._transitions = ZoneTransitionDetector(self._zones)
        self._passings_enabled: bool = bool(
            meta_cfg.get("passings", {}).get("enabled", True)
        )

        # Person↔object proximity (the safety/AGV distance signal). Enabled by
        # default; the horizon defaults to the SAME knob the dashboard's
        # distance lines use so wire and screen agree.
        prox_cfg = meta_cfg.get("proximity", {})
        self._proximity_enabled: bool = bool(prox_cfg.get("enabled", True))
        self._proximity_max_m = float(prox_cfg.get(
            "max_distance_m",
            cfg["detection"].get("person_pallet_max_distance_m", 6.0)))
        self._proximity_interval_s = float(prox_cfg.get("refresh_interval_s", 0.5))
        self._proximity_last_ts = 0.0
        self._proximity_had_pairs = False

        # Per-camera raw observations for display consumers (the dashboard
        # renders these instead of running its own detector — ONE perception).
        # Enabled by default; opt-out via ``metadata.observations.enabled``.
        self._observations_enabled: bool = bool(
            meta_cfg.get("observations", {}).get("enabled", True))

        # Zone state (retained per-zone object list — the WMS/FMS signal).
        # Enabled by default; opt-out via ``metadata.zone_state.enabled: false``.
        zone_state_cfg = meta_cfg.get("zone_state", {})
        if zone_state_cfg.get("enabled", True):
            self._zone_state: ZoneStateTracker | None = ZoneStateTracker(
                self._zones,
                refresh_interval_s=float(zone_state_cfg.get("refresh_interval_s", 1.0)),
            )
        else:
            self._zone_state = None

        # Image snapshots on zone-passing events (Phase C). Opt-in via
        # ``metadata.images.enabled: true`` in backbone.yaml. JPEG bytes go to
        # disk only; the published message carries the URL, never raw bytes.
        images_cfg = meta_cfg.get("images", {})
        if images_cfg.get("enabled", False):
            self._snapshot_writer: SnapshotWriter | None = SnapshotWriter(
                out_dir=str(images_cfg["out_dir"]),
                url_base=str(images_cfg.get("url_base", "file://")),
                jpeg_quality=int(images_cfg.get("jpeg_quality", 85)),
            )
            # ``on_event`` (not ``on``): PyYAML 1.1 parses a bare ``on`` key as
            # the boolean True, which round-trips to a ``true:`` key and corrupts
            # the config. The trigger value is one of "enter" / "leave" / "both".
            self._images_on: str = str(images_cfg.get("on_event", "enter"))
            logger.info(
                "orchestrator: snapshot-writer enabled → out_dir=%s, on_event=%s",
                images_cfg["out_dir"],
                self._images_on,
            )
        else:
            self._snapshot_writer = None
            self._images_on = "enter"

        # Latency / frame counters. ``_frames_by_camera`` counts frames as they
        # arrive from each source (the true per-camera capture rate, before
        # synchronisation) — the diagnostics heartbeat diffs it into per-camera
        # fps for the operator STATUS panel.
        self._latency_total = LatencyMeter("capture_to_publish", window=2048)
        self._frame_count = 0
        self._frames_by_camera: dict[str, int] = {cam_id: 0 for cam_id in self._sources}

        # DiagnosticsPublisher — optional, enabled by default.
        diag_cfg = meta_cfg.get("diagnostics", {})
        # Store the gate threshold so _build_config_message uses the SAME value as
        # the heartbeat; both must agree on what "rms_ok" means.
        self._rms_gate_px: float = float(diag_cfg.get("rms_gate_px", 2.0))
        if diag_cfg.get("enabled", True):
            self._diagnostics: DiagnosticsPublisher | None = DiagnosticsPublisher(
                self,
                self._publisher,
                node_id=self._node_id,
                interval_sec=float(diag_cfg.get("interval_sec", 5.0)),
                rms_gate_px=self._rms_gate_px,
            )
        else:
            self._diagnostics = None

        # Threads (created lazily in run).
        self._ingestion_threads: list[threading.Thread] = []
        self._pipeline_thread: threading.Thread | None = None

    # ---- exposed for the latency probe and tests ----

    @property
    def latency_meter(self) -> LatencyMeter:
        return self._latency_total

    @property
    def mode(self) -> str:
        """``"single_cam_homography"`` (Mode 1) or ``"dual_cam_homography_triangulation"`` (Mode 2)."""
        return self._mode

    @property
    def source_status(self) -> dict[str, str]:
        """Per-source liveness: ``"alive"`` / ``"exited"`` / ``"crashed"``."""
        return dict(self._source_status)

    @property
    def rig(self) -> CameraRig:
        return self._rig

    @property
    def publisher(self) -> Publisher:
        return self._publisher

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def frames_by_camera(self) -> dict[str, int]:
        """Frames ingested per camera since start (source rate, pre-sync)."""
        return dict(self._frames_by_camera)

    @property
    def zone_count(self) -> int:
        """Number of zones loaded from ``zones_path``."""
        return len(self._zones)

    @property
    def subscription_count(self) -> int:
        """Number of active triangulation subscription rules."""
        return len(self._subscriptions.rules)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def _build_config_message(self) -> ConfigMessage:
        """Build a ``ConfigMessage`` from current orchestrator state for the retained MQTT advert."""
        cam_views = self._rig.items()
        if cam_views:
            rms_ok = all(v.reprojection_rms_px <= self._rms_gate_px for v in cam_views.values())
        else:
            rms_ok = False
        cal_mode = 1 if self._mode == "single_cam_homography" else 2
        calibration = CalibrationFactCheck(loaded=True, rms_ok=rms_ok, mode=cal_mode)

        zone_specs = []
        for name in self._zones.names:
            z = self._zones[name]
            zone_specs.append(
                ZoneSpec(
                    name=z.name,
                    kind=z.kind,
                    type=z.type,
                    severity=z.severity,
                    polygon=z.polygon.tolist(),
                )
            )

        return ConfigMessage(
            ts=now(),
            node_id=self._node_id,
            area=self._area,
            mode=self._mode,
            cameras=list(self._rig.camera_ids),
            zones=zone_specs,
            calibration=calibration,
        )

    # ---- run / shutdown ----

    def run(self) -> None:
        """Start ingestion + pipeline threads. Blocks until stop_event is set."""
        for src in self._sources.values():
            if hasattr(src, "start"):
                src.start()
            t = threading.Thread(
                target=self._ingestion_loop,
                args=(src,),
                name=f"ingest-{src.camera_id}",
                daemon=True,
            )
            t.start()
            self._ingestion_threads.append(t)

        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            name="pipeline",
            daemon=True,
        )
        self._pipeline_thread.start()

        # Publish the retained config advertisement and start the heartbeat.
        try:
            self._publisher.publish_config(self._build_config_message())
        except Exception:
            logger.warning("orchestrator: failed to publish config advertisement", exc_info=True)
        # Retained empty state for every configured zone, so the whole zone/
        # folder is discoverable by subscription before anything moves.
        if self._zone_state is not None:
            try:
                for state in self._zone_state.initial_states(now()):
                    self._publisher.publish_zone_state(ZoneStateMessage.from_state(state))
            except Exception:
                logger.warning("orchestrator: failed to publish initial zone states", exc_info=True)
        if self._diagnostics is not None:
            self._diagnostics.start()

        # Re-assert OUR signal disposition now that every pipeline is up:
        # loading the GStreamer nvcodec elements (CUDA context creation inside
        # the source threads) was observed to clobber the process's SIGTERM
        # handler installed before run() — the Python handler then never fired
        # and the process hung in the wait below until the supervisor's
        # SIGKILL (observed live with decoder=nvdec; software decode was
        # unaffected). Re-installing after startup restores clean shutdown.
        if self._signals_installed:
            self.install_signal_handlers()

        # Wait for shutdown. Polling wait (not a bare wait()) so the main
        # thread returns to the bytecode loop regularly — Python signal
        # handlers can only run there.
        while not self._stop_event.wait(timeout=0.5):
            pass
        self._shutdown()

    def request_shutdown(self) -> None:
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: FrameType | None) -> None:
            logger.info("orchestrator: caught signal %d, shutting down", signum)
            self.request_shutdown()
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        self._signals_installed = True

    def _shutdown(self) -> None:
        # Stop sources IN PARALLEL — each stop() joins its GStreamer thread
        # (bounded), so a sequential loop over N cameras multiplies the wall
        # time. Parallel stoppers make it max(join), not sum: the supervisor's
        # SIGTERM grace (~3.5 s) must comfortably contain the whole teardown
        # or the process gets SIGKILLed before the clean MQTT disconnect.
        def _stop_source(src: Any) -> None:
            try:
                src.stop()
            except Exception:
                logger.warning("source %s failed to stop cleanly",
                               src.camera_id, exc_info=True)

        stoppers = [
            threading.Thread(target=_stop_source, args=(src,), daemon=True,
                             name=f"stop-{cam_id}")
            for cam_id, src in self._sources.items()
        ]
        for t in stoppers:
            t.start()
        for t in stoppers:
            t.join(timeout=2.0)
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=2.0)
        if self._diagnostics is not None:
            self._diagnostics.stop()
        self._publisher.close()
        logger.info("orchestrator: shutdown complete; %d frames processed", self._frame_count)

    # ---- ingestion ----

    def _ingestion_loop(self, source: Any) -> None:
        """Pull frames from one source, submit them to the synchronizer.

        A per-source crash does **not** kill the pipeline: the remaining
        sources continue, and the synchronizer's degraded solo-emit path
        keeps `Track2D` flowing from the surviving cameras. systemd handles
        the case where every source dies.
        """
        cam_id = source.camera_id
        try:
            for frame in source.frames():
                if self._stop_event.is_set():
                    break
                self._frames_by_camera[cam_id] = self._frames_by_camera.get(cam_id, 0) + 1
                pair = self._sync.submit(frame)
                if pair is not None:
                    self._bus.publish(pair)
            self._source_status[cam_id] = "exited"
            logger.info("ingestion %s: source exited cleanly", cam_id)
        except Exception:
            self._source_status[cam_id] = "crashed"
            logger.error(
                "ingestion %s: source crashed; pipeline continues with the rest",
                cam_id,
                exc_info=True,
            )

    # ---- pipeline ----

    def _pipeline_loop(self) -> None:
        """Consume FramePairs from the bus and run the full processing stack."""
        while not self._stop_event.is_set():
            try:
                pair = self._bus.get_latest(timeout=0.5)   # newest pair only — never a stale backlog
            except queue.Empty:
                continue
            try:
                self.step(pair)
            except Exception:
                logger.error("pipeline step crashed; stopping", exc_info=True)
                self._stop_event.set()
                return

    def _scale_detections_to_calibration(self, detections_by_camera, pair) -> None:
        """Map detections from INGEST-frame pixels to CALIBRATION-frame pixels.

        Ingestion may deliver downscaled frames (per-source ``output_wh``) to cut
        decode/convert/copy cost, but every geometric consumer downstream —
        FootProjector (H), KeypointAssociator (triangulation), occupancy — is
        calibrated at the camera's native resolution. This is the single scale
        boundary: past here, all pixel coordinates are calibration-frame. No-op
        (and near-zero cost) when frames are already at calibration size.
        """
        for cam_id, dets in detections_by_camera.items():
            frame = pair.frames.get(cam_id)
            if frame is None or not dets or cam_id not in self._rig:
                continue
            fh, fw = frame.image.shape[:2]
            cw, ch = self._rig[cam_id].image_size_wh
            if (int(fw), int(fh)) == (int(cw), int(ch)):
                continue
            sx, sy = cw / float(fw), ch / float(fh)
            for d in dets:
                x0, y0, x1, y1 = d.bbox_xyxy
                d.bbox_xyxy = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                u, v = d.foot_uv
                d.foot_uv = (u * sx, v * sy)
                if d.keypoints_uv is not None:          # (K, 3) = u, v, conf
                    kp = np.asarray(d.keypoints_uv, dtype=np.float64).copy()
                    kp[:, 0] *= sx
                    kp[:, 1] *= sy
                    d.keypoints_uv = kp
                if d.mask is not None:
                    # Keep the mask in the SAME space as the bbox — occupancy
                    # compares mask/bbox areas across detections. Masks may be
                    # CROP-relative (zone scope): scale the array by the same
                    # factors and shift the crop origin; a full-frame mask
                    # (offset None) scales to the calibration frame exactly
                    # as before.
                    import cv2
                    mh, mw = d.mask.shape[:2]
                    d.mask = cv2.resize(
                        d.mask.astype(np.uint8),
                        (max(1, round(mw * sx)), max(1, round(mh * sy))),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    if d.mask_offset_xy is not None:
                        d.mask_offset_xy = (round(d.mask_offset_xy[0] * sx),
                                            round(d.mask_offset_xy[1] * sy))

    def _publish_observations(self, detections_by_camera: dict, pair: Any) -> None:
        """Per-camera raw detections for display consumers (UDP sink only).

        ONE perception: the dashboard renders these instead of running its own
        detector. Coordinates are CALIBRATION-frame pixels (``frame_wh`` says
        so on the wire); pallet detections carry the same occupancy verdict the
        tracker enrichment uses; masks (when decoded) travel as simplified
        polygons, never bitmaps.
        """
        from backbone.homography.pallet_occupancy import PALLET_CLASSES

        for cam_id, dets in detections_by_camera.items():
            if cam_id not in self._rig:
                continue
            pallets = [d for d in dets if str(d.cls).lower() in PALLET_CLASSES]
            occ_by_det: dict[int, tuple] = {}
            if pallets:
                try:
                    states = self._occupancy.frame_states(dets)
                    for pal, (_pm, st, content, conf) in zip(pallets, states, strict=False):
                        occ_by_det[id(pal)] = (st, content, conf)
                except Exception:
                    logger.debug("observations: occupancy hint failed", exc_info=True)
            obs = []
            for d in dets:
                st, content, conf = occ_by_det.get(id(d), (None, None, 0.0))
                poly = (mask_to_polygon(d.mask, d.mask_offset_xy)
                        if d.mask is not None else None)
                obs.append(ObservationDet(
                    cls=str(d.cls), confidence=float(d.confidence),
                    bbox_xyxy=tuple(float(v) for v in d.bbox_xyxy),
                    foot_uv=(float(d.foot_uv[0]), float(d.foot_uv[1])),
                    occupancy_state=st, occupancy_content=content,
                    occupancy_confidence=float(conf or 0.0),
                    mask_poly=tuple((float(x), float(y)) for x, y in poly)
                              if poly else None,
                ))
            w, h = self._rig[cam_id].image_size_wh
            self._publisher.publish_observations(ObservationsMessage(
                ts=pair.capture_ts, camera_id=cam_id,
                frame_wh=(int(w), int(h)), dets=tuple(obs)))

    def step(self, pair: Any) -> tuple[list[Track2D], list[Any]]:
        """Process one ``FramePair`` end-to-end. Returns the published 2D + 3D tracks.

        Exposed so tests can drive the pipeline synchronously without
        spinning up threads.
        """
        # --- detection ---
        if self._detector is not None:
            detections_by_camera = self._detector.detect(pair)
            self._scale_detections_to_calibration(detections_by_camera, pair)
            if self._observations_enabled:
                try:
                    self._publish_observations(detections_by_camera, pair)
                except Exception:
                    logger.warning("orchestrator: observations publish failed",
                                   exc_info=True)
        else:
            # Zone-based system, no zones configured: no object detection at
            # all — the pose path below still produces person tracks.
            detections_by_camera = {cid: [] for cid in pair.frames}
        all_detections = [d for dets in detections_by_camera.values() for d in dets]
        # Person-pose runs as a second detector; its detections join the homography
        # path (foot → floor → Track2D) but NOT the triangulation association below
        # (that stays object-detector-only via `detections_by_camera`).
        # `pose_every_n` > 1 amortises the pose model across pairs — person tracks
        # coast on their Kalman prediction between pose frames (ByteTrack tolerates
        # `max_lost_frames` misses), buying the object pipeline its frame budget.
        if (self._person_detector is not None
                and self._frame_count % self._pose_every_n == 0):
            try:
                pose_by_camera = self._person_detector.detect(pair)
                self._scale_detections_to_calibration(pose_by_camera, pair)
                for dets in pose_by_camera.values():
                    all_detections.extend(dets)
            except Exception:
                logger.warning("orchestrator: person detection failed", exc_info=True)

        # --- homography ---
        floor_pairs = self._projector.project_batch(all_detections)
        fused = self._fusion.fuse(floor_pairs)
        gated = self._disagreement_gate.check(fused)
        observations = [
            (o.cls, o.xy_m, o.confidence, o.cameras_seeing) for o in gated
        ]
        raw_tracks = self._tracker.update(pair.capture_ts, observations)
        tracks_2d = self._stabilizer.stabilize(raw_tracks)

        # --- pallet occupancy enrichment (empty/full) ---
        # Uses this frame's detections (with masks/bboxes) to classify each pallet
        # track; sets occupancy_* on the Track2D before it's published.
        self._occupancy.enrich(tracks_2d, detections_by_camera)

        # --- publish 2D + zone transitions + zone state ---
        # Zone membership is computed ONCE per track and shared by the
        # transition detector (events) and the state tracker (absolute state).
        need_membership = self._passings_enabled or self._zone_state is not None
        memberships: dict[int, tuple[str, ...]] = {}
        for track in tracks_2d:
            self._publisher.publish_track_2d(track)
            if not need_membership:
                continue
            membership = self._zones.which(track.xy_m)
            memberships[track.track_id] = membership
            if self._passings_enabled:
                for ev in self._transitions.update(
                    track.track_id, track.cls, track.xy_m, pair.capture_ts,
                    membership=membership,
                ):
                    self._publisher.publish_event(ev)
                    # Snapshot on enter/leave/both — JPEG written to disk only;
                    # bus message carries the URL, never raw bytes.
                    if self._snapshot_writer is not None and (
                        self._images_on == "both"
                        or self._images_on == ev.direction
                    ):
                        cam_id = next(iter(pair.frames))
                        image = pair.frames[cam_id].image
                        url = self._snapshot_writer.write(
                            image, ev.track_id, ev.zone, ev.ts
                        )
                        if url is not None:
                            self._publisher.publish_image_ref(
                                ev.track_id, ev.cls, ev.zone, ev.ts, url
                            )
        if self._passings_enabled:
            self._transitions.forget({t.track_id for t in tracks_2d})

        # --- zone state (retained per-zone object list) ---
        if self._zone_state is not None:
            zone_states = self._zone_state.update(
                [(t, memberships.get(t.track_id, ())) for t in tracks_2d],
                pair.capture_ts,
            )
            for state in zone_states:
                self._publisher.publish_zone_state(ZoneStateMessage.from_state(state))

        # --- person↔object proximity (safety/AGV distance on the wire) ---
        # Floor distances between every person track and every object track
        # within the horizon; published throttled + retained on MQTT. An
        # explicit empty message clears the topic when the last pair leaves.
        if (self._proximity_enabled
                and pair.capture_ts - self._proximity_last_ts >= self._proximity_interval_s):
            self._proximity_last_ts = pair.capture_ts
            persons = [t for t in tracks_2d if t.cls == "person"]
            objects = [t for t in tracks_2d if t.cls != "person"]
            prox_pairs = []
            for person in persons:
                px, py = person.xy_m
                for obj in objects:
                    ox, oy = obj.xy_m
                    dist = float(np.hypot(px - ox, py - oy))
                    if dist <= self._proximity_max_m:
                        prox_pairs.append(ProximityPair(
                            person_track_id=person.track_id,
                            object_track_id=obj.track_id,
                            object_cls=obj.cls,
                            distance_m=round(dist, 3),
                            person_xy_m=(px, py),
                            object_xy_m=(ox, oy),
                        ))
            if prox_pairs or self._proximity_had_pairs:
                self._publisher.publish_proximity(ProximityMessage(
                    ts=pair.capture_ts,
                    max_distance_m=self._proximity_max_m,
                    pairs=tuple(prox_pairs),
                ))
            self._proximity_had_pairs = bool(prox_pairs)

        # --- triangulation (Mode 2 only; subscription-driven) ---
        tracks_3d: list[Any] = []
        if self._mode == "dual_cam_homography_triangulation":
            subscribed = self._subscriptions.filter(tracks_2d, reference_ts=pair.capture_ts)
            for track, rule in subscribed:
                obs_uv = self._associator.resolve_foot_uv(track, detections_by_camera)
                track_3d = None
                if len(obs_uv) >= 2:
                    # Real 2-view triangulation — reprojection-gated.
                    xyz = self._triangulator.triangulate_point(obs_uv)
                    if xyz is None:
                        continue
                    if not self._reproj_gate.check(xyz, obs_uv):
                        continue
                    track_3d = self._tracker_3d.update(
                        track_id=track.track_id,
                        xyz_obs=xyz,
                        capture_ts=pair.capture_ts,
                        cameras_seeing=tuple(obs_uv.keys()),
                        cls=track.cls,
                        max_reproj_error_px=self._reproj_gate.last_max_error_px,
                    )
                elif len(obs_uv) == 1 and rule.match.allow_single_view:
                    # Single-view floor fallback (occlusion): promote the 2D floor
                    # position to (X, Y, 0) so 3D stays continuous through the gap,
                    # flagged single_view + downgraded confidence. No triangulation
                    # → no reprojection error.
                    xyz = (float(track.xy_m[0]), float(track.xy_m[1]), 0.0)
                    track_3d = self._tracker_3d.update(
                        track_id=track.track_id,
                        xyz_obs=xyz,
                        capture_ts=pair.capture_ts,
                        cameras_seeing=tuple(obs_uv.keys()),
                        cls=track.cls,
                        max_reproj_error_px=0.0,
                        single_view=True,
                        confidence=self._single_view_confidence,
                    )
                if track_3d is None:
                    continue
                tracks_3d.append(track_3d)
                self._publisher.publish_track_3d(track_3d)

            # GC 3D Kalmans whose 2D parent disappeared.
            self._tracker_3d.gc({t.track_id for t in tracks_2d})

        # Latency bookkeeping — measured against the SINGLE capture-time clock.
        self._latency_total.record_ms(elapsed_ms(pair.capture_ts))
        self._frame_count += 1
        return tracks_2d, tracks_3d


__all__ = ["Orchestrator"]
