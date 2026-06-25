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
from backbone.comms.schemas import CalibrationFactCheck, ConfigMessage, ZoneSpec
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
from backbone.shared.snapshot_writer import SnapshotWriter
from backbone.shared.timestamps import LatencyMeter, elapsed_ms, now
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
        # `inference_imgsz` is the runtime input size (the dashboard slider). It maps
        # to the detector's square `input_size`; effective only on a dynamic ONNX.
        imgsz = det_cfg.pop("inference_imgsz", None)
        if imgsz:
            det_cfg["input_size"] = (int(imgsz), int(imgsz))
        self._detector = detector_registry.create(det_plugin, **det_cfg)

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
                self._person_detector = detector_registry.create(
                    "yolo_onnx_pose", onnx_path=pose_path, confidence_threshold=pose_conf,
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
            self._images_on: str = str(images_cfg.get("on", "enter"))
            logger.info(
                "orchestrator: snapshot-writer enabled → out_dir=%s, on=%s",
                images_cfg["out_dir"],
                self._images_on,
            )
        else:
            self._snapshot_writer = None
            self._images_on = "enter"

        # Latency / frame counter.
        self._latency_total = LatencyMeter("capture_to_publish", window=2048)
        self._frame_count = 0

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
        if self._diagnostics is not None:
            self._diagnostics.start()

        # Wait for shutdown.
        self._stop_event.wait()
        self._shutdown()

    def request_shutdown(self) -> None:
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: FrameType | None) -> None:
            logger.info("orchestrator: caught signal %d, shutting down", signum)
            self.request_shutdown()
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _shutdown(self) -> None:
        for src in self._sources.values():
            try:
                src.stop()
            except Exception:
                logger.warning("source %s failed to stop cleanly", src.camera_id, exc_info=True)
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=5.0)
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
                pair = self._bus.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.step(pair)
            except Exception:
                logger.error("pipeline step crashed; stopping", exc_info=True)
                self._stop_event.set()
                return

    def step(self, pair: Any) -> tuple[list[Track2D], list[Any]]:
        """Process one ``FramePair`` end-to-end. Returns the published 2D + 3D tracks.

        Exposed so tests can drive the pipeline synchronously without
        spinning up threads.
        """
        # --- detection ---
        detections_by_camera = self._detector.detect(pair)
        all_detections = [d for dets in detections_by_camera.values() for d in dets]
        # Person-pose runs as a second detector; its detections join the homography
        # path (foot → floor → Track2D) but NOT the triangulation association below
        # (that stays object-detector-only via `detections_by_camera`).
        if self._person_detector is not None:
            try:
                for dets in self._person_detector.detect(pair).values():
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

        # --- publish 2D + zone transitions ---
        for track in tracks_2d:
            self._publisher.publish_track_2d(track)
            if self._passings_enabled:
                for ev in self._transitions.update(
                    track.track_id, track.cls, track.xy_m, pair.capture_ts
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
