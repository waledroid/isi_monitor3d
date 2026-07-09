"""In-process host for the perception producer (Direction 1, dev-box topology).

When ``backbone.yaml`` says ``ingestion.mode: points``, the Backbone is a
pure metric engine and SOMEONE must produce detections. On the dev box that
someone is the dashboard: this manager builds a ``perception.PerceptionCore``
backed by the shared camera hub (ONE RTSP decode per camera, shared with the
display panels) and runs it while the Backbone runs — started after a
successful START, stopped on STOP.

Headless deployments run ``python -m perception`` as its own service instead;
this module is glue, not logic — the loop lives in the FastAPI-free
``perception`` package.
"""

from __future__ import annotations

import logging
import threading

import yaml

from .camera_hub import get_hub

logger = logging.getLogger(__name__)


class PerceptionHost:
    """Lifecycle wrapper: reads backbone.yaml, holds hub acquisitions, and
    starts/stops the core with the Backbone."""

    def __init__(self, backbone_config_path) -> None:
        self._config_path = backbone_config_path
        self._core = None
        self._streams: list = []
        self._lock = threading.Lock()

    @staticmethod
    def _load_cfg(path) -> dict:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}

    def points_mode(self) -> bool:
        try:
            cfg = self._load_cfg(self._config_path)
            return str(cfg.get("ingestion", {}).get("mode", "frames")) == "points"
        except Exception:
            return False

    def start(self) -> bool:
        """Start the producer if (and only if) the config says points mode.
        Returns True when a producer is running after the call."""
        with self._lock:
            if self._core is not None and self._core.running:
                return True
            try:
                cfg = self._load_cfg(self._config_path)
                if str(cfg.get("ingestion", {}).get("mode", "frames")) != "points":
                    return False

                # Hold a hub reader per camera for the producer's lifetime —
                # the SAME decoded stream the display panels use.
                hub = get_hub()
                streams = {}
                for cam_id, cam_cfg in cfg.get("cameras", {}).items():
                    src = dict(cam_cfg.get("source", {}))
                    plugin = src.pop("name", "rtsp")
                    streams[cam_id] = hub.acquire(cam_id, plugin, src)
                self._streams = list(streams.values())

                def frame_provider(camera_id: str):
                    stream = streams.get(camera_id)
                    return stream.latest_real_frame_with_ts() if stream else None

                from perception import build_perception_core
                self._core = build_perception_core(
                    cfg, frame_provider, producer_id="monitor_web")
                self._core.start()
                logger.info("perception host: producer RUNNING (in-process, hub-backed)")
                return True
            except Exception:
                logger.warning("perception host: start failed", exc_info=True)
                self._release_streams()
                self._core = None
                return False

    def stop(self) -> None:
        with self._lock:
            core, self._core = self._core, None
            if core is not None:
                try:
                    core.stop()
                except Exception:
                    logger.warning("perception host: stop failed", exc_info=True)
            self._release_streams()

    def _release_streams(self) -> None:
        hub = get_hub()
        for stream in self._streams:
            try:
                hub.release(stream)
            except Exception:
                logger.debug("perception host: stream release failed", exc_info=True)
        self._streams = []

    def status(self) -> dict:
        core = self._core
        if core is None:
            return {"running": False}
        return {
            "running": core.running,
            "sets_sent": dict(core.sets_sent),
            "last_tick_ms": round(core.last_tick_ms, 1),
            "last_error": core.last_error,
        }
