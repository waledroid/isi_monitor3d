"""``python -m perception --config config/backbone.yaml`` — standalone producer.

The headless topology: perception runs as its own service (its own RTSP
sources via the Backbone's FrameSource plugins), the Backbone runs as the
metric engine (``ingestion.mode: points``), and the dashboard is optional —
a pure consumer. Example systemd pairing:

    # /etc/systemd/system/isi-perception.service
    [Service]
    ExecStart=/path/to/env/bin/python -m perception --config /path/to/backbone.yaml
    Restart=always

    # /etc/systemd/system/isi-backbone.service
    [Service]
    ExecStart=/path/to/env/bin/python -m backbone.runtime --config /path/to/backbone.yaml
    Restart=always

Both read the SAME backbone.yaml, so the config fingerprint matches by
construction. SIGINT/SIGTERM stop cleanly.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="isi perception producer")
    parser.add_argument("--config", required=True, help="path to backbone.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logger = logging.getLogger("perception.main")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    if str(cfg.get("ingestion", {}).get("mode", "frames")) != "points":
        raise SystemExit(
            "perception: backbone.yaml has ingestion.mode != 'points' — the "
            "Backbone owns perception in frames mode; a standalone producer "
            "would double every model and decode. Set ingestion.mode: points.")

    # Own the cameras directly: one FrameSource per camera (same plugins the
    # Backbone uses in frames mode), latest-frame slots, capture_ts preserved.
    # Each pump RECONNECTS on EOS/error — a camera dropping its RTSP session
    # (observed on the Dahua) must mean a gap in that camera's detection sets
    # (the engine degrades, then recovers), never a dead producer. The old
    # exit-on-EOS pump left a half-torn GStreamer source running, which later
    # segfaulted the whole process.
    import backbone.ingestion  # noqa: F401 — fire @register
    from backbone.core.interfaces import frame_source_registry

    latest: dict[str, tuple] = {}
    lock = threading.Lock()
    stop = threading.Event()

    def pump(cam_id: str, src_cfg_full: dict) -> None:
        backoff = 2.0
        while not stop.is_set():
            src = None
            streamed_since = None
            try:
                src_cfg = dict(src_cfg_full)
                plugin = src_cfg.pop("name")
                src = frame_source_registry.create(
                    plugin, camera_id=cam_id, **src_cfg)
                if hasattr(src, "start"):
                    src.start()
                import time as _time
                streamed_since = _time.monotonic()
                for frame in src.frames():
                    if stop.is_set():
                        break
                    with lock:
                        latest[cam_id] = (frame.image, frame.capture_ts)
                if not stop.is_set():
                    logger.warning("perception: %s stream ended (EOS) — reconnecting", cam_id)
            except Exception:
                logger.warning("perception: source %s error — reconnecting", cam_id,
                               exc_info=True)
            finally:
                if src is not None and hasattr(src, "stop"):
                    try:
                        src.stop()
                    except Exception:
                        logger.debug("perception: %s stop failed", cam_id, exc_info=True)
            if stop.is_set():
                return
            import time as _time
            if streamed_since is not None and _time.monotonic() - streamed_since > 30.0:
                backoff = 2.0        # a healthy run resets the backoff
            stop.wait(backoff)
            backoff = min(backoff * 2.0, 15.0)

    def frame_provider(camera_id: str):
        with lock:
            return latest.get(camera_id)

    from perception import build_perception_core
    core = build_perception_core(cfg, frame_provider, producer_id="perception-standalone")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    for cam_id, cam_cfg in cfg["cameras"].items():
        threading.Thread(target=pump, args=(cam_id, dict(cam_cfg["source"])),
                         daemon=True, name=f"pump-{cam_id}").start()
    core.start()
    logger.info("perception: standalone producer up (%d cameras)", len(cfg["cameras"]))
    try:
        while not stop.wait(timeout=0.5):
            pass
    finally:
        core.stop()
        logger.info("perception: standalone producer stopped")


if __name__ == "__main__":
    main()
