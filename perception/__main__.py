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
    import backbone.ingestion  # noqa: F401 — fire @register
    from backbone.core.interfaces import frame_source_registry

    latest: dict[str, tuple] = {}
    lock = threading.Lock()
    sources = []
    for cam_id, cam_cfg in cfg["cameras"].items():
        src_cfg = dict(cam_cfg["source"])
        plugin = src_cfg.pop("name")
        sources.append(frame_source_registry.create(
            plugin, camera_id=cam_id, **src_cfg))

    def pump(src) -> None:
        try:
            for frame in src.frames():
                with lock:
                    latest[src.camera_id] = (frame.image, frame.capture_ts)
        except Exception:
            logger.warning("perception: source %s died", src.camera_id,
                           exc_info=True)

    def frame_provider(camera_id: str):
        with lock:
            return latest.get(camera_id)

    from perception import build_perception_core
    core = build_perception_core(cfg, frame_provider, producer_id="perception-standalone")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    for src in sources:
        if hasattr(src, "start"):
            src.start()
        threading.Thread(target=pump, args=(src,), daemon=True,
                         name=f"pump-{src.camera_id}").start()
    core.start()
    logger.info("perception: standalone producer up (%d cameras)", len(sources))
    try:
        while not stop.wait(timeout=0.5):
            pass
    finally:
        core.stop()
        for src in sources:
            if hasattr(src, "stop"):
                try:
                    src.stop()
                except Exception:
                    pass
        logger.info("perception: standalone producer stopped")


if __name__ == "__main__":
    main()
