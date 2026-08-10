"""``python -m isistream --config config/backbone.yaml`` — standalone producer.

The headless topology: isistream runs as its own service (its own RTSP
sources via the Backbone's FrameSource plugins), the Backbone runs as the
metric engine (``ingestion.mode: points``), and the dashboard is optional —
a pure consumer. Example systemd pairing:

    # /etc/systemd/system/isistream.service
    [Service]
    ExecStart=/path/to/env/bin/python -m isistream --config /path/to/backbone.yaml
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
    parser = argparse.ArgumentParser(description="isistream — the perception producer")
    parser.add_argument("--config", required=True, help="path to backbone.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logger = logging.getLogger("isistream.main")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    if str(cfg.get("ingestion", {}).get("mode", "frames")) != "points":
        raise SystemExit(
            "isistream: backbone.yaml has ingestion.mode != 'points' — the "
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

    # `isistream.detect_substream: false` ignores the per-camera detect_source
    # blocks (detection back on the main stream) without losing their URLs.
    # (pre-rename `perception:` key still reads.)
    isis_cfg = cfg.get("isistream", cfg.get("perception", {})) or {}
    # Compressed video passthrough: tee each RTSP camera's ORIGINAL
    # H.264/H.265 bitstream (post-depay, pre-decode) and serve it on a
    # per-camera unix socket (isistream/nal_relay.py). The dashboard relays
    # it to browsers for hardware decode — no per-frame JPEG encode on the
    # display path. Relays are keyed per camera and OUTLIVE source rebuilds:
    # a reconnect keeps clients connected; the new stream resumes at its
    # next keyframe naturally (relay clients gate on keyframes anyway).
    video_passthrough = bool(isis_cfg.get("video_passthrough", True))
    relays: dict = {}
    bus_writers: dict = {}   # cam_id → FrameShmWriter (guarded main-thread unlink)

    def pump(cam_id: str, src_cfg_full: dict, *, feed_detect: bool = True,
             feed_bus: bool = True) -> None:
        """One reconnecting capture loop. Default: one stream feeds BOTH the
        detection slot and the display frame bus. With a per-camera
        ``detect_source`` (the camera's SUBSTREAM), two pumps split the
        duties: the substream feeds detection (smaller frames → cheaper
        preprocessing), the main stream feeds only the frame bus (full-detail
        display). Detections stay correct either way — DetectionSetMessage
        declares its own frame_wh and the engine's scale boundary maps any
        resolution to calibration pixels.
        """
        backoff = 2.0
        # The shared frame bus: every decoded frame is published to
        # /dev/shm so display consumers (the dashboard's camera hub) read
        # THIS decode instead of opening a second RTSP session per camera —
        # one ingest, one decode, DeepStream-style fan-out.
        from backbone.shared.frame_shm import FrameShmWriter
        bus_writer = FrameShmWriter(cam_id) if feed_bus else None
        if bus_writer is not None:
            bus_writers[cam_id] = bus_writer
        # Bitstream tap: DISPLAY pump only (feed_bus), rtsp plugin only (raw
        # sources — v4l2/replay — have no compressed bitstream to tee). The
        # NalRelay is created lazily on the first tapped AU because the codec
        # is only known after the source's probe (src.nal_codec at start()).
        nal_tap = None
        if video_passthrough and feed_bus and src_cfg_full.get("name") == "rtsp":
            from isistream.nal_relay import NalRelay
            src_holder: list = [None]
            tap_disabled = [False]

            def nal_tap(au: bytes, capture_ts: float, keyframe: bool) -> None:
                if tap_disabled[0]:
                    return
                relay = relays.get(cam_id)
                if relay is None:
                    codec = getattr(src_holder[0], "nal_codec", None) or "h264"
                    try:
                        relay = NalRelay(cam_id, codec)
                    except OSError:
                        logger.warning(
                            "isistream: %s NAL relay bind failed — video "
                            "passthrough disabled for this camera", cam_id,
                            exc_info=True)
                        tap_disabled[0] = True
                        return
                    relays[cam_id] = relay
                relay.push(au, capture_ts, keyframe)
        while not stop.is_set():
            src = None
            streamed_since = None
            try:
                src_cfg = dict(src_cfg_full)
                plugin = src_cfg.pop("name")
                if nal_tap is not None:
                    src_cfg["nal_tap"] = nal_tap
                src = frame_source_registry.create(
                    plugin, camera_id=cam_id, **src_cfg)
                if nal_tap is not None:
                    src_holder[0] = src
                if hasattr(src, "start"):
                    src.start()
                import time as _time
                streamed_since = _time.monotonic()
                for frame in src.frames():
                    if stop.is_set():
                        break
                    if feed_detect:
                        with lock:
                            latest[cam_id] = (frame.image, frame.capture_ts)
                    if bus_writer is not None:
                        try:
                            bus_writer.write(frame.image, frame.capture_ts)
                        except Exception:
                            logger.debug("isistream: %s frame-bus write failed",
                                         cam_id, exc_info=True)
                if not stop.is_set():
                    logger.warning("isistream: %s stream ended (EOS) — reconnecting", cam_id)
            except Exception:
                logger.warning("isistream: source %s error — reconnecting", cam_id,
                               exc_info=True)
            finally:
                if src is not None and hasattr(src, "stop"):
                    try:
                        src.stop()
                    except Exception:
                        logger.debug("isistream: %s stop failed", cam_id, exc_info=True)
            if stop.is_set():
                break
            import time as _time
            if streamed_since is not None and _time.monotonic() - streamed_since > 30.0:
                backoff = 2.0        # a healthy run resets the backoff
            stop.wait(backoff)
            backoff = min(backoff * 2.0, 15.0)
        # Deliberate shutdown: unlink the bus so readers see 'absent'
        # immediately instead of waiting out the staleness window.
        if bus_writer is not None:
            bus_writer.unlink()

    def frame_provider(camera_id: str):
        with lock:
            return latest.get(camera_id)

    from isistream import build_isistream_core
    core = build_isistream_core(cfg, frame_provider, producer_id="isistream")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    use_substream = bool(isis_cfg.get("detect_substream", True))
    for cam_id, cam_cfg in cfg["cameras"].items():
        detect_src = cam_cfg.get("detect_source") if use_substream else None
        if detect_src:
            # Substream split: detection reads the camera's SUBSTREAM (e.g.
            # 704p — half the preprocessing per frame), the frame bus serves
            # the MAIN stream for full-detail display.
            logger.info("isistream: %s split streams — detect on substream, "
                        "display on main", cam_id)
            threading.Thread(target=pump,
                             args=(cam_id, dict(detect_src)),
                             kwargs={"feed_detect": True, "feed_bus": False},
                             daemon=True, name=f"pump-{cam_id}-detect").start()
            threading.Thread(target=pump,
                             args=(cam_id, dict(cam_cfg["source"])),
                             kwargs={"feed_detect": False, "feed_bus": True},
                             daemon=True, name=f"pump-{cam_id}-display").start()
        else:
            threading.Thread(target=pump, args=(cam_id, dict(cam_cfg["source"])),
                             daemon=True, name=f"pump-{cam_id}").start()
    core.start()
    logger.info("isistream: standalone producer up (%d cameras)", len(cfg["cameras"]))
    try:
        while not stop.wait(timeout=0.5):
            pass
    finally:
        core.stop()
        # Unlink the frame bus from the MAIN thread — the daemon pumps never
        # reach their own cleanup on interpreter exit, and readers should see
        # 'absent' instantly instead of waiting out the staleness window.
        # Through the WRITER's inode-guarded unlink, never a raw os.unlink:
        # an overlapping restart otherwise deletes the SUCCESSOR instance's
        # live bus and its readers see 'absent' forever (the double-RTSP
        # fallback observed on the rig).
        for writer in bus_writers.values():
            try:
                writer.unlink()
            except Exception:
                pass
        # Same deal for the NAL relays: close from the MAIN thread so the
        # unix-socket paths are unlinked on deliberate shutdown.
        for relay in relays.values():
            try:
                relay.close()
            except Exception:
                logger.debug("isistream: relay close failed", exc_info=True)
        logger.info("isistream: standalone producer stopped")


if __name__ == "__main__":
    main()
