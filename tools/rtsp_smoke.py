"""Manual smoke test for ``RtspFrameSource`` against a real RTSP URL.

Usage:
    python tools/rtsp_smoke.py rtsp://192.168.1.10/Streaming/Channels/102
    python tools/rtsp_smoke.py rtsp://... --latency 100 --frames 30 --camera-id cam_a

Prints, for each frame:
    frame_idx, shape, capture_ts, latency_ms (= now - capture_ts)

The latency reading is the *Backbone-observable* delay from the appsink to
the moment we get the frame back in Python. With the conda env's GStreamer
pipeline and a 100 ms RTSP latency buffer, expect single-digit ms after
the first frame.
"""

from __future__ import annotations

import argparse
import sys
import time

from backbone.ingestion.rtsp import RtspFrameSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="rtsp:// URL")
    parser.add_argument("--camera-id", default="smoke")
    parser.add_argument("--latency", type=int, default=100, help="rtspsrc latency (ms)")
    parser.add_argument("--frames", type=int, default=10, help="how many frames to print")
    parser.add_argument(
        "--startup-timeout", type=float, default=15.0,
        help="seconds to wait for the pipeline to reach PLAYING",
    )
    args = parser.parse_args(argv)

    src = RtspFrameSource(
        camera_id=args.camera_id,
        url=args.url,
        latency_ms=args.latency,
        startup_timeout_s=args.startup_timeout,
    )
    print(f"[smoke] starting pipeline for {args.url}", flush=True)
    src.start()
    print("[smoke] PLAYING. Pulling frames...", flush=True)

    try:
        seen = 0
        for frame in src.frames():
            now = time.time()
            latency_ms = (now - frame.capture_ts) * 1000.0
            print(
                f"[smoke] idx={frame.frame_idx:>4d}  shape={frame.image.shape}  "
                f"capture_ts={frame.capture_ts:.3f}  latency={latency_ms:6.1f} ms  "
                f"dropped_since_start={src.dropped_count}",
                flush=True,
            )
            seen += 1
            if seen >= args.frames:
                break
    except KeyboardInterrupt:
        print("[smoke] interrupted", flush=True)
    finally:
        src.stop()
        print("[smoke] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
