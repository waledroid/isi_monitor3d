"""Capture-to-publish latency probe.

Two modes:

1. **Online:** run the orchestrator against the configured pipeline (real
   RTSP or replay), print p50 / p95 / p99 of ``time.time() - capture_ts``
   on a fixed cadence. KPI: p95 < 200 ms.

2. **Offline-instrument:** for already-running pipelines, listen on the UDP
   metadata channel and compute the same latency over received envelopes.
   The envelope's ``ts`` is the propagated ``capture_ts``.

The "single capture-time clock" principle means the only legitimate latency
measurement is ``time.time() - <capture_ts from envelope>``. Anything else
silently lies about the system's responsiveness.

Usage:
    # Online — run the orchestrator and probe in-process.
    python tools/latency_probe.py online --config config/backbone.yaml --seconds 60

    # Offline — subscribe to a running Backbone's UDP feed.
    python tools/latency_probe.py listen --host 0.0.0.0 --port 9001 --seconds 60
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from backbone.shared.timestamps import LatencyMeter


def _report(meter: LatencyMeter) -> None:
    p = meter.percentiles()
    print(
        f"[{meter.name}] n={p['n']:>5d}  p50={p['p50']:7.2f} ms  "
        f"p95={p['p95']:7.2f} ms  p99={p['p99']:7.2f} ms",
        flush=True,
    )


def _online(args: argparse.Namespace) -> int:
    # Lazy import — orchestrator pulls heavy deps (cv2, onnxruntime).
    from backbone.runtime import Orchestrator

    orch = Orchestrator(args.config)
    orch.install_signal_handlers()

    runner = threading.Thread(target=orch.run, daemon=True, name="orchestrator")
    runner.start()

    deadline = time.time() + args.seconds
    next_report = time.time() + args.interval
    try:
        while time.time() < deadline and not orch.stop_event.is_set():
            if time.time() >= next_report:
                _report(orch.latency_meter)
                next_report = time.time() + args.interval
            time.sleep(0.05)
    finally:
        orch.request_shutdown()
        runner.join(timeout=10.0)
    _report(orch.latency_meter)
    return 0


def _listen(args: argparse.Namespace) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)
    meter = LatencyMeter("udp_to_now", window=4096)

    deadline = time.time() + args.seconds
    next_report = time.time() + args.interval
    print(f"[listen] bound on {args.host}:{args.port}", flush=True)
    try:
        while time.time() < deadline:
            try:
                payload, _ = sock.recvfrom(8192)
            except TimeoutError:
                if time.time() >= next_report:
                    _report(meter)
                    next_report = time.time() + args.interval
                continue
            now = time.time()
            try:
                obj = json.loads(payload.decode("utf-8"))
                ts = float(obj["ts"])
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
                continue
            meter.record_ms((now - ts) * 1000.0)
            if time.time() >= next_report:
                _report(meter)
                next_report = time.time() + args.interval
    finally:
        sock.close()
    _report(meter)
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    """Read a recorded log of (capture_ts, publish_ts) pairs and report percentiles."""
    pairs: list[tuple[float, float]] = []
    for line in Path(args.path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pairs.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    latencies = np.array([(pub - cap) * 1000.0 for cap, pub in pairs])
    if latencies.size == 0:
        print("no samples in input")
        return 1
    print(
        f"n={latencies.size}  p50={np.percentile(latencies, 50):.2f} ms  "
        f"p95={np.percentile(latencies, 95):.2f} ms  "
        f"p99={np.percentile(latencies, 99):.2f} ms",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_online = sub.add_parser("online", help="run orchestrator + probe in-process")
    p_online.add_argument("--config", required=True)
    p_online.add_argument("--seconds", type=int, default=60)
    p_online.add_argument("--interval", type=float, default=5.0)
    p_online.set_defaults(func=_online)

    p_listen = sub.add_parser("listen", help="subscribe to UDP feed")
    p_listen.add_argument("--host", default="0.0.0.0")
    p_listen.add_argument("--port", type=int, default=9001)
    p_listen.add_argument("--seconds", type=int, default=60)
    p_listen.add_argument("--interval", type=float, default=5.0)
    p_listen.set_defaults(func=_listen)

    p_agg = sub.add_parser("aggregate", help="read a recorded (capture, publish) log")
    p_agg.add_argument("path")
    p_agg.set_defaults(func=_aggregate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
