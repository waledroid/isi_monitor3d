"""The single capture-time clock.

Every Frame carries ``capture_ts`` — the NTP-aligned moment the RTSP source
produced the frame, in seconds since the Unix epoch (float). All latency
measurements anywhere in the Backbone reference this clock; ``time.time()``
at the publish site is never used.

This module exposes the minimum surface needed for that discipline: the
"now" function in the same scale as ``capture_ts`` and a small percentile
collector for per-stage latency reporting.

Why this is a module of its own:
    The latency KPI (<200 ms p95, end-to-end) is meaningful only if every
    stage measures against the same reference. Routing every "now()" through
    this module prevents accidental mixing of wall-clock and monotonic-clock
    values across stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def now() -> float:
    """Return wall time in the same units as ``Frame.capture_ts``."""
    return time.time()


def elapsed_ms(capture_ts: float, reference: float | None = None) -> float:
    """Milliseconds since the given capture timestamp."""
    ref = now() if reference is None else reference
    return (ref - capture_ts) * 1000.0


@dataclass(slots=True)
class LatencyMeter:
    """Ring-buffer percentile collector for a single pipeline stage.

    Use one per stage (capture → detection → homography → publish).
    Reports p50/p95/p99 over the most recent ``window`` samples; older
    samples drop off without copying.
    """

    name: str
    window: int = 1024
    _buf: np.ndarray = field(init=False)
    _count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._buf = np.zeros(self.window, dtype=np.float32)

    def record_ms(self, value_ms: float) -> None:
        idx = self._count % self.window
        self._buf[idx] = value_ms
        self._count += 1

    def record_since(self, capture_ts: float) -> None:
        self.record_ms(elapsed_ms(capture_ts))

    @property
    def samples(self) -> int:
        return min(self._count, self.window)

    def percentiles(self) -> dict[str, float]:
        n = self.samples
        if n == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
        active = self._buf[:n]
        return {
            "p50": float(np.percentile(active, 50)),
            "p95": float(np.percentile(active, 95)),
            "p99": float(np.percentile(active, 99)),
            "n": n,
        }

    def reset(self) -> None:
        self._buf.fill(0)
        self._count = 0
