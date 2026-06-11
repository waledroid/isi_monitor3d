"""``timestamps`` helpers."""

from __future__ import annotations

import time

import pytest

from backbone.shared.timestamps import LatencyMeter, elapsed_ms, now


def test_now_is_wall_clock() -> None:
    assert abs(now() - time.time()) < 0.1


def test_elapsed_ms_against_explicit_reference() -> None:
    assert elapsed_ms(100.0, reference=100.05) == pytest.approx(50.0, abs=1e-6)


def test_latency_meter_empty() -> None:
    m = LatencyMeter("test")
    p = m.percentiles()
    assert p["n"] == 0
    assert p["p50"] == 0.0


def test_latency_meter_percentiles() -> None:
    m = LatencyMeter("test", window=100)
    for v in range(1, 101):
        m.record_ms(float(v))
    p = m.percentiles()
    assert p["n"] == 100
    assert p["p50"] == pytest.approx(50.5, rel=1e-3)
    assert p["p95"] == pytest.approx(95.05, rel=1e-3)


def test_latency_meter_ring_buffer_drops_old_samples() -> None:
    m = LatencyMeter("test", window=4)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        m.record_ms(v)
    p = m.percentiles()
    # Only the most recent 4 (3, 4, 5, 6) are present.
    assert p["n"] == 4
    assert p["p50"] == pytest.approx(4.5, rel=1e-3)
