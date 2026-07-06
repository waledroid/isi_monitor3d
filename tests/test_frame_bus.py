"""``FrameBus`` — single-subscriber bounded queue with drop-old semantics."""

from __future__ import annotations

import queue

import numpy as np
import pytest

from backbone.core.types import Frame, FramePair
from backbone.ingestion.frame_bus import FrameBus


def _pair(ts: float, idx: int = 0) -> FramePair:
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    frames = {"cam_a": Frame("cam_a", ts, idx, img), "cam_b": Frame("cam_b", ts, idx, img)}
    return FramePair(capture_ts=ts, frame_idx=idx, frames=frames)


def test_consumer_receives_published_items_in_order() -> None:
    bus = FrameBus(default_maxsize=4)
    bus.publish(_pair(1.0, 1))
    bus.publish(_pair(2.0, 2))
    assert bus.get_nowait().capture_ts == 1.0
    assert bus.get_nowait().capture_ts == 2.0
    assert bus.get_nowait() is None


def test_drop_old_when_full_increments_counter() -> None:
    bus = FrameBus(default_maxsize=2)
    bus.publish(_pair(1.0, 1))
    bus.publish(_pair(2.0, 2))
    bus.publish(_pair(3.0, 3))  # drops ts=1
    bus.publish(_pair(4.0, 4))  # drops ts=2
    received = [bus.get_nowait().capture_ts for _ in range(2)]
    assert received == [3.0, 4.0]
    assert bus.dropped == 2


def test_publish_after_close_is_silent_noop() -> None:
    bus = FrameBus()
    bus.close()
    bus.publish(_pair(1.0))  # must not raise
    assert bus.get_nowait() is None


def test_get_with_timeout_raises_queue_empty() -> None:
    bus = FrameBus()
    with pytest.raises(queue.Empty):
        bus.get(timeout=0.05)


def test_depth_tracks_pending() -> None:
    bus = FrameBus(default_maxsize=4)
    assert bus.depth == 0
    bus.publish(_pair(1.0))
    bus.publish(_pair(2.0))
    assert bus.depth == 2
    bus.get_nowait()
    assert bus.depth == 1


def test_get_latest_returns_sole_item() -> None:
    bus = FrameBus()
    bus.publish(_pair(1.0, 1))
    assert bus.get_latest(timeout=0.1).frame_idx == 1
    assert bus.dropped == 0


def test_get_latest_skips_to_newest_and_counts_drops() -> None:
    """The latest-only read: with two queued pairs the older is discarded
    (counted as dropped) and the consumer gets the NEWEST."""
    bus = FrameBus()
    bus.publish(_pair(1.0, 1))
    bus.publish(_pair(2.0, 2))
    assert bus.get_latest(timeout=0.1).frame_idx == 2
    assert bus.dropped == 1
    assert bus.depth == 0


def test_get_latest_times_out_like_get() -> None:
    bus = FrameBus()
    with pytest.raises(queue.Empty):
        bus.get_latest(timeout=0.05)
