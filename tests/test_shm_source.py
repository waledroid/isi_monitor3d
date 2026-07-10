"""``shm`` FrameSource plugin — yields once per NEW bus frame, ends on staleness."""

from __future__ import annotations

import threading
import time

import numpy as np

from backbone.core.interfaces import frame_source_registry
from backbone.ingestion.shm_source import ShmFrameSource
from backbone.shared.frame_shm import FrameShmWriter


def test_registered_as_shm():
    import backbone.ingestion  # noqa: F401
    assert "shm" in frame_source_registry.names()


def test_yields_new_frames_and_ends_when_stale(tmp_path):
    w = FrameShmWriter("cam_s", directory=str(tmp_path))
    src = ShmFrameSource("cam_s", directory=str(tmp_path),
                         poll_s=0.002, stale_after_s=0.4)
    got = []

    def consume():
        for f in src.frames():
            got.append((f.frame_idx, f.capture_ts))

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    for _ in range(3):
        w.write(img, time.time())
        time.sleep(0.05)
    # Stop writing → the source must END (stale) so consumers can fall back.
    t.join(timeout=3.0)
    assert not t.is_alive(), "frames() must return once the bus goes stale"
    assert len(got) == 3
    assert [i for i, _ in got] == [0, 1, 2]
    assert got[0][1] < got[1][1] < got[2][1]      # true capture_ts, monotonic
    src.stop()
    w.close()


def test_stop_unblocks(tmp_path):
    src = ShmFrameSource("cam_none", directory=str(tmp_path),
                         poll_s=0.002, stale_after_s=30.0)
    t = threading.Thread(target=lambda: list(src.frames()), daemon=True)
    t.start()
    time.sleep(0.1)
    src.stop()
    t.join(timeout=2.0)
    assert not t.is_alive()
