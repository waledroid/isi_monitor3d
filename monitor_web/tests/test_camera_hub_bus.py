"""Camera hub x shared frame bus — prefer the bus, fall back when stale."""

from __future__ import annotations

import time

import numpy as np
import pytest
from backbone.shared.frame_shm import FrameShmWriter


@pytest.fixture()
def shm_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ISI3D_SHM_DIR", str(tmp_path))
    return tmp_path


def test_hub_streams_from_bus_without_opening_rtsp(shm_dir):
    from monitor_web.camera_hub import CameraStream

    writer = FrameShmWriter("cam_bus")
    img = np.full((24, 32, 3), 7, dtype=np.uint8)
    writer.write(img, time.time())

    # An unreachable RTSP URL: if the hub tried to build its own source it
    # would sit on placeholders — a real frame proves it came from the bus.
    stream = CameraStream("cam_bus", "rtsp", {"url": "rtsp://192.0.2.1/nope"})
    stream.ensure_pump()
    try:
        deadline = time.time() + 3.0
        got = None
        while got is None and time.time() < deadline:
            writer.write(img, time.time())      # keep the bus fresh
            got = stream.latest_real_frame_with_ts()
            time.sleep(0.02)
        assert got is not None, "hub never picked up the bus frame"
        frame, ts = got
        assert np.array_equal(frame, img)
        assert ts > 0
    finally:
        stream.stop()
        writer.unlink()
