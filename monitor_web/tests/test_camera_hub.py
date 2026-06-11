"""``CameraHub`` — one shared source per camera, fanned out to many viewers."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from monitor_web import camera_hub


class _FakeFrame:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image


class _FakeSource:
    """A live-source stand-in: yields frames forever until stopped."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._stop = threading.Event()
        self.stopped = False

    def start(self) -> None:
        pass

    def frames(self):
        n = 0
        while not self._stop.is_set():
            n += 1
            yield _FakeFrame(np.full((4, 4, 3), n % 256, dtype=np.uint8))
            time.sleep(0.005)

    def stop(self) -> None:
        self._stop.set()
        self.stopped = True


@pytest.fixture
def patched_hub(monkeypatch):
    """A fresh hub whose ``_build_source`` returns counting fake sources."""
    builds: list[_FakeSource] = []

    def fake_build(camera_id, plugin, src_cfg):
        src = _FakeSource(camera_id)
        builds.append(src)
        return src

    monkeypatch.setattr(camera_hub, "_build_source", fake_build)
    hub = camera_hub.CameraHub()
    yield hub, builds
    hub.shutdown()


def _first_frame(stream, timeout=2.0):
    """Pull one frame from a reader generator with a hard timeout."""
    result = {}
    gen = stream.read()

    def pull():
        result["frame"] = next(gen)

    t = threading.Thread(target=pull, daemon=True)
    t.start()
    t.join(timeout)
    gen.close()
    assert "frame" in result, "reader produced no frame"
    return result["frame"]


def test_multiple_viewers_share_one_source(patched_hub):
    hub, builds = patched_hub
    cfg = {"url": "rtsp://cam"}
    s1 = hub.acquire("cam_a", "rtsp", cfg)
    s2 = hub.acquire("cam_a", "rtsp", cfg)

    assert s1 is s2                          # same shared stream
    time.sleep(0.1)                          # let the single pump build its source
    assert len(builds) == 1                  # ONE underlying session, not two

    # Both viewers receive frames from the shared slot.
    f1 = _first_frame(s1)
    f2 = _first_frame(s2)
    assert f1.shape[2] == 3 and f2.shape[2] == 3

    hub.release(s1)
    hub.release(s2)


def test_config_change_rebuilds_source(patched_hub):
    hub, builds = patched_hub
    s_a = hub.acquire("cam_a", "rtsp", {"url": "rtsp://old"})
    time.sleep(0.05)
    s_b = hub.acquire("cam_a", "rtsp", {"url": "rtsp://new"})   # URL changed
    time.sleep(0.05)

    assert s_a is not s_b                     # rebuilt
    assert len(builds) == 2                   # old session stopped, new one opened
    assert builds[0].stopped is True          # the old source was released
    hub.release(s_b)


def test_idle_retire_releases_source(patched_hub, monkeypatch):
    hub, _builds = patched_hub
    monkeypatch.setattr(camera_hub, "IDLE_GRACE_S", 0.1)
    stream = hub.acquire("cam_a", "rtsp", {"url": "rtsp://cam"})
    time.sleep(0.1)
    hub.release(stream)                        # last viewer leaves → idle timer armed

    time.sleep(0.4)                            # past the grace window
    assert "cam_a" not in hub._streams         # retired from the registry
    assert stream._stop.is_set()               # and the source was stopped


def test_returning_viewer_cancels_idle_shutdown(patched_hub, monkeypatch):
    hub, _builds = patched_hub
    monkeypatch.setattr(camera_hub, "IDLE_GRACE_S", 0.3)
    stream = hub.acquire("cam_a", "rtsp", {"url": "rtsp://cam"})
    hub.release(stream)                        # arms a 0.3s idle timer
    again = hub.acquire("cam_a", "rtsp", {"url": "rtsp://cam"})   # viewer returns

    assert again is stream                     # same source kept alive
    time.sleep(0.4)
    assert "cam_a" in hub._streams             # idle shutdown was cancelled
    assert not stream._stop.is_set()
    hub.release(again)
