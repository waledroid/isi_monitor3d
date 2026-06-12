"""`/ws/video` — the multiplexed WebSocket video transport.

Hermetic: `_build_stream` is monkeypatched to a synthetic frame generator, so no
camera/model/GPU is needed. Pins the protocol (sub → binary frames with the
``idLen | id | JPEG`` framing; unknown stream → JSON error; unsub → pump stops
and the generator unwinds its ``finally``) and the id→pipeline mapping.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.api import routes_ws_video
from monitor_web.app import create_app
from monitor_web.config import Settings


@pytest.fixture
def client(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
        "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": 0}]},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as c:
        yield c


def _frame_gen(released: threading.Event, n: int = 1000):
    """Synthetic frame iterator that records its unwind (the hub-release path)."""
    try:
        for i in range(n):
            yield np.full((24, 32, 3), i % 255, dtype=np.uint8)
            time.sleep(0.005)
    finally:
        released.set()


def test_subscribe_receives_framed_jpegs(client, monkeypatch):
    released = threading.Event()
    monkeypatch.setattr(routes_ws_video, "_build_stream",
                        lambda state, sid: _frame_gen(released))
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "cam:cam_a"})
        data = ws.receive_bytes()
        id_len = data[0]
        assert data[1:1 + id_len].decode() == "cam:cam_a"
        jpeg = data[1 + id_len:]
        assert jpeg[:2] == b"\xff\xd8"          # JPEG SOI
        # Frames keep coming while subscribed.
        assert ws.receive_bytes()[0] == id_len
        ws.send_json({"unsub": "cam:cam_a"})
    assert released.wait(2.0), "unsub/close must unwind the frame generator"


def test_unknown_stream_gets_json_error(client):
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "nonsense"})
        msg = ws.receive_json()
        assert "error" in msg and msg["stream"] == "nonsense"
        # An unconfigured camera also errors instead of opening anything.
        ws.send_json({"sub": "cam:cam_zz"})
        msg = ws.receive_json()
        assert "error" in msg and msg["stream"] == "cam:cam_zz"


def test_disconnect_stops_all_pumps(client, monkeypatch):
    released = threading.Event()
    monkeypatch.setattr(routes_ws_video, "_build_stream",
                        lambda state, sid: _frame_gen(released))
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "zone:zp_x"})
        ws.receive_bytes()                       # at least one frame flowed
    assert released.wait(2.0), "socket close must stop the pump and release the stream"


def test_build_stream_id_mapping(monkeypatch):
    """The id grammar maps to the right pipeline builders with the right flags."""
    calls: list = []
    monkeypatch.setattr(routes_ws_video, "build_cam_stream",
                        lambda state, cam, detect, warp: calls.append(("cam", cam, detect, warp)))
    monkeypatch.setattr(routes_ws_video, "build_zone_stream",
                        lambda state, pid: calls.append(("zone", pid)))
    monkeypatch.setattr(routes_ws_video, "build_unified_stream",
                        lambda state: calls.append(("unified",)))
    routes_ws_video._build_stream(None, "cam:cam_a")
    routes_ws_video._build_stream(None, "cam:cam_b:warp")
    routes_ws_video._build_stream(None, "zone:zp_1")
    routes_ws_video._build_stream(None, "unified")
    assert calls == [("cam", "cam_a", True, False), ("cam", "cam_b", False, True),
                     ("zone", "zp_1"), ("unified",)]
    with pytest.raises(LookupError):
        routes_ws_video._build_stream(None, "cam:cam_a:bogus")
    with pytest.raises(LookupError):
        routes_ws_video._build_stream(None, "mp4:x")
