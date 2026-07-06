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


# ---- credit-gated sends (latest-frame-only to a slow client) ---------------


def _counter_frame_gen(period_s: float = 0.005, n: int = 2000):
    """Frames whose uniform pixel value encodes the frame index (mod 250) —
    lets a test read WHICH frame it received after JPEG round-trip."""
    for i in range(n):
        yield np.full((24, 32, 3), i % 250, dtype=np.uint8)
        time.sleep(period_s)


def _frame_counter(data: bytes) -> int:
    import cv2
    id_len = data[0]
    jpeg = np.frombuffer(data[1 + id_len:], dtype=np.uint8)
    img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    return round(float(img.mean()))


def test_credit_mode_sends_only_newest_on_ack(client, monkeypatch):
    """After the first ack the server sends one frame per credit; while the
    client withholds acks the one-slot holder keeps overwriting, so the frame
    delivered on the NEXT ack is the NEWEST — never the stale backlog."""
    monkeypatch.setattr(routes_ws_video, "ACK_REFILL_S", 60.0)   # refill can't fire
    monkeypatch.setattr(routes_ws_video, "_build_stream",
                        lambda state, sid: _counter_frame_gen())
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "cam:cam_a"})
        ws.send_json({"ack": "cam:cam_a"})       # priming ack → credit mode
        first = _frame_counter(ws.receive_bytes())
        # Possibly one more frame from the priming window; drain nothing else —
        # now STOP acking and let many frames pass on the server (generous
        # window: CI/parallel-test load slows the 5 ms generator).
        time.sleep(0.5)
        ws.send_json({"ack": "cam:cam_a"})
        second = _frame_counter(ws.receive_bytes())
        # Latest-only proof: the delivered frame skipped far ahead of first+1.
        assert second - first > 5, (
            f"expected a jump to the newest frame, got {first} -> {second} "
            f"(a FIFO backlog would deliver ~{first + 1})")


def test_credit_refill_recovers_lost_acks(client, monkeypatch):
    """With acks lost (client never acks again), the refill timer keeps a slow
    trickle flowing instead of freezing the panel forever."""
    monkeypatch.setattr(routes_ws_video, "ACK_REFILL_S", 0.05)
    monkeypatch.setattr(routes_ws_video, "_build_stream",
                        lambda state, sid: _counter_frame_gen())
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "cam:cam_a"})
        ws.send_json({"ack": "cam:cam_a"})       # enter credit mode
        for _ in range(3):                        # no further acks — refill only
            data = ws.receive_bytes()
            assert data[1:1 + data[0]].decode() == "cam:cam_a"


def test_ack_for_unknown_stream_is_ignored(client, monkeypatch):
    """An ack for a stream that was just unsubscribed (resubscribe race) must
    not error or kill the socket."""
    released = threading.Event()
    monkeypatch.setattr(routes_ws_video, "_build_stream",
                        lambda state, sid: _frame_gen(released))
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"ack": "zone:gone"})       # unknown sid — silently ignored
        ws.send_json({"sub": "zone:zp_x"})
        ws.receive_bytes()                        # socket still healthy
