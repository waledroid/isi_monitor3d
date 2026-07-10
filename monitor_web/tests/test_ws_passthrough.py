"""Compressed-video passthrough relay (`camh264:` over /ws/video).

Hermetic: a FAKE unix-socket server plays isistream's NalRelay (header + AU
frames per the fixed contract in `isistream/nal_relay.py`); no camera, codec,
or GPU. Pins the ws-payload framing (type 0 INIT / type 1 AU, ts+flags
round-trip), the immediate unavailable-INIT fallback signal when the socket
is absent, and the end-to-end envelope over a real /ws/video connection.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.api.routes_ws_video import (
    PT_AU,
    PT_INIT,
    nal_socket_path,
    relay_nal_session,
    run_nal_relay,
)
from monitor_web.app import create_app
from monitor_web.config import Settings

_HEADER = {"v": 1, "codec": "h264", "camera_id": "cam_a"}
_AUS = [
    (1234.5, 1, b"\x00\x00\x00\x01\x67spspps\x00\x00\x00\x01\x65keyframe"),
    (1234.55, 0, b"\x00\x00\x00\x01\x41delta"),
    (1234.6, 0, b"\x00\x00\x00\x01\x41delta2"),
]


def _wire_blob(header: dict = _HEADER, aus=_AUS) -> bytes:
    """Serialize the upstream unix-socket stream exactly per the contract."""
    hdr = json.dumps(header).encode("utf-8")
    blob = struct.pack("<I", len(hdr)) + hdr
    for ts, flags, au in aus:
        blob += struct.pack("<IdB", 9 + len(au), ts, flags) + au
    return blob


def _serve_once(sock_path: str, blob: bytes,
                hold_open: threading.Event | None = None) -> threading.Thread:
    """Fake NalRelay: accept ONE client, send ``blob``, optionally hold the
    connection open until ``hold_open`` is set, then close."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def _serve() -> None:
        try:
            conn, _ = srv.accept()
            conn.sendall(blob)
            if hold_open is not None:
                hold_open.wait(timeout=10.0)
            conn.close()
        finally:
            srv.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def test_relay_session_forwards_init_and_aus(tmp_path: Path):
    """The pump forwards the header as an available INIT and every AU frame
    byte-identically (ts/flags/Annex-B round-trip) — tested directly, no ws."""
    sock_path = str(tmp_path / "nal.sock")
    server = _serve_once(sock_path, _wire_blob())
    payloads: list[bytes] = []
    relay_nal_session(sock_path, payloads.append, threading.Event())
    server.join(timeout=5.0)

    assert len(payloads) == 1 + len(_AUS)
    assert payloads[0][0] == PT_INIT
    init = json.loads(payloads[0][1:])
    assert init == {**_HEADER, "available": True}
    for payload, (ts, flags, au) in zip(payloads[1:], _AUS, strict=True):
        assert payload[0] == PT_AU
        got_ts, got_flags = struct.unpack_from("<dB", payload, 1)
        assert got_ts == ts
        assert got_flags == flags
        assert payload[10:] == au


def test_missing_socket_emits_one_unavailable_init(tmp_path: Path):
    """No unix socket (isistream stopped / frames mode) → exactly ONE
    unavailable INIT per outage, immediately — the JS fallback trigger."""
    payloads: list[bytes] = []
    stop = threading.Event()
    t = threading.Thread(
        target=run_nal_relay,
        args=(str(tmp_path / "absent.sock"), "cam_a", payloads.append, stop),
        kwargs={"backoff_s": (0.01,)},
        daemon=True,
    )
    t.start()
    time.sleep(0.2)          # many retry cycles at 10 ms backoff
    stop.set()
    t.join(timeout=5.0)

    assert len(payloads) == 1, "one INIT per outage, not one per retry"
    assert payloads[0][0] == PT_INIT
    init = json.loads(payloads[0][1:])
    assert init["available"] is False
    assert init["camera_id"] == "cam_a"
    assert init["reason"]


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


def test_ws_camh264_relays_over_video_socket(client, tmp_path: Path, monkeypatch):
    """End to end through /ws/video: sub camh264:cam_a → envelope-framed INIT
    then AU payloads, in order, no acks needed."""
    monkeypatch.setenv("ISI3D_NAL_DIR", str(tmp_path))
    hold = threading.Event()
    _serve_once(nal_socket_path("cam_a"), _wire_blob(), hold_open=hold)
    try:
        with client.websocket_connect("/ws/video") as ws:
            ws.send_json({"sub": "camh264:cam_a"})
            frames = []
            for _ in range(1 + len(_AUS)):
                data = ws.receive_bytes()
                id_len = data[0]
                assert data[1:1 + id_len].decode() == "camh264:cam_a"
                frames.append(data[1 + id_len:])
            assert frames[0][0] == PT_INIT
            assert json.loads(frames[0][1:])["available"] is True
            for payload, (ts, flags, au) in zip(frames[1:], _AUS, strict=True):
                assert payload[0] == PT_AU
                got_ts, got_flags = struct.unpack_from("<dB", payload, 1)
                assert (got_ts, got_flags, payload[10:]) == (ts, flags, au)
            ws.send_json({"unsub": "camh264:cam_a"})
    finally:
        hold.set()


def test_ws_camh264_unknown_camera_errors(client):
    """An unconfigured camera errors like every other stream kind."""
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "camh264:cam_zz"})
        msg = ws.receive_json()
        assert "error" in msg and msg["stream"] == "camh264:cam_zz"


def test_ws_camh264_absent_socket_signals_unavailable(client, tmp_path: Path,
                                                      monkeypatch):
    """Socket absent at subscribe time → the client's FIRST payload is the
    unavailable INIT, so the JS can fall back to JPEG immediately."""
    monkeypatch.setenv("ISI3D_NAL_DIR", str(tmp_path))   # empty dir — no socket
    with client.websocket_connect("/ws/video") as ws:
        ws.send_json({"sub": "camh264:cam_a"})
        data = ws.receive_bytes()
        id_len = data[0]
        assert data[1:1 + id_len].decode() == "camh264:cam_a"
        payload = data[1 + id_len:]
        assert payload[0] == PT_INIT
        init = json.loads(payload[1:])
        assert init["available"] is False
        ws.send_json({"unsub": "camh264:cam_a"})
