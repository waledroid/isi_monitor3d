"""``NalRelay`` — unix-socket bitstream relay (compressed video passthrough).

Hermetic: a fake pusher stands in for the GStreamer tap (no GStreamer, no
camera). Coverage:
    * Wire framing round-trips (u32-length JSON header, then
      u32 | f64 capture_ts | u8 flags | Annex-B AU frames).
    * Delivery starts at the first keyframe — a client connecting mid-stream
      never receives deltas from before its first keyframe.
    * Multiple simultaneous clients each get the full stream.
    * A slow client (backlog over the cap) is disconnected; others live on.
    * close() unlinks the socket path; a stale path is unlinked on bind.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import time

import pytest

from isistream.nal_relay import NalRelay, nal_socket_path


@pytest.fixture(autouse=True)
def _nal_dir(tmp_path, monkeypatch):
    """Route the unix sockets into the test's tmp dir, never /tmp."""
    monkeypatch.setenv("ISI3D_NAL_DIR", str(tmp_path))
    return tmp_path


# ---- client-side wire helpers -------------------------------------------------


def _read_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("relay closed the connection")
        data += chunk
    return data


def _connect(camera_id: str) -> tuple[socket.socket, dict]:
    """Connect and read the header — after which the client is registered
    (the header is sent by the client's sender thread post-registration)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(nal_socket_path(camera_id))
    (hlen,) = struct.unpack("<I", _read_exact(sock, 4))
    header = json.loads(_read_exact(sock, hlen).decode("utf-8"))
    return sock, header


def _read_au(sock: socket.socket) -> tuple[float, bool, bytes]:
    (flen,) = struct.unpack("<I", _read_exact(sock, 4))
    rest = _read_exact(sock, flen)
    capture_ts, flags = struct.unpack("<dB", rest[:9])
    return capture_ts, bool(flags & 0x01), rest[9:]


# ---- tests ---------------------------------------------------------------------


def test_header_and_framing_round_trip() -> None:
    relay = NalRelay("cam_a", "h264")
    try:
        sock, header = _connect("cam_a")
        assert header == {"v": 1, "codec": "h264", "camera_id": "cam_a"}

        key_au = b"\x00\x00\x00\x01\x67spspps\x00\x00\x00\x01\x65idr"
        delta_au = b"\x00\x00\x00\x01\x41delta"
        relay.push(key_au, 1234.5, True)
        relay.push(delta_au, 1234.54, False)

        ts, keyframe, au = _read_au(sock)
        assert (ts, keyframe, au) == (1234.5, True, key_au)
        ts, keyframe, au = _read_au(sock)
        assert (ts, keyframe, au) == (1234.54, False, delta_au)
        sock.close()
    finally:
        relay.close()


def test_h265_codec_in_header() -> None:
    relay = NalRelay("cam_b", "h265")
    try:
        sock, header = _connect("cam_b")
        assert header["codec"] == "h265"
        sock.close()
    finally:
        relay.close()


def test_delivery_starts_at_first_keyframe() -> None:
    """Deltas pushed before the client's first keyframe must be skipped —
    a decoder can only start at a keyframe (with SPS/PPS in front of it)."""
    relay = NalRelay("cam_a", "h264")
    try:
        sock, _ = _connect("cam_a")
        relay.push(b"delta-1", 1.0, False)
        relay.push(b"delta-2", 2.0, False)
        relay.push(b"KEY", 3.0, True)
        relay.push(b"delta-3", 4.0, False)

        ts, keyframe, au = _read_au(sock)
        assert (ts, keyframe, au) == (3.0, True, b"KEY")
        ts, keyframe, au = _read_au(sock)
        assert (ts, keyframe, au) == (4.0, False, b"delta-3")
        sock.close()
    finally:
        relay.close()


def test_multiple_clients_each_get_the_stream() -> None:
    relay = NalRelay("cam_a", "h264")
    try:
        sock1, _ = _connect("cam_a")
        sock2, _ = _connect("cam_a")
        relay.push(b"KEY", 10.0, True)
        relay.push(b"delta", 11.0, False)
        for sock in (sock1, sock2):
            assert _read_au(sock) == (10.0, True, b"KEY")
            assert _read_au(sock) == (11.0, False, b"delta")
        sock1.close()
        sock2.close()
    finally:
        relay.close()


def test_late_joiner_waits_for_next_keyframe() -> None:
    """A client connecting mid-GOP gets nothing until the next keyframe."""
    relay = NalRelay("cam_a", "h264")
    try:
        sock1, _ = _connect("cam_a")
        relay.push(b"KEY-1", 1.0, True)
        assert _read_au(sock1) == (1.0, True, b"KEY-1")

        sock2, _ = _connect("cam_a")     # joins mid-GOP
        relay.push(b"delta-1", 2.0, False)
        relay.push(b"KEY-2", 3.0, True)

        assert _read_au(sock1) == (2.0, False, b"delta-1")
        assert _read_au(sock1) == (3.0, True, b"KEY-2")
        # sock2's FIRST AU is the keyframe — the mid-GOP delta never arrives.
        assert _read_au(sock2) == (3.0, True, b"KEY-2")
        sock1.close()
        sock2.close()
    finally:
        relay.close()


def test_slow_client_disconnected_fast_client_survives() -> None:
    relay = NalRelay("cam_a", "h264", max_backlog_bytes=64 * 1024)
    try:
        slow, _ = _connect("cam_a")
        fast, _ = _connect("cam_a")
        # The slow client stops reading; push well past backlog + socket
        # buffers. The fast client drains as it goes.
        au = b"K" * 8192
        deadline = time.monotonic() + 10.0
        disconnected = False
        for i in range(512):
            relay.push(au, float(i), True)
            try:
                fast.settimeout(5.0)
                got = _read_au(fast)
                assert got[2] == au
            except ConnectionError:
                pytest.fail("fast client was disconnected")
            if time.monotonic() > deadline:
                break
        # The slow client must observe EOF/reset once it finally reads
        # (a recv timeout would mean it is still connected — a failure).
        slow.settimeout(5.0)
        try:
            while True:
                _read_au(slow)
        except TimeoutError:
            pytest.fail("slow client still connected (recv timed out, no EOF)")
        except (ConnectionError, OSError):
            disconnected = True
        assert disconnected, "slow client was never disconnected"
        fast.close()
        slow.close()
    finally:
        relay.close()


def test_push_without_clients_is_a_noop() -> None:
    relay = NalRelay("cam_a", "h264")
    try:
        relay.push(b"KEY", 1.0, True)    # must not raise or block
    finally:
        relay.close()


def test_close_unlinks_socket_path() -> None:
    relay = NalRelay("cam_a", "h264")
    path = relay.path
    assert os.path.exists(path)
    relay.close()
    assert not os.path.exists(path)
    relay.close()                        # idempotent


def test_stale_socket_path_unlinked_on_bind() -> None:
    """A crashed producer leaves the socket file behind — the next relay
    must bind over it, not die with EADDRINUSE."""
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(nal_socket_path("cam_a"))
    stale.close()                        # closes the fd, leaves the path
    relay = NalRelay("cam_a", "h264")    # must not raise
    try:
        sock, header = _connect("cam_a")
        assert header["camera_id"] == "cam_a"
        sock.close()
    finally:
        relay.close()


def test_push_after_close_is_a_noop() -> None:
    relay = NalRelay("cam_a", "h264")
    relay.close()
    relay.push(b"KEY", 1.0, True)        # must not raise
