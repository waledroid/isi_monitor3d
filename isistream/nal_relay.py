"""``NalRelay`` — per-camera unix-socket server for the compressed-video tap.

The camera already compressed the video; re-encoding it for display is pure
waste. isistream tees each camera's ORIGINAL H.264/H.265 bitstream right
after RTP depay (``RtspFrameSource(nal_tap=...)``) and this relay serves the
access units on a unix socket at ``/tmp/isi3d_nal_<camera_id>.sock``. The
dashboard connects as a client and forwards the bitstream to browsers for
hardware decode — no JPEG encode anywhere on the display path. The frame bus
and the detection path are untouched.

Wire format (all little-endian):

    HEADER (once per connection):
        u32 length | JSON utf8 {"v": 1, "codec": "h264"|"h265", "camera_id": ...}
    AU frame (repeated):
        u32 length-of-rest | f64 capture_ts | u8 flags (bit0=keyframe) | Annex-B AU

Per-client delivery starts at the FIRST KEYFRAME after connect (deltas before
it are skipped) — combined with the tap's ``config-interval=-1`` parser
(SPS/PPS/VPS re-injected before every keyframe), any client can start
decoding immediately from its first AU frame.

Threading: ``push()`` is called from the GStreamer tap callback and must
never block — it only appends to per-client queues under a short lock. One
daemon sender thread per client drains its queue; a client whose backlog
exceeds ``max_backlog_bytes`` (~8 MB) is disconnected (its socket is shut
down, which also unblocks a sender stuck in ``sendall``).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
from collections import deque

__all__ = ["NalRelay", "nal_socket_path"]

logger = logging.getLogger(__name__)

_MAX_BACKLOG_BYTES = 8 * 1024 * 1024

# u32 length-of-rest, f64 capture_ts, u8 flags — then the Annex-B AU bytes.
_AU_HEADER = struct.Struct("<IdB")
_FLAG_KEYFRAME = 0x01


def nal_socket_path(camera_id: str) -> str:
    """Socket path for one camera — /tmp, overridable via ISI3D_NAL_DIR (tests)."""
    return os.path.join(os.environ.get("ISI3D_NAL_DIR", "/tmp"),
                        f"isi3d_nal_{camera_id}.sock")


class _Client:
    """One connected consumer: its socket, pending queue, and delivery state."""

    __slots__ = ("backlog", "cond", "conn", "dead", "pending", "started")

    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn
        self.pending: deque[bytes] = deque()
        self.backlog = 0            # queued bytes not yet handed to the kernel
        self.cond = threading.Condition()
        self.started = False        # becomes True at the first keyframe
        self.dead = False


class NalRelay:
    """Unix-socket server relaying one camera's Annex-B access units."""

    def __init__(self, camera_id: str, codec: str, *,
                 max_backlog_bytes: int = _MAX_BACKLOG_BYTES) -> None:
        self.camera_id = camera_id
        self.codec = codec
        self.path = nal_socket_path(camera_id)
        self._max_backlog = int(max_backlog_bytes)
        header_json = json.dumps(
            {"v": 1, "codec": codec, "camera_id": camera_id}).encode("utf-8")
        self._header = struct.pack("<I", len(header_json)) + header_json
        self._clients: list[_Client] = []
        self._clients_lock = threading.Lock()
        self._closed = threading.Event()

        try:
            os.unlink(self.path)        # a stale path from a crashed run
        except OSError:
            pass
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._server.bind(self.path)
            self._server.listen(8)
        except OSError:
            self._server.close()
            raise
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True,
            name=f"nal-relay-{camera_id}")
        self._accept_thread.start()
        logger.info("NalRelay[%s]: serving %s bitstream on %s",
                    camera_id, codec, self.path)

    # ---- producer side (GStreamer tap callback thread) ----

    def push(self, au: bytes, capture_ts: float, keyframe: bool) -> None:
        """Queue one access unit to every connected client. Never blocks."""
        if self._closed.is_set():
            return
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            return
        frame = _AU_HEADER.pack(_AU_HEADER.size - 4 + len(au), capture_ts,
                                _FLAG_KEYFRAME if keyframe else 0) + au
        for c in clients:
            with c.cond:
                if c.dead:
                    continue
                if not c.started:
                    if not keyframe:
                        continue        # no deltas before the first keyframe
                    c.started = True
                c.pending.append(frame)
                c.backlog += len(frame)
                if c.backlog > self._max_backlog:
                    c.dead = True
                    logger.warning(
                        "NalRelay[%s]: disconnecting slow client "
                        "(backlog %d bytes)", self.camera_id, c.backlog)
                    # Shutdown (non-blocking) also unblocks a sender thread
                    # stuck in sendall against a full socket buffer.
                    try:
                        c.conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                c.cond.notify()

    # ---- server internals ----

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                break                   # server socket closed
            client = _Client(conn)
            with self._clients_lock:
                self._clients.append(client)
            threading.Thread(
                target=self._sender, args=(client,), daemon=True,
                name=f"nal-relay-{self.camera_id}-client").start()

    def _sender(self, c: _Client) -> None:
        try:
            c.conn.sendall(self._header)
            while True:
                with c.cond:
                    while not c.pending and not c.dead and not self._closed.is_set():
                        c.cond.wait(timeout=0.5)
                    if c.dead or self._closed.is_set():
                        break
                    frame = c.pending.popleft()
                    c.backlog -= len(frame)
                c.conn.sendall(frame)
        except OSError:
            pass                        # client went away mid-send
        finally:
            with c.cond:
                c.dead = True
            try:
                c.conn.close()
            except OSError:
                pass
            with self._clients_lock:
                try:
                    self._clients.remove(c)
                except ValueError:
                    pass

    def close(self) -> None:
        """Stop serving: drop every client, close the socket, unlink the path."""
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._server.close()        # unblocks the accept loop
        except OSError:
            pass
        with self._clients_lock:
            clients = list(self._clients)
        for c in clients:
            with c.cond:
                c.dead = True
                c.cond.notify()
            try:
                c.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            os.unlink(self.path)
        except OSError:
            pass
