"""WebSocket `/ws/video` — ALL dashboard video over ONE multiplexed connection.

Why: the dashboard used one MJPEG `<img>` per panel (big view + up to 3 zone
panels + hidden tabs). Each MJPEG stream permanently occupies one of the
browser's ~6 HTTP/1.1 connections per host, so with enough panels a settings
POST (or a page reload) could not get a connection slot and the whole UI froze.
WebSocket connections are exempt from that cap, and one socket carries every
panel's frames — the connection-limit class of bugs is gone structurally.

Protocol:
  client → server (text/JSON): {"sub": "<stream-id>"} / {"unsub": "<stream-id>"}
                               / {"ack": "<stream-id>"}
  server → client (binary):    uint8 idLen | stream-id utf8 | JPEG bytes
  server → client (text/JSON): {"error": ..., "stream": ...} for a bad sub

Stream ids: ``cam:<camera_id>`` (live detect view), ``cam:<camera_id>:warp``
(rectified verification view), ``zone:<patch_id>``, ``unified``, and
``camh264:<camera_id>`` (compressed-video passthrough — the camera's ORIGINAL
H.264/H.265 bitstream relayed from isistream's per-camera unix socket for
browser-side WebCodecs decode; see the passthrough section below).

Each subscription reuses the SAME sync frame pipelines as the MJPEG endpoints
(`build_cam_stream` / `build_zone_stream` / `build_unified_stream`) running in a
dedicated daemon thread, publishing the LATEST JPEG into a one-slot holder —
drop-oldest semantics, so a slow client never builds a queue on the server.

CREDIT-GATED sends (latest-frame-only end to end): TCP backpressure never
reflects the browser's RENDER rate — browsers buffer received WS frames in
memory regardless, so a stalled tab used to accumulate frames and display
ever-older ones (the demo lag bug). The client acks each frame it actually
swaps into the <img> (at requestAnimationFrame time); the server sends a
stream's next frame only while it has credit (window CREDIT_WINDOW — one frame
rendering + one in flight to hide RTT). Between sends the one-slot holder
keeps overwriting with newer frames, so what eventually goes out is always the
NEWEST. A subscription runs in legacy free-run mode until its first ack (old
clients / the MJPEG-parity tests keep working); ``ACK_REFILL_S`` recovers lost
acks and gives hidden tabs a slow keepalive instead of a frozen stale image.
Zone/unified/warp streams are source-paced (camera fps), so credit never
binds them. The hidden MP4 dev viewer and curl debugging keep the MJPEG
endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import threading
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..video_stream import encode_jpeg
from .routes_video import (
    _load_cameras_from_backbone_yaml,
    build_cam_stream,
    build_unified_stream,
    build_zone_stream,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Max unacknowledged frames in flight per stream (credit mode). 2 = one frame
# being rendered + one crossing the wire, so a healthy client never stalls on
# RTT while a stalled one stops receiving after 2 frames.
CREDIT_WINDOW = 2
# With a pending frame and no send for this long, send one anyway — recovers
# lost acks / dead render loops; a hidden tab gets a ~0.5 fps keepalive.
ACK_REFILL_S = 2.0


def _build_stream(state, stream_id: str):
    """Map a stream id to its frame iterator. Raises ``LookupError`` on an
    unknown id / unconfigured camera or zone."""
    kind, _, rest = stream_id.partition(":")
    if kind == "cam" and rest:
        camera_id, _, flag = rest.partition(":")
        if flag not in ("", "warp"):
            raise LookupError(f"unknown cam stream flag {flag!r}")
        # The dashboard cam view always runs the detect overlay; the warp
        # variant is the plain rectified verification view (no detection).
        return build_cam_stream(state, camera_id, detect=(flag != "warp"),
                                warp=(flag == "warp"))
    if kind == "zone" and rest:
        return build_zone_stream(state, rest)
    if kind == "unified" and not rest:
        return build_unified_stream(state)
    raise LookupError(f"unknown stream id {stream_id!r}")


class _Subscription:
    """One subscribed stream: a daemon thread pumps the sync frame iterator,
    JPEG-encodes, and parks the newest frame in a one-slot holder (older frames
    are simply overwritten — drop-oldest). ``credit``/``credit_mode`` belong to
    the sender coroutine's pacing (see module docstring)."""

    def __init__(self, stream_id: str, frames, loop: asyncio.AbstractEventLoop,
                 wake: asyncio.Event):
        self.stream_id = stream_id
        self._frames = frames
        self._loop = loop
        self._wake = wake
        self._stop = threading.Event()
        self._latest: bytes | None = None
        # Credit state (owned by the event loop — only touched there).
        self.credit_mode = False
        self.credit = 0
        self.last_send = 0.0
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name=f"wsvideo[{stream_id}]")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def has_frame(self) -> bool:
        """Non-consuming peek — used by the refill check."""
        return self._latest is not None

    def take(self) -> bytes | None:
        """Pop the newest frame (None when nothing new since the last take)."""
        buf, self._latest = self._latest, None
        return buf

    def _pump(self) -> None:
        try:
            for image in self._frames:
                if self._stop.is_set():
                    break
                try:
                    self._latest = encode_jpeg(image)
                except (ValueError, RuntimeError):
                    continue
                self._loop.call_soon_threadsafe(self._wake.set)
        except Exception:
            logger.warning("ws/video[%s]: stream pump died", self.stream_id,
                           exc_info=True)
        finally:
            # Unwind the generator chain NOW (hub.release in its finally) instead
            # of waiting for GC — the camera viewer count must drop when the
            # client unsubscribes, not "eventually".
            close = getattr(self._frames, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass


def _frame_message(stream_id: str, jpeg: bytes) -> bytes:
    sid = stream_id.encode("utf-8")
    return bytes([len(sid)]) + sid + jpeg


# ---------------------------------------------------------------------------
# Compressed-video passthrough (``camh264:<camera_id>``)
#
# isistream serves each camera's ORIGINAL Annex-B bitstream on a unix socket
# (`isistream/nal_relay.py` — the upstream contract; the path convention is
# duplicated here on purpose, monitor_web never imports isistream). We relay
# the access units over this websocket unchanged; the browser hardware-decodes
# them (WebCodecs) — no JPEG encode anywhere on that display path.
#
# WS payload (after the usual ``idLen | stream-id`` envelope), little-endian:
#   payload[0] = 0 (INIT)  → rest is JSON utf8
#       {"v": 1, "codec": "h264"|"h265", "camera_id": ..., "available": true}
#       or {"available": false, "reason": ...} when the unix socket is
#       absent/unreachable — sent once per outage so the JS falls back
#       immediately instead of waiting on a dead subscription.
#   payload[0] = 1 (AU)    → rest is f64 capture_ts | u8 flags (bit0=keyframe)
#       | Annex-B AU bytes — byte-identical to the unix-socket frame body, so
#       the relay is a pure prefix-and-forward.
#
# Deliberately NO credit/ack gating and NO latest-frame dropping here: a video
# decoder needs the continuous stream (every delta counts). If the send
# backlog explodes (browser can't drain), we emit an unavailable INIT and stop
# — the JS falls back to the JPEG path.
# ---------------------------------------------------------------------------

PT_INIT = 0
PT_AU = 1
# u32 length prefixes on the unix socket (little-endian).
_U32 = struct.Struct("<I")
# Sanity caps for the length-prefixed reads — a corrupt prefix must not make
# us allocate gigabytes.
_PT_MAX_HEADER = 64 * 1024
_PT_MAX_AU = 32 * 1024 * 1024
# Queued-but-unsent WS payloads before we declare backlog and bail out.
_PT_MAX_QUEUE = 512
# Reconnect backoff to the unix socket while subscribed (stays at the last).
_PT_BACKOFF_S = (0.5, 1.0, 2.0, 4.0)


def nal_socket_path(camera_id: str) -> str:
    """Socket path for one camera — mirrors isistream's convention exactly
    (``/tmp/isi3d_nal_<camera_id>.sock``, overridable via ISI3D_NAL_DIR)."""
    return os.path.join(os.environ.get("ISI3D_NAL_DIR", "/tmp"),
                        f"isi3d_nal_{camera_id}.sock")


def _recv_exact(sock: socket.socket, n: int, stop: threading.Event) -> bytes | None:
    """Read exactly ``n`` bytes; None on EOF or stop. The socket carries a
    short timeout so a stop is honoured within ~0.5 s even on a silent peer."""
    buf = bytearray()
    while len(buf) < n:
        if stop.is_set():
            return None
        try:
            chunk = sock.recv(n - len(buf))
        except TimeoutError:
            continue
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def relay_nal_session(sock_path: str, emit, stop: threading.Event) -> None:
    """ONE connection to the NAL unix socket: forward its header as an INIT
    payload (type 0, ``available: true``) then every AU frame as an AU payload
    (type 1). Returns on EOF/stop; raises ``OSError`` when the socket is
    absent, refuses, or violates the framing contract."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.connect(sock_path)
        raw = _recv_exact(sock, _U32.size, stop)
        if raw is None:
            return
        (hlen,) = _U32.unpack(raw)
        if not 0 < hlen <= _PT_MAX_HEADER:
            raise OSError(f"bad header length {hlen}")
        hdr_bytes = _recv_exact(sock, hlen, stop)
        if hdr_bytes is None:
            return
        try:
            header = json.loads(hdr_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise OSError(f"bad header json: {exc}") from exc
        header["available"] = True
        emit(bytes([PT_INIT]) + json.dumps(header).encode("utf-8"))
        while not stop.is_set():
            raw = _recv_exact(sock, _U32.size, stop)
            if raw is None:
                return
            (flen,) = _U32.unpack(raw)
            if not 0 < flen <= _PT_MAX_AU:
                raise OSError(f"bad AU length {flen}")
            body = _recv_exact(sock, flen, stop)   # f64 ts | u8 flags | AU
            if body is None:
                return
            emit(bytes([PT_AU]) + body)
    finally:
        try:
            sock.close()
        except OSError:
            pass


def run_nal_relay(sock_path: str, camera_id: str, emit, stop: threading.Event,
                  *, backoff_s: tuple[float, ...] = _PT_BACKOFF_S) -> None:
    """Reconnect-with-backoff wrapper around :func:`relay_nal_session`.

    On the FIRST failure of an outage it emits one unavailable INIT (so the
    JS can fall back immediately), then keeps retrying quietly while
    subscribed. A later successful connect re-emits an available INIT and the
    stream restarts at a keyframe (upstream contract), so a client that stuck
    around can resume decoding.
    """
    failures = 0
    while not stop.is_set():
        try:
            relay_nal_session(sock_path, emit, stop)
            failures = 0            # clean session (connect worked); retry soon
            delay = backoff_s[0]
        except OSError as exc:
            if failures == 0:
                emit(bytes([PT_INIT]) + json.dumps({
                    "v": 1, "camera_id": camera_id,
                    "available": False, "reason": str(exc) or type(exc).__name__,
                }).encode("utf-8"))
            delay = backoff_s[min(failures, len(backoff_s) - 1)]
            failures += 1
        stop.wait(delay)


class _PassthroughSub:
    """One ``camh264:`` subscription: a daemon thread relays the camera's NAL
    unix socket into a FIFO of ws payloads (NEVER latest-only — the decoder
    needs every AU). The shared sender drains the FIFO in order, outside the
    credit machinery. A backlog blow-up emits an unavailable INIT and stops
    the pump — the JS falls back to JPEG."""

    fifo = True

    def __init__(self, stream_id: str, camera_id: str,
                 loop: asyncio.AbstractEventLoop, wake: asyncio.Event):
        self.stream_id = stream_id
        self._loop = loop
        self._wake = wake
        self._stop = threading.Event()
        self._pending: list[bytes] = []          # guarded by _lock
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=run_nal_relay,
            args=(nal_socket_path(camera_id), camera_id, self._emit, self._stop),
            daemon=True, name=f"wsvideo[{stream_id}]")
        self._camera_id = camera_id

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, payload: bytes) -> None:
        with self._lock:
            if len(self._pending) >= _PT_MAX_QUEUE:
                # The ws sender isn't keeping up — a video stream can't drop
                # frames, so end the passthrough honestly instead of streaming
                # ever-older video. One final INIT tells the JS to fall back.
                logger.warning("ws/video[%s]: passthrough backlog (%d payloads) — "
                               "stopping, client falls back to JPEG",
                               self.stream_id, len(self._pending))
                self._pending.clear()
                self._pending.append(bytes([PT_INIT]) + json.dumps({
                    "v": 1, "camera_id": self._camera_id,
                    "available": False, "reason": "backlog",
                }).encode("utf-8"))
                self._stop.set()
            else:
                self._pending.append(payload)
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            pass    # event loop already closed (shutdown race) — nothing to wake

    def take(self) -> bytes | None:
        with self._lock:
            return self._pending.pop(0) if self._pending else None


def _create_subscription(state, stream_id: str, loop: asyncio.AbstractEventLoop,
                         wake: asyncio.Event):
    """Subscription factory for every stream kind. ``LookupError`` on an
    unknown id — same surface the JPEG kinds have always had."""
    kind, _, rest = stream_id.partition(":")
    if kind == "camh264":
        if not rest or rest not in _load_cameras_from_backbone_yaml(
                state.settings.backbone_config_path):
            raise LookupError(f"camera {rest!r} not configured")
        return _PassthroughSub(stream_id, rest, loop, wake)
    return _Subscription(stream_id, _build_stream(state, stream_id), loop, wake)


@router.websocket("/ws/video")
async def ws_video(ws: WebSocket) -> None:
    """One socket per dashboard tab; subscriptions come and go with the panels."""
    await ws.accept()
    state = ws.app.state
    loop = asyncio.get_running_loop()
    wake = asyncio.Event()
    subs: dict[str, _Subscription] = {}

    async def sender() -> None:
        while True:
            # Timeout drives the ACK_REFILL_S check even when no new frame or
            # ack wakes us (e.g. a hidden tab that stopped acking).
            try:
                await asyncio.wait_for(wake.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            wake.clear()
            now = time.monotonic()
            for sub in list(subs.values()):
                if getattr(sub, "fifo", False):
                    # Passthrough: drain IN ORDER, no credit, no dropping — a
                    # video decoder needs the continuous stream.
                    while True:
                        buf = sub.take()
                        if buf is None:
                            break
                        try:
                            await ws.send_bytes(_frame_message(sub.stream_id, buf))
                        except RuntimeError:
                            return
                    continue
                if not sub.has_frame():
                    continue
                may_send = (
                    not sub.credit_mode
                    or sub.credit > 0
                    or now - sub.last_send >= ACK_REFILL_S
                )
                if not may_send:
                    # Leave the frame in the holder: the pump keeps overwriting
                    # it with newer, so the eventual send is the NEWEST frame.
                    continue
                buf = sub.take()
                if buf is None:
                    continue
                if sub.credit_mode:
                    sub.credit = max(0, sub.credit - 1)
                sub.last_send = now
                try:
                    await ws.send_bytes(_frame_message(sub.stream_id, buf))
                except RuntimeError:
                    # Socket closed between the check and the send (page
                    # navigated away mid-frame). Normal shutdown race — end the
                    # sender quietly; the endpoint's finally cancels/cleans up.
                    return

    send_task = asyncio.create_task(sender())
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (ValueError, KeyError):
                continue
            except RuntimeError:
                # Socket already closed (app shutdown races the receive) —
                # a normal end of life, not an error worth a traceback.
                return
            if not isinstance(msg, dict):
                continue
            sid = str(msg.get("sub") or "")
            if sid and sid not in subs:
                try:
                    sub = _create_subscription(state, sid, loop, wake)
                except LookupError as exc:
                    await ws.send_text(json.dumps({"error": str(exc), "stream": sid}))
                    continue
                subs[sid] = sub
                sub.start()
                continue
            sid = str(msg.get("unsub") or "")
            if sid:
                sub = subs.pop(sid, None)
                if sub is not None:
                    sub.stop()
                continue
            sid = str(msg.get("ack") or "")
            if sid:
                sub = subs.get(sid)
                # Unknown sid = resubscribe race; passthrough subs are outside
                # the credit machinery by design — both are ignored.
                if sub is not None and not getattr(sub, "fifo", False):
                    sub.credit_mode = True
                    sub.credit = min(CREDIT_WINDOW, sub.credit + 1)
                    wake.set()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning("ws/video: unexpected error, closing", exc_info=True)
    finally:
        send_task.cancel()
        for sub in subs.values():
            sub.stop()
