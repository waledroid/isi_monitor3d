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
(rectified verification view), ``zone:<patch_id>``, ``unified``.

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
Zone/unified/warp streams are display_fps-capped upstream, so credit never
binds them. The hidden MP4 dev viewer and curl debugging keep the MJPEG
endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..video_stream import encode_jpeg
from .routes_video import build_cam_stream, build_unified_stream, build_zone_stream

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
                await ws.send_bytes(_frame_message(sub.stream_id, buf))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (ValueError, KeyError):
                continue
            if not isinstance(msg, dict):
                continue
            sid = str(msg.get("sub") or "")
            if sid and sid not in subs:
                try:
                    frames = _build_stream(state, sid)
                except LookupError as exc:
                    await ws.send_text(json.dumps({"error": str(exc), "stream": sid}))
                    continue
                sub = _Subscription(sid, frames, loop, wake)
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
                if sub is not None:      # unknown sid = resubscribe race; ignore
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
