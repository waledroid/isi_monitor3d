"""WebSocket `/ws/video` — ALL dashboard video over ONE multiplexed connection.

Why: the dashboard used one MJPEG `<img>` per panel (big view + up to 3 zone
panels + hidden tabs). Each MJPEG stream permanently occupies one of the
browser's ~6 HTTP/1.1 connections per host, so with enough panels a settings
POST (or a page reload) could not get a connection slot and the whole UI froze.
WebSocket connections are exempt from that cap, and one socket carries every
panel's frames — the connection-limit class of bugs is gone structurally.

Protocol:
  client → server (text/JSON): {"sub": "<stream-id>"} / {"unsub": "<stream-id>"}
  server → client (binary):    uint8 idLen | stream-id utf8 | JPEG bytes
  server → client (text/JSON): {"error": ..., "stream": ...} for a bad sub

Stream ids: ``cam:<camera_id>`` (live detect view), ``cam:<camera_id>:warp``
(rectified verification view), ``zone:<patch_id>``, ``unified``.

Each subscription reuses the SAME sync frame pipelines as the MJPEG endpoints
(`build_cam_stream` / `build_zone_stream` / `build_unified_stream`) running in a
dedicated daemon thread, publishing the LATEST JPEG into a one-slot holder —
drop-oldest semantics, so a slow client never builds a queue. The async sender
drains the holders to the socket. GPU usage is identical to the MJPEG path
(those generators already ran on worker threads); the hidden MP4 dev viewer and
curl debugging keep the MJPEG endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..video_stream import encode_jpeg
from .routes_video import build_cam_stream, build_unified_stream, build_zone_stream

logger = logging.getLogger(__name__)

router = APIRouter()


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
    are simply overwritten — drop-oldest)."""

    def __init__(self, stream_id: str, frames, loop: asyncio.AbstractEventLoop,
                 wake: asyncio.Event):
        self.stream_id = stream_id
        self._frames = frames
        self._loop = loop
        self._wake = wake
        self._stop = threading.Event()
        self._latest: bytes | None = None
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name=f"wsvideo[{stream_id}]")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

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
            await wake.wait()
            wake.clear()
            for sub in list(subs.values()):
                buf = sub.take()
                if buf is not None:
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
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning("ws/video: unexpected error, closing", exc_info=True)
    finally:
        send_task.cancel()
        for sub in subs.values():
            sub.stop()
