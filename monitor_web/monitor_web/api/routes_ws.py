"""WebSocket `/ws/tracks` — fan out incoming UDP envelopes to live clients."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backbone.metadata.schemas import Track2DMessage, Track3DMessage
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


def _serialize(msg: Any) -> dict:
    """Turn a pydantic envelope back into JSON-able dict for the browser."""
    if isinstance(msg, (Track2DMessage, Track3DMessage)):
        return msg.model_dump(mode="json")
    return msg


@router.websocket("/ws/tracks")
async def ws_tracks(ws: WebSocket) -> None:
    """One WebSocket = one fan-out subscriber. Lifespan is the connection."""
    await ws.accept()
    broadcast: asyncio.Queue = ws.app.state.broadcast_queue
    bus = ws.app.state.bus

    # Replay the most recent snapshot so the freshly-connected client paints
    # immediately rather than waiting for the next UDP packet.
    snap = bus.snapshot()
    for msg in list(snap.last_track2d_by_id.values()) + list(snap.last_track3d_by_id.values()):
        try:
            await ws.send_json(_serialize(msg))
        except WebSocketDisconnect:
            return

    try:
        while True:
            msg = await broadcast.get()
            try:
                await ws.send_json(_serialize(msg))
            except WebSocketDisconnect:
                return
    except WebSocketDisconnect:
        return
    except Exception:
        logger.warning("ws/tracks: unexpected error, closing", exc_info=True)
        try:
            await ws.close()
        except RuntimeError:
            pass
