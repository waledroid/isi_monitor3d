"""WebSocket `/ws/tracks` — fan out incoming UDP envelopes to live clients."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backbone.comms.schemas import Track2DMessage, Track3DMessage
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


def _serialize(msg: Any) -> dict | None:
    """JSON-able dict for the browser, or ``None`` for envelope types this
    socket doesn't carry (diagnostics, zone_state, config…). The broadcast
    queue fans out EVERY bus envelope — passing an unknown pydantic model
    through raw made ``send_json`` raise and killed the socket."""
    if isinstance(msg, (Track2DMessage, Track3DMessage)):
        return msg.model_dump(mode="json")
    return None


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
        payload = _serialize(msg)
        if payload is None:
            continue
        try:
            await ws.send_json(payload)
        except WebSocketDisconnect:
            return

    try:
        while True:
            msg = await broadcast.get()
            payload = _serialize(msg)
            if payload is None:
                continue
            try:
                await ws.send_json(payload)
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
