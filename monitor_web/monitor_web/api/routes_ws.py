"""WebSockets `/ws/tracks` (UDP track fan-out) + `/ws/overlays` (per-camera
observation feed for client-side overlay drawing on the passthrough video)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backbone.comms.schemas import ObservationsMessage, Track2DMessage, Track3DMessage
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# /ws/overlays poll rate — a few KB of JSON per camera at ~15 Hz.
OVERLAYS_POLL_S = 1.0 / 15.0


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


# ---------------------------------------------------------------------------
# /ws/overlays — the client-side overlay feed for the compressed-video
# passthrough: the browser decodes the camera bitstream itself, so it needs
# the raw per-camera observations (boxes, masks, skeletons) to draw on top.
# ---------------------------------------------------------------------------


def _overlay_payload(msg: ObservationsMessage) -> dict:
    """Lean JSON for one camera's observations. Coordinates stay in the
    message's ``frame_wh`` space; optional fields are omitted when absent."""
    dets = []
    for d in msg.dets:
        det: dict[str, Any] = {
            "cls": d.cls,
            "confidence": d.confidence,
            "bbox_xyxy": list(d.bbox_xyxy),
            "foot_uv": list(d.foot_uv),
        }
        if d.mask_poly is not None:
            det["mask_poly"] = [list(p) for p in d.mask_poly]
        if d.keypoints_uv is not None:
            det["keypoints_uv"] = [list(k) for k in d.keypoints_uv]
        dets.append(det)
    return {
        "camera_id": msg.camera_id,
        "ts": msg.ts,
        "frame_wh": list(msg.frame_wh),
        "dets": dets,
    }


@router.websocket("/ws/overlays")
async def ws_overlays(ws: WebSocket) -> None:
    """Per-connection POLLING sender, deliberately NOT the broadcast queue
    (that queue is track-only by design). The client sends
    ``{"cameras": ["cam_a", ...]}`` (re-sendable to change the set); we poll
    the bus snapshot ~15 Hz and push each camera's observations ONLY when its
    ``ts`` changed — silent while the Backbone is stopped or the frame is
    unchanged."""
    await ws.accept()
    bus = ws.app.state.bus
    cameras: list[str] = []
    last_ts: dict[str, float] = {}
    recv = asyncio.create_task(ws.receive_json())
    try:
        while True:
            done, _ = await asyncio.wait({recv}, timeout=OVERLAYS_POLL_S)
            if done:
                try:
                    msg = recv.result()     # re-raises WebSocketDisconnect
                except ValueError:          # non-JSON text — ignore, keep going
                    msg = None
                recv = asyncio.create_task(ws.receive_json())
                if isinstance(msg, dict) and isinstance(msg.get("cameras"), list):
                    cameras = [str(c) for c in msg["cameras"]]
                    last_ts = {c: t for c, t in last_ts.items() if c in cameras}
            snap = bus.snapshot()
            for cam in cameras:
                obs = snap.observations_by_camera.get(cam)
                if obs is None or last_ts.get(cam) == obs.ts:
                    continue
                last_ts[cam] = obs.ts
                await ws.send_json(_overlay_payload(obs))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning("ws/overlays: unexpected error, closing", exc_info=True)
        try:
            await ws.close()
        except RuntimeError:
            pass
    finally:
        recv.cancel()
