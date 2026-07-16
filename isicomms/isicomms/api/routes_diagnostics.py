"""GET /diagnostics — per-node diagnostics heartbeat summary."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


@router.get("/diagnostics", dependencies=[Depends(require_token)])
async def diagnostics(request: Request) -> JSONResponse:
    """Per-node diagnostics: freshness + last heartbeat payload."""
    subscriber = request.app.state.subscriber
    settings = request.app.state.settings
    now = time.time()

    result = []
    for node_id, node in subscriber.snapshot_nodes().items():
        fresh = subscriber.node_alive(node_id, now, settings.node_stale_after_s)
        result.append({
            "node_id": node_id,
            "fresh": fresh,
            "diagnostics": node.last_diagnostics.model_dump() if node.last_diagnostics else None,
        })

    return JSONResponse({"nodes": result, "count": len(result)})
