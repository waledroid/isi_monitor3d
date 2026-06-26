"""GET /config — per-node raw config advertisement."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


@router.get("/config", dependencies=[Depends(require_token)])
async def config(request: Request) -> JSONResponse:
    """Per-node config: the retained ConfigMessage payload or null if not yet seen."""
    subscriber = request.app.state.subscriber

    result = []
    for node_id, node_state in subscriber.snapshot_nodes().items():
        result.append({
            "node_id": node_id,
            "topic_version": node_state.topic_version,
            "config": node_state.config.model_dump() if node_state.config else None,
        })

    return JSONResponse({"nodes": result, "count": len(result)})
