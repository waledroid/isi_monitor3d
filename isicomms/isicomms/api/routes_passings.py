"""GET /passings — recent zone-passing events across all nodes.

``GET /passages`` is a French alias for the same endpoint (AGV engineers on
site work in French); both names stay supported — /passings is the frozen
canonical, /passages is additive.

Query params:
  ?limit=N    max events to return (clamped to passings_buffer)
  ?node=<id>  filter to a single node
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


@router.get("/passings", dependencies=[Depends(require_token)])
@router.get("/passages", dependencies=[Depends(require_token)],
            include_in_schema=False)   # FR alias — same handler, same shape
async def passings(
    request: Request,
    limit: int | None = Query(default=None, ge=1),
    node: str | None = Query(default=None),
) -> JSONResponse:
    """Newest-last list of zone-passing events, each tagged with node_id."""
    subscriber = request.app.state.subscriber
    settings = request.app.state.settings
    cap = settings.passings_buffer

    collected = []
    for node_id, node_state in subscriber.snapshot_nodes().items():
        if node is not None and node_id != node:
            continue
        for msg in node_state.passings:
            item = msg.model_dump()
            item["node_id"] = node_id
            collected.append(item)

    # Sort by ts ascending (newest last).
    collected.sort(key=lambda x: x.get("ts", 0.0))

    # Clamp to passings_buffer, then apply the optional limit.
    collected = collected[-cap:]
    if limit is not None:
        collected = collected[-limit:]

    return JSONResponse({"passings": collected, "count": len(collected)})
