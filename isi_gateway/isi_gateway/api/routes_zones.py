"""GET /zones — union of all nodes' config zones (global warehouse map)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


@router.get("/zones", dependencies=[Depends(require_token)])
async def zones(request: Request) -> JSONResponse:
    """Union of all zones from every node's retained config advertisement."""
    subscriber = request.app.state.subscriber

    result = []
    seen: set[tuple[str, str]] = set()

    for node_id, node_state in subscriber.snapshot_nodes().items():
        if node_state.config is None:
            continue
        area = node_state.config.area
        for zspec in node_state.config.zones:
            key = (node_id, zspec.name)
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "node_id": node_id,
                "area": area,
                "name": zspec.name,
                "kind": zspec.kind,
                "type": zspec.type,
                "severity": zspec.severity,
                "polygon": zspec.polygon,
            })

    return JSONResponse({"zones": result, "count": len(result)})
