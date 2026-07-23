"""GET /zones — union of all nodes' config zones (global warehouse map).

Each zone item is enriched with the node's latest retained ``ZoneStateMessage``
(``objects`` / ``count`` / ``state_ts``) — the same payload shape as the MQTT
``zone/<zone>`` topic, so a WMS/FMS can consume either transport identically.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


def _zone_entry(node_id: str, area: str, zspec, state) -> dict:
    """One zone item: spec fields + the latest zone-state payload fields."""
    return {
        "node_id": node_id,
        "area": area,
        "name": zspec.name,
        # Stable topic-segment identity — lets the /ui schema tree (and any
        # consumer) annotate id-keyed zone topics with the display name.
        "zone_id": zspec.zone_id,
        "kind": zspec.kind,
        "type": zspec.type,
        "severity": zspec.severity,
        "polygon": zspec.polygon,
        # Live contents from the retained zone/<zone> state (None = no state
        # received yet — distinct from an explicit empty list).
        "objects": [o.model_dump(mode="json") for o in state.objects] if state else None,
        "count": state.count if state else None,
        "state_ts": state.ts if state else None,
    }


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
            state = (node_state.zone_state_by_zone.get(zspec.zone_id)
                     or node_state.zone_state_by_zone.get(zspec.name))
            result.append(_zone_entry(node_id, area, zspec, state))

    return JSONResponse({"zones": result, "count": len(result)})


@router.get("/zones/{name}", dependencies=[Depends(require_token)])
async def zone_by_name(name: str, request: Request) -> JSONResponse:
    """One zone across all nodes that define it: spec + latest state per node."""
    subscriber = request.app.state.subscriber

    entries = []
    for node_id, node_state in subscriber.snapshot_nodes().items():
        if node_state.config is None:
            continue
        for zspec in node_state.config.zones:
            if zspec.name != name:
                continue
            state = (node_state.zone_state_by_zone.get(zspec.zone_id)
                     or node_state.zone_state_by_zone.get(name))
            entries.append(_zone_entry(node_id, node_state.config.area, zspec, state))

    if not entries:
        raise HTTPException(status_code=404, detail=f"zone {name!r} not found on any node")
    return JSONResponse({"name": name, "zones": entries, "count": len(entries)})
