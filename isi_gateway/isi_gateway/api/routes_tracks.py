"""GET /tracks — flat track list across all nodes, with optional filters.

Filters:
  ?node=<node_id>   exact node match
  ?cls=<class>      exact class match
  ?zone=<name>      point-in-polygon: keep tracks whose floor position lies inside
                    the named zone (searched across all nodes' config advertisements)
"""

from __future__ import annotations

import logging

import numpy as np
from backbone.shared.zones import Zone
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from .auth import require_token

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_zone(zone_name: str, nodes_snapshot: dict) -> Zone | None:
    """Build a ``Zone`` from the first node config that names ``zone_name``."""
    for node in nodes_snapshot.values():
        if node.config is None:
            continue
        for zspec in node.config.zones:
            if zspec.name == zone_name:
                try:
                    polygon = np.asarray(zspec.polygon, dtype=np.float64)
                    return Zone(
                        name=zspec.name,
                        kind=zspec.kind,
                        type=zspec.type,
                        severity=zspec.severity,
                        polygon=polygon,
                    )
                except Exception as exc:
                    logger.warning("routes_tracks: bad zone polygon %r: %s", zone_name, exc)
                    return None
    return None


@router.get("/tracks", dependencies=[Depends(require_token)])
async def tracks(
    request: Request,
    node: str | None = Query(default=None),
    cls: str | None = Query(default=None),
    zone: str | None = Query(default=None),
) -> JSONResponse:
    """Flat list of the most recent track from every node, each tagged with node_id."""
    subscriber = request.app.state.subscriber
    nodes_snapshot = subscriber.snapshot_nodes()

    # Build zone filter once (searches all node configs).
    zone_filter: Zone | None = None
    if zone is not None:
        zone_filter = _build_zone(zone, nodes_snapshot)
        if zone_filter is None:
            logger.debug("routes_tracks: zone %r not found in any node config", zone)

    result = []
    for node_id, node_state in nodes_snapshot.items():
        # Node filter.
        if node is not None and node_id != node:
            continue

        # 2D tracks.
        for _track_id, msg in node_state.last_track2d_by_id.items():
            if cls is not None and msg.cls != cls:
                continue
            if zone_filter is not None:
                if not zone_filter.contains((msg.xy_m[0], msg.xy_m[1])):
                    continue
            item = msg.model_dump()
            item["node_id"] = node_id
            result.append(item)

        # 3D tracks (also checked against the 2D floor position via xyz_m[:2]).
        for _track_id, msg in node_state.last_track3d_by_id.items():
            if cls is not None and msg.cls != cls:
                continue
            if zone_filter is not None:
                if not zone_filter.contains((msg.xyz_m[0], msg.xyz_m[1])):
                    continue
            item = msg.model_dump()
            item["node_id"] = node_id
            result.append(item)

    return JSONResponse({"tracks": result, "count": len(result)})
