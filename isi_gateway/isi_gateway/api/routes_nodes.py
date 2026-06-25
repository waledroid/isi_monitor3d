"""GET /nodes — per-node summary (alive/stale, mode, cameras, latency/fps)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


@router.get("/nodes", dependencies=[Depends(require_token)])
async def nodes(request: Request) -> JSONResponse:
    """List all known nodes with their current status."""
    subscriber = request.app.state.subscriber
    settings = request.app.state.settings
    now = time.time()

    result = []
    for node_id, node in subscriber.snapshot_nodes().items():
        alive = subscriber.node_alive(node_id, now, settings.node_stale_after_s)
        status = "alive" if alive else "stale"

        # Populate from config advertisement if present.
        area: str | None = None
        mode: str | None = None
        cameras: list[str] = []
        if node.config is not None:
            area = node.config.area
            mode = node.config.mode
            cameras = list(node.config.cameras)

        # Populate from latest diagnostics if present.
        latency_ms: float | None = None
        fps: float | None = None
        if node.last_diagnostics is not None:
            d = node.last_diagnostics
            latency_ms = d.latency_ms.p95
            fps = d.fps

        result.append({
            "node_id": node_id,
            "area": area,
            "status": status,
            "last_seen": node.last_seen,
            "mode": mode,
            "cameras": cameras,
            "latency_ms": latency_ms,
            "fps": fps,
        })

    return JSONResponse({"nodes": result, "count": len(result)})
