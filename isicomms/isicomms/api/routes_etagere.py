"""GET /etagere — every node's latest étagère (bin-rack) cell matrices."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


def _entry(node_id: str, msg) -> dict:
    matrix = [["unknown"] * msg.cols for _ in range(msg.rows)]
    for c in msg.cells:
        if 1 <= c.r <= msg.rows and 1 <= c.c <= msg.cols:
            matrix[c.r - 1][c.c - 1] = c.state
    return {
        "node_id": node_id, "zone_id": msg.zone_id, "name": msg.name,
        "camera_id": msg.camera_id, "rows": msg.rows, "cols": msg.cols,
        "cells": [c.model_dump() for c in msg.cells], "matrix": matrix, "ts": msg.ts,
    }


def _all(request: Request) -> list[dict]:
    out = []
    for node_id, st in request.app.state.subscriber.snapshot_nodes().items():
        for msg in st.etagere_by_zone.values():
            out.append(_entry(node_id, msg))
    return out


@router.get("/etagere", dependencies=[Depends(require_token)])
async def etagere(request: Request) -> JSONResponse:
    items = _all(request)
    return JSONResponse({"etageres": items, "count": len(items)})


@router.get("/etagere/{zone_id}", dependencies=[Depends(require_token)])
async def etagere_one(zone_id: str, request: Request) -> JSONResponse:
    for e in _all(request):
        if e["zone_id"] == zone_id:
            return JSONResponse(e)
    raise HTTPException(status_code=404, detail=f"unknown étagère {zone_id!r}")
