"""Étagère zones — config authoring (GET/POST /api/etagere) + live cell state.

The dashboard AUTHORS config/etagere.yaml; isistream CONSUMES it (per-cell
crop inference). No detector here. A save hot-restarts a running isistream
exactly like the detection-model save does (see routes_config.py's
config-save handler for the identical restart closure).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from backbone.shared.etagere import (
    EtagereConfig,
    cells_from_corners,
    load_etagere_config,
    resolve_config_path,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from .routes_config import _write_yaml_atomic

logger = logging.getLogger(__name__)
router = APIRouter()


def _path(request: Request) -> Path:
    cfg = request.app.state.settings
    by = Path(cfg.backbone_config_path)
    try:
        backbone_cfg = yaml.safe_load(by.read_text()) or {}
    except (OSError, yaml.YAMLError):
        backbone_cfg = {}
    return resolve_config_path(backbone_cfg, by)


@router.get("/api/etagere")
def get_etagere(request: Request) -> dict[str, Any]:
    cfg = load_etagere_config(_path(request))
    return cfg.model_dump(mode="json")


class AutosplitBody(BaseModel):
    corners: list[list[float]]
    rows: int = 3
    cols: int = 3


@router.post("/api/etagere/autosplit")
def autosplit(body: AutosplitBody) -> dict[str, Any]:
    if len(body.corners) != 4:
        raise HTTPException(status_code=422, detail="corners must have exactly 4 points")
    cells = cells_from_corners(body.corners, body.rows, body.cols)
    out = []
    for cell in cells:
        d = cell.model_dump(mode="json")
        # Bilinear interpolation leaves float noise (e.g. 60.00000000000001) on
        # axis-aligned quads — round to sub-pixel precision so the JS client
        # (and this endpoint's own callers) get clean, comparable numbers.
        d["rect"] = [round(v, 6) for v in d["rect"]]
        out.append(d)
    return {"cells": out}


@router.post("/api/etagere")
def post_etagere(body: dict[str, Any], request: Request) -> dict[str, Any]:
    try:
        cfg = EtagereConfig.model_validate(body)
    except ValidationError as e:
        # include_context=False drops pydantic's raw exception objects from
        # the "ctx" field of custom (model_validator) errors, which are not
        # JSON-serializable and would 500 the error response itself.
        raise HTTPException(status_code=422, detail=e.errors(include_context=False)) from e
    _write_yaml_atomic(_path(request), cfg.model_dump(mode="json"))

    # Direction 1 hot-apply: a RUNNING isistream producer reads etagere.yaml
    # only at spawn, so a config change takes a producer restart — same
    # closure as the detection-model save in routes_config.py.
    host = getattr(request.app.state, "isistream", None)
    if host is not None and host.points_mode() and host.status().get("running"):
        logger.info("etagere: restarting isistream to apply the new config")

        def _restart() -> None:
            try:
                host.stop()
                host.start()
            except Exception:
                logger.warning("etagere: isistream restart failed", exc_info=True)

        threading.Thread(target=_restart, name="etagere-isistream-restart", daemon=True).start()

    return cfg.model_dump(mode="json")


@router.get("/api/etagere/state")
def etagere_state(request: Request) -> dict[str, Any]:
    bus = request.app.state.bus
    snap = bus.snapshot()
    states: dict[str, Any] = {}
    for zone_id, msg in snap.etagere_by_zone.items():
        matrix = [["unknown"] * msg.cols for _ in range(msg.rows)]
        for c in msg.cells:
            if 1 <= c.r <= msg.rows and 1 <= c.c <= msg.cols:
                matrix[c.r - 1][c.c - 1] = c.state
        states[zone_id] = {
            "name": msg.name,
            "camera_id": msg.camera_id,
            "rows": msg.rows,
            "cols": msg.cols,
            "matrix": matrix,
            "cells": [c.model_dump() for c in msg.cells],
            "ts": msg.ts,
        }
    return {"states": states}
