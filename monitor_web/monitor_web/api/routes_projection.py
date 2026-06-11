"""Pixel ↔ floor projection endpoints (S17).

Two POST endpoints back the dashboard's "draw zones on the camera image" UX
(``draw_mode.js``) and the persistent zone overlay rendered on top of every
CAM tab (``camera_zones.js``). Both go through the Backbone's own
:mod:`backbone.shared.geometry` helpers + :class:`backbone.shared.camera_rig.CameraRig`
so the math is identical to the production homography pipeline — no JS
duplication of OpenCV.

* ``POST /api/project/pixel-to-floor`` — each ``(u, v)`` undistorted via
  ``cv2.undistortPoints`` then mapped through ``H`` to world metres. Operator
  clicks on the CAM image; we store the resulting ``(X, Y)`` in ``zones.yaml``.
* ``POST /api/project/floor-to-pixel`` — each world ``(X, Y)`` mapped back to
  ``(u, v)`` via ``H^-1``. The dashboard projects every zone polygon to each
  camera so the operator sees the zone outlined on the live feed.

Both require a valid ``calibration.json`` (Mode 1 ``single_cam_4pt`` or Mode 2
``multical_full``). The path is taken from ``Settings.calibration_path`` if set,
else from ``backbone.yaml`` 's ``calibration_path`` key. Missing / unloadable
calibration → HTTP 503.

Distortion note: ``floor_to_pixel`` returns *ideal* (undistorted) pixels. For
Mode 1 (``D = 0``) this is exact. For Mode 2 the small re-distortion error
near frame edges is documented but not applied here — operator-drawn zones
typically sit in the centre of the frame where ``D`` is sub-pixel anyway.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import yaml
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import floor_to_pixel, pixel_to_floor, undistort_points
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---- request/response schemas ----


class PixelToFloorBody(BaseModel):
    camera_id: str = Field(..., min_length=1)
    points: list[tuple[float, float]] = Field(..., min_length=1)


class FloorToPixelBody(BaseModel):
    camera_id: str = Field(..., min_length=1)
    polygon: list[tuple[float, float]] = Field(..., min_length=1)


# ---- calibration resolution + caching ----


def _resolve_calibration_path(cfg) -> Path | None:
    """Mirror of ``_resolve_zones_path`` — explicit override wins, else read
    from ``backbone.yaml``'s ``calibration_path`` key, else ``None``."""
    if cfg.calibration_path is not None:
        return Path(cfg.calibration_path)
    bb_path = Path(cfg.backbone_config_path)
    if not bb_path.exists():
        return None
    try:
        data = yaml.safe_load(bb_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("calibration_path")
    return Path(raw) if raw else None


@functools.lru_cache(maxsize=4)
def _load_rig_cached(path_str: str, mtime_ns: int) -> CameraRig:
    """Load a CameraRig keyed by (path, mtime). Re-edits to the file invalidate
    the cache automatically because ``os.stat().st_mtime_ns`` changes."""
    return CameraRig.from_file(path_str)


def _resolve_rig(cfg) -> CameraRig:
    path = _resolve_calibration_path(cfg)
    if path is None or not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No calibration.json available. Run "
                "`python -m calibration.calibrate single-cam` (Mode 1) or "
                "`calibrate-all` (Mode 2) and set calibration_path in backbone.yaml."
            ),
        )
    try:
        mtime = path.stat().st_mtime_ns
        return _load_rig_cached(str(path.resolve()), mtime)
    except Exception as exc:
        logger.exception("projection: failed to load %s", path)
        raise HTTPException(status_code=503, detail=f"calibration.json unreadable: {exc}") from exc


def _camera_view(rig: CameraRig, camera_id: str):
    if camera_id not in rig:
        raise HTTPException(
            status_code=404,
            detail=f"camera_id {camera_id!r} not in calibration "
                   f"(available: {list(rig.camera_ids)})",
        )
    return rig[camera_id]


# ---- handlers ----


@router.post("/api/project/pixel-to-floor")
async def pixel_to_floor_endpoint(body: PixelToFloorBody, request: Request) -> JSONResponse:
    """Undistort + map source-pixel points to world floor metres ``(X, Y)``.

    Input pixels are assumed to be in the **source frame** (the camera's
    native resolution, not the dashboard's displayed size). The JS side
    handles the ``object-fit: cover`` display→source mapping.
    """
    import numpy as np

    cfg = request.app.state.settings
    rig = _resolve_rig(cfg)
    view = _camera_view(rig, body.camera_id)
    pts_uv = np.asarray(body.points, dtype=np.float64)
    pts_undist = undistort_points(pts_uv, view.K, view.D)
    pts_world = pixel_to_floor(pts_undist, view.H)
    return JSONResponse({
        "camera_id": body.camera_id,
        "points": [[float(x), float(y)] for x, y in pts_world],
    })


@router.post("/api/project/floor-to-pixel")
async def floor_to_pixel_endpoint(body: FloorToPixelBody, request: Request) -> JSONResponse:
    """Map world floor ``(X, Y)`` metres back to source-pixel coords.

    Returns the polygon + the camera's source frame ``image_size_wh`` so the
    client can scale to its displayed size (with ``object-fit: cover``).
    """
    import numpy as np

    cfg = request.app.state.settings
    rig = _resolve_rig(cfg)
    view = _camera_view(rig, body.camera_id)
    pts_world = np.asarray(body.polygon, dtype=np.float64)
    pts_uv = floor_to_pixel(pts_world, view.H)
    w, h = view.image_size_wh
    return JSONResponse({
        "camera_id": body.camera_id,
        "image_size": [int(w), int(h)],
        "points": [[float(u), float(v)] for u, v in pts_uv],
    })


@router.get("/api/project/cameras")
async def list_calibrated_cameras(request: Request) -> JSONResponse:
    """Lightweight 'which cameras can I project against?' query.

    Used by the dashboard to show/hide the per-camera draw target. Returns
    ``{cameras: ["cam_a", "cam_b"], mode: "multical_full"|"single_cam_4pt"}``
    or ``{cameras: [], error: "..."}`` when no calibration is available.
    """
    cfg = request.app.state.settings
    try:
        rig = _resolve_rig(cfg)
    except HTTPException as exc:
        return JSONResponse(
            {"cameras": [], "mode": None, "error": str(exc.detail)},
            status_code=200,   # graceful degrade — the modal just hides CAM targets
        )
    payload: dict[str, Any] = {
        "cameras": list(rig.camera_ids),
        "mode": rig.calibration_mode,
        "image_sizes": {cam_id: list(rig[cam_id].image_size_wh) for cam_id in rig.camera_ids},
    }
    return JSONResponse(payload)
