"""Warehouse layout twin endpoints + rectified-floor snapshot for tracing.

Consumer-side: reads the calibration (via routes_video helpers) and the layout
YAML. No Backbone import.
"""
from __future__ import annotations

import base64
import logging

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import dashboard_config
from ..floor_rectify import (
    cropped_bounds,
    rectify_frame,
    rectify_params_for_frame,
    world_rect_to_pixel_box,
)
from ..warehouse_map import validate_map
from .routes_video import _load_cameras_from_backbone_yaml, _warp_camera, grab_real_frame

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/warehouse-map")
async def get_warehouse_map(request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    return JSONResponse(validate_map(dashboard_config.read_section(cfg, "warehouse_map")))


@router.post("/api/warehouse-map")
async def post_warehouse_map(request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    body = await request.json()
    try:
        validated = validate_map(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dashboard_config.write_section(cfg, "warehouse_map", validated)
    return JSONResponse(validated)


@router.get("/api/warp-snapshot/{camera_id}")
async def warp_snapshot(
    camera_id: str, request: Request, crop: str | None = None,
) -> JSONResponse:
    """One rectified floor frame + its metric bounds, for use as a tracing underlay.

    Returns ``{image (b64 jpeg|null), x_min, y_min, px_per_m, out_wh}``. 404 when the
    camera isn't configured or isn't calibrated for the current mode.

    ``?crop=x0,y0,x1,y1`` (world metres) crops the rectified image to that work-area
    rectangle — a real pixel crop of the returned JPEG, with ``x_min``/``y_min``/``out_wh``
    shifted to the cropped sub-image so the client places it correctly.
    """
    cfg = request.app.state.settings
    cameras = _load_cameras_from_backbone_yaml(cfg.backbone_config_path)
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"camera {camera_id!r} not configured")
    cam = _warp_camera(cfg, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="camera not calibrated for current mode")

    # Grab ONE REAL live frame FIRST (skipping the hub's "connecting…" placeholder,
    # which a single read would otherwise capture while the source warms up), then
    # rectify at the frame's ACTUAL resolution using the same shared helper the CAM
    # warp uses — so the MAP underlay is identical to the rectified image in the CAM
    # tab (the frame-size guard handles a stream that differs from the calibration
    # size). If no real frame arrives within the timeout, fall back to the calibration
    # size so the metric bounds are still returned (image: null, never a 500).
    raw = None
    try:
        raw = grab_real_frame(camera_id, cameras[camera_id].get("source", {}), timeout=4.0)
    except Exception as exc:  # snapshot image is optional; bounds still returned below
        logger.warning("warp-snapshot %s: no frame (%s)", camera_id, exc)

    frame_wh = (int(raw.shape[1]), int(raw.shape[0])) if raw is not None else tuple(cam.image_size_wh)
    params = rectify_params_for_frame(cam.H, cam.image_size_wh, frame_wh)
    if params is None:
        raise HTTPException(status_code=409, detail="degenerate rectification (re-calibrate)")
    M, out_wh, bounds = params["M"], params["out_wh"], params["bounds"]

    warped = None
    if raw is not None:
        warped = rectify_frame(raw, cam.K, cam.D, cam.H, out_wh=out_wh, M=M)

    # Optional crop to a work-area world rectangle. A genuine pixel crop of the
    # rectified image; the returned bounds shift to the sub-image so the client
    # places it where the work area is and can expand it to full width.
    if crop:
        try:
            x0, y0, x1, y1 = (float(v) for v in crop.split(","))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="crop must be 'x0,y0,x1,y1'") from exc
        box = world_rect_to_pixel_box(bounds, (x0, y0, x1, y1))
        if box is not None:
            if warped is not None:
                u0, v0, u1, v1 = box
                warped = warped[v0:v1, u0:u1]
            bounds = cropped_bounds(bounds, box)

    image_b64 = None
    if warped is not None:
        ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    return JSONResponse({
        "image": image_b64,
        "x_min": bounds["x_min"], "y_min": bounds["y_min"],
        "px_per_m": bounds["px_per_m"], "out_wh": list(bounds["out_wh"]),
    })


# ---- detection twin: the live map mirror of the cam view(s) -------------------

@router.get("/api/map/twin")
async def map_twin(request: Request) -> JSONResponse:
    """Floor-projected snapshot of the dashboard's OWN detections — the 3D map's
    realtime digital twin of the cam view(s).

    Sources: the background ZoneDetectionWorker snapshot per camera (objects from
    the zones + people from full-frame pose, all from ONE frame) and the drawn
    zone-patch polygons. Everything is projected pixel→floor through the current
    mode's calibration: Mode 1 = cam_a only (mirrors CAM 1), Mode 2 = both cameras
    in the shared world frame (mirrors the unified view).

    ``{"available": false}`` when there is no calibration or no zone worker —
    the map then has nothing detection-driven to draw (graceful degrade)."""
    import numpy as np
    from backbone.shared.geometry import pixel_to_floor, undistort_points

    from ..zone_worker import SNAPSHOT_MAX_AGE_S  # noqa: F401  (freshness doc'd there)
    from .routes_projection import _resolve_rig
    from .routes_zone_patches import load_patches

    cfg = request.app.state.settings
    manager = getattr(request.app.state, "zone_manager", None)
    try:
        rig = _resolve_rig(cfg)
    except HTTPException as exc:
        return JSONResponse({"available": False, "reason": str(exc.detail)})
    if manager is None:
        return JSONResponse({"available": False, "reason": "no zone worker"})

    def project(view, pts_uv, frame_wh):
        """Pixel points (at the live frame size) → floor metres. Points are scaled
        to the CALIBRATION frame size first so K/D/H apply exactly."""
        pts = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)
        cal_w, cal_h = float(view.image_size_wh[0]), float(view.image_size_wh[1])
        if frame_wh and (int(frame_wh[0]), int(frame_wh[1])) != (int(cal_w), int(cal_h)):
            pts = pts * np.array([cal_w / float(frame_wh[0]), cal_h / float(frame_wh[1])])
        return pixel_to_floor(undistort_points(pts, view.K, view.D), view.H)

    objects, people, zones_out = [], [], []
    cams_used = []
    for cam_id in rig.camera_ids:
        view = rig[cam_id]
        snap = manager.fresh_snapshot(cam_id)
        if snap is not None:
            cams_used.append(cam_id)
            fwh = snap.get("frame_wh")
            for zone_id, dets in (snap.get("zones") or {}).items():
                feet = [(d.foot_uv if d.foot_uv is not None
                         else ((d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0, d.bbox_xyxy[3]))
                        for d in dets]
                if feet:
                    world = project(view, feet, fwh)
                    for d, (wx, wy) in zip(dets, world, strict=False):
                        objects.append({"cls": str(d.cls), "conf": float(d.confidence),
                                        "xy_m": [float(wx), float(wy)],
                                        "zone_id": zone_id, "camera": cam_id})
            feet = [p["foot_uv"] for p in (snap.get("people") or [])]
            if feet:
                world = project(view, feet, fwh)
                for p, (wx, wy) in zip(snap.get("people") or [], world, strict=False):
                    people.append({"conf": float(p.get("confidence", 0.0)),
                                   "xy_m": [float(wx), float(wy)], "camera": cam_id})
        # Zone-patch polygons for this camera → floor outlines (drawn-size guarded).
        for patch in load_patches(cfg):
            if str(patch.get("camera") or "cam_a") != cam_id:
                continue
            poly = patch.get("polygon")
            if not isinstance(poly, list) or len(poly) < 3:
                continue
            world = project(view, [[float(u), float(v)] for u, v in poly],
                            patch.get("frame_wh"))
            zones_out.append({
                "id": str(patch.get("id")), "name": patch.get("name") or "",
                "color": patch.get("color") or "#ff3b3b", "camera": cam_id,
                "polygon_m": [[float(x), float(y)] for x, y in world],
            })

    return JSONResponse({
        "available": True,
        "mode": 2 if len(rig.camera_ids) >= 2 else 1,
        "cameras": cams_used,
        "objects": objects,
        "people": people,
        "zones": zones_out,
    })
