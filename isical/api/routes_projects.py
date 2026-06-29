"""Project CRUD + phase-board status."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.project import (
    CAMERA_IDS,
    EXTRINSIC_TARGET_MIN,
    CameraSpec,
    create_project,
    delete_project,
    list_projects,
    load_project,
    save_project,
)
from ..core.runners import calibration_summary, intrinsic_summary, phase_status
from .deps import project_cfg, project_dir

router = APIRouter()


class CameraIn(BaseModel):
    type: Literal["rtsp", "usb"] = "rtsp"
    url: str = ""
    device: str = ""


class CreateBody(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    cam_a: CameraIn
    cam_b: CameraIn | None = None


@router.get("/api/projects")
async def projects(request: Request) -> dict:
    data_dir = request.app.state.settings.data_dir
    out = []
    for name in list_projects(data_dir):
        cfg = load_project(data_dir / name)
        out.append({"name": name, "cameras": cfg.configured_cameras(),
                    "mode2": cfg.is_mode2()})
    return {"projects": out}


@router.post("/api/projects")
async def create(request: Request, body: CreateBody) -> dict:
    cams = {"cam_a": CameraSpec(id="cam_a", **body.cam_a.model_dump())}
    if body.cam_b is not None and (body.cam_b.url or body.cam_b.device):
        cams["cam_b"] = CameraSpec(id="cam_b", **body.cam_b.model_dump())
    if not cams["cam_a"].configured():
        raise HTTPException(status_code=422, detail="cam_a needs an RTSP url or USB device")
    try:
        path = create_project(request.app.state.settings.data_dir, body.name, cams)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "path": str(path)}


@router.delete("/api/projects/{name}")
async def delete(request: Request, name: str) -> dict:
    s = request.app.state.settings
    try:
        delete_project(s.data_dir, name, runs_dir=s.runs_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/api/p/{name}/status")
async def status(request: Request, name: str) -> dict:
    d = project_dir(request, name)
    return phase_status(d)


@router.get("/api/p/{name}/intrinsic-summary")
async def intr_summary(request: Request, name: str) -> dict:
    """Per-camera intrinsics from work/intrinsic.json (after the Intrinsic solve).

    Returns ``{"rms_gate_px", "cameras": {cam: {image_size, fx, fy, cx, cy,
    K, dist, rms}}}`` when solved, or ``{"cameras": {}}`` when not yet solved
    (the UI hides the intrinsic-results panel in that case).
    """
    d = project_dir(request, name)
    return intrinsic_summary(d)


@router.get("/api/p/{name}/calibration-summary")
async def cal_summary(request: Request, name: str) -> dict:
    """Vital calibration facts from the SOLVE (reprojection RMS + geometry)."""
    d = project_dir(request, name)
    return {"summary": calibration_summary(d)}


class CaptureConfigBody(BaseModel):
    extrinsic_target: int = Field(ge=EXTRINSIC_TARGET_MIN)


@router.get("/api/p/{name}/capture-config")
async def get_capture_config(request: Request, name: str) -> dict:
    _d, cfg = project_cfg(request, name)
    return {"extrinsic_target": cfg.capture.extrinsic_target,
            "target_per_camera": cfg.capture.target_per_camera,
            "extrinsic_target_min": EXTRINSIC_TARGET_MIN}


@router.put("/api/p/{name}/capture-config")
async def put_capture_config(request: Request, name: str,
                             body: CaptureConfigBody) -> dict:
    """Operator-settable capture targets (currently the extrinsic pair count).

    Floored at ``EXTRINSIC_TARGET_MIN`` (the BA is ill-conditioned below it)."""
    d, cfg = project_cfg(request, name)
    cfg.capture.extrinsic_target = max(EXTRINSIC_TARGET_MIN, int(body.extrinsic_target))
    save_project(d, cfg)
    return {"ok": True, "extrinsic_target": cfg.capture.extrinsic_target}


class CamerasBody(BaseModel):
    cam_a: CameraIn
    cam_b: CameraIn | None = None


@router.get("/api/p/{name}/cameras")
async def get_cameras(request: Request, name: str) -> dict:
    _d, cfg = project_cfg(request, name)
    return {cid: cfg.cameras[cid].model_dump() if cid in cfg.cameras else None
            for cid in CAMERA_IDS}


@router.put("/api/p/{name}/cameras")
async def put_cameras(request: Request, name: str, body: CamerasBody) -> dict:
    """Edit the rig's cameras (e.g. add cam_b once the second camera is mounted)."""
    d, cfg = project_cfg(request, name)
    cam_a = CameraSpec(id="cam_a", **body.cam_a.model_dump())
    if not cam_a.configured():
        raise HTTPException(status_code=422, detail="cam_a needs an RTSP url or USB device")
    cams = {"cam_a": cam_a}
    if body.cam_b is not None and (body.cam_b.url or body.cam_b.device):
        cams["cam_b"] = CameraSpec(id="cam_b", **body.cam_b.model_dump())
    cfg.cameras = cams
    save_project(d, cfg)
    # make sure the per-camera capture dirs exist for any newly-added camera
    for cid in cams:
        (d / "intrinsic" / cid).mkdir(parents=True, exist_ok=True)
        (d / "extrinsic" / cid).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "cameras": cfg.configured_cameras(), "mode2": cfg.is_mode2()}
