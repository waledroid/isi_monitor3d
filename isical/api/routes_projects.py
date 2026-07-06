"""Project CRUD + phase-board status."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.project import (
    CAMERA_IDS,
    EXTRINSIC_TARGET_MIN,
    CameraSpec,
    board_cm_from_config,
    board_config_from_cm,
    create_project,
    delete_project,
    list_projects,
    load_project,
    save_project,
)
from ..core.runners import (
    calibration_matrices,
    calibration_summary,
    extrinsic_summary,
    intrinsic_summary,
    phase_status,
    targetless_calibration_matrices,
    targetless_report,
)
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


@router.get("/api/p/{name}/extrinsic-summary")
async def extr_summary(request: Request, name: str) -> dict:
    """Per-camera extrinsics ([R | t] pose) from calibration.json (after solve).

    Returns ``{"rms_gate_px", "baseline_m"?, "cameras": {cam: {R, t, rms}}}``
    when solved, or ``{"cameras": {}}`` when not (the UI hides the panel).
    """
    d = project_dir(request, name)
    return extrinsic_summary(d)


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


class BoardConfigBody(BaseModel):
    tag_length_cm: float = Field(gt=0)
    tag_gap_cm: float = Field(ge=0)
    # ChArUco (intrinsics + floor anchor) — optional so existing callers that
    # only submit the AprilGrid fields keep working. The metric solve depends
    # on square_cm: the c1 rig shipped with the 3.5 cm DEFAULT while the
    # physical board square is ~12.2 cm — a silent 3.5x scale error in every
    # extrinsic/floor solve. Measure the printed board and enter it here.
    square_cm: float | None = Field(default=None, gt=0)
    marker_cm: float | None = Field(default=None, gt=0)


@router.get("/api/p/{name}/board-config")
async def get_board_config(request: Request, name: str) -> dict:
    """Board measurements for the capture forms, in cm (operator units).

    AprilGrid: reverses the stored ``board.tag_length_m`` / ``tag_spacing`` back
    to the ruler measurements. ChArUco: ``square_cm`` / ``marker_cm`` for the
    intrinsics + floor board."""
    _d, cfg = project_cfg(request, name)
    cm = board_cm_from_config(cfg.board.tag_length_m, cfg.board.tag_spacing)
    return {**cm,
            "tag_length_m": cfg.board.tag_length_m,
            "tag_spacing": cfg.board.tag_spacing,
            "square_cm": round(cfg.board.square_length_m * 100.0, 3),
            "marker_cm": round(cfg.board.marker_length_m * 100.0, 3)}


@router.put("/api/p/{name}/board-config")
async def put_board_config(request: Request, name: str, body: BoardConfigBody) -> dict:
    """Persist board geometry from cm measurements (422 on bad input)."""
    d, cfg = project_cfg(request, name)
    try:
        derived = board_config_from_cm(body.tag_length_cm, body.tag_gap_cm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    cfg.board.tag_length_m = derived["tag_length_m"]
    cfg.board.tag_spacing = derived["tag_spacing"]
    if body.square_cm is not None:
        marker_cm = body.marker_cm if body.marker_cm is not None \
            else body.square_cm * (cfg.board.marker_length_m / cfg.board.square_length_m)
        if not marker_cm < body.square_cm:
            raise HTTPException(status_code=422,
                                detail="ChArUco marker must be smaller than the square")
        cfg.board.square_length_m = body.square_cm / 100.0
        cfg.board.marker_length_m = marker_cm / 100.0
        derived = {**derived,
                   "square_length_m": cfg.board.square_length_m,
                   "marker_length_m": cfg.board.marker_length_m}
    save_project(d, cfg)
    return {"ok": True, **derived}


class ExtrinsicMethodBody(BaseModel):
    method: Literal["aprilgrid", "targetless"]


@router.get("/api/p/{name}/extrinsic-method")
async def get_extrinsic_method(request: Request, name: str) -> dict:
    """The extrinsic calibration method: AprilGrid (target) or targetless."""
    _d, cfg = project_cfg(request, name)
    return {"method": cfg.extrinsic_method}


@router.put("/api/p/{name}/extrinsic-method")
async def put_extrinsic_method(request: Request, name: str,
                               body: ExtrinsicMethodBody) -> dict:
    """Select the extrinsic method. Targetless is experimental; AprilGrid is the
    default/fallback. Routes the extrinsic solve (routes_jobs) accordingly."""
    d, cfg = project_cfg(request, name)
    cfg.extrinsic_method = body.method
    save_project(d, cfg)
    return {"ok": True, "method": cfg.extrinsic_method}


@router.get("/api/p/{name}/targetless-report")
async def get_targetless_report(request: Request, name: str) -> dict:
    """The 3-level targetless validation report (or null when not yet solved)."""
    d = project_dir(request, name)
    return {"report": targetless_report(d)}


@router.get("/api/p/{name}/calibration-matrices")
async def get_calibration_matrices(request: Request, name: str) -> dict:
    """Per-camera R (3x3) / t (3x1) + reprojection RMS from calibration.json.

    Feeds the targetless notebook's Result cell (prints the solved matrices as
    text). ``matrices`` is null until the Extrinsic phase has been solved."""
    d = project_dir(request, name)
    return {"matrices": calibration_matrices(d)}


@router.get("/api/p/{name}/targetless-matrices")
async def get_targetless_matrices(request: Request, name: str) -> dict:
    """Per-camera R/t + RMS from the TARGETLESS output (calibration_targetless.json).

    Feeds the targetless notebook's Result cell — independent of the board
    calibration.json. ``matrices`` is null until the targetless solve has run."""
    d = project_dir(request, name)
    return {"matrices": targetless_calibration_matrices(d)}


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
