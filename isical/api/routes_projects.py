"""Project CRUD + phase-board status."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.project import (
    CameraSpec,
    create_project,
    delete_project,
    list_projects,
    load_project,
)
from ..core.runners import phase_status
from .deps import project_dir

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
