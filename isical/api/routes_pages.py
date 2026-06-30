"""HTML pages — projects list, the phase board, and the live capture view."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .deps import project_cfg

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(request=request, name="projects.html")


@router.get("/p/{name}", response_class=HTMLResponse)
async def board(request: Request, name: str) -> HTMLResponse:
    _d, cfg = project_cfg(request, name)
    return request.app.state.templates.TemplateResponse(
        request=request, name="phases.html",
        context={"project": name, "cameras": cfg.configured_cameras()})


@router.get("/p/{name}/capture/{phase}", response_class=HTMLResponse)
async def capture_page(request: Request, name: str, phase: str) -> HTMLResponse:
    from ..core.project import EXTRINSIC_TARGET_MIN, board_cm_from_config
    _d, cfg = project_cfg(request, name)
    cm = board_cm_from_config(cfg.board.tag_length_m, cfg.board.tag_spacing)
    return request.app.state.templates.TemplateResponse(
        request=request, name="capture.html",
        context={"project": name, "phase": phase,
                 "cameras": cfg.configured_cameras(),
                 "extrinsic_target": cfg.capture.extrinsic_target,
                 "extrinsic_target_min": EXTRINSIC_TARGET_MIN,
                 "tag_length_cm": round(cm["tag_length_cm"], 2),
                 "tag_gap_cm": round(cm["tag_gap_cm"], 2)})
