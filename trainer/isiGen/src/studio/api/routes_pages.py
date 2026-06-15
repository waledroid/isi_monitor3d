"""HTML pages — server-rendered shells; the JS fills them via the APIs."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .deps import project_cfg

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def projects_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "projects.html", {})


def _phase_page(request: Request, name: str, template: str):
    _, cfg = project_cfg(request, name)
    return request.app.state.templates.TemplateResponse(
        request, template, {"project": name,
                            "classes": [c.model_dump() for c in cfg.classes]})


@router.get("/p/{name}", response_class=HTMLResponse)
async def phases_page(request: Request, name: str):
    return _phase_page(request, name, "phases.html")


@router.get("/p/{name}/curate", response_class=HTMLResponse)
async def curate_page(request: Request, name: str):
    return _phase_page(request, name, "curate.html")


@router.get("/p/{name}/maps", response_class=HTMLResponse)
async def maps_page(request: Request, name: str):
    return _phase_page(request, name, "maps.html")


@router.get("/p/{name}/masks", response_class=HTMLResponse)
async def masks_page(request: Request, name: str):
    return _phase_page(request, name, "masks.html")


@router.get("/p/{name}/captions", response_class=HTMLResponse)
async def captions_page(request: Request, name: str):
    return _phase_page(request, name, "captions.html")
