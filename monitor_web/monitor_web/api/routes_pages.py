"""Page routes — serve the dashboard HTML."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..i18n import available_langs, load_bundle

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    cfg = request.app.state.settings
    templates = request.app.state.templates
    lang = cfg.default_lang
    try:
        strings = load_bundle(lang)
    except FileNotFoundError:
        strings = {}
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "lang": lang,
            "available_langs": available_langs(),
            "t": strings,
            "ws_url": "/ws/tracks",
        },
    )
