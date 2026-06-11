"""HTMX-friendly logs partial — returns the latest log lines as HTML."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..log_format import format_lines

router = APIRouter()


@router.get("/api/logs", response_class=HTMLResponse)
async def logs_partial(request: Request, n: int = 200) -> HTMLResponse:
    supervisor = request.app.state.supervisor
    templates = request.app.state.templates
    n = max(1, min(int(n), 2000))
    return templates.TemplateResponse(
        request=request,
        name="_logs.html",
        context={"rows": format_lines(supervisor.log_lines(n))},
    )
