"""The probe UI: ``GET /ui`` (live view + AGV test cards) + ``GET /recent``.

``/ui`` serves a single self-contained HTML page (inline CSS/JS, no static
mount, no build step — wheel-safe by construction) with three parts: the
schema tree, live cards (nodes / zones / tracks / consumers, polled every
2 s), and the AGV system-test cards (on-demand RUN checks that show the
exact REST endpoint / MQTT topic to use — the former ``/test`` console,
merged in). The page SHELL carries no data, so it is served without a token;
its JavaScript sends ``Authorization: Bearer`` from a localStorage token box
when the deployment requires one.

``/test`` (referenced by the AGV integration guide) redirects to ``/ui``,
preserving the query string so ``/test?run=all`` still auto-runs the checks.

``/recent`` exposes the subscriber's raw-message ring buffer plus the
ingestion counters (received / dropped) that previously had no REST surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .auth import require_token
from .ui_page import UI_HTML

router = APIRouter()

# Served bare-only (not under /v1) and token-free — see module docstring.
page_router = APIRouter()


@router.get("/recent", dependencies=[Depends(require_token)])
async def recent(request: Request,
                 limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """The last ``limit`` raw MQTT messages (newest last) + ingest counters."""
    sub = request.app.state.subscriber
    messages = sub.recent(limit)
    return JSONResponse({
        "messages": messages,
        "topics": sub.topics(),
        "stats": sub.stats(),
        "count": len(messages),
    })


@page_router.get("/ui", include_in_schema=False)
async def ui() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@page_router.get("/test", include_in_schema=False)
async def test_console(request: Request) -> RedirectResponse:
    """The former AGV test console — now part of ``/ui`` (see module docstring)."""
    query = request.url.query
    return RedirectResponse(url="/ui" + (f"?{query}" if query else ""),
                            status_code=307)
