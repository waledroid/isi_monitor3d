"""isiGen Studio — FastAPI app factory (monitor_web's pattern).

The Studio is the per-phase VISUALIZER + job launcher: projects, phase board,
curate gallery, maps viewer (with SAM2 prompt canvas), caption editor. Heavy
work runs through the single-worker JobRunner — the same `core/runners.py`
functions the CLI scripts call.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import (
    routes_captions,
    routes_curate,
    routes_jobs,
    routes_maps,
    routes_media,
    routes_pages,
    routes_projects,
)
from .config import Settings
from .jobs import JobRunner

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    jobs = JobRunner(cfg.runs_dir, log_buffer=cfg.job_log_buffer)
    templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        jobs.start()
        try:
            yield
        finally:
            jobs.stop()

    app = FastAPI(title="isiGen Studio", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg
    app.state.jobs = jobs
    app.state.templates = templates

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon() -> Response:
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")

    # Static assets revalidate on every request (cheap 304s) so CSS/JS edits
    # show up on plain reload — same trap-avoidance as monitor_web.
    @app.middleware("http")
    async def _revalidate_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.include_router(routes_pages.router)
    app.include_router(routes_projects.router)
    app.include_router(routes_curate.router)
    app.include_router(routes_maps.router)
    app.include_router(routes_captions.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_media.router)
    return app
