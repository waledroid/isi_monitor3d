"""isical Studio — FastAPI app factory (isiGen/monitor_web pattern).

The Studio is the calibration project board + live capture view + solve launcher.
The capture loop runs via routes_capture (a CaptureManager); the Multical solves
run through the single-worker JobRunner (the same core/runners.py functions).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import routes_capture, routes_jobs, routes_pages, routes_projects
from .capture.session import CaptureManager
from .config import Settings
from .jobs import JobRunner

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    jobs = JobRunner(cfg.runs_dir, log_buffer=cfg.job_log_buffer)
    capture = CaptureManager()
    templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        jobs.start()
        try:
            yield
        finally:
            capture.stop_all()
            jobs.stop()

    app = FastAPI(title="isical — Calibration Studio", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg
    app.state.jobs = jobs
    app.state.capture = capture
    app.state.templates = templates

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon() -> Response:
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")

    @app.middleware("http")
    async def _revalidate_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.include_router(routes_pages.router)
    app.include_router(routes_projects.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_capture.router)
    return app
