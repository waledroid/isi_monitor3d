"""Phase-solve jobs (the Multical runs) + job status — mirrors isiGen routes_jobs."""

from __future__ import annotations

from functools import partial

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..core.project import load_project
from ..core.runners import (
    preview_targetless_matches,
    run_export,
    run_extrinsic,
    run_extrinsic_targetless,
    run_intrinsic,
)
from .deps import project_dir

router = APIRouter()


def _extrinsic_runner(d, body):
    """Dispatch the extrinsic solve on the project's configured method.

    AprilGrid (default/fallback) → run_extrinsic; targetless (experimental) →
    run_extrinsic_targetless. The method persists in calib.yaml (extrinsic_method).
    """
    method = load_project(d).extrinsic_method
    fn = run_extrinsic_targetless if method == "targetless" else run_extrinsic
    return partial(fn, d)


_PHASES = {
    "intrinsic": lambda d, body: partial(run_intrinsic, d),
    "extrinsic": _extrinsic_runner,
    "export": lambda d, body: partial(run_export, d,
                                      install=bool(body and body.install)),
    # DIAGNOSTIC only — renders per-pair feature-match previews under work/ scratch.
    # Never solves / writes calibration_targetless.json / needs scale references.
    "targetless-diag": lambda d, body: partial(preview_targetless_matches, d),
}


class RunBody(BaseModel):
    install: bool | None = None       # export only — also copy to config/mode2/


@router.post("/api/p/{name}/run/{phase}")
async def run_phase(request: Request, name: str, phase: str,
                    body: RunBody | None = None) -> dict:
    d = project_dir(request, name)
    factory = _PHASES.get(phase)
    if factory is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown phase {phase!r} (one of {sorted(_PHASES)})")
    fn = factory(d, body)
    try:
        job = request.app.state.jobs.submit(name, phase, fn)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "job": job.to_dict()}


@router.get("/api/jobs")
async def jobs(request: Request) -> dict:
    return {"jobs": request.app.state.jobs.list()}


@router.get("/api/jobs/{job_id}/log")
async def job_log(request: Request, job_id: str) -> dict:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job": job.to_dict(), "log": list(job.log)}
