"""Phase-run + job-status API. Phase bodies are core/runners functions —
identical to the CLI paths."""

from __future__ import annotations

from functools import partial

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...core.runners import (
    reset_phase,
    run_captions,
    run_control_maps,
    run_export,
    run_filter,
    run_generation,
    run_lora,
    run_masks,
    run_scaffolds,
    run_strength_test,
)
from .deps import project_dir

router = APIRouter()

_PHASES = {
    "maps": lambda d: partial(run_control_maps, d),
    "masks": lambda d: partial(run_masks, d),
    "captions": lambda d: partial(run_captions, d),
    "lora": lambda d: partial(run_lora, d),
    "scaffolds": lambda d: partial(run_scaffolds, d),
    "generate": lambda d: partial(run_generation, d),
    "strength_test": lambda d: partial(run_strength_test, d),   # mint-page sweep
    "filter": lambda d: partial(run_filter, d),
    "export": lambda d: partial(run_export, d),
}


class RunBody(BaseModel):
    max_steps: int | None = None        # LoRA only — overrides + persists step count
    count: int | None = None            # scaffolds only — # synthetic images (1 per scaffold)
    paste_count: int | list | None = None   # scaffolds only — objects pasted per scene
    placement: str | None = None        # scaffolds only — copy_paste placement (original/random)
    strength: float | None = None       # generate only — inpaint strength (copy_paste path)
    clip_filter: bool | None = None     # export only — CLIP-filter mints (False → export_noclip/)
    prompt_detector: str | None = None  # masks only — auto-prompt detector ONNX path
                                        # ("" / "none" clears, None leaves configured)


@router.post("/api/p/{name}/run/{phase}")
async def run_phase(request: Request, name: str, phase: str,
                    body: RunBody | None = None) -> dict:
    d = project_dir(request, name)
    factory = _PHASES.get(phase)
    if factory is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown/not-yet-runnable phase {phase!r} "
                                   f"(runnable: {sorted(_PHASES)})")
    if phase == "lora":
        fn = partial(run_lora, d, max_steps=(body.max_steps if body else None),
                     runs_dir=request.app.state.settings.runs_dir)
    elif phase == "scaffolds":
        fn = partial(run_scaffolds, d,
                     count=(body.count if body else None),
                     paste_count=(body.paste_count if body else None),
                     placement=(body.placement if body else None))
    elif phase == "masks":
        fn = partial(run_masks, d,
                     prompt_detector=(body.prompt_detector if body else None))
    elif phase == "generate":
        fn = partial(run_generation, d, strength=(body.strength if body else None))
    elif phase == "export":
        fn = partial(run_export, d,
                     clip_filter=(body.clip_filter if body and body.clip_filter is not None
                                  else True))           # default: export WITH clip
    else:
        fn = factory(d)
    try:
        job = request.app.state.jobs.submit(name, phase, fn)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "job": job.to_dict()}


@router.post("/api/p/{name}/reset/{phase}")
async def reset(request: Request, name: str, phase: str) -> dict:
    """Wipe a phase's outputs so it can be re-run cleanly (not a job — fast)."""
    d = project_dir(request, name)
    try:
        summary = reset_phase(d, phase,
                              runs_dir=request.app.state.settings.runs_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "reset": summary}


@router.get("/api/jobs")
async def jobs(request: Request) -> dict:
    return {"jobs": request.app.state.jobs.list()}


@router.get("/api/jobs/{job_id}/log")
async def job_log(request: Request, job_id: str) -> dict:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return {"job": job.to_dict(), "log": list(job.log)}
