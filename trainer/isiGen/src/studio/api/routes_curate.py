"""Curate gallery API — record listing + per-record edits + folder ingest."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...core.runners import run_curate
from .deps import manifest, project_cfg, project_dir

router = APIRouter()


@router.get("/api/p/{name}/records")
async def records(request: Request, name: str) -> dict:
    _, m = manifest(request, name)
    return {"records": [r.model_dump(mode="json") for _, r in sorted(m.records.items())]}


class RecordPatch(BaseModel):
    class_name: str | None = None
    excluded: bool | None = None
    notes: str | None = None


@router.patch("/api/p/{name}/records/{rid}")
async def patch_record(request: Request, name: str, rid: str, body: RecordPatch) -> dict:
    _d, m = manifest(request, name)
    rec = m.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {rid!r} not found")
    if body.class_name is not None:
        _, cfg = project_cfg(request, name)
        cfg.class_by_name(body.class_name)        # validates
        rec.class_name = body.class_name
    if body.excluded is not None:
        rec.excluded = body.excluded
    if body.notes is not None:
        rec.notes = body.notes
    m.upsert(rec)
    m.save()
    return {"ok": True, "record": rec.model_dump(mode="json")}


class IngestBody(BaseModel):
    source: str                                   # server-side folder path
    class_name: str | None = None
    auto_class: bool = False


@router.post("/api/p/{name}/ingest")
async def ingest(request: Request, name: str, body: IngestBody) -> dict:
    d = project_dir(request, name)
    jobs = request.app.state.jobs
    try:
        job = jobs.submit(name, "curate",
                          lambda: run_curate(d, source=body.source,
                                             class_name=body.class_name,
                                             auto_class=body.auto_class))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "job": job.to_dict()}
