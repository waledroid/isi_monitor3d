"""Project CRUD + phase status summary (the phase-board counts)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.manifest import Manifest
from ...core.project import (
    ClassSpec,
    create_project,
    delete_project,
    list_projects,
    load_project,
)
from .deps import project_dir

router = APIRouter()


class CreateProjectBody(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    classes: list[ClassSpec] = Field(min_length=1)


@router.get("/api/projects")
async def projects(request: Request) -> dict:
    data_dir = request.app.state.settings.data_dir
    out = []
    for name in list_projects(data_dir):
        d = data_dir / name
        cfg = load_project(d)
        m = Manifest.load(d)
        out.append({"name": name,
                    "classes": [c.model_dump() for c in cfg.classes],
                    "records": len(m.records)})
    return {"projects": out}


@router.post("/api/projects")
async def create(request: Request, body: CreateProjectBody) -> dict:
    try:
        path = create_project(request.app.state.settings.data_dir,
                              body.name, body.classes)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "path": str(path)}


@router.delete("/api/projects/{name}")
async def delete(request: Request, name: str) -> dict:
    settings = request.app.state.settings
    try:
        delete_project(settings.data_dir, name, runs_dir=settings.runs_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


def _lora_trained(project_dir, name: str, runs_dir) -> bool:
    """A project's LoRA is 'done' if its configured weights exist, or any of its
    runs/lora/<name>_* dirs holds a weights file."""
    from pathlib import Path
    cfg = load_project(project_dir)
    weights = (cfg.phase("generation") or {}).get("lora_weights")
    if weights:
        p = Path(weights)
        if p.is_file() or (p / "pytorch_lora_weights.safetensors").is_file():
            return True
    return any((d / "pytorch_lora_weights.safetensors").is_file()
               for d in (Path(runs_dir) / "lora").glob(f"{name}_*"))


@router.get("/api/p/{name}/status")
async def status(request: Request, name: str) -> dict:
    d = project_dir(request, name)
    m = Manifest.load(d)
    recs = list(m.records.values())
    active = [r for r in recs if not r.excluded]
    return {
        "lora_trained": _lora_trained(d, name, request.app.state.settings.runs_dir),
        "records": len(recs),
        "excluded": sum(r.excluded for r in recs),
        "by_class": {c: sum(r.class_name == c for r in active)
                     for c in {r.class_name for r in active}},
        "depth": sum(r.depth_map is not None for r in active),
        "canny": sum(r.canny_map is not None for r in active),
        "masked": sum(r.mask is not None for r in active),
        "prompted": sum(bool(r.mask_prompts) for r in active),
        "needs_review": sum(r.needs_review for r in active),
        "captioned": sum(r.caption_path is not None and not getattr(r, "synthetic", False)
                         for r in active),
        "caption_edited": sum(r.caption_edited for r in active),
        "scaffolds": _scaffold_counts(d),
        "synthetic": sum(bool(getattr(r, "synthetic", False)) for r in recs),
        "clip_scored": sum(getattr(r, "clip_score", None) is not None for r in recs),
        "exported": (d / "export" / "yolo_seg" / "data.yaml").exists(),
    }


def _scaffold_counts(project_dir) -> dict:
    from ...core.runners import load_scaffold_index
    entries = load_scaffold_index(project_dir)
    return {"total": len(entries),
            "pending": sum(e.get("status") == "pending" for e in entries),
            "generated": sum(e.get("status") == "generated" for e in entries)}
