"""Maps viewer API — SAM2 prompt editing per record."""

from __future__ import annotations

import cv2
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.manifest import MaskPrompt
from ...core.models import list_detector_onnx
from ...core.project import load_project
from .deps import manifest, project_dir

router = APIRouter()


class PromptsBody(BaseModel):
    prompts: list[MaskPrompt] = Field(default_factory=list)


class DetectBody(BaseModel):
    onnx_path: str
    confidence_threshold: float | None = None


@router.put("/api/p/{name}/records/{rid}/prompts")
async def put_prompts(request: Request, name: str, rid: str, body: PromptsBody) -> dict:
    _, m = manifest(request, name)
    rec = m.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {rid!r} not found")
    rec.mask_prompts = body.prompts
    rec.mask = None              # prompts changed → mask must be recomputed
    rec.needs_review = False
    m.upsert(rec)
    m.save()
    return {"ok": True, "prompts": [p.model_dump() for p in rec.mask_prompts]}


@router.get("/api/p/{name}/detector-models")
async def detector_models(request: Request, name: str) -> dict:
    """List trained detector ONNX (isidet) for the auto-prompt dropdown + the
    currently-configured selection."""
    d = project_dir(request, name)
    project = load_project(d)
    current = ((project.phase("masking").get("prompt_detector") or {}).get("onnx_path"))
    return {"models": list_detector_onnx(), "current": current}


@router.post("/api/p/{name}/records/{rid}/detect-prompts")
async def detect_prompts(request: Request, name: str, rid: str, body: DetectBody) -> dict:
    """Run a detector on ONE image → box prompts for the canvas (no mask yet).

    The operator can edit/save the returned prompts before masking. Builds the
    detector for this single call (a manual per-image action); the Run-masks job
    reuses one loaded detector across all images instead."""
    from ...stages.detection import build_prompt_detector

    d, m = manifest(request, name)
    rec = m.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {rid!r} not found")
    img = cv2.imread(str(d / rec.image))
    if img is None:
        raise HTTPException(status_code=404, detail=f"unreadable image for {rid!r}")
    project = load_project(d)
    class_names = [c.name for c in project.classes]
    kw = {}
    if body.confidence_threshold is not None:
        kw["confidence_threshold"] = float(body.confidence_threshold)
    try:
        detector = build_prompt_detector(body.onnx_path, **kw)
        detector.load()
        try:
            prompts = detector.detect(img, class_names)
        finally:
            detector.close()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # surface decode/ORT errors to the UI
        raise HTTPException(status_code=500, detail=f"detection failed: {exc}") from exc
    return {"ok": True, "prompts": [p.model_dump() for p in prompts]}
