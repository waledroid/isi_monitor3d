"""Maps viewer API — SAM2 prompt editing per record."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.manifest import MaskPrompt
from .deps import manifest

router = APIRouter()


class PromptsBody(BaseModel):
    prompts: list[MaskPrompt] = Field(default_factory=list)


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
