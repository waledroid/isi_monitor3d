"""Caption editor API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .deps import manifest

router = APIRouter()


@router.get("/api/p/{name}/records/{rid}/caption")
async def get_caption(request: Request, name: str, rid: str) -> dict:
    d, m = manifest(request, name)
    rec = m.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {rid!r} not found")
    text = ""
    if rec.caption_path and (d / rec.caption_path).exists():
        text = (d / rec.caption_path).read_text().strip()
    return {"caption": text, "edited": rec.caption_edited}


class CaptionBody(BaseModel):
    caption: str


@router.put("/api/p/{name}/records/{rid}/caption")
async def put_caption(request: Request, name: str, rid: str, body: CaptionBody) -> dict:
    d, m = manifest(request, name)
    rec = m.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {rid!r} not found")
    rel = rec.caption_path or f"captions/{rec.id}.txt"
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_text(body.caption.strip() + "\n")
    rec.caption_path = rel
    rec.caption_edited = True            # re-runs will never overwrite this
    m.upsert(rec)
    m.save()
    return {"ok": True}
