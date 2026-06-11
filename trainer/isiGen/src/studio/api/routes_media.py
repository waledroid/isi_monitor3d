"""Media serving — images / maps / thumbnails, id-validated against the
manifest (no path traversal: every path comes from the manifest, never the URL)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .deps import manifest

router = APIRouter()

_KINDS = {"image": "image", "depth": "depth_map", "canny": "canny_map", "mask": "mask"}


@router.get("/media/{name}/{kind}/{rid}")
async def media(request: Request, name: str, kind: str, rid: str):
    d, m = manifest(request, name)
    rec = m.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {rid!r} not found")
    if kind == "thumb":
        from ..thumbs import thumb_path
        try:
            p = thumb_path(d, rec.id, rec.image,
                           max_px=request.app.state.settings.thumb_max_px)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"thumb failed: {exc}") from exc
        return FileResponse(p)
    field = _KINDS.get(kind)
    if field is None:
        raise HTTPException(status_code=404, detail=f"unknown media kind {kind!r}")
    rel = getattr(rec, field, None)
    if not rel or not (d / rel).exists():
        raise HTTPException(status_code=404, detail=f"no {kind} for {rid!r}")
    return FileResponse(d / rel)
