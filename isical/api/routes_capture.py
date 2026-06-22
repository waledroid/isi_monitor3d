"""Live capture control + the annotated MJPEG stream.

Capture is NOT a JobRunner job — it's an interactive live loop owned by the
CaptureManager (one session at a time). Start opens the cameras + auto-snaps;
the phase solve (run/{phase}) is the separate JobRunner job.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .deps import project_cfg, project_dir

router = APIRouter()

_PHASES = ("intrinsic", "extrinsic")


@router.post("/api/p/{name}/capture/{phase}/start")
async def start(request: Request, name: str, phase: str) -> dict:
    if phase not in _PHASES:
        raise HTTPException(status_code=404, detail=f"capture phase must be one of {_PHASES}")
    d, cfg = project_cfg(request, name)
    if not cfg.configured_cameras():
        raise HTTPException(status_code=422, detail="no cameras configured")
    if phase == "extrinsic" and len(cfg.configured_cameras()) < 2:
        raise HTTPException(status_code=422,
                            detail="extrinsic needs both cameras configured")
    try:
        st = request.app.state.capture.start(name, d, cfg, phase)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"capture start failed: {exc}") from exc
    return {"ok": True, "status": st}


@router.post("/api/p/{name}/capture/{phase}/stop")
async def stop(request: Request, name: str, phase: str) -> dict:
    project_dir(request, name)
    request.app.state.capture.stop_current()
    return {"ok": True}


@router.get("/api/p/{name}/capture/status")
async def status(request: Request, name: str) -> dict:
    project_dir(request, name)
    sess = request.app.state.capture.active(name)
    if sess is None:
        return {"active": False}
    return {"active": True, **sess.status()}


@router.get("/stream/{name}/{cam}")
def stream(request: Request, name: str, cam: str) -> StreamingResponse:
    project_dir(request, name)
    gen = request.app.state.capture.mjpeg(name, cam)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")
