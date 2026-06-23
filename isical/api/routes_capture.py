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


def _resolve_cameras(cfg, phase: str, cam: str | None) -> list[str]:
    """Which cameras a capture run covers. Intrinsic → the chosen single camera
    (or all if none given); extrinsic → always both (synchronized pairs)."""
    configured = cfg.configured_cameras()
    if not configured:
        raise HTTPException(status_code=422, detail="no cameras configured")
    if phase == "extrinsic":
        if len(configured) < 2:
            raise HTTPException(status_code=422, detail="extrinsic needs both cameras configured")
        return configured
    if cam is not None:
        if cam not in configured:
            raise HTTPException(status_code=404, detail=f"camera {cam!r} not configured")
        return [cam]
    return configured


@router.post("/api/p/{name}/capture/{phase}/start")
async def start(request: Request, name: str, phase: str, cam: str | None = None) -> dict:
    if phase not in _PHASES:
        raise HTTPException(status_code=404, detail=f"capture phase must be one of {_PHASES}")
    d, cfg = project_cfg(request, name)
    cameras = _resolve_cameras(cfg, phase, cam)
    try:
        st = request.app.state.capture.start(name, d, cfg, phase, cameras=cameras)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"capture start failed: {exc}") from exc
    return {"ok": True, "status": st}


@router.post("/api/p/{name}/capture/{phase}/restart")
async def restart(request: Request, name: str, phase: str, cam: str | None = None) -> dict:
    """Wipe this phase's captures (for the selected camera, or all) then start fresh."""
    if phase not in _PHASES:
        raise HTTPException(status_code=404, detail=f"capture phase must be one of {_PHASES}")
    d, cfg = project_cfg(request, name)
    cameras = _resolve_cameras(cfg, phase, cam)
    from ..capture.session import wipe_phase_captures
    request.app.state.capture.stop_current()
    removed = wipe_phase_captures(d, phase, cameras)
    try:
        st = request.app.state.capture.start(name, d, cfg, phase, cameras=cameras)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"capture start failed: {exc}") from exc
    return {"ok": True, "removed": removed, "status": st}


@router.post("/api/p/{name}/capture/{phase}/stop")
async def stop(request: Request, name: str, phase: str) -> dict:
    project_dir(request, name)
    request.app.state.capture.stop_current()
    return {"ok": True}


@router.get("/api/p/{name}/sync-probe")
def sync_probe(request: Request, name: str, seconds: float = 4.0) -> dict:
    """LIVE stream-sync probe (NOT a calibration output): per-camera FPS + the
    inter-camera capture-timestamp skew. Sync def → runs in the threadpool so the
    few-second probe doesn't block the event loop."""
    d, cfg = project_cfg(request, name)
    if request.app.state.capture.active(name) is not None:
        raise HTTPException(status_code=409, detail="stop the live capture first (cameras busy)")
    from ..capture.probe import probe_streams
    try:
        return probe_streams(d, cfg, seconds=seconds)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/p/{name}/capture/status")
async def status(request: Request, name: str) -> dict:
    project_dir(request, name)
    sess = request.app.state.capture.active(name)
    if sess is None:
        return {"active": False}
    return {"active": True, **sess.status()}


@router.post("/api/p/{name}/floor/{cam}")
async def floor_shot(request: Request, name: str, cam: str) -> dict:
    """Grab one ChArUco-on-floor shot for a camera (the world anchor for extrinsics)."""
    d, cfg = project_cfg(request, name)
    if cam not in cfg.configured_cameras():
        raise HTTPException(status_code=404, detail=f"camera {cam!r} not configured")
    if request.app.state.capture.active(name) is not None:
        raise HTTPException(status_code=409,
                            detail="stop the live capture first (the camera is busy)")
    from ..capture.session import grab_floor_shot
    try:
        res = grab_floor_shot(d, cfg, cam)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"floor shot failed: {exc}") from exc
    return {"ok": True, **res}


@router.get("/stream/{name}/{cam}")
def stream(request: Request, name: str, cam: str) -> StreamingResponse:
    project_dir(request, name)
    gen = request.app.state.capture.mjpeg(name, cam)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")
