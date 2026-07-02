"""Live capture control + the annotated MJPEG stream.

Capture is NOT a JobRunner job — it's an interactive live loop owned by the
CaptureManager (one session at a time). Start opens the cameras + auto-snaps;
the phase solve (run/{phase}) is the separate JobRunner job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .deps import project_cfg, project_dir

router = APIRouter()

# Capture phases. "floor" is the single-button, both-camera synchronized
# ChArUco-on-floor auto-snap (the world anchor for the AprilGrid extrinsics).
_PHASES = ("intrinsic", "extrinsic", "floor")
_PAIR_PHASES = ("extrinsic", "floor")


def _resolve_cameras(cfg, phase: str, cam: str | None) -> list[str]:
    """Which cameras a capture run covers. Intrinsic → the chosen single camera
    (or all if none given); extrinsic/floor → always both (synchronized pairs)."""
    configured = cfg.configured_cameras()
    if not configured:
        raise HTTPException(status_code=422, detail="no cameras configured")
    if phase in _PAIR_PHASES:
        if len(configured) < 2:
            raise HTTPException(status_code=422,
                                detail=f"{phase} needs both cameras configured")
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


# ---------------------------------------------------------------------------
# Targetless capture — self-contained textured-scene stereo-pair capture.
# Wholly separate from the board 'extrinsic' phase above: its own session, its own
# targetless/{cam}/ storage, a manual (no-board-gate) capture-pair trigger + live
# texture readout. Never touches extrinsic/ or the board calibration.json.
# ---------------------------------------------------------------------------


@router.post("/api/p/{name}/targetless/capture/start")
async def targetless_start(request: Request, name: str) -> dict:
    """Open BOTH cameras for targetless capture (live texture readout, manual pairs)."""
    d, cfg = project_cfg(request, name)
    if len(cfg.configured_cameras()) < 2:
        raise HTTPException(status_code=422, detail="targetless needs both cameras configured")
    try:
        st = request.app.state.capture.start_targetless(name, d, cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"targetless start failed: {exc}") from exc
    return {"ok": True, "status": st}


@router.post("/api/p/{name}/targetless/capture/stop")
async def targetless_stop(request: Request, name: str) -> dict:
    project_dir(request, name)
    request.app.state.capture.stop_targetless()
    return {"ok": True}


@router.post("/api/p/{name}/targetless/capture-pair")
async def targetless_capture_pair(request: Request, name: str) -> dict:
    """Manually snap one synchronized stereo pair into targetless/{cam}/."""
    project_dir(request, name)
    sess = request.app.state.capture.targetless(name)
    if sess is None:
        raise HTTPException(status_code=409, detail="targetless capture not running — start it first")
    return sess.capture_pair()


@router.get("/api/p/{name}/targetless/capture/status")
async def targetless_status(request: Request, name: str) -> dict:
    project_dir(request, name)
    sess = request.app.state.capture.targetless(name)
    if sess is None:
        return {"active": False}
    return {"active": True, **sess.status()}


@router.get("/targetless-stream/{name}/{cam}")
def targetless_stream(request: Request, name: str, cam: str) -> StreamingResponse:
    project_dir(request, name)
    gen = request.app.state.capture.targetless_mjpeg(name, cam)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/api/p/{name}/targetless-shots")
def list_targetless_shots(request: Request, name: str) -> dict:
    """The captured targetless stereo pairs (both cameras), for the notebook gallery.

    Read-only. Returns ``{pair_count, cameras: {cam: {count, files:[...]}}}`` — the
    gallery pairs cam_a[i] with cam_b[i] side by side, refreshing as pairs land.
    """
    d, cfg = project_cfg(request, name)
    cams: dict[str, dict] = {}
    for cam in cfg.configured_cameras():
        cam_dir = d / "targetless" / cam
        files = sorted(p.name for p in cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []
        cams[cam] = {"count": len(files), "files": files}
    pair_count = min((v["count"] for v in cams.values()), default=0)
    return {"pair_count": pair_count, "cameras": cams}


@router.get("/targetless-shot/{name}/{cam}/{file}")
def targetless_shot_image(request: Request, name: str, cam: str, file: str) -> FileResponse:
    """Serve one captured targetless jpg. Read-only, path-guarded."""
    if not _SHOT_FILE_RE.match(file):
        raise HTTPException(status_code=404, detail="not found")
    d = project_dir(request, name)
    base = (d / "targetless" / cam).resolve()
    target = (base / file).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="image/jpeg")


_TARGETLESS_DIAG_REL = "work/targetless_diag"


@router.get("/api/p/{name}/targetless-match-preview")
def list_targetless_match_preview(request: Request, name: str) -> dict:
    """List the rendered per-pair feature-match DIAGNOSTIC previews (cell ②).

    Read-only. Reads ``work/targetless_diag/`` (populated by the ``targetless-diag``
    job) and returns, per pair, the match + keypoints image filenames and the
    ``N matches, M RANSAC inliers`` counts from the sidecar. Never triggers the
    matcher and never touches board files.
    """
    d = project_dir(request, name)
    diag = d / _TARGETLESS_DIAG_REL
    pairs: list[dict] = []
    if diag.is_dir():
        labels = sorted({p.name[len("pair_"):-len("_matches.jpg")]
                         for p in diag.glob("pair_*_matches.jpg")})
        for label in labels:
            match_name = f"pair_{label}_matches.jpg"
            kp_name = f"pair_{label}_keypoints.jpg"
            meta_path = diag / f"pair_{label}.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, ValueError):
                    meta = {}
            entry = {"pair": label, "matches": match_name,
                     "n_matches": meta.get("n_matches"), "n_inliers": meta.get("n_inliers")}
            if (diag / kp_name).is_file():
                entry["keypoints"] = kp_name
            pairs.append(entry)
    return {"count": len(pairs), "pairs": pairs}


@router.get("/targetless-diag/{name}/{file}")
def targetless_diag_image(request: Request, name: str, file: str) -> FileResponse:
    """Serve one rendered feature-match diagnostic jpg. Read-only, path-guarded."""
    if not _SHOT_FILE_RE.match(file):
        raise HTTPException(status_code=404, detail="not found")
    d = project_dir(request, name)
    base = (d / _TARGETLESS_DIAG_REL).resolve()
    target = (base / file).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="image/jpeg")


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


@router.post("/api/p/{name}/floor/{cam}/preview")
async def floor_preview_start(request: Request, name: str, cam: str) -> dict:
    """Open a single-camera live ChArUco preview so the operator can aim the floor
    shot. Stream it via /floor-stream/{name}/{cam}; capture with POST /floor/{cam}."""
    d, cfg = project_cfg(request, name)
    if cam not in cfg.configured_cameras():
        raise HTTPException(status_code=404, detail=f"camera {cam!r} not configured")
    try:
        request.app.state.capture.start_floor(name, d, cfg, cam)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"floor preview failed: {exc}") from exc
    return {"ok": True, "camera": cam}


@router.post("/api/p/{name}/floor/{cam}/preview/stop")
async def floor_preview_stop(request: Request, name: str) -> dict:
    project_dir(request, name)
    request.app.state.capture.stop_floor()
    return {"ok": True}


@router.post("/api/p/{name}/floor/{cam}")
async def floor_shot(request: Request, name: str, cam: str) -> dict:
    """Grab one ChArUco-on-floor shot for a camera (the world anchor for extrinsics).

    If a floor preview is live for this camera, grab from its already-open source
    (so preview + grab never double-open the camera). Otherwise fall back to a
    standalone open/settle/grab — but never while a full capture session holds the
    cameras (409)."""
    d, cfg = project_cfg(request, name)
    if cam not in cfg.configured_cameras():
        raise HTTPException(status_code=404, detail=f"camera {cam!r} not configured")
    fp = request.app.state.capture.floor(name, cam)
    if fp is not None:
        try:
            res = fp.grab()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"floor shot failed: {exc}") from exc
        return {"ok": True, **res}
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


@router.get("/floor-stream/{name}/{cam}")
def floor_stream(request: Request, name: str, cam: str) -> StreamingResponse:
    project_dir(request, name)
    gen = request.app.state.capture.floor_mjpeg(name, cam)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/stream/{name}/{cam}")
def stream(request: Request, name: str, cam: str) -> StreamingResponse:
    project_dir(request, name)
    gen = request.app.state.capture.mjpeg(name, cam)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


_SHOT_FILE_RE = re.compile(r"^[A-Za-z0-9_\-]+\.jpg$")


def _shot_meta(jpg: Path, cfg) -> dict:
    """Metadata for one shot, reading the sidecar or backfilling via ChArUco detection.

    Backfill detects once on the saved jpg and caches the sidecar, so already-captured
    projects (no sidecars) work without re-capture. Never raises into the request.
    """
    side = jpg.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except (OSError, ValueError):
            pass
    meta = {"corners": 0, "centroid": None, "blur_var": 0.0}
    try:
        import cv2

        from ..capture.detect import CharucoBoardDetector
        from ..core.project import charuco_spec
        img = cv2.imread(str(jpg))
        if img is not None:
            det = CharucoBoardDetector(charuco_spec(cfg.board)).detect(img)
            meta = {
                "corners": int(det.n),
                "centroid": [float(det.centroid[0]), float(det.centroid[1])] if det.centroid else None,
                "blur_var": float(det.blur_var),
            }
            try:
                side.write_text(json.dumps(meta))
            except OSError:
                pass
    except Exception:
        pass
    return meta


@router.get("/api/p/{name}/shots/{phase}/{cam}")
def list_shots(request: Request, name: str, phase: str, cam: str) -> dict:
    if phase not in ("intrinsic", "extrinsic"):
        raise HTTPException(status_code=404, detail="phase must be intrinsic|extrinsic")
    d, cfg = project_cfg(request, name)
    if cam not in cfg.configured_cameras():
        raise HTTPException(status_code=404, detail=f"camera {cam!r} not configured")
    cam_dir = d / phase / cam
    jpgs = sorted(cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []
    shots = [{"file": p.name, **_shot_meta(p, cfg)} for p in jpgs]
    target = cfg.capture.target_per_camera if phase == "intrinsic" else cfg.capture.extrinsic_target
    return {"target": target, "count": len(shots),
            "blur_min_var": float(cfg.capture.blur_min_var), "shots": shots}


# ---------------------------------------------------------------------------
# Targetless extrinsics — scale-reference marking + stage-image serving
# ---------------------------------------------------------------------------


class ScaleRefIn(BaseModel):
    p1_a: tuple[float, float]
    p1_b: tuple[float, float]
    p2_a: tuple[float, float]
    p2_b: tuple[float, float]
    distance_m: float
    # display-only: whether the b-points were auto-filled from a matched keypoint
    # (snap-assist) vs clicked manually. The calibration runner ignores it — the
    # solve-relevant shape (p1_a,p1_b,p2_a,p2_b,distance_m) is unchanged.
    snapped: bool = False


class ScaleRefsBody(BaseModel):
    references: list[ScaleRefIn]


_SCALE_REFS_REL = "work/scale_references.json"
_STAGE_DIR_REL = "work/targetless_stages"
_FEATURE_MATCHES_REL = "work/feature_matches.json"


def _load_stereo_pair_paths(d: Path) -> tuple[Path, Path] | None:
    """First captured synchronized (cam_a, cam_b) jpg pair, or None if incomplete.

    Reads the targetless method's OWN captures (targetless/) — the snap-assist
    matcher must match on the same textured scenes the targetless solve uses, not
    the board AprilGrid captures.
    """
    a_dir, b_dir = d / "targetless" / "cam_a", d / "targetless" / "cam_b"
    a = sorted(a_dir.glob("*.jpg")) if a_dir.is_dir() else []
    b = sorted(b_dir.glob("*.jpg")) if b_dir.is_dir() else []
    if not a or not b:
        return None
    return a[0], b[0]


def _compute_feature_matches(d: Path) -> dict:
    """Run the SuperPoint+LightGlue matcher on the first captured stereo pair.

    Returns ``{matches: [{a:[x,y], b:[x,y], score}], count, reason}``. This is the
    snap-assist source for cell ③: each entry is a verified cam_a↔cam_b pixel
    correspondence the operator can snap a landmark click to.

    Graceful degradation (the manual marking path never breaks): if the pair isn't
    captured yet, the ONNX weights aren't vendored, or the matcher errors, returns
    ``count: 0`` with a human ``reason`` so the UI shows "snap unavailable — mark
    manually" and falls back to the full 4-click flow.

    A stored synthetic set (``work/feature_matches.json``) — if present — is served
    verbatim, so hermetic tests / offline demos can exercise snap without weights.
    """
    cached = d / _FEATURE_MATCHES_REL
    if cached.exists():
        try:
            data = json.loads(cached.read_text())
            ms = data.get("matches", [])
            return {"matches": ms, "count": len(ms),
                    "reason": data.get("reason", "cached synthetic matches")}
        except (OSError, ValueError):
            pass  # fall through to a live compute

    pair = _load_stereo_pair_paths(d)
    if pair is None:
        return {"matches": [], "count": 0,
                "reason": "no captured stereo pair yet — mark landmarks manually"}

    try:
        import cv2

        from calibration.feature_extrinsics import (
            MatcherWeightsMissing,
            OnnxSuperPointLightGlue,
        )
    except Exception as exc:  # pragma: no cover - import guard
        return {"matches": [], "count": 0,
                "reason": f"matcher unavailable ({exc}) — mark landmarks manually"}

    img_a = cv2.imread(str(pair[0]))
    img_b = cv2.imread(str(pair[1]))
    if img_a is None or img_b is None:
        return {"matches": [], "count": 0,
                "reason": "captured pair unreadable — mark landmarks manually"}

    models_dir = Path(__file__).resolve().parents[2] / "models"
    try:
        matcher = OnnxSuperPointLightGlue(models_dir=models_dir)
        pa, pb, scores = matcher.match(img_a, img_b)
    except MatcherWeightsMissing:
        return {"matches": [], "count": 0,
                "reason": "snap unavailable — matcher ONNX weights not vendored; "
                          "mark landmarks manually"}
    except Exception as exc:
        return {"matches": [], "count": 0,
                "reason": f"matcher failed ({exc}) — mark landmarks manually"}

    matches = []
    for i in range(len(pa)):
        matches.append({
            "a": [float(pa[i][0]), float(pa[i][1])],
            "b": [float(pb[i][0]), float(pb[i][1])],
            "score": float(scores[i]) if scores is not None and i < len(scores) else 1.0,
        })
    return {"matches": matches, "count": len(matches),
            "reason": "" if matches else "no correspondences found — mark landmarks manually"}


@router.get("/api/p/{name}/feature-matches")
def get_feature_matches(request: Request, name: str) -> dict:
    """Verified cam_a↔cam_b correspondences for snap-assisted scale-reference marking.

    Sync def → the (on-rig) ONNX matcher runs in the threadpool, off the event loop.
    Hermetically (no weights / no captured pair) it returns ``count: 0`` with a
    ``reason`` and cell ③ falls back to the manual 4-click marking. The stored
    ``ScaleReference`` shape is unchanged — these coordinates only auto-fill points.
    """
    d = project_dir(request, name)
    return _compute_feature_matches(d)


@router.get("/api/p/{name}/scale-references")
async def get_scale_references(request: Request, name: str) -> dict:
    """The operator-marked floor scale references for the targetless flow."""
    d = project_dir(request, name)
    path = d / _SCALE_REFS_REL
    refs = json.loads(path.read_text()) if path.exists() else []
    return {"references": refs, "count": len(refs)}


@router.put("/api/p/{name}/scale-references")
async def put_scale_references(request: Request, name: str, body: ScaleRefsBody) -> dict:
    """Persist ≥3 measured floor point-pairs (targetless scale). Marked interactively
    by clicking the pair on the images + entering each measured metres value."""
    d = project_dir(request, name)
    for r in body.references:
        if r.distance_m <= 0:
            raise HTTPException(status_code=422, detail="each reference distance_m must be > 0")
    (d / "work").mkdir(parents=True, exist_ok=True)
    (d / _SCALE_REFS_REL).write_text(
        json.dumps([r.model_dump() for r in body.references], indent=2))
    return {"ok": True, "count": len(body.references),
            "enough": len(body.references) >= 3}


@router.get("/api/p/{name}/targetless-stages")
async def list_targetless_stages(request: Request, name: str) -> dict:
    """Which of the 5 key-stage images exist for this project (post-solve)."""
    d = project_dir(request, name)
    stage_dir = d / _STAGE_DIR_REL
    names = sorted(p.stem for p in stage_dir.glob("*.jpg")) if stage_dir.is_dir() else []
    return {"stages": names}


@router.get("/targetless-stage/{name}/{stage}")
def targetless_stage_image(request: Request, name: str, stage: str) -> FileResponse:
    """Serve one annotated key-stage image (pair/matches/scale_refs/triangulation/result)."""
    if not re.match(r"^[A-Za-z0-9_\-]+$", stage):
        raise HTTPException(status_code=404, detail="not found")
    d = project_dir(request, name)
    base = (d / _STAGE_DIR_REL).resolve()
    target = (base / f"{stage}.jpg").resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="image/jpeg")


@router.get("/api/p/{name}/floor-shots")
def list_floor_shots(request: Request, name: str) -> dict:
    """Read-only presence of the per-camera floor-anchor ChArUco shots (multi-shot).

    The single-button [FLOOR] flow auto-snaps synchronized ChArUco pairs into
    ``floor/<cam>/NNN.jpg`` (a per-camera DIR, one flat placement per index). A
    legacy single ``floor/<cam>.jpg`` file is still recognised. Returns
    ``{target, cameras: {cam: {present, count, files:[...]}}}`` — the Floor-anchor
    cell shows the captured placements or an "awaiting floor shots" placeholder.
    Never writes.
    """
    from ..capture.session import FLOOR_TARGET
    from ..core.project import floor_shots
    d, cfg = project_cfg(request, name)
    cams: dict[str, dict] = {}
    for cam in cfg.configured_cameras():
        shots = floor_shots(d, cam)
        # files are addressed relative to floor/ so the image server can find them
        files = [str(p.relative_to(d / "floor")) for p in shots]
        cams[cam] = {"present": bool(shots), "count": len(shots), "files": files}
    return {"target": FLOOR_TARGET, "cameras": cams}


_FLOOR_FILE_RE = re.compile(r"^[A-Za-z0-9_\-]+(/[A-Za-z0-9_\-]+)?\.jpg$")


@router.get("/floor-shot/{name}/{file:path}")
def floor_shot_image(request: Request, name: str, file: str) -> FileResponse:
    """Serve one captured floor-anchor jpg. Read-only, path-guarded.

    Accepts both the legacy flat ``<cam>.jpg`` and the new ``<cam>/NNN.jpg`` layout
    (one level deep). Resolved-path containment guards against traversal.
    """
    if not _FLOOR_FILE_RE.match(file):
        raise HTTPException(status_code=404, detail="not found")
    d = project_dir(request, name)
    base = (d / "floor").resolve()
    target = (base / file).resolve()
    if not (target == base or base in target.parents) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="image/jpeg")


@router.get("/shots/{name}/{phase}/{cam}/{file}")
def shot_image(request: Request, name: str, phase: str, cam: str, file: str) -> FileResponse:
    if phase not in ("intrinsic", "extrinsic") or not _SHOT_FILE_RE.match(file):
        raise HTTPException(status_code=404, detail="not found")
    d = project_dir(request, name)
    base = (d / phase / cam).resolve()
    target = (base / file).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="image/jpeg")
