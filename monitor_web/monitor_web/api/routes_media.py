"""Hidden MP4 dev viewer (S12.2) — replay a media-folder MP4 *through the detector*.

DEV-ONLY, password-gated. This is the one **documented exception** to monitor_web's
"don't import the Backbone detector" rule (see CLAUDE.md): so a recorded MP4 can be
reviewed with detection overlays, it loads `backbone.detection.YoloOnnxDetector`
in-process and streams annotated MJPEG. The password gate is obscurity for a
localhost tool — NOT real auth (the stream/list endpoints aren't session-gated).

Endpoints:
  POST /api/unlock           {password} -> {ok}
  GET  /api/media/mp4        -> {files: [relpath, ...]}
  GET  /stream/mp4/{name}    -> multipart MJPEG with detection boxes drawn
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..detection_overlay import (
    annotate_frame,
    boxes_enabled,
    get_detector,
    masks_enabled,
    nodes_enabled,
    occupancy_enabled,
)
from ..video_stream import JPEG_BOUNDARY, encode_mjpeg_frame

logger = logging.getLogger(__name__)

router = APIRouter()


class UnlockBody(BaseModel):
    password: str = ""


# ---- helpers ----


def _media_root(cfg) -> Path:
    return Path(cfg.media_dir).resolve()


def _annotated_mjpeg(path: Path, detector, cam_id: str = "mp4",
                     show_nodes: bool = True, show_masks: bool = True,
                     show_boxes: bool = True, pose_detector=None,
                     show_occupancy: bool = False) -> Iterator[bytes]:
    """Decode the MP4 and yield MJPEG chunks, paced to real time. If ``detector``
    is None (no model configured), play raw — boxes appear once a model is set.
    ``pose_detector`` (optional) overlays person poses + foot nodes alongside.
    ``show_occupancy`` draws the pallet empty/full badge — image-space (approach A),
    so it needs no calibration and works on any MP4 (unlike the metric distance
    lines, which require a calibrated camera the MP4 isn't tied to)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps if fps and fps > 1 else 1.0 / 25.0   # pace to real time
    try:
        while True:
            t0 = time.time()
            ok, image = cap.read()
            if not ok:
                break
            if detector is not None or pose_detector is not None:
                annotate_frame(image, detector, cam_id=cam_id,
                               show_nodes=show_nodes, show_masks=show_masks,
                               show_boxes=show_boxes, pose_detector=pose_detector,
                               show_occupancy=show_occupancy)
            try:
                yield encode_mjpeg_frame(image)
            except (ValueError, RuntimeError):
                pass
            # Real-time pacing: sleep off whatever the read/detect/encode didn't use.
            lag = frame_interval - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
    finally:
        cap.release()


# ---- routes ----


@router.post("/api/unlock")
async def unlock(body: UnlockBody, request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    ok = bool(body.password) and body.password == cfg.mp4_unlock_password
    return JSONResponse({"ok": ok})


@router.get("/api/media/mp4")
async def media_mp4(request: Request) -> JSONResponse:
    """List every *.mp4 under media_dir (default: the whole project), recursively.
    Hidden dirs (.git, .venv, __pycache__, …) are pruned for speed + cleanliness."""
    cfg = request.app.state.settings
    root = _media_root(cfg)
    files: list[str] = []
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]  # prune hidden
            for fn in filenames:
                if fn.lower().endswith(".mp4") and not fn.endswith("Zone.Identifier"):
                    files.append(os.path.relpath(os.path.join(dirpath, fn), root))
        files.sort()
    return JSONResponse({"files": files})


@router.get("/stream/mp4/{name:path}")
async def stream_mp4(name: str, request: Request) -> StreamingResponse:
    # Only serve .mp4 (this also rejects traversal probes like ../../etc/passwd).
    if not name.lower().endswith(".mp4"):
        raise HTTPException(400, "only .mp4 files are served")
    cfg = request.app.state.settings
    root = _media_root(cfg)
    target = (root / name).resolve()
    if root != target and root not in target.parents:   # escaped media_dir
        raise HTTPException(404, f"not found: {name}")
    if not target.is_file():
        raise HTTPException(404, f"not found: {name}")
    # No model configured? Degrade to raw playback (no boxes) instead of erroring,
    # so the video always plays; detections appear once a model is set.
    try:
        detector = get_detector(cfg)
    except HTTPException as exc:
        logger.warning("MP4 viewer: %s — raw playback without detections", exc.detail)
        detector = None
    return StreamingResponse(
        _annotated_mjpeg(target, detector,
                         show_nodes=nodes_enabled(cfg),
                         show_masks=masks_enabled(cfg),
                         show_boxes=boxes_enabled(cfg),
                         pose_detector=None,
                         show_occupancy=occupancy_enabled(cfg)),
        media_type=f"multipart/x-mixed-replace; boundary={JPEG_BOUNDARY}",
    )
