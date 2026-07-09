"""MJPEG re-streaming endpoint per camera.

The Backbone's `RtspFrameSource` is re-used as a consumer-side library to
pull frames; we encode them as JPEG and stream as ``multipart/x-mixed-replace``
so the browser's native MJPEG support renders them in ``<img>``.

Cameras are read from the same ``backbone.yaml`` the Backbone subprocess uses.
Each ``GET /stream/video/{camera_id}`` spawns its own RtspFrameSource — they
share the underlying RTSP URL (RTSP is one-to-many).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..camera_hub import get_hub
from ..detection_overlay import (
    annotate_frame,
    boxes_enabled,
    display_fps,
    distance_line_style,
    distances_enabled,
    get_async_pose,
    masks_enabled,
    nodes_enabled,
    occupancy_enabled,
    person_pallet_max_m,
)
from ..floor_rectify import (
    build_fit_rectify_matrix,
    composite_bev,
    rectify_frame,
    rectify_params_for_frame,
    shared_bev_layout,
)
from ..video_stream import JPEG_BOUNDARY, mjpeg_stream
from .routes_calibrate import _mode_calibration_path
from .routes_projection import _load_rig_cached
from .routes_zone_patches import find_patch, patch_pixel_box, patch_rect

logger = logging.getLogger(__name__)

router = APIRouter()

# Display streams don't need full camera FPS — cap inference/compositing to bound
# GPU load when several detect/patch/unified streams run at once. Frames above the
# cap are dropped (the hub already coalesces to the latest), not queued.
DISPLAY_FPS = 10.0


def _cap_fps(frames: Iterator, fps: float = DISPLAY_FPS) -> Iterator:
    """Yield at most ``fps`` frames/sec, dropping the rest — so the (expensive)
    per-frame detection/warp downstream runs at most ``fps`` times a second."""
    min_dt = 1.0 / float(fps)
    last = 0.0
    for image in frames:
        now = time.monotonic()
        if now - last < min_dt:
            continue
        last = now
        yield image


def _backbone_running(state) -> bool:
    """True iff the Backbone supervisor is RUNNING. Gates whether the camera views do
    in-process detection: off (raw feed) until START, on once the Backbone spawns.
    Takes the app state (works from both HTTP requests and WebSocket handlers)."""
    sup = getattr(state, "supervisor", None)
    try:
        return bool(sup) and sup.state == "running"
    except Exception:
        return False


# Zone detection now lives in the background ZoneDetectionWorker (zone_worker.py) —
# one thread per camera, all zones on the same frame, one atomic snapshot. These
# re-exports keep existing imports/tests stable.
from ..zone_worker import (  # noqa: E402, F401
    _drop_persons,
    _zone_objects,
)


def _load_cameras_from_backbone_yaml(path: Path) -> dict[str, dict]:
    """Return {camera_id: source_config} from the Backbone's YAML."""
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    cams = data.get("cameras", {})
    return cams if isinstance(cams, dict) else {}


def _frame_iter(camera_id: str, source_cfg: dict) -> Iterator:
    """Yield decoded frames for a camera from the shared :class:`CameraHub`.

    All viewers of a camera fan out from ONE long-lived source (one RTSP/V4L2
    session) instead of each ``<img>`` opening its own — which used to exhaust
    the Hikvision's concurrent-session cap (and is impossible for a V4L2 device).
    The hub also owns reconnect/placeholder resilience, so a slow or briefly
    failing camera shows "connecting…" and self-heals rather than killing the
    response. Client disconnect raises ``GeneratorExit`` at the next ``yield``,
    unwinding the ``finally`` so the viewer is released (and the source retired
    once the last viewer leaves).
    """
    src_cfg = dict(source_cfg)
    plugin = src_cfg.pop("name", "rtsp")
    hub = get_hub()
    stream = hub.acquire(camera_id, plugin, src_cfg)
    try:
        yield from stream.read()
    finally:
        hub.release(stream)


def grab_real_frame(camera_id: str, source_cfg: dict, timeout: float = 4.0):
    """Return ONE genuine decoded frame for a camera, or ``None`` on timeout.

    For one-shot consumers (the MAP warp-snapshot). Unlike ``_frame_iter``'s
    first ``read()``, this skips the pump's "connecting…" placeholder via the
    hub's explicit ``wait_for_real_frame`` flag, so the caller never rectifies
    the placeholder. Acquires/releases the shared stream like ``_frame_iter`` so
    the source is never leaked."""
    src_cfg = dict(source_cfg)
    plugin = src_cfg.pop("name", "rtsp")
    hub = get_hub()
    stream = hub.acquire(camera_id, plugin, src_cfg)
    try:
        return stream.wait_for_real_frame(timeout=timeout)
    finally:
        hub.release(stream)


def _detect_iter(frames: Iterator, cfg, camera_id: str, *, is_running=None,
                 get_zone_dets=None) -> Iterator:
    """Annotate each cam frame before MJPEG-encoding.

    Gated by ``is_running`` (the Backbone-running check): until START, yields the RAW
    frame — no detection, pose or lines — so the cam view just confirms the camera.

    STRICT rule: the cam view NEVER runs a full-frame object detector. The only
    model that runs on the full frame here is HUMAN POSE; objects come SOLELY from
    the background :class:`~monitor_web.zone_worker.ZoneDetectionWorker`'s snapshot
    (``get_zone_dets`` closure: one coherent frame's worth, already person-free and
    cross-zone deduped at publish time) — naturally empty when the camera has no
    zones.

    POSE IS ASYNC: the video loop never waits on the pose model. A per-camera
    background worker (``AsyncPoseRunner``) infers on the newest frame at
    whatever rate the GPU allows; every rendered frame overlays the LATEST
    completed skeletons. The view therefore stays at the camera rate even while
    the GPU is busy with the live Backbone; skeletons refresh at the
    pose-achievable rate. The separate "Zones FPS" preference governs only the
    zone workers, not pose.
    Zone-worker detections and distance lines are re-drawn on every frame too
    (cheap — the worker already ran the inference; we just draw its cached result)."""
    pose = dist_view = dets = None
    for image in frames:
        if is_running is not None and not is_running():
            # Backbone stopped → raw camera feed only. DROP the previous
            # iteration's refs: a suspended generator's locals otherwise pin
            # the pose engine (its CUDA session) and the dets' full-frame
            # masks after STOP, defeating reset_detector()'s memory release
            # for as long as the panel stays open.
            pose = dist_view = dets = None
            yield image
            continue
        dist_view = _warp_camera(cfg, camera_id) if distances_enabled(cfg) else None
        dist_style = distance_line_style(cfg)
        # Async pose: the runner infers in a background worker on the newest
        # frame; the video loop just overlays the latest skeletons — the view
        # stays at camera rate even when the GPU is busy with the Backbone.
        pose = get_async_pose(cfg, camera_id)
        dets = get_zone_dets() if get_zone_dets is not None else []
        # show_occupancy stays False on the CAM views: the raw machine labels
        # ('palette_vide' / 'palette_carton_…') read as noise here — the human
        # summary lives in the COMMUNICATION zone cards. Zone panels + the MP4
        # dev viewer keep the badge (still governed by the Settings toggle).
        out = annotate_frame(image, None, cam_id=camera_id, detections=dets,
                             show_nodes=nodes_enabled(cfg), show_masks=masks_enabled(cfg),
                             show_boxes=boxes_enabled(cfg), pose_detector=pose,
                             dist_view=dist_view, dist_max_m=person_pallet_max_m(cfg),
                             show_occupancy=False, dist_style=dist_style)
        # No fused-track ring markers here (the '#id cls' amber/green circles
        # were retired as clutter, like the mirrored rings before them): every
        # zone has a detecting TWIN on the other camera, so boxes/masks appear
        # in both views natively, and the metric proof lives in the Settings
        # triangulation test + the warp view's unified-track overlay.
        yield out


def _warp_camera(cfg, camera_id: str):
    """The calibrated camera view for the current mode, or ``None`` if this camera
    isn't calibrated. Best-effort (no raise): auto-warp must degrade to the raw
    feed, never break the stream. Reads the CURRENT-mode calibration file so the
    warp matches the operational mode (1 = 4pt, 2 = Multical)."""
    cal_path = _mode_calibration_path(cfg)
    if not cal_path.exists():
        return None
    try:
        rig = _load_rig_cached(str(cal_path.resolve()), cal_path.stat().st_mtime_ns)
    except Exception:
        return None
    return rig[camera_id] if camera_id in rig else None


def _draw_unified_tracks(frame, bounds, bus) -> None:
    """Overlay the FUSED (one-identity) tracks on a rectified floor view — Mode-2
    calibration visibility. Each unified track is drawn at its world (X, Y) mapped
    to the warp's pixels (u=(X-x_min)·ppm, v=(Y-y_min)·ppm); 3D-triangulated tracks
    are ringed green, 2D-only amber. Mutates ``frame``. Best-effort: stale/empty
    bus → nothing drawn."""
    try:
        if not bus.is_fresh(1.5):
            return
        snap = bus.snapshot()
    except Exception:
        return
    ppm, xm, ym = bounds["px_per_m"], bounds["x_min"], bounds["y_min"]
    ow, oh = bounds["out_wh"]
    t3d = snap.last_track3d_by_id
    pos: dict[int, tuple[float, float]] = {
        tid: (m.xy_m[0], m.xy_m[1]) for tid, m in snap.last_track2d_by_id.items()}
    pos.update({tid: (m.xyz_m[0], m.xyz_m[1]) for tid, m in t3d.items()})  # 3D wins
    for tid, (x, y) in pos.items():
        u, v = round((x - xm) * ppm), round((y - ym) * ppm)
        if not (0 <= u < ow and 0 <= v < oh):
            continue
        color = (80, 230, 120) if tid in t3d else (80, 170, 255)  # green=3D, amber=2D
        cv2.circle(frame, (u, v), 9, color, 2)
        cv2.circle(frame, (u, v), 2, color, -1)
        cv2.putText(frame, f"#{tid}", (u + 11, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


def _warp_detect_iter(frames: Iterator, cfg, camera_id: str, cam, M, out_wh, do_detect: bool,
                      is_running=None, bus=None, mode2: bool = False) -> Iterator:
    """Warp each frame to the bird's-eye floor view (M = S·H) at the auto-fit
    output size ``out_wh`` (``do_detect`` is accepted for signature stability
    but no longer runs inference — ONE perception; see below). It draws ON THE
    RECTIFIED frame and draw boxes, so detection continues over the warped view.
    Detector re-fetched per frame (cached) so a model swap applies live; falls
    back to the plain warp if no model is resolvable.

    In Mode 2 (``mode2`` + a ``bus``), also overlays the FUSED tracks on the
    rectified floor — visibility that calibration unifies both cameras into one
    identity space (an object seen by either camera lands at the same floor spot).

    Frame-size guard: when the live camera delivers frames at a different
    resolution than the calibration's ``image_size_wh``, M/out_wh (and the
    world→pixel ``bounds``) are recomputed on the first frame so the full source
    content is visible instead of being clipped to the calibration-size box.
    """
    _M, _out_wh, _bounds = M, out_wh, None
    _checked = False
    for image in frames:
        if not _checked:
            _checked = True
            ih, iw = image.shape[:2]
            params = rectify_params_for_frame(cam.H, cam.image_size_wh, (iw, ih))
            if params is not None:
                _M, _out_wh, _bounds = params["M"], params["out_wh"], params["bounds"]
        warped = rectify_frame(image, cam.K, cam.D, cam.H, out_wh=_out_wh, M=_M)
        running = is_running is None or is_running()
        # ONE PERCEPTION: the warp view no longer runs its own full-frame
        # detection/pose (it was the last hidden dashboard inference path —
        # per-frame get_detector + annotate in this pump; an exception there
        # killed the pump, so the verify view went blank exactly while the
        # Backbone ran). Calibration verification needs the flattened floor +
        # the FUSED tracks below, nothing else.
        # Mode-2 visibility: unified (fused) tracks on the rectified floor.
        if mode2 and bus is not None and _bounds is not None and running:
            _draw_unified_tracks(warped, _bounds, bus)
        yield warped


def _to_crop(d, x0: int, y0: int, ch: int, cw: int):
    """Translate one detection into crop coordinates (shift bbox/foot/polygon
    by the crop origin; slice the mask). The inverse of the worker's
    `_remap_det`, used by the panel renderer to draw the worker's snapshot on
    the cropped view. Duck-typed on purpose: snapshot dets are core
    ``Detection`` objects in local mode but plain namespaces from the wire's
    observations in backbone mode (which carry ``mask_poly``, no bitmaps and
    no camera/ts fields) — the drawer only reads attributes."""
    from types import SimpleNamespace
    bx = d.bbox_xyxy
    bbox = (bx[0] - x0, bx[1] - y0, bx[2] - x0, bx[3] - y0)
    foot = None if d.foot_uv is None else (d.foot_uv[0] - x0, d.foot_uv[1] - y0)
    mask = None
    if getattr(d, "mask", None) is not None:
        mask = d.mask[y0:y0 + ch, x0:x0 + cw]
    poly = getattr(d, "mask_poly", None)
    return SimpleNamespace(
        cls=d.cls, confidence=d.confidence, bbox_xyxy=bbox, foot_uv=foot,
        keypoints_uv=getattr(d, "keypoints_uv", None), mask=mask,
        mask_poly=[[px - x0, py - y0] for px, py in poly] if poly else None,
        occupancy_state=getattr(d, "occupancy_state", None),
        occupancy_content=getattr(d, "occupancy_content", None),
        occupancy_confidence=getattr(d, "occupancy_confidence", 0.0))


_ZONE_STATUS_LABELS = {
    "no_vram": "MODEL UNAVAILABLE (VRAM)",
    "error": "DETECTION ERROR",
}


def _zone_render_iter(frames: Iterator, cfg, camera_id: str, rect, stored_wh,
                      infer_size: int = 320, is_running=None, get_dets=None,
                      get_status=None) -> Iterator:
    """Pure RENDERER for a zone panel — NO detection in the HTTP path. Crops each
    frame to the zone's bounding rect, draws the background worker's published
    detections for this zone (translated to crop coords), then downsizes to the
    panel display size. Backbone stopped → raw crop (the pre-START state). A zone
    the worker disabled (circuit breaker) gets its status banner drawn instead of
    silently looking object-free."""
    for image in frames:
        ih, iw = image.shape[:2]
        box = patch_pixel_box(rect, stored_wh, (iw, ih))
        if box is None:
            continue
        x0, y0, x1, y1 = box
        crop = image[y0:y1, x0:x1].copy()        # copy: annotate draws in place
        ch, cw = crop.shape[:2]
        running = is_running() if is_running is not None else True
        if running:
            dets = [_to_crop(d, x0, y0, ch, cw) for d in (get_dets() if get_dets else [])]
            crop = annotate_frame(crop, None, cam_id=camera_id, detections=dets,
                                  show_nodes=nodes_enabled(cfg), show_masks=masks_enabled(cfg),
                                  show_boxes=boxes_enabled(cfg), pose_detector=None,
                                  show_occupancy=occupancy_enabled(cfg))
            label = _ZONE_STATUS_LABELS.get(get_status() if get_status else "")
            if label:
                cv2.putText(crop, label, (8, max(20, ch - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(crop, label, (8, max(20, ch - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 1, cv2.LINE_AA)
        # Downscale to the panel display size (longest side = infer_size) for
        # bandwidth parity with the old fed-image stream.
        longest = max(ch, cw)
        if longest > infer_size and longest > 0:
            s = infer_size / float(longest)
            crop = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))),
                              interpolation=cv2.INTER_AREA)
        yield crop


def build_zone_stream(state, patch_id: str) -> Iterator:
    """Frame iterator for one zone-patch panel (crop + worker detections + status
    banner). Shared by the MJPEG endpoint and /ws/video. Raises ``LookupError``
    when the ROI or its camera isn't configured."""
    cfg = state.settings
    patch = find_patch(cfg, patch_id)
    if patch is None:
        raise LookupError(f"zone patch {patch_id!r} not found")
    camera_id = patch.get("camera", "cam_a")
    cameras = _load_cameras_from_backbone_yaml(cfg.backbone_config_path)
    if camera_id not in cameras:
        raise LookupError(f"camera {camera_id!r} not configured")
    src_cfg = cameras[camera_id].get("source", {})
    rect = patch_rect(patch)   # polygon bounding box (or stored rect)
    if rect is None:
        raise LookupError(f"zone patch {patch_id!r} has no rect/polygon")
    manager = getattr(state, "zone_manager", None)
    frames = _cap_fps(_frame_iter(camera_id, src_cfg), display_fps(cfg))
    return _zone_render_iter(
        frames, cfg, camera_id, rect, patch.get("frame_wh"),
        infer_size=int(patch.get("infer_size") or 320),
        is_running=lambda: _backbone_running(state),
        get_dets=(lambda: manager.zone_dets(patch_id)) if manager is not None else None,
        get_status=(lambda: manager.zone_status(patch_id)) if manager is not None else None,
    )


@router.get("/stream/zone/{patch_id}")
async def zone_patch_stream(patch_id: str, request: Request) -> StreamingResponse:
    """Live MJPEG of one zone-patch ROI: the camera feed cropped to the watch box with
    the background worker's detections overlaid (the panel never detects itself).
    Reuses the shared CameraHub session, so it adds no extra RTSP load. 404 if the
    ROI or its camera isn't configured."""
    try:
        frames = build_zone_stream(request.app.state, patch_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        mjpeg_stream(frames),
        media_type=f"multipart/x-mixed-replace; boundary={JPEG_BOUNDARY}",
    )


def _unified_iter(cam_ids, src_cfgs, views, fps=DISPLAY_FPS) -> Iterator:
    """Composite a bird's-eye unified view from the configured cameras (Mode 2).

    Driven by the first camera's frame cadence; every OTHER camera contributes its
    latest real frame, or is skipped while its feed is down — so the unified view
    degrades gracefully to whatever cameras are actually live (Phase D). The shared
    floor layout is recomputed from the contributing frames each tick (cheap vs the
    warp), so cameras appearing/disappearing re-fit the canvas automatically."""
    hub = get_hub()
    streams: dict[str, object] = {}
    for cid in cam_ids:
        scfg = dict(src_cfgs[cid])
        plugin = scfg.pop("name", "rtsp")
        streams[cid] = hub.acquire(cid, plugin, scfg)
    primary = cam_ids[0]
    try:
        for frame_primary in _cap_fps(streams[primary].read(), fps):
            frames = {primary: frame_primary}
            for cid in cam_ids[1:]:
                f = streams[cid].latest_real_frame()
                if f is not None:
                    frames[cid] = f
            order = [cid for cid in cam_ids if cid in frames]
            cams_for_layout = [
                (views[cid].H, tuple(views[cid].image_size_wh),
                 (frames[cid].shape[1], frames[cid].shape[0]))
                for cid in order
            ]
            layout = shared_bev_layout(cams_for_layout)
            if layout is None:
                yield frames[primary]          # degenerate calibration → raw primary
                continue
            bounds, matrices = layout
            layers = [
                (frames[cid], views[cid].K, views[cid].D, M)
                for cid, M in zip(order, matrices, strict=True)
            ]
            yield composite_bev(layers, bounds["out_wh"])
    finally:
        for s in streams.values():
            hub.release(s)


def build_unified_stream(state) -> Iterator:
    """Frame iterator for the Mode-2 unified bird's-eye composite. Shared by the
    MJPEG endpoint and /ws/video. Raises ``LookupError`` when unavailable."""
    cfg = state.settings
    cameras = _load_cameras_from_backbone_yaml(cfg.backbone_config_path)
    cal_path = _mode_calibration_path(cfg)
    if not cal_path.exists():
        raise LookupError("no Mode-2 calibration — run calibrate-all")
    try:
        rig = _load_rig_cached(str(cal_path.resolve()), cal_path.stat().st_mtime_ns)
    except Exception as exc:
        raise LookupError(f"calibration unreadable: {exc}") from exc
    cam_ids = [c for c in ("cam_a", "cam_b") if c in cameras and c in rig]
    if len(cam_ids) < 2:
        raise LookupError("unified view needs 2 configured+calibrated cameras")
    views = {cid: rig[cid] for cid in cam_ids}
    src_cfgs = {cid: cameras[cid].get("source", {}) for cid in cam_ids}
    return _unified_iter(cam_ids, src_cfgs, views, display_fps(cfg))


@router.get("/stream/unified")
async def unified_stream(request: Request) -> StreamingResponse:
    """Mode-2 unified bird's-eye view: every configured camera warped to the shared
    metric floor and composited into one top-down picture. Needs the Mode-2 (joint)
    calibration so the cameras share one world frame. 404 if <2 cameras are
    configured-and-calibrated. Degrades to the live cameras if one feed is down."""
    try:
        frames = build_unified_stream(request.app.state)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        mjpeg_stream(frames),
        media_type=f"multipart/x-mixed-replace; boundary={JPEG_BOUNDARY}",
    )


def build_cam_stream(state, camera_id: str, *, detect: bool = False,
                     warp: bool = False) -> Iterator:
    """Frame iterator for a camera view. Shared by the MJPEG endpoint and
    /ws/video. Raises ``LookupError`` when the camera isn't configured.

    - ``detect`` overlays pose (+ zone-worker detections + distance lines) once
      the Backbone runs — needs **no calibration**.
    - ``warp`` auto-rectifies through the current mode's calibration homography;
      best-effort (uncalibrated camera → raw feed).
    """
    cfg = state.settings
    cameras = _load_cameras_from_backbone_yaml(cfg.backbone_config_path)
    if camera_id not in cameras:
        raise LookupError(f"camera {camera_id!r} not configured")
    src_cfg = cameras[camera_id].get("source", {})
    # Gate in-process detection on the Backbone RUNNING — checked PER FRAME (via the
    # `is_running` closure) so an already-open cam stream flips from raw → detection
    # the instant START fires, no reload. Before START the cam views show only the
    # RAW feed (confirm the camera; status AMBER "ready — press START"); pose +
    # distance lines begin with the Backbone. Also spares idle GPU/RAM.
    is_running = lambda: _backbone_running(state)   # noqa: E731
    warp_cam = _warp_camera(cfg, camera_id) if warp else None
    frames = _frame_iter(camera_id, src_cfg)
    if warp_cam is not None:
        # Auto-fit: rectify the WHOLE frame into a canvas sized to its warped
        # bounding box (+Y down, right-side-up). Regains content the old fixed
        # window cropped and tightens the black margin to just the unavoidable
        # perspective wedges.
        M, out_wh = build_fit_rectify_matrix(warp_cam.H, warp_cam.image_size_wh)
        # Mode 2 (≥2 cameras) → overlay the fused/unified tracks on the warp for
        # calibration visibility (replaces the retired unified BEV render).
        mode2 = len(cameras) >= 2
        # Cap before the (expensive) warp+detect; raw passthrough below stays smooth.
        frames = _warp_detect_iter(_cap_fps(frames, display_fps(cfg)) if detect else frames,
                                   cfg, camera_id, warp_cam, M, out_wh, do_detect=detect,
                                   is_running=is_running,
                                   bus=getattr(state, "bus", None), mode2=mode2)
    elif detect:
        # Per-frame detector lookup (see _detect_iter) so model changes apply live.
        # NO _cap_fps here: the source is already capped at capture_fps (Camera FPS)
        # by the camera-hub. _detect_iter runs POSE on every frame, so pose inherits
        # the Camera-FPS cam-view rate (Zones FPS / display_fps is zones-only now).
        manager = getattr(state, "zone_manager", None)
        frames = _detect_iter(
            frames, cfg, camera_id,
            is_running=is_running,
            get_zone_dets=(lambda: manager.camera_dets(camera_id)) if manager is not None else None,
        )
    return frames


@router.get("/stream/video/{camera_id}")
async def video_stream(
    camera_id: str,
    request: Request,
    detect: bool = False,
    warp: bool = False,
) -> StreamingResponse:
    """Live MJPEG for a camera.

    - ``?detect=1`` overlays pose + zone-worker detections in-process (+ foot
      nodes if Settings allows) — needs **no calibration**.
    - ``?warp=1`` auto-rectifies the feed to a bird's-eye floor view through the
      current mode's calibration homography (the frontend adds this only when the
      camera is calibrated). The output is auto-fit to the warped frame's bounds
      so no content is cropped. Best-effort: if the camera isn't calibrated, falls
      back to the raw (+detect) feed — the stream never 503s.
    """
    try:
        frames = build_cam_stream(request.app.state, camera_id, detect=detect, warp=warp)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        mjpeg_stream(frames),
        media_type=f"multipart/x-mixed-replace; boundary={JPEG_BOUNDARY}",
    )
