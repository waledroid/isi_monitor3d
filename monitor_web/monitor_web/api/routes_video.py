"""MJPEG re-streaming endpoint per camera.

The Backbone's `RtspFrameSource` is re-used as a consumer-side library to
pull frames; we encode them as JPEG and stream as ``multipart/x-mixed-replace``
so the browser's native MJPEG support renders them in ``<img>``.

Cameras are read from the same ``backbone.yaml`` the Backbone subprocess uses.
Each ``GET /stream/video/{camera_id}`` spawns its own RtspFrameSource — they
share the underlying RTSP URL (RTSP is one-to-many).
"""

from __future__ import annotations

import functools
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
    distance_line_style,
    distances_enabled,
    floor_zones_enabled,
    get_async_pose,
    masks_enabled,
    nodes_enabled,
    person_pallet_max_m,
    zone_fill_dim_enabled,
)
from ..floor_rectify import (
    build_fit_rectify_matrix,
    composite_bev,
    rectify_frame,
    rectify_params_for_frame,
    shared_bev_layout,
)
from ..floor_zone_sync import _zones_yaml_path
from ..video_stream import JPEG_BOUNDARY, mjpeg_stream
from ..zone_projection import (
    clip_to_zones_metric,
    draw_zone_outlines,
    project_zone_hulls,
    project_zone_polygons,
    scale_polygons,
    zone_stencil,
)
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
                 get_zone_dets=None, wire_pose=None, zone_ctx=None,
                 axis_overlay=None) -> Iterator:
    """Annotate each cam frame before MJPEG-encoding.

    Gated by ``is_running`` (the Backbone-running check): until START, yields the RAW
    frame — no detection, pose or lines — so the cam view just confirms the camera.

    STRICT rule: the cam view NEVER runs a full-frame object detector. The only
    model that runs on the full frame here is HUMAN POSE; objects come from the
    ``get_zone_dets(image)`` closure — in points mode the producer's zone-scoped
    observations straight off the bus (:class:`~monitor_web.pose_overlay.WireObjectSource`,
    person-free, boxes wherever detection fires, independent of the pixel-space
    zone_patches that gate the ZONE PANELS); in frames mode the background
    :class:`~monitor_web.zone_worker.ZoneDetectionWorker`'s patch-scoped snapshot
    (naturally empty when the camera has no zones).

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
    _stencil_cache, _stencil_wh = None, None
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
        # Skeleton source: in points mode (Direction 1) the producer's pose
        # rides the observations echo — render it with ZERO dashboard
        # inference (wire_pose). Frames mode keeps the async runner (the
        # dashboard is then the only pose in the system for display).
        pose = wire_pose if wire_pose is not None else get_async_pose(cfg, camera_id)
        dets = get_zone_dets(image) if get_zone_dets is not None else []
        # Zone-based: membership is METRIC — each foot projects to the zone's
        # own plane (the engine's own geometry) and must land inside a zone
        # polygon ± the producer's configured tolerance
        # (detection.zone_membership_tol_m), so both cameras agree with each
        # other AND with the fused zone state (a pixel-polygon test dropped
        # boundary objects on one camera while the other showed them). Masks are stencil-bounded to
        # the zones' EXTRUDED HULLS (tight laterally, tall enough for the
        # objects standing in the zone). Persons/skeletons are the deliberate global
        # safety exception — never zone-clipped. No zones ⇒ no object boxes.
        scaled_zones = stencil = None
        if zone_ctx is not None:
            fh, fw = image.shape[:2]
            cw, ch = zone_ctx["calib_wh"]
            scaled_zones = scale_polygons(zone_ctx["polys"], fw / cw, fh / ch)
            dets = clip_to_zones_metric(dets, zone_ctx["rig"], camera_id, (fw, fh),
                                        zone_ctx["zones"],
                                        tol_m=zone_ctx.get("tol_m", 0.15))
            if _stencil_wh != (fw, fh):     # zones are fixed per stream build
                _stencil_cache = zone_stencil(
                    (fh, fw), scale_polygons(zone_ctx["hulls"], fw / cw, fh / ch))
                _stencil_wh = (fw, fh)
            stencil = _stencil_cache
        # show_occupancy stays False on the CAM views: the raw machine labels
        # ('palette_vide' / 'palette_carton_…') read as noise here — the human
        # summary lives in the COMMUNICATION zone cards. Zone panels + the MP4
        # dev viewer keep the badge (still governed by the Settings toggle).
        try:
            out = annotate_frame(image, None, cam_id=camera_id, detections=dets,
                                 show_nodes=nodes_enabled(cfg), show_masks=masks_enabled(cfg),
                                 show_boxes=boxes_enabled(cfg), pose_detector=pose,
                                 dist_view=dist_view, dist_max_m=person_pallet_max_m(cfg),
                                 show_occupancy=False, dist_style=dist_style,
                                 mask_clip=stencil)
        except Exception:
            # An overlay failure must NEVER kill the stream: the pump treats a
            # generator exception as terminal and the panel freezes until the
            # operator reloads. Show the raw frame instead, log throttled.
            now_s = time.monotonic()
            if now_s - _detect_iter._last_warn_s > 30.0:
                _detect_iter._last_warn_s = now_s
                logger.warning("cam %s: overlay failed — showing the raw frame "
                               "(throttled 30s)", camera_id, exc_info=True)
            out = image
        # The floor-zone boundaries on top of the annotated frame (Settings
        # toggle, off by default — the operator's dashed zone patches remain
        # the always-visible boundary). The CLIP above is not optional: a
        # zone-based system never shows detections outside the zones.
        if scaled_zones and floor_zones_enabled(cfg):
            draw_zone_outlines(out, scaled_zones)
        # 3D-localization axis + height badge at every FRESH two-view Track3D
        # fix on the bus (the cam-view twin of the floor map's gizmo). The
        # overlay is a no-op for Mode-1/uncalibrated cameras and never raises.
        if axis_overlay is not None:
            axis_overlay.draw(out)
        # No fused-track ring markers here (the '#id cls' amber/green circles
        # were retired as clutter, like the mirrored rings before them): every
        # zone has a detecting TWIN on the other camera, so boxes/masks appear
        # in both views natively, and the metric proof lives in the Settings
        # triangulation test + the warp view's unified-track overlay.
        yield out


_detect_iter._last_warn_s = 0.0   # overlay-failure log throttle (see above)


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


def _warp_detect_iter(frames: Iterator, cfg, camera_id: str, cam, M, out_wh,
                      is_running=None) -> Iterator:
    """Warp each frame to the bird's-eye floor view (M = S·H) at the auto-fit
    RECTIFIED frame. Pure geometry — no inference, no overlays: the fused
    #id ring markers were retired here just like on the plain cam views
    (operator feedback: clutter); the metric proof lives in the Settings
    triangulation test and the floor map.

    Frame-size guard: when the live camera delivers frames at a different
    resolution than the calibration's ``image_size_wh``, M/out_wh are
    recomputed on the first frame so the full source content is visible
    instead of being clipped to the calibration-size box.
    """
    _M, _out_wh = M, out_wh
    _checked = False
    for image in frames:
        if not _checked:
            _checked = True
            ih, iw = image.shape[:2]
            params = rectify_params_for_frame(cam.H, cam.image_size_wh, (iw, ih))
            if params is not None:
                _M, _out_wh = params["M"], params["out_wh"]
        yield rectify_frame(image, cam.K, cam.D, cam.H, out_wh=_out_wh, M=_M)


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


def _zone_render_iter(frames: Iterator, cfg, camera_id: str, rect, stored_wh,
                      display_px: int = 320, is_running=None, get_dets=None,
                      polygon=None, hull_calib=None, calib_wh=None,
                      fill_poly_calib=None) -> Iterator:
    """Pure RENDERER for a zone panel — NO detection in the HTTP path. Crops each
    frame to the zone's bounding rect, draws the background worker's published
    detections for this zone (translated to crop coords; masks stencil-clipped
    to the zone's EXTRUDED HULL when calibrated — the drawn polygon is flat and
    cuts the body off tall objects — else to the drawn polygon), then downsizes
    to the panel display size. Backbone stopped → raw crop (pre-START state)."""
    import numpy as np

    from ..zone_worker import _scaled_polygon
    _stencil, _stencil_key = None, None
    _dim, _dim_key = None, None
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
            # show_occupancy=False: the machine state labels ('palette_vide',
            # 'palette_carton_…') are decision data, not display — they live
            # on in the COMMUNICATION cards and MQTT; the zone view stays
            # clean (same rule the CAM views already follow).
            stencil = None
            if hull_calib is not None or polygon is not None:
                if _stencil_key != (iw, ih, x0, y0, ch, cw):
                    if hull_calib is not None and calib_wh:
                        poly = np.asarray(hull_calib, dtype=np.float64) * [
                            iw / float(calib_wh[0]), ih / float(calib_wh[1])]
                    else:
                        poly = _scaled_polygon({"polygon": polygon,
                                                "frame_wh": stored_wh}, (iw, ih))
                    if poly is not None and len(poly) >= 3:
                        m = np.zeros((ch, cw), dtype=np.uint8)
                        cv2.fillPoly(m, [np.asarray(
                            poly - [x0, y0], dtype=np.int32)], 255)
                        _stencil = m
                    _stencil_key = (iw, ih, x0, y0, ch, cw)
                stencil = _stencil
            # Fill-dim (Settings ▸ Display, off by default): darken the pixels
            # the producer's polygon fill blanks before inference — the panel
            # then shows the detector's true field of view. Uses the projected
            # FLOOR polygon (what the fill actually keeps at zero headroom),
            # not the extruded hull; slightly conservative vs the producer's
            # ~0.3 m dilation. Applied BEFORE annotate so overlays stay bright.
            if zone_fill_dim_enabled(cfg) and (
                    fill_poly_calib is not None or polygon is not None):
                if _dim_key != (iw, ih, x0, y0, ch, cw):
                    if fill_poly_calib is not None and calib_wh:
                        poly = np.asarray(fill_poly_calib, dtype=np.float64) * [
                            iw / float(calib_wh[0]), ih / float(calib_wh[1])]
                    else:
                        poly = _scaled_polygon({"polygon": polygon,
                                                "frame_wh": stored_wh}, (iw, ih))
                    _dim = None
                    if poly is not None and len(poly) >= 3:
                        m = np.zeros((ch, cw), dtype=np.uint8)
                        cv2.fillPoly(m, [np.asarray(
                            poly - [x0, y0], dtype=np.int32)], 255)
                        _dim = m == 0
                    _dim_key = (iw, ih, x0, y0, ch, cw)
                if _dim is not None:
                    crop[_dim] = (crop[_dim] * 0.45).astype(np.uint8)
            crop = annotate_frame(crop, None, cam_id=camera_id, detections=dets,
                                  show_nodes=nodes_enabled(cfg), show_masks=masks_enabled(cfg),
                                  show_boxes=boxes_enabled(cfg), pose_detector=None,
                                  show_occupancy=False, mask_clip=stencil)
        # Downscale to the panel display size (longest side = display_px) for
        # bandwidth parity with the old fed-image stream.
        longest = max(ch, cw)
        if longest > display_px and longest > 0:
            s = display_px / float(longest)
            crop = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))),
                              interpolation=cv2.INTER_AREA)
        yield crop


def build_zone_stream(state, patch_id: str) -> Iterator:
    """Frame iterator for one zone-patch panel (crop + worker detections). Shared
    by the MJPEG endpoint and /ws/video. Raises ``LookupError`` when the ROI or
    its camera isn't configured."""
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
    # The zone's extruded hull in THIS camera (mask stencil) — the zone id is
    # the base patch id; a twin panel shares its base's floor zone.
    hull_calib = calib_wh = fill_poly_calib = None
    ctx = _camera_zone_ctx(cfg, camera_id)
    if ctx is not None:
        zid = str(patch.get("twin_of") or patch_id)
        for h_zid, _n, hull in ctx["hulls"]:
            if h_zid == zid:
                hull_calib, calib_wh = hull, ctx["calib_wh"]
                break
        # The FLOOR polygon (not the hull) — the fill-dim overlay mirrors the
        # producer's polygon fill, which keeps only the floor footprint.
        for p_zid, _n, poly in ctx["polys"]:
            if p_zid == zid:
                fill_poly_calib, calib_wh = poly, ctx["calib_wh"]
                break
    frames = _frame_iter(camera_id, src_cfg)   # source-paced; no display cap
    return _zone_render_iter(
        frames, cfg, camera_id, rect, patch.get("frame_wh"),
        is_running=lambda: _backbone_running(state),
        get_dets=(lambda: manager.zone_dets(patch_id)) if manager is not None else None,
        polygon=patch.get("polygon"),
        hull_calib=hull_calib, calib_wh=calib_wh,
        fill_poly_calib=fill_poly_calib,
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
    return _unified_iter(cam_ids, src_cfgs, views, DISPLAY_FPS)


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


@functools.lru_cache(maxsize=4)
def _load_zones_cached(path_str: str, mtime_ns: int):
    """Load the floor ZoneRegistry keyed by (path, mtime) — edits invalidate."""
    from backbone.shared.zones import ZoneRegistry
    return ZoneRegistry.load(path_str)


def _camera_zone_ctx(cfg, camera_id: str) -> dict | None:
    """Everything the zone-based cam view needs, or ``None`` (uncalibrated /
    no zones ⇒ no object boxes). Computed once per stream build (a config
    save rebuilds the stream), not per frame.

    - ``polys``   projected floor-zone polygons, calibration px (outlines)
    - ``rig``     the calibration rig (plane-aware metric membership needs
                  every camera's view, not just this one — see
                  ``ZoneAwareProjector``)
    - ``zones``   the metric ZoneRegistry (membership polygons, in metres)
    - ``hulls``   the zones' extruded hulls, calibration px (the mask stencil)
    - ``calib_wh`` the calibration frame size (display scaling)
    """
    cal_path = _mode_calibration_path(cfg)
    if not cal_path.exists():
        return None
    try:
        rig = _load_rig_cached(str(cal_path.resolve()), cal_path.stat().st_mtime_ns)
        zpath = _zones_yaml_path(cfg)
        zones = _load_zones_cached(str(zpath), zpath.stat().st_mtime_ns)
        polys = project_zone_polygons(rig, zones, camera_id)
        if not polys or camera_id not in rig:
            return None
        w, h = rig[camera_id].image_size_wh
        hulls = project_zone_hulls(rig, zones, camera_id)
        from ..zone_projection import membership_tol_m
        return {"polys": polys, "rig": rig, "zones": zones,
                "hulls": hulls, "calib_wh": (int(w), int(h)),
                "tol_m": membership_tol_m(cfg.backbone_config_path)}
    except Exception:
        logger.warning("cam %s: floor-zone projection failed", camera_id, exc_info=True)
        return None


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
        frames = _warp_detect_iter(frames,
                                   cfg, camera_id, warp_cam, M, out_wh,
                                   is_running=is_running)
    elif detect:
        # Per-frame detector lookup (see _detect_iter) so model changes apply live.
        # NO _cap_fps here: the source is already capped at capture_fps (Camera FPS)
        # by the camera-hub. _detect_iter runs POSE on every frame, so pose inherits
        # the Camera-FPS cam-view rate (there is no separate zones rate cap).
        manager = getattr(state, "zone_manager", None)
        perception = getattr(state, "isistream", None)
        wire_pose = None
        # Cam-view object boxes. Points mode (the deployed default): draw
        # isistream's zone-scoped observations DIRECTLY off the bus
        # (WireObjectSource) — boxes appear wherever detection fires, with NO
        # dependency on the pixel-space zone_patches that gate the ZONE PANELS.
        # (An empty zone_patches.yaml otherwise leaves the patch-scoped worker
        # idle, so a perfectly good detection never reaches the cam view.)
        # Frames mode keeps the patch-scoped worker snapshot.
        get_zone_dets = None
        zone_ctx = None
        if perception is not None and perception.points_mode():
            from ..pose_overlay import WireObjectSource, WirePoseSource
            bus = getattr(state, "bus", None)
            wire_pose = WirePoseSource(lambda: bus, camera_id)
            get_zone_dets = WireObjectSource(lambda: bus, camera_id).objects
            # Zone-based cam view: metric membership + outline + mask stencil
            # (isistream detects in the zone's larger bounding-box crop, so
            # its observations spill past the polygon — see _detect_iter).
            zone_ctx = _camera_zone_ctx(cfg, camera_id)
        elif manager is not None:
            def get_zone_dets(_img, _m=manager, _c=camera_id):
                return _m.camera_dets(_c)
        # 3D axis + height badge from the bus's Track3D fixes (Mode 2 only in
        # practice — the overlay self-disables on Mode-1 placeholder
        # extrinsics). Built once per stream build so rvec/tvec are cached.
        from ..track3d_overlay import CamAxisOverlay
        axis_bus = getattr(state, "bus", None)

        def _axis_zones(_c=cfg):
            # (path, mtime)-cached — the badge anchors to a raised zone's
            # declared plane (Zone.z_base_m) instead of under the floor.
            try:
                zpath = _zones_yaml_path(_c)
                return _load_zones_cached(str(zpath), zpath.stat().st_mtime_ns)
            except Exception:
                return None

        axis_overlay = CamAxisOverlay(_warp_camera(cfg, camera_id),
                                      lambda: axis_bus, zones_getter=_axis_zones)
        frames = _detect_iter(
            frames, cfg, camera_id,
            is_running=is_running,
            get_zone_dets=get_zone_dets,
            wire_pose=wire_pose,
            zone_ctx=zone_ctx,
            axis_overlay=axis_overlay,
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
