"""Zone-patch ROIs — pixel-space "watch boxes" drawn directly on a camera frame.

Each ROI is a display crop of the live feed for the ZONE panels; its contents come
from the Backbone's per-camera observations (one perception — the dashboard runs no
detector). The zone worker groups those wire detections into these polygons for the
COMMUNICATION cards. Stored in ``zone_patches.yaml`` next to ``backbone.yaml``;
monitor_web-only (the Backbone never reads these — they're a UI monitoring aid, not
metric floor zones). Rects are in SOURCE-frame pixels, with the frame size they were
drawn at so the crop stays correct if the live resolution differs.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from .. import dashboard_config

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_PATCHES = 6   # max zones the operator can create (mirrors the dashboard cap)

# Auto-derived cross-camera TWIN patches (occlusion persistence): a zone drawn
# on one camera gets a server-generated twin on the other camera covering the
# SAME floor region, so both workers detect it and an occluded camera never
# blinds the zone. Twins are regenerated on every save — never operator-edited.
TWIN_SUFFIX = "__twin"


def is_twin(patch: dict) -> bool:
    return bool(patch.get("twin_of"))


def load_patches(cfg) -> list[dict]:
    """Return the stored ROI list (``[]`` when none configured). Reads the merged
    ``zone_patches`` section of the unified dashboard config."""
    patches = dashboard_config.read_section(cfg, "zone_patches").get("patches", [])
    return patches if isinstance(patches, list) else []


def find_patch(cfg, patch_id: str) -> dict | None:
    """Look up one ROI by id."""
    return next((p for p in load_patches(cfg) if str(p.get("id")) == str(patch_id)), None)


def patch_rect(patch: dict) -> list[float] | None:
    """Axis-aligned bounding rect ``[x0,y0,x1,y1]`` (source px) for a patch — derived
    from its polygon if present (the drawn shape), else the stored rect. The polygon
    is the display boundary; the crop fed to detection is this bounding rectangle."""
    poly = patch.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    return patch.get("rect")


def patch_pixel_box(rect, stored_wh, frame_wh):
    """Map a stored ROI ``rect`` (source px, drawn at ``stored_wh``) to an integer
    pixel box ``(x0, y0, x1, y1)`` on a frame of ``frame_wh`` — scaling when the live
    frame size differs (the frame-size guard). Clamped to the frame; ``None`` if the
    box is degenerate (< 2 px on a side)."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    if stored_wh and (int(stored_wh[0]), int(stored_wh[1])) != (fw, fh):
        sw, sh = float(stored_wh[0]) or fw, float(stored_wh[1]) or fh
        sx, sy = fw / sw, fh / sh
        x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
    x0i, x1i = max(0, min(fw, round(x0))), max(0, min(fw, round(x1)))
    y0i, y1i = max(0, min(fh, round(y0))), max(0, min(fh, round(y1)))
    if x1i - x0i < 2 or y1i - y0i < 2:
        return None
    return (x0i, y0i, x1i, y1i)


class PatchRect(BaseModel):
    id: str
    name: str = ""
    camera: str = "cam_a"
    # The drawn shape, in SOURCE-frame pixels. A polygon (>=3 pts) is the display
    # boundary (red overlay on the cam); `rect` is its axis-aligned bounding box and
    # is what gets cropped + fed to detection. Either may be supplied — rect is
    # auto-derived from the polygon when omitted (and a bare rect stays valid).
    rect: list[float] | None = Field(default=None, min_length=4, max_length=4)
    polygon: list[list[float]] | None = None                     # [[u,v], ...] source px
    frame_wh: list[int] | None = None                            # [W,H] drawn at (guard)
    color: str | None = None    # outline colour on the cam overlay (hex); None = red
    # Per-zone DISPLAY confidence floor on the Backbone's wire detections (one
    # perception — the dashboard never infers). None = a sane default floor.
    confidence: float | None = None
    # Set on server-derived cross-camera twins (the base patch's id). Twins in
    # a POST are dropped and regenerated — never operator-authoritative.
    twin_of: str | None = None

    # Tolerate legacy per-zone inference keys (model/infer_size/sahi/enhance/
    # max_fps) still sitting in older saved YAML — they are ignored now that the
    # dashboard runs no local inference; a resave drops them harmlessly.
    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _derive_and_clamp(self) -> PatchRect:
        if (not self.rect or len(self.rect) != 4) and self.polygon and len(self.polygon) >= 3:
            xs = [float(p[0]) for p in self.polygon]
            ys = [float(p[1]) for p in self.polygon]
            self.rect = [min(xs), min(ys), max(xs), max(ys)]
        if not self.rect or len(self.rect) != 4:
            raise ValueError("zone patch needs a rect or a polygon of >=3 points")
        return self


class PatchesBody(BaseModel):
    # 2x headroom so a client that round-trips the derived twins isn't
    # rejected; the handler enforces MAX_PATCHES on OPERATOR patches only.
    patches: list[PatchRect] = Field(default_factory=list, max_length=MAX_PATCHES * 2)


def _patch_ghost(patch: dict, rig) -> dict | None:
    """Project a patch's polygon into the OTHER camera's pixels (Mode-2 ghost).

    Round-trips through the floor: own-camera pixels → undistort → ``H_own`` →
    world metres → ``H_other``⁻¹ → other-camera pixels. Points are scaled from
    the patch's stored ``frame_wh`` to the calibration frame first. Returns
    ``{"camera", "polygon", "image_wh"}`` or ``None`` when the patch camera
    isn't calibrated, there is no second camera, or the projection lands
    entirely outside the other view (no overlap).
    """
    import numpy as np
    from backbone.shared.geometry import (
        densify_polygon,
        floor_to_pixel,
        has_metric_camera_model,
        pixel_to_floor,
        project_floor_polygon_distorted,
        undistort_points_checked,
    )

    own_id = str(patch.get("camera", ""))
    cam_ids = list(rig.camera_ids)
    if own_id not in cam_ids or len(cam_ids) < 2:
        return None
    other_id = next(c for c in cam_ids if c != own_id)
    own, other = rig[own_id], rig[other_id]

    poly = patch.get("polygon")
    if not (isinstance(poly, list) and len(poly) >= 3):
        rect = patch_rect(patch)
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    # A straight edge in one camera is CURVED after the floor round-trip
    # (homography + lens distortion) — sample the edges, don't just map corners.
    pts = densify_polygon(np.asarray(poly, dtype=np.float64), segments_per_edge=8)

    # Stored pixels → the own camera's calibration frame.
    stored_wh = patch.get("frame_wh") or list(own.image_size_wh)
    ow, oh = own.image_size_wh
    pts[:, 0] *= ow / float(stored_wh[0] or ow)
    pts[:, 1] *= oh / float(stored_wh[1] or oh)

    try:
        # Border pixels of a strong barrel lens can DIVERGE under undistortion
        # (x=75 → x=-1412), authoring absurd floor points that spike the
        # projected outline — keep only the samples whose undistortion
        # round-trips (a zone hugging the frame edge keeps its valid part).
        und, valid = undistort_points_checked(pts, own.K, own.D)
        if valid.sum() < 6:
            return None
        world = pixel_to_floor(und[valid], own.H)
        # Distortion-aware: the ghost is drawn over the RAW (distorted) live
        # frame — pinhole coords would drift 100+ px near the edges of a
        # strong barrel lens. H-only fallback for placeholder extrinsics.
        if has_metric_camera_model(other.K, other.R, other.t):
            # The FORWARD polynomial of the same lens folds beyond a critical
            # radius (cam_b folds at normalized r=1.11, its corners sit at
            # 1.03) — a zone spilling past the other camera's field would get
            # samples projected back INSIDE the frame at folded positions,
            # warping the twin. Clip the floor polygon to the reliably-
            # projectable field first (the visible overlap is bounded by the
            # field rim, not just the zone's own boundary).
            ghost = project_floor_polygon_distorted(
                world, other.K, other.D, other.R, other.t, other.image_size_wh)
            if ghost is None or len(ghost) < 3:
                return None
        else:
            ghost = floor_to_pixel(world, other.H)
    except Exception:
        return None
    gw, gh = other.image_size_wh
    inside = ((ghost[:, 0] >= 0) & (ghost[:, 0] < gw)
              & (ghost[:, 1] >= 0) & (ghost[:, 1] < gh))
    if not bool(inside.any()):
        return None   # no overlap with the other view
    return {
        "camera": other_id,
        "polygon": [[float(u), float(v)] for u, v in ghost],
        "image_wh": [int(gw), int(gh)],
    }


def _load_rig(cfg):
    """The current-mode calibration rig, or ``None`` (best-effort)."""
    try:
        from .routes_calibrate import _mode_calibration_path
        from .routes_projection import _load_rig_cached
        cal_path = _mode_calibration_path(cfg)
        if cal_path.exists():
            return _load_rig_cached(str(cal_path.resolve()), cal_path.stat().st_mtime_ns)
    except Exception:
        pass
    return None


def _make_twin(patch: dict, rig) -> dict | None:
    """Derive the cross-camera twin of a patch: the ghost polygon (same floor
    region, other camera), clipped to that camera's frame, carrying the same
    name/model/confidence so both workers detect the same physical zone."""
    ghost = _patch_ghost(patch, rig)
    if ghost is None:
        return None
    import numpy as np
    gw, gh = ghost["image_wh"]
    poly = np.asarray(ghost["polygon"], dtype=np.float64)
    poly[:, 0] = np.clip(poly[:, 0], 0.0, gw - 1.0)
    poly[:, 1] = np.clip(poly[:, 1], 0.0, gh - 1.0)
    # Degenerate after clipping (zone barely overlaps the other view) → no twin.
    if (poly.max(axis=0) - poly.min(axis=0)).min() < 8.0:
        return None
    return {
        "id": f"{patch['id']}{TWIN_SUFFIX}",
        "name": patch.get("name", ""),
        "camera": ghost["camera"],
        "polygon": [[float(u), float(v)] for u, v in poly],
        "frame_wh": [int(gw), int(gh)],
        "color": patch.get("color"),
        "confidence": patch.get("confidence"),
        "twin_of": patch["id"],
    }


def regenerate_twins(cfg, zone_manager=None) -> int:
    """Recompute every zone's cross-camera twin from the CURRENT calibration.

    Twin polygons are stored at save time — they do NOT track calibration
    changes by themselves. Call this whenever the active calibration switches
    (alignment fine-tune toggled/refit, a different calibration.json selected)
    so the outlines and the twin detection crops move with the new geometry.
    Returns the number of twins written.
    """
    stored = load_patches(cfg)
    user_patches = [dict(p) for p in stored if not is_twin(p)]
    out = list(user_patches)
    n_twins = 0
    rig = _load_rig(cfg)
    if rig is not None and len(list(rig.camera_ids)) >= 2:
        for p in user_patches:
            try:
                twin = _make_twin(p, rig)
            except Exception:
                twin = None
            if twin is not None:
                out.append(twin)
                n_twins += 1
    dashboard_config.write_section(cfg, "zone_patches", {"patches": out})
    try:
        from ..floor_zone_sync import sync_floor_zones_from_patches
        sync_floor_zones_from_patches(cfg, patches=out, rig=rig)
    except Exception:
        logger.warning("regenerate_twins: floor-zone sync failed", exc_info=True)
    if zone_manager is not None:
        try:
            zone_manager.reload()
        except Exception:
            logger.warning("regenerate_twins: worker reload failed", exc_info=True)
    logger.info("zone-patches: regenerated %d twin(s) for the current calibration",
                n_twins)
    return n_twins


@router.get("/api/zone-patches")
# Sync def on purpose: ghost projection runs cv2/numpy per patch —
# threadpool, not the event loop (same rule as routes_config readers).
def get_zone_patches(request: Request) -> dict:
    """The stored patches (operator zones + their auto-derived cross-camera
    twins). Patches WITHOUT a twin are enriched with a ``ghost`` — the outline
    projected into the other camera; twinned zones don't need one (the twin IS
    the real, detecting outline over there)."""
    cfg = request.app.state.settings
    patches = [dict(p) for p in load_patches(cfg)]
    rig = _load_rig(cfg)
    if rig is not None:
        twinned = {p.get("twin_of") for p in patches if is_twin(p)}
        for p in patches:
            if is_twin(p) or p.get("id") in twinned:
                continue
            try:
                p["ghost"] = _patch_ghost(p, rig)
            except Exception:
                p["ghost"] = None
    return {"patches": patches}


@router.get("/api/zone-patches/state")
def zone_patches_state(request: Request) -> dict:
    """Live per-patch contents from the background zone workers.

    Feeds the COMMUNICATION panel's zone cards for the operator's camera zones
    (zone patches) — works in Mode 1 AND Mode 2, no calibration or broker
    needed (the workers detect locally). The per-patch payload mirrors the
    world-zone ``zone_state`` shape (``objects`` with cls/confidence and a
    pallet ``occupancy_state``) so the cards render both zone systems
    identically. A patch with no fresh worker coverage (Backbone stopped,
    worker error) is simply absent → its card stays dim.
    """
    from ..detection_overlay import _PALLET_CLASSES, image_occupancy

    cfg = request.app.state.settings
    mgr = getattr(request.app.state, "zone_manager", None)

    def _objects_for(dets) -> list[dict]:
        occ_by_det = {id(pal): label for pal, label in image_occupancy(dets)}
        objects = []
        for d in dets:
            cls_l = str(getattr(d, "cls", "")).lower()
            entry: dict = {
                "cls": "palette" if cls_l in _PALLET_CLASSES else cls_l,
                "confidence": float(getattr(d, "confidence", 0.0)),
            }
            # Backbone-sourced observations already CARRY the occupancy verdict
            # (the same one the tracker/MQTT uses) — prefer it, so the cards
            # and the wire literally cannot disagree. Locally-detected dets
            # fall through to the image-side classification as before.
            carried = getattr(d, "occupancy_state", None)
            if carried in ("empty", "full"):
                entry["occupancy_state"] = carried
                content = getattr(d, "occupancy_content", None)
                if carried == "full" and content:
                    entry["occupancy_content"] = [str(content)]
                objects.append(entry)
                continue
            label = occ_by_det.get(id(d))
            if label is not None:
                entry["occupancy_state"] = "empty" if label == "palette_vide" else "full"
                if label != "palette_vide" and label.startswith("palette_"):
                    # 'palette_carton_polybag' → ["carton", "polybag"]: what the
                    # pallet carries, for the human-readable zone cards.
                    entry["occupancy_content"] = label[len("palette_"):].split("_")
            objects.append(entry)
        return objects

    def _resolve_pallet_conflicts(objects: list[dict]) -> list[dict]:
        """Same WMS/FMS rule as the Backbone's ``ZoneStateTracker`` (MQTT
        payloads): a double-detected pallet with conflicting readings must not
        surface both — when the zone holds a LOADED pallet, concurrent 'empty'
        pallet readings are dropped. Unknown occupancy is kept."""
        has_loaded = any(
            o["cls"] == "palette" and o.get("occupancy_state") == "full"
            for o in objects
        )
        if not has_loaded:
            return objects
        return [
            o for o in objects
            if not (o["cls"] == "palette" and o.get("occupancy_state") == "empty")
        ]

    def _merge(base: list[dict], twin: list[dict]) -> list[dict]:
        """Occlusion-proof union of the two cameras' views of the SAME zone:
        per class, keep whichever camera saw MORE (both seeing one object must
        not double-count; one camera occluded → the other's list carries)."""
        by_cls_a: dict[str, list[dict]] = {}
        by_cls_b: dict[str, list[dict]] = {}
        for o in base:
            by_cls_a.setdefault(o["cls"], []).append(o)
        for o in twin:
            by_cls_b.setdefault(o["cls"], []).append(o)
        merged: list[dict] = []
        for cls in sorted(set(by_cls_a) | set(by_cls_b)):
            a, b = by_cls_a.get(cls, []), by_cls_b.get(cls, [])
            merged.extend(a if len(a) >= len(b) else b)
        return merged

    states: dict[str, dict] = {}
    if mgr is not None:
        patches = load_patches(cfg)
        for p in patches:
            if is_twin(p):
                continue    # folded into the base zone below
            pid = str(p.get("id"))
            twin_id = f"{pid}{TWIN_SUFFIX}"
            status = mgr.zone_status(pid)
            twin_status = mgr.zone_status(twin_id)
            if not status and not twin_status:   # neither camera covers it
                continue
            objects = _resolve_pallet_conflicts(_merge(
                _objects_for(mgr.zone_dets(pid)) if status else [],
                _objects_for(mgr.zone_dets(twin_id)) if twin_status else [],
            ))
            states[pid] = {"objects": objects, "count": len(objects),
                           "status": status or twin_status}
    return {"states": states}


@router.post("/api/zone-patches")
def post_zone_patches(body: PatchesBody, request: Request) -> dict:
    # Sync handler on purpose — write_section fsyncs and mgr.reload() can join
    # worker threads; in the threadpool neither stalls the event loop.
    cfg = request.app.state.settings
    # Twins are DERIVED — drop any that round-tripped from the client, then
    # regenerate them fresh from the operator patches (calibration permitting),
    # so both cameras detect every zone and occlusion of one camera never
    # blinds it.
    user_patches = [p.model_dump() for p in body.patches if not p.twin_of]
    if len(user_patches) > MAX_PATCHES:
        from fastapi import HTTPException
        raise HTTPException(status_code=422,
                            detail=f"max {MAX_PATCHES} zones (excluding twins)")
    stored = list(user_patches)
    rig = _load_rig(cfg)
    if rig is not None and len(list(rig.camera_ids)) >= 2:
        for p in user_patches:
            try:
                twin = _make_twin(p, rig)
            except Exception:
                twin = None
            if twin is not None:
                stored.append(twin)
    doc = {"patches": stored}
    dashboard_config.write_section(cfg, "zone_patches", doc)
    logger.info("zone-patches: saved %d ROI(s) (+%d twins)",
                len(user_patches), len(stored) - len(user_patches))
    # Derive the Backbone's FLOOR zones from the same drawings (one drawing →
    # cards AND zone_state/proximity MQTT). Best-effort; applies at next START.
    try:
        from ..floor_zone_sync import sync_floor_zones_from_patches
        sync_floor_zones_from_patches(cfg, patches=stored, rig=rig)
    except Exception:
        logger.warning("zone-patches: floor-zone sync failed", exc_info=True)
    # Sync the background detection workers to the new zone set (start/update/stop
    # per camera). getattr guard: apps built without the manager stay safe.
    mgr = getattr(request.app.state, "zone_manager", None)
    if mgr is not None:
        mgr.reload()
    return {"ok": True, "count": len(body.patches)}
