"""Derive the Backbone's FLOOR zones from the operator's camera zone patches.

The operator draws zones ONCE, on the camera image (Settings ▸ zone patches —
"Zone 1", "Zone 2", …). Those pixel polygons drive the dashboard's zone
workers and COMMUNICATION cards directly; the Backbone's zone engine
(zone-scoped detection, retained ``zone_state`` MQTT, proximity) speaks floor
METRES. This module closes the gap: every patch save projects each non-twin
patch polygon through the current calibration (undistort → H) to floor
coordinates and rewrites the derived entries of ``zones.yaml`` — same names as
the cards, so the panel and the MQTT bus tell one story.

Rules:
- Twins are skipped (a twin is the same physical zone seen by the other
  camera — one floor zone, not two).
- Derived entries carry ``derived_from: zone_patch`` (ignored by the Backbone
  loader); entries WITHOUT the marker (hand-authored / map-drawn) are
  preserved untouched.
- Best-effort by design: no calibration, an uncalibrated camera, or a bad
  polygon skips that patch with a log line — a zone save must never fail
  because projection couldn't run. The Backbone reads ``zones.yaml`` at START,
  so changes apply on the next START.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_MARKER = "zone_patch"


def _zones_yaml_path(cfg) -> Path:
    """``zones_path`` from backbone.yaml, else ``zones.yaml`` beside it."""
    from .model_store import read_backbone

    raw = (read_backbone(cfg) or {}).get("zones_path")
    if raw:
        return Path(raw)
    return Path(cfg.backbone_config_path).resolve().parent / "zones.yaml"


def _patch_pixel_polygon(patch: dict) -> list[list[float]] | None:
    """The drawn boundary in source pixels: polygon if present, else the rect
    corners. ``None`` when the patch carries neither (nothing to project)."""
    poly = patch.get("polygon")
    if poly and len(poly) >= 3:
        return [[float(u), float(v)] for u, v in poly]
    rect = patch.get("rect")
    if rect and len(rect) == 4:
        x0, y0, x1, y1 = (float(v) for v in rect)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return None


def _project_patch_to_floor(patch: dict, view) -> list[list[float]] | None:
    """Patch pixels (at their stored ``frame_wh``) → floor metres, or None."""
    from backbone.shared.geometry import pixel_to_floor, undistort_points

    poly = _patch_pixel_polygon(patch)
    if poly is None:
        return None
    pts = np.asarray(poly, dtype=np.float64)
    cw, ch = view.image_size_wh
    stored = patch.get("frame_wh")
    if stored and len(stored) == 2 and stored[0] and stored[1]:
        fw, fh = float(stored[0]), float(stored[1])
        if (int(fw), int(fh)) != (int(cw), int(ch)):
            pts = pts * [cw / fw, ch / fh]
    world = pixel_to_floor(undistort_points(pts, view.K, view.D), view.H)
    if not np.isfinite(world).all():
        return None
    return [[round(float(x), 3), round(float(y), 3)] for x, y in world]


def sync_floor_zones_from_patches(cfg, patches: list[dict] | None = None,
                                  rig=None) -> int:
    """Rewrite ``zones.yaml``'s derived entries from the current zone patches.

    Returns the number of floor zones written (0 when nothing was projectable
    — the file's derived section is then cleared so deleted patches disappear,
    unless no calibration was available at all, in which case the file is left
    untouched).
    """
    from . import dashboard_config

    if patches is None:
        doc = dashboard_config.read_section(cfg, "zone_patches") or {}
        patches = list(doc.get("patches") or [])
    user_patches = [p for p in patches if not p.get("twin_of")]

    if rig is None:
        from .api.routes_zone_patches import _load_rig
        rig = _load_rig(cfg)
    if rig is None:
        logger.info("floor_zone_sync: no calibration — zones.yaml left untouched")
        return 0

    path = _zones_yaml_path(cfg)
    try:
        existing = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    except (OSError, yaml.YAMLError):
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing_zones = [z for z in (existing.get("zones") or []) if isinstance(z, dict)]
    # kind/type/severity live on the FLOOR zone (the patch carries none) —
    # remember them per id so a rename/redraw never resets a danger zone.
    prev_by_id = {str(z.get("id") or "").strip(): z
                  for z in existing_zones if z.get("id")}

    derived: list[dict] = []
    used_names: set[str] = set()
    for p in user_patches:
        cam = p.get("camera")
        if cam not in rig:
            logger.info("floor_zone_sync: %r skipped (camera %r not calibrated)",
                        p.get("name") or p.get("id"), cam)
            continue
        try:
            floor = _project_patch_to_floor(p, rig[cam])
        except Exception:
            logger.warning("floor_zone_sync: projection failed for %r",
                           p.get("name") or p.get("id"), exc_info=True)
            floor = None
        if floor is None or len(floor) < 3:
            continue
        name = str(p.get("name") or p.get("id") or "zone").strip() or "zone"
        if name in used_names:                       # ZoneRegistry rejects dupes
            name = f"{name} ({p.get('id', '')})"
        used_names.add(name)
        # STABLE identity: reuse the patch id (``zp_…``) so the pixel patch and
        # this floor zone share ONE immutable id — never positional, never
        # reused after a delete. The label (``name``) is free to change; the id
        # is what AGVs/WMS/MQTT key on. Legacy patches without an id fall back to
        # the loader's name-slug (still deterministic).
        pid = str(p.get("id") or "").strip()
        prev = prev_by_id.get(pid, {})
        entry: dict = {}
        if pid:
            entry["id"] = pid
        entry.update({
            "name": name,
            "type": str(prev.get("type") or "palette"),
            "kind": str(prev.get("kind") or "palette"),
            "severity": str(prev.get("severity") or "info"),
            "polygon": floor,
            "derived_from": _MARKER,                 # ignored by the loader
        })
        derived.append(entry)

    # ---- reconcile: the patch and its floor zone are ONE object -------------
    # Ownership is by ID, not by the derived_from marker (legacy writers left
    # entries unmarked, which used to strand them forever: a rename then added
    # a second entry beside the stale one, and a delete left an orphan).
    #   - an entry whose id was re-projected this pass  → replaced above
    #   - a patch-born entry (zp_ id) whose patch is gone → deleted with it
    #   - a patch-born entry whose patch LIVES but failed projection
    #     (uncalibrated camera, bad polygon) → kept until projection succeeds
    #   - anything else (hand-authored, non-zp id)       → preserved untouched
    current_ids = {str(p.get("id") or "").strip()
                   for p in user_patches if p.get("id")}
    projected_ids = {e["id"] for e in derived if "id" in e}

    def _keep(z: dict) -> bool:
        zid = str(z.get("id") or "").strip()
        if zid in projected_ids:
            return False
        if zid.startswith("zp_"):
            return zid in current_ids
        return z.get("derived_from") != _MARKER

    manual = [z for z in existing_zones if _keep(z)]

    payload = {"zones": manual + derived}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)
    except OSError:
        logger.warning("floor_zone_sync: could not write %s", path, exc_info=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return 0
    logger.info("floor_zone_sync: %d floor zone(s) derived from patches → %s "
                "(+%d manual kept; applies at next backbone START)",
                len(derived), path.name, len(manual))
    return len(derived)
