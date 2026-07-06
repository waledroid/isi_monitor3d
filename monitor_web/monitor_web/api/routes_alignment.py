"""Post-calibration floor-alignment fine-tune (operator point-pair tool).

The operator clicks N corresponding FLOOR points on both cameras; the server
authors each click to world metres through the CURRENT base calibration, fits
the rigid floor correction (``calibration.refine``), and writes a DERIVED
``calibration_refined.json`` next to the base — the base solve is never
modified. A toggle points the system (dashboard rig cache + backbone.yaml via
the Mode-2 path override) at the refined or the base file.

Persistence (dashboard config section ``alignment_refinement``):
    pairs        — the raw clicked pixel pairs (survive recalibration; the
                   physical points don't move, so a REFIT after a new solve
                   just re-authors them through the new base)
    fit          — the fitted correction + residuals + the base file's mtime
                   (staleness detection: base re-solved since the fit)
    enabled      — whether the system points at the refined file
    base_path / refined_path — the two artifacts the toggle flips between.

Gates: >= 3 pairs (4 recommended); max post-fit residual 0.15 m (bad picks —
wrong correspondences or clicks on elevated objects — must not silently bend
the calibration).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import dashboard_config

logger = logging.getLogger(__name__)
router = APIRouter()

_SECTION = "alignment_refinement"
_REFINED_NAME = "calibration_refined.json"
MAX_FIT_RESIDUAL_M = 0.15
REFERENCE_CAM = "cam_a"     # fixed 2-slot rig: cam_b is corrected onto cam_a
TARGET_CAM = "cam_b"


class PointPair(BaseModel):
    cam_a: tuple[float, float]
    cam_b: tuple[float, float]


class FitBody(BaseModel):
    pairs: list[PointPair] = Field(..., min_length=3, max_length=16)
    # Frame sizes the clicks were measured on (browser streams are downscaled
    # copies of the calibrated sensor). Omitted = calibration-frame pixels.
    frame_wh: dict[str, tuple[int, int]] | None = None


class EnableBody(BaseModel):
    enabled: bool


def _store(cfg) -> dict:
    return dashboard_config.read_section(cfg, _SECTION) or {}


def _save(cfg, doc: dict) -> None:
    dashboard_config.write_section(cfg, _SECTION, doc)


def _base_calibration_path(cfg) -> Path:
    """The BASE (un-refined) calibration. When the refinement is enabled the
    current Mode-2 path points at the refined file — refitting must always go
    back to the stored base."""
    from .routes_calibrate import _mode_calibration_path
    current = _mode_calibration_path(cfg)
    doc = _store(cfg)
    refined = doc.get("refined_path")
    base = doc.get("base_path")
    if refined and base and str(current) == str(refined) and Path(base).exists():
        return Path(base)
    return current


def _load_base_rig(cfg):
    from .routes_projection import _load_rig_cached
    path = _base_calibration_path(cfg)
    if not path.exists():
        raise HTTPException(status_code=503,
                            detail="no calibration.json — solve the rig first")
    rig = _load_rig_cached(str(path.resolve()), path.stat().st_mtime_ns)
    for cid in (REFERENCE_CAM, TARGET_CAM):
        if cid not in rig:
            raise HTTPException(status_code=422,
                                detail=f"alignment fine-tune needs both cameras; "
                                       f"{cid!r} missing from the calibration")
    return path, rig


def _point_current_calibration(cfg, path: Path, zone_manager=None) -> None:
    """Flip what the system reads: dashboard rig caches key on (path, mtime);
    backbone.yaml's calibration_path applies on next START. Stored zone TWINS
    were computed under the previous geometry — regenerate them so the
    cross-camera outlines and crops actually move with the new calibration."""
    from .routes_calibrate import _register_calibration_in_backbone_yaml
    from .routes_config import _merge_ui_settings
    from .routes_zone_patches import regenerate_twins
    _merge_ui_settings(cfg, {"mode2_calibration_path": str(path)})
    _register_calibration_in_backbone_yaml(cfg, path)
    regenerate_twins(cfg, zone_manager)


@router.get("/api/alignment")
async def get_alignment(request: Request) -> dict:
    """Current fine-tune state: stored pairs, fit, toggle, staleness."""
    cfg = request.app.state.settings
    doc = _store(cfg)
    stale = False
    fit = doc.get("fit")
    base = doc.get("base_path")
    if fit and base and Path(base).exists():
        stale = Path(base).stat().st_mtime_ns != fit.get("base_mtime_ns")
    return {
        "pairs": doc.get("pairs", []),
        "fit": fit,
        "enabled": bool(doc.get("enabled")),
        "stale": stale,
        "reference": REFERENCE_CAM,
        "target": TARGET_CAM,
        "base_path": base,
        "refined_path": doc.get("refined_path"),
    }


def _fit_and_write(cfg, pairs: list[dict], frame_wh: dict | None = None,
                   zone_manager=None) -> dict:
    """Author pairs → fit → gate → write the refined artifact → persist."""
    import numpy as np
    from backbone.shared.geometry import pixel_to_floor, undistort_points
    from calibration.refine import apply_floor_alignment, fit_rigid_floor_alignment
    from calibration.schema import CalibrationFile

    base_path, rig = _load_base_rig(cfg)
    va, vb = rig[REFERENCE_CAM], rig[TARGET_CAM]

    pa = np.asarray([p["cam_a"] for p in pairs], dtype=np.float64)
    pb = np.asarray([p["cam_b"] for p in pairs], dtype=np.float64)
    # Scale browser-stream clicks into each camera's calibration frame; the
    # STORED pairs are always calibration-frame (refit is frame-independent).
    if frame_wh:
        for cid, pts, view in ((REFERENCE_CAM, pa, va), (TARGET_CAM, pb, vb)):
            fw_fh = frame_wh.get(cid)
            if fw_fh:
                cw, ch = view.image_size_wh
                fw, fh = fw_fh
                if fw and fh and (int(fw), int(fh)) != (int(cw), int(ch)):
                    pts *= [cw / float(fw), ch / float(fh)]
        pairs = [{"cam_a": [float(u), float(v)], "cam_b": [float(x), float(y)]}
                 for (u, v), (x, y) in zip(pa, pb, strict=True)]
    xa = pixel_to_floor(undistort_points(pa, va.K, va.D), va.H)
    xb = pixel_to_floor(undistort_points(pb, vb.K, vb.D), vb.H)
    before = [float(d) for d in np.linalg.norm(xa - xb, axis=1)]

    fit = fit_rigid_floor_alignment(xb, xa)   # correct cam_b ONTO cam_a
    if fit.max_residual_m > MAX_FIT_RESIDUAL_M:
        raise HTTPException(
            status_code=422,
            detail=(f"fit residual {fit.max_residual_m*100:.0f} cm exceeds "
                    f"{MAX_FIT_RESIDUAL_M*100:.0f} cm — the picked points don't "
                    f"correspond (check they are the SAME physical spots, ON the "
                    f"floor, not on boxes). Nothing was changed."),
        )

    base = CalibrationFile.read(base_path)
    refined = apply_floor_alignment(base, TARGET_CAM, fit)
    refined_path = base_path.parent / _REFINED_NAME
    refined.write(refined_path)

    doc = _store(cfg)
    doc.update({
        "pairs": pairs,
        "fit": {**fit.as_dict(), "base_mtime_ns": base_path.stat().st_mtime_ns},
        "base_path": str(base_path),
        "refined_path": str(refined_path),
        "enabled": bool(doc.get("enabled")),
    })
    _save(cfg, doc)
    # If already enabled, the refined file just changed in place — re-point so
    # backbone.yaml is stamped and the rig cache (keyed on mtime) refreshes.
    if doc["enabled"]:
        _point_current_calibration(cfg, refined_path, zone_manager)
    logger.info("alignment: fitted %s (before mean %.3f m -> max residual %.3f m)",
                fit.as_dict(), float(np.mean(before)), fit.max_residual_m)
    return {
        "ok": True,
        "fit": doc["fit"],
        "before_error_m": [round(b, 4) for b in before],
        "refined_path": str(refined_path),
        "enabled": doc["enabled"],
    }


@router.post("/api/alignment/fit")
def fit_alignment(body: FitBody, request: Request) -> dict:
    """Fit from freshly picked pairs and write the refined calibration."""
    cfg = request.app.state.settings
    return _fit_and_write(cfg, [p.model_dump() for p in body.pairs],
                          frame_wh=body.frame_wh,
                          zone_manager=getattr(request.app.state, "zone_manager", None))


@router.post("/api/alignment/refit")
def refit_alignment(request: Request) -> dict:
    """Re-fit the STORED pairs against the current base calibration — the
    'remake after a new solve' path (physical points didn't move; the residual
    gate catches the case where the cameras did)."""
    cfg = request.app.state.settings
    pairs = _store(cfg).get("pairs") or []
    if len(pairs) < 3:
        raise HTTPException(status_code=422, detail="no stored pairs — pick points first")
    return _fit_and_write(cfg, pairs,
                          zone_manager=getattr(request.app.state, "zone_manager", None))


@router.post("/api/alignment/enable")
def enable_alignment(body: EnableBody, request: Request) -> dict:
    """Toggle: point the system at the refined (on) or base (off) calibration."""
    cfg = request.app.state.settings
    doc = _store(cfg)
    refined = doc.get("refined_path")
    base = doc.get("base_path")
    if body.enabled and not (refined and Path(refined).exists()):
        raise HTTPException(status_code=422, detail="no refined calibration — fit first")
    if not body.enabled and not (base and Path(base).exists()):
        raise HTTPException(status_code=422, detail="base calibration missing")
    _point_current_calibration(cfg, Path(refined if body.enabled else base),
                               getattr(request.app.state, "zone_manager", None))
    doc["enabled"] = body.enabled
    _save(cfg, doc)
    return {"ok": True, "enabled": body.enabled,
            "calibration_path": refined if body.enabled else base}


@router.delete("/api/alignment")
def clear_alignment(request: Request) -> dict:
    """Forget the fine-tune: repoint the base calibration and clear the store.
    The refined artifact file is left on disk (harmless; regenerated on refit)."""
    cfg = request.app.state.settings
    doc = _store(cfg)
    base = doc.get("base_path")
    if doc.get("enabled") and base and Path(base).exists():
        _point_current_calibration(cfg, Path(base),
                                   getattr(request.app.state, "zone_manager", None))
    _save(cfg, {})
    return {"ok": True}
