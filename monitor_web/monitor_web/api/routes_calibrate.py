"""In-dashboard calibration endpoints (S18).

Mode 1 (single-camera pallet) lands in S18.A:

* ``GET  /api/calibrate/status``           — drives the toolbar ruler button colour.
* ``POST /api/calibrate/single-cam``       — pallet 4-corner click flow.

The pallet method takes 4 pixel clicks of a pallet's corners + the pallet's
factory-spec dimensions, computes ``H`` via
:func:`calibration.calibrate_single_cam.build_single_camera_calibration` (the
same code path the CLI uses), and merges the result into ``calibration.json``.
If an existing calibration covers other cameras, those entries are preserved.

Multical (Mode 2, full 3D) endpoints land in S18.B and are out of scope here.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml
from backbone.shared.camera_rig import CameraRig
from calibration.calibrate_single_cam import (
    PointPair,
    SingleCamCalibrationError,
    build_single_camera_calibration,
)
from calibration.schema import (
    CALIBRATION_MODE_SINGLE_CAM_4PT,
    CalibrationFile,
)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

router = APIRouter()


# ---- schemas ----


class SingleCamBody(BaseModel):
    camera_id: str = Field(..., min_length=1)
    image_size: tuple[int, int] = Field(..., description="(width, height) in source-frame px")
    pallet_width_m: float = Field(..., gt=0, description="Pallet long side in metres (e.g. 1.2 for EUR)")
    pallet_height_m: float = Field(..., gt=0, description="Pallet short side in metres (e.g. 0.8 for EUR)")
    corners_uv: list[tuple[float, float]] = Field(
        ..., min_length=4, max_length=12,
        description=(
            "Source-frame pixel coords, clicked in order: the 4 pallet corners "
            "(TL → TR → BR → BL) first, then any extra floor points. 4 = exactly "
            "determined (residual gate can't fire); 5+ is overdetermined and validated."
        ),
    )
    extra_world_xy: list[tuple[float, float]] | None = Field(
        default=None,
        description=(
            "World (X, Y) metres for each EXTRA point beyond the 4 pallet corners, "
            "in the pallet frame (origin = pallet TL corner). Length must equal "
            "len(corners_uv) - 4. Required when more than 4 corners are sent."
        ),
    )
    residual_threshold_m: float = Field(default=0.10, gt=0)

    @field_validator("image_size")
    @classmethod
    def positive_image_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError("image_size must have positive width and height")
        return v

    @model_validator(mode="after")
    def extras_match_corner_count(self):
        n_extra = len(self.corners_uv) - 4
        n_world = len(self.extra_world_xy or [])
        if n_extra > 0 and n_world != n_extra:
            raise ValueError(
                f"extra_world_xy must have {n_extra} (X, Y) pairs "
                f"(one per point beyond the 4 pallet corners)"
            )
        return self


# ---- helpers ----


def _read_backbone(cfg) -> dict:
    path = Path(cfg.backbone_config_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _configured_cameras(cfg) -> list[str]:
    data = _read_backbone(cfg)
    cams = data.get("cameras", {})
    return list(cams.keys()) if isinstance(cams, dict) else []


def _current_mode(cfg) -> int:
    """Operational mode from the configured camera count: 1 camera → Mode 1
    (single-cam 4pt), 2+ → Mode 2 (Multical 3D)."""
    return 1 if len(_configured_cameras(cfg)) <= 1 else 2


def _mode_calibration_path(cfg, mode: int | None = None) -> Path:
    """The calibration file for a given mode (defaults to the current mode).

    Mode 1 and Mode 2 keep separate files under per-mode folders beside
    ``backbone.yaml`` so switching camera count in Settings re-applies the saved
    calibration for that mode without re-calibrating: ``mode1/calibration.json`` /
    ``mode2/calibration.json``.
    """
    if mode is None:
        mode = _current_mode(cfg)
    base = Path(cfg.backbone_config_path).resolve().parent
    return base / f"mode{mode}" / "calibration.json"


def _write_json_atomic(path: Path, payload: str) -> None:
    """Mirror of ``routes_config._write_yaml_atomic`` for JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _register_calibration_in_backbone_yaml(cfg, cal_path: Path) -> None:
    """Point backbone.yaml's ``calibration_path`` at ``cal_path`` (the current
    mode's calibration file) so the orchestrator loads the right one on START.

    Overwrites unconditionally: the dashboard owns this path and keeps it in sync
    with the operational mode (camera count), so a stale value from a previous
    mode must be corrected, not preserved."""
    bb_path = Path(cfg.backbone_config_path)
    if not bb_path.exists():
        return
    try:
        data = yaml.safe_load(bb_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(data, dict) or data.get("calibration_path") == str(cal_path):
        return
    data["calibration_path"] = str(cal_path)
    try:
        bb_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    except OSError:
        # non-fatal — the calibration file is written either way; operator can
        # manually wire `calibration_path:` if this fails.
        logger.warning("could not stamp calibration_path into %s", bb_path)


# ---- handlers ----


@router.get("/api/calibrate/status")
async def calibrate_status(request: Request) -> JSONResponse:
    """Report calibration state for the CURRENT mode (camera count).

    Mode 1 and Mode 2 keep separate calibration files; this reads the one for the
    current mode so the button colour + auto-warp reflect "is THIS mode usable".

    Schema:
        mode: 1 | 2                                   — operational mode (camera count)
        calibrated_cameras: ["cam_a", ...]            — entries in the current-mode file
        configured_cameras: ["cam_a", "cam_b"]        — cameras in backbone.yaml
        is_fully_calibrated: bool                     — every configured cam is calibrated
        calibration_mode: "single_cam_4pt" | "multical_full" | null
        calibration_path: str                         — current-mode file (may not exist yet)
    """
    cfg = request.app.state.settings
    configured = _configured_cameras(cfg)
    cal_path = _mode_calibration_path(cfg)
    calibrated: list[str] = []
    cal_mode: str | None = None
    if cal_path.exists():
        try:
            rig = CameraRig.from_file(cal_path)
            calibrated = list(rig.camera_ids)
            cal_mode = rig.calibration_mode
        except Exception as exc:
            logger.warning("calibrate_status: %s unreadable: %s", cal_path, exc)

    is_full = bool(configured) and set(configured) <= set(calibrated)
    return JSONResponse({
        "mode": _current_mode(cfg),
        "calibrated_cameras": calibrated,
        "configured_cameras": configured,
        "is_fully_calibrated": is_full,
        "calibration_mode": cal_mode,
        "calibration_path": str(cal_path),
    })


@router.post("/api/calibrate/single-cam")
async def calibrate_single_cam(body: SingleCamBody, request: Request) -> JSONResponse:
    """Pallet calibration (Mode 1) for a single camera.

    Operator places a pallet flat on the floor, clicks its 4 corners on the live
    camera frame, and submits {camera_id, pallet dims, corners_uv (TL→TR→BR→BL)}.
    We build 4 (pixel, world) pairs anchored at the pallet's TL corner and call
    the existing :func:`build_single_camera_calibration`. The result is merged
    into ``calibration.json`` so multi-camera setups keep their other entries.
    """
    cfg = request.app.state.settings

    # World coords from the pallet: TL=(0,0), TR=(w,0), BR=(w,h), BL=(0,h), then
    # any extra tape-measured floor points (same pallet-origin frame). 5+ points
    # overdetermine the homography so the residual gate can catch a bad pick.
    w, h = float(body.pallet_width_m), float(body.pallet_height_m)
    world_xy = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    world_xy += [(float(x), float(y)) for (x, y) in (body.extra_world_xy or [])]
    pairs = [
        PointPair(pixel_uv=(float(u), float(v)), world_xy_m=(x, y))
        for (u, v), (x, y) in zip(body.corners_uv, world_xy, strict=True)
    ]

    # Fit + sanity-gate.
    try:
        new_cal = build_single_camera_calibration(
            camera_id=body.camera_id,
            image_size_wh=(int(body.image_size[0]), int(body.image_size[1])),
            pairs=pairs,
            floor_origin_note=(
                f"Pallet corners ({w:.3f} x {h:.3f} m); origin = TL corner"
            ),
            residual_threshold_m=float(body.residual_threshold_m),
        )
    except SingleCamCalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Single-cam pallet calibration is the Mode-1 artefact — always write the
    # Mode-1 file (not whatever calibration_path currently points at), so it's
    # kept separate from a Mode-2 Multical file and re-applied only in Mode 1.
    target = _mode_calibration_path(cfg, mode=1)
    merged = new_cal
    if target.exists():
        try:
            existing = CalibrationFile.read(target)
            existing.cameras[body.camera_id] = new_cal.cameras[body.camera_id]
            # Pallet calibration always tags single_cam_4pt — if the file was
            # previously Multical-full this is a downgrade, but the pallet
            # entries supersede the Multical ones the operator just replaced.
            existing.calibration_mode = CALIBRATION_MODE_SINGLE_CAM_4PT
            existing.created_at = new_cal.created_at
            existing.floor_anchor_method = "4pt_floor"
            existing.floor_origin_note = new_cal.floor_origin_note
            merged = existing
        except Exception as exc:
            logger.warning(
                "calibrate_single_cam: existing %s unreadable (%s); overwriting",
                target, exc,
            )

    try:
        _write_json_atomic(target, merged.to_json())
    except OSError as exc:
        logger.exception("calibrate_single_cam: write failed")
        raise HTTPException(status_code=500, detail=f"write failed: {exc}") from exc

    # Keep backbone.yaml pointed at this file only while Mode 1 is the active
    # mode; in Mode 2 the calibration_path must stay on the Multical file.
    if _current_mode(cfg) == 1:
        _register_calibration_in_backbone_yaml(cfg, target)

    cam = merged.cameras[body.camera_id]
    return JSONResponse({
        "ok": True,
        "max_residual_m": float(cam.reprojection_rms_px),
        "calibration_mode": merged.calibration_mode,
        "calibration_path": str(target),
        "calibrated_cameras": list(merged.cameras.keys()),
    })


@router.post("/api/calibrate/clear")
async def calibrate_clear(request: Request) -> JSONResponse:
    """Remove the CURRENT mode's calibration (the green-button "clear" action).

    Deletes the current-mode calibration file so the CAM feeds stop auto-warping
    and the button returns to white; the other mode's saved calibration is left
    untouched. The operator can then recalibrate the current mode.
    """
    cfg = request.app.state.settings
    target = _mode_calibration_path(cfg)
    removed = False
    if target.exists():
        try:
            target.unlink()
            removed = True
        except OSError as exc:
            logger.exception("calibrate_clear: failed to remove %s", target)
            raise HTTPException(status_code=500, detail=f"clear failed: {exc}") from exc
    return JSONResponse({
        "ok": True,
        "removed": removed,
        "mode": _current_mode(cfg),
        "calibration_path": str(target),
        "calibrated_cameras": [],
    })
