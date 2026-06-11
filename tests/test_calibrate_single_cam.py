"""``calibrate_single_cam`` — Mode 1 4-point floor calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import pixel_to_floor
from calibration.calibrate_single_cam import (
    PointPair,
    SingleCamCalibrationError,
    _parse_pair_arg,
    build_single_camera_calibration,
    fit_single_camera_homography,
)
from calibration.schema import (
    CALIBRATION_MODE_SINGLE_CAM_4PT,
    CalibrationFile,
)


def _good_pairs() -> list[PointPair]:
    """Four floor corners of a 4 m x 2 m rectangle, projected through a known H.

    We pretend a top-down camera produces a simple scale-and-shift mapping:
        world_x = (pixel_u - 500) * 0.01    (1 cm per pixel)
        world_y = (pixel_v - 500) * 0.01
    Pick pixel corners, compute world from this rule, feed back to the fitter.
    """
    pixel_corners = [(100, 100), (900, 100), (900, 700), (100, 700)]
    return [
        PointPair(
            pixel_uv=(float(u), float(v)),
            world_xy_m=((u - 500) * 0.01, (v - 500) * 0.01),
        )
        for u, v in pixel_corners
    ]


# ---- fit math ----


def test_fit_round_trip_zero_residual() -> None:
    H, max_residual = fit_single_camera_homography(_good_pairs())
    assert max_residual < 1e-6
    # Project each input pixel through H, recover the world coord.
    pairs = _good_pairs()
    pixels = np.asarray([p.pixel_uv for p in pairs])
    world = pixel_to_floor(pixels, H)
    for got, exp in zip(world, [p.world_xy_m for p in pairs], strict=True):
        np.testing.assert_allclose(got, exp, atol=1e-9)


def test_fit_rejects_too_few_pairs() -> None:
    pairs = _good_pairs()[:3]
    with pytest.raises(SingleCamCalibrationError, match="at least 4"):
        fit_single_camera_homography(pairs)


def test_fit_rejects_collinear_points() -> None:
    """4 colinear world points → cv2.findHomography returns None."""
    pairs = [
        PointPair(pixel_uv=(100.0, 100.0), world_xy_m=(0.0, 0.0)),
        PointPair(pixel_uv=(200.0, 200.0), world_xy_m=(1.0, 0.0)),
        PointPair(pixel_uv=(300.0, 300.0), world_xy_m=(2.0, 0.0)),
        PointPair(pixel_uv=(400.0, 400.0), world_xy_m=(3.0, 0.0)),
    ]
    with pytest.raises(SingleCamCalibrationError, match=r"collinear|None"):
        fit_single_camera_homography(pairs)


def _good_pairs_overdetermined() -> list[PointPair]:
    """Same world↔pixel mapping as ``_good_pairs`` but with 5 points so the
    fit is overdetermined and residuals are observable."""
    pixel_corners = [(100, 100), (900, 100), (900, 700), (100, 700), (500, 400)]
    return [
        PointPair(
            pixel_uv=(float(u), float(v)),
            world_xy_m=((u - 500) * 0.01, (v - 500) * 0.01),
        )
        for u, v in pixel_corners
    ]


def test_fit_sanity_gate_rejects_inconsistent_pairs() -> None:
    """With ≥5 points (overdetermined fit), a mis-typed world coord produces
    measurable residuals and the sanity gate fires.

    Note: with EXACTLY 4 points, the fit is exactly-determined and residuals
    are always ~0 — the operator gets no protection there. This is a
    documented limitation; use 5+ points if you want the sanity gate to bite.
    """
    pairs = _good_pairs_overdetermined()
    # Inject a 1 m error on the 5th point's world coord.
    pairs[4] = PointPair(pixel_uv=pairs[4].pixel_uv, world_xy_m=(5.0, 5.0))
    with pytest.raises(SingleCamCalibrationError, match="residual"):
        fit_single_camera_homography(pairs)


def test_fit_custom_threshold_is_respected() -> None:
    pairs = _good_pairs_overdetermined()
    # 50 cm error on the 5th point's X.
    pairs[4] = PointPair(
        pixel_uv=pairs[4].pixel_uv,
        world_xy_m=(pairs[4].world_xy_m[0] + 0.5, pairs[4].world_xy_m[1]),
    )
    # Tight threshold: rejected. Loose threshold: accepted.
    with pytest.raises(SingleCamCalibrationError):
        fit_single_camera_homography(pairs, residual_threshold_m=0.05)
    H, _ = fit_single_camera_homography(pairs, residual_threshold_m=0.60)
    assert H.shape == (3, 3)


def test_exactly_four_points_fits_with_zero_residual_always() -> None:
    """Document the 4-point limitation: residual is always ~0 because the
    homography is exactly-determined. The sanity gate cannot catch operator
    errors with only 4 points — recommend ≥5 in the field."""
    pairs = _good_pairs()
    # Inject a 1 m world-coord error on point 3.
    pairs[2] = PointPair(pixel_uv=pairs[2].pixel_uv, world_xy_m=(5.0, 5.0))
    # No error is raised — H just bends to fit the bogus point.
    H, max_residual = fit_single_camera_homography(pairs)
    assert max_residual < 1e-6
    assert H.shape == (3, 3)


# ---- CalibrationFile output ----


def test_build_single_camera_calibration_writes_mode1_file() -> None:
    cal = build_single_camera_calibration(
        camera_id="cam_a",
        image_size_wh=(1920, 1080),
        pairs=_good_pairs(),
        floor_origin_note="4 rack corners, tape-measured 2026-05-18",
    )
    assert cal.calibration_mode == CALIBRATION_MODE_SINGLE_CAM_4PT
    assert cal.floor_anchor_method == "4pt_floor"
    cam = cal.cameras["cam_a"]
    assert cam.image_size_wh == (1920, 1080)
    # K, D, R, t, P are placeholders.
    np.testing.assert_allclose(cam.K_np(), np.eye(3))
    np.testing.assert_allclose(cam.D_np(), np.zeros(5))
    np.testing.assert_allclose(cam.R_np(), np.eye(3))
    np.testing.assert_allclose(cam.t_np(), np.zeros(3))
    # H is real.
    assert cam.H_np().shape == (3, 3)
    # reprojection_rms_px field repurposed to hold the max world-residual (in meters).
    assert cam.reprojection_rms_px < 1e-6


def test_camera_rig_can_load_a_single_cam_calibration(tmp_path: Path) -> None:
    cal = build_single_camera_calibration(
        camera_id="cam_a", image_size_wh=(1920, 1080), pairs=_good_pairs(),
    )
    path = tmp_path / "calibration.json"
    path.write_text(cal.to_json())
    rig = CameraRig.from_file(path)
    assert rig.calibration_mode == CALIBRATION_MODE_SINGLE_CAM_4PT
    assert rig.camera_ids == ("cam_a",)


def test_calibration_file_roundtrip_preserves_mode(tmp_path: Path) -> None:
    cal = build_single_camera_calibration(
        camera_id="cam_a", image_size_wh=(1920, 1080), pairs=_good_pairs(),
    )
    path = tmp_path / "calibration.json"
    path.write_text(cal.to_json())
    reloaded = CalibrationFile.read(path)
    assert reloaded.calibration_mode == CALIBRATION_MODE_SINGLE_CAM_4PT


def test_old_calibration_without_mode_defaults_to_multical(tmp_path: Path) -> None:
    """Schema back-compat: a calibration.json from before S8 has no
    ``calibration_mode`` field — must load as ``multical_full``."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "version": 1,
        "created_at": "2026-05-12T00:00:00Z",
        "floor_anchor_method": "charuco_floor",
        "floor_origin_note": "old format",
        "cameras": {},
    }))
    cal = CalibrationFile.read(path)
    assert cal.calibration_mode == "multical_full"


# ---- CLI arg parsing ----


def test_parse_pair_arg_well_formed() -> None:
    p = _parse_pair_arg("100.5,200.0,1.0,2.0")
    assert p.pixel_uv == (100.5, 200.0)
    assert p.world_xy_m == (1.0, 2.0)


def test_parse_pair_arg_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="4 floats"):
        _parse_pair_arg("100,200,1")


def test_parse_pair_arg_rejects_non_float() -> None:
    with pytest.raises(ValueError, match="parse"):
        _parse_pair_arg("100,200,x,2")
