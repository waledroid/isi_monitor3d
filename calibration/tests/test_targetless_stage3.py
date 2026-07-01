"""Hermetic Stage-3 tests: targetless orchestration + viz + validation report.

No ONNX, no rig, no network. A synthetic two-view scene (known R, t + a 3D cloud,
with the floor references lying on a look-down plane) is fed through
:class:`FakeMatcher`; we assert the orchestration chains Stage-1 + Stage-2 +
``assemble_calibration`` into a valid ``calibration.json``, the 5 key-stage
visualizations return valid annotated images, and the 3-level validation report
computes the geometric level and marks the metric / end-to-end levels pending when
no measurements / detections are supplied (and evaluates them when they are).
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.feature_extrinsics import FakeMatcher, ScaleReference
from calibration.feature_viz import (
    draw_feature_matches,
    draw_result_overlay,
    draw_scale_references,
    draw_stereo_pair,
    draw_triangulation_floor,
)
from calibration.schema import CalibrationFile
from calibration.targetless_orchestration import solve_targetless
from calibration.targetless_validation import (
    ObjectMeasurement,
    build_validation_report,
    geometric_level,
    metric_level,
)

# --- synthetic look-down rig: floor is a plane the cameras look down onto -----

K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)
IMAGE_SIZE = (1280, 960)


def _rodrigues(axis, angle_deg):
    import cv2
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(axis * np.deg2rad(angle_deg))
    return R


R_TRUE = _rodrigues([0.0, 1.0, 0.0], 12.0)
T_TRUE = np.array([0.7, 0.0, 0.0])


def _project(pts3, R, t):
    cam = pts3 @ R.T + t.reshape(1, 3)
    uv = (K @ (cam / cam[:, 2:3]).T).T
    return uv[:, :2]


def _scene(rng, n=200):
    """A cloud plus a set of FLOOR points on a plane ~5 m ahead (for the plane fit)."""
    # General cloud (for pose recovery).
    cloud = np.column_stack([
        rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(4, 9, n)])
    # Floor points on a plane y=1.5 (a horizontal floor below the cameras).
    fn = 40
    floor = np.column_stack([
        rng.uniform(-2, 2, fn), np.full(fn, 1.5), rng.uniform(4, 8, fn)])
    return cloud, floor


def _refs_on_floor(floor_pts, pairs):
    refs = []
    for i, j in pairs:
        a1 = _project(floor_pts[i:i + 1], np.eye(3), np.zeros(3))[0]
        a2 = _project(floor_pts[j:j + 1], np.eye(3), np.zeros(3))[0]
        b1 = _project(floor_pts[i:i + 1], R_TRUE, T_TRUE)[0]
        b2 = _project(floor_pts[j:j + 1], R_TRUE, T_TRUE)[0]
        dist = float(np.linalg.norm(floor_pts[i] - floor_pts[j]))
        refs.append(ScaleReference(tuple(a1), tuple(b1), tuple(a2), tuple(b2), dist))
    return refs


def _solve(seed=7):
    rng = np.random.default_rng(seed)
    cloud, floor = _scene(rng)
    pts_a = _project(cloud, np.eye(3), np.zeros(3))
    pts_b = _project(cloud, R_TRUE, T_TRUE)
    matcher = FakeMatcher(pts_a=pts_a, pts_b=pts_b)
    refs = _refs_on_floor(floor, [(0, 10), (1, 20), (2, 30), (5, 35)])
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    return solve_targetless(
        image_pairs=[(img, img)], matcher=matcher,
        K_a=K, D_a=D, K_b=K, D_b=D,
        references=refs, image_size_wh=IMAGE_SIZE,
    ), refs


# --- orchestration -----------------------------------------------------------


def test_orchestration_produces_valid_calibration(tmp_path):
    res, _refs = _solve()
    assert isinstance(res.calibration, CalibrationFile)
    assert set(res.calibration.cameras) == {"cam_a", "cam_b"}
    assert res.floor_anchor.method == "planefit"

    out = tmp_path / "calibration.json"
    res.calibration.write(out)
    reloaded = CalibrationFile.read(out)
    for cid in ("cam_a", "cam_b"):
        cam = reloaded.cameras[cid]
        assert cam.H_np().shape == (3, 3)
        assert cam.P_np().shape == (3, 4)
        assert np.isfinite(cam.H_np()).all()
        assert np.isfinite(cam.P_np()).all()


def test_orchestration_recovers_metric_baseline():
    res, _refs = _solve()
    cam_b = res.feature.solution.cameras["cam_b"]
    assert np.linalg.norm(cam_b.t_in_rig) == pytest.approx(np.linalg.norm(T_TRUE), rel=3e-2)


def test_orchestration_stage_images_present_and_valid():
    res, _refs = _solve()
    assert set(res.stage_images) == {"pair", "matches", "scale_refs", "triangulation", "result"}
    for name, img in res.stage_images.items():
        assert img.ndim == 3 and img.shape[2] == 3, name
        assert img.dtype == np.uint8, name
        assert img.shape[0] > 0 and img.shape[1] > 0, name


# --- visualizations (standalone, annotations present) ------------------------


def _blank(w=320, h=240):
    return np.full((h, w, 3), 30, dtype=np.uint8)


def test_draw_stereo_pair_is_side_by_side():
    a, b = _blank(), _blank()
    out = draw_stereo_pair(a, b, pair_index=3)
    assert out.shape[1] >= a.shape[1] * 2  # side-by-side
    assert out.dtype == np.uint8


def test_draw_feature_matches_colours_inliers_and_outliers():
    a, b = _blank(), _blank()
    pa = np.array([[50, 50], [100, 100], [150, 120]], float)
    pb = np.array([[60, 55], [110, 105], [140, 130]], float)
    mask = np.array([True, True, False])
    base = np.hstack([a, b]).copy()
    out = draw_feature_matches(a, b, pa, pb, mask)
    # Something was drawn (differs from the blank canvas).
    assert not np.array_equal(out[:, :, :], base)
    # Both an inlier (green) and outlier (red) colour appear.
    flat = out.reshape(-1, 3)
    assert np.any((flat[:, 1] > 150) & (flat[:, 2] < 120))   # green-ish
    assert np.any((flat[:, 2] > 150) & (flat[:, 1] < 120))   # red-ish


def test_draw_scale_references_marks_flagged_outlier():
    a, b = _blank(), _blank()
    refs = [ScaleReference((10, 10), (12, 10), (60, 60), (62, 60), 1.0),
            ScaleReference((20, 20), (22, 20), (80, 80), (82, 80), 2.0)]
    out = draw_scale_references(a, b, refs, outliers=[1])
    assert out.shape[1] >= a.shape[1] * 2
    flat = out.reshape(-1, 3)
    assert np.any((flat[:, 2] > 150) & (flat[:, 1] < 120))   # red (flagged) present


def test_draw_triangulation_floor_highlights_inliers_and_grid():
    a = _blank(IMAGE_SIZE[0], IMAGE_SIZE[1])   # match K's principal point so pts land in-frame
    pts = np.array([[0.0, 0.5, 5.0], [0.2, 0.5, 5.5], [-0.3, 0.5, 6.0]])
    mask = np.array([True, True, False])
    out = draw_triangulation_floor(
        a, pts, K, floor_inlier_mask=mask,
        plane_normal=np.array([0.0, 1.0, 0.0]), plane_offset=-0.5)
    assert out.dtype == np.uint8
    flat = out.reshape(-1, 3)
    assert np.any((flat[:, 1] > 150) & (flat[:, 2] < 120))   # green inliers


def test_draw_result_overlay_pending_and_pass():
    a = _blank()
    out_pending = draw_result_overlay(a, ["reproj 0.3 px"], passed=None)
    out_pass = draw_result_overlay(a, ["reproj 0.3 px"], passed=True)
    assert out_pending.shape == a.shape
    assert not np.array_equal(out_pending, out_pass)


# --- validation report -------------------------------------------------------


def _synthetic_calib_dict(baseline=0.7):
    return {
        "cameras": {
            "cam_a": {"R": np.eye(3).tolist(), "t": [0, 0, 0], "reprojection_rms_px": 0.3},
            "cam_b": {"R": np.eye(3).tolist(), "t": [-baseline, 0, 0],
                      "reprojection_rms_px": 0.4},
        },
        "floor_anchor_method": "planefit",
    }


def test_geometric_level_computes_baseline_and_rms():
    geo = geometric_level(_synthetic_calib_dict(), reprojection_rms_px=0.35)
    assert geo["baseline_m"] == pytest.approx(0.7, rel=1e-6)
    assert geo["worst_camera_rms_px"] == 0.4
    assert geo["checks"]["reprojection_rms"] is True
    assert geo["has_reference"] is False


def test_geometric_level_diffs_against_reference():
    geo = geometric_level(
        _synthetic_calib_dict(0.7), reference_calib=_synthetic_calib_dict(0.72),
        reprojection_rms_px=0.35)
    assert geo["has_reference"] is True
    assert "baseline_diff_rel" in geo
    assert geo["checks"]["baseline_agreement"] is True   # within 5 %


def test_metric_level_pending_without_measurements():
    met = metric_level(None)
    assert met["status"] == "pending"
    assert met["passed"] is None


def test_metric_level_evaluates_distance_measurements():
    ms = [ObjectMeasurement("pallet-gap", "distance", measured_m=2.00, estimated_m=2.01),
          ObjectMeasurement("rack-h", "height", measured_m=1.50, estimated_m=1.70)]
    met = metric_level(ms)
    assert met["status"] == "evaluated"
    assert met["items"][0]["passed"] is True     # 1 cm error
    assert met["items"][1]["passed"] is False    # 20 cm error > 5 cm gate
    assert met["passed"] is False


def test_report_accepted_false_when_metric_pending():
    rep = build_validation_report(_synthetic_calib_dict(), reprojection_rms_px=0.35)
    assert rep.accepted is False
    assert rep.acceptance_reason == "metric_pending"
    assert any("PENDING" in ln for ln in rep.summary_lines)


def test_report_accepted_true_when_metric_kpis_met():
    ms = [ObjectMeasurement("d1", "distance", measured_m=2.0, estimated_m=2.01),
          ObjectMeasurement("d2", "distance", measured_m=3.0, estimated_m=2.98),
          ObjectMeasurement("p1", "position",
                            point_a=(1.0, 2.0, 0.0), estimated_point=(1.02, 2.01, 0.0))]
    rep = build_validation_report(
        _synthetic_calib_dict(), reference_calib=_synthetic_calib_dict(0.71),
        reprojection_rms_px=0.35, measurements=ms)
    assert rep.metric["passed"] is True
    assert rep.accepted is True
    assert rep.acceptance_reason == "metric_kpis_met"


def test_report_end_to_end_evaluates_when_supplied():
    from calibration.targetless_validation import end_to_end_level
    e2e = end_to_end_level({"distances": [
        {"label": "fork-pallet", "measured_m": 1.20, "fused_m": 1.22}]})
    assert e2e["status"] == "evaluated"
    assert e2e["distances"][0]["passed"] is True
