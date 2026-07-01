"""Hermetic tests for the targetless stereo extrinsic solve (Stage 1).

No ONNX weights, no rig, no network. A synthetic two-view scene (known R, t + a
3D cloud projected to two cameras) is fed through :class:`FakeMatcher`; we assert
the Essential/``recoverPose`` recovery of R + T-direction, metric scale from
measured references (with cross-validation flagging a bad reference), BA
convergence to a low reprojection RMS, and that the emitted ``MultiCalSolution``
flows through ``assemble_calibration`` into a valid ``calibration.json`` with sane
H/P (mirroring ``tests/test_e2e_triangulation_synthetic.py``).

The real ONNX matcher is only checked for its missing-weights error path.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.shared.geometry import pixel_to_floor, project_world_to_pixel
from calibration.calibrate import FloorAnchor, assemble_calibration
from calibration.feature_extrinsics import (
    FakeMatcher,
    MatcherWeightsMissing,
    OnnxSuperPointLightGlue,
    ScaleReference,
    estimate_metric_scale,
    recover_relative_pose,
    solve_feature_extrinsics,
)
from calibration.schema import CalibrationFile

# --- synthetic rig -----------------------------------------------------------

K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)
IMAGE_SIZE = (1280, 960)


def _rodrigues(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    import cv2

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(axis * np.deg2rad(angle_deg))
    return R


# Ground-truth cam_b ← cam_a pose: ~15° yaw, baseline ~0.8 m along +x.
R_TRUE = _rodrigues(np.array([0.0, 1.0, 0.0]), 15.0)
T_TRUE = np.array([0.8, 0.0, 0.0])


def _scene_points(rng: np.random.Generator, n: int = 200) -> np.ndarray:
    """3D cloud in cam_a frame, in front of both cameras, metric metres."""
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = rng.uniform(4.0, 9.0, n)
    return np.column_stack([x, y, z])


def _project(pts3: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project cam_a-frame 3D points into a camera at [R|t] with intrinsics K."""
    cam = pts3 @ R.T + t.reshape(1, 3)
    uv = (K @ (cam / cam[:, 2:3]).T).T
    return uv[:, :2]


def _synthetic_matches(rng: np.random.Generator, n: int = 200):
    pts3 = _scene_points(rng, n)
    pts_a = _project(pts3, np.eye(3), np.zeros(3))
    pts_b = _project(pts3, R_TRUE, T_TRUE)
    return pts3, pts_a, pts_b


def _refs_from_points(pts3: np.ndarray, pairs: list[tuple[int, int]]) -> list[ScaleReference]:
    refs = []
    for i, j in pairs:
        a1 = _project(pts3[i : i + 1], np.eye(3), np.zeros(3))[0]
        a2 = _project(pts3[j : j + 1], np.eye(3), np.zeros(3))[0]
        b1 = _project(pts3[i : i + 1], R_TRUE, T_TRUE)[0]
        b2 = _project(pts3[j : j + 1], R_TRUE, T_TRUE)[0]
        dist = float(np.linalg.norm(pts3[i] - pts3[j]))
        refs.append(ScaleReference(tuple(a1), tuple(b1), tuple(a2), tuple(b2), dist))
    return refs


# --- pose recovery -----------------------------------------------------------


def test_recover_pose_matches_ground_truth_rotation_and_direction() -> None:
    rng = np.random.default_rng(0)
    _pts3, pts_a, pts_b = _synthetic_matches(rng)

    R, t_dir, inliers = recover_relative_pose(pts_a, pts_b, K, D, K, D)

    # Rotation within ~1 degree.
    import cv2

    ang = np.rad2deg(np.linalg.norm(cv2.Rodrigues(R @ R_TRUE.T)[0]))
    assert ang < 1.0
    # Translation direction (unit) aligns with the true baseline direction.
    cos = float(np.dot(t_dir / np.linalg.norm(t_dir), T_TRUE / np.linalg.norm(T_TRUE)))
    assert cos > 0.999
    assert inliers.sum() > 150


# --- metric scale ------------------------------------------------------------


def test_metric_scale_recovers_true_baseline() -> None:
    rng = np.random.default_rng(1)
    pts3, pts_a, pts_b = _synthetic_matches(rng)
    R, t_dir, _ = recover_relative_pose(pts_a, pts_b, K, D, K, D)

    refs = _refs_from_points(pts3, [(0, 10), (1, 20), (2, 30), (3, 40)])
    est = estimate_metric_scale(refs, K, D, K, D, R, t_dir)

    metric_t = est.scale * (t_dir / np.linalg.norm(t_dir))
    # Recovered metric baseline length ~ 0.8 m.
    assert np.linalg.norm(metric_t) == pytest.approx(np.linalg.norm(T_TRUE), rel=1e-2)
    assert est.outliers == []


def test_scale_cross_validation_flags_bad_reference() -> None:
    rng = np.random.default_rng(2)
    pts3, pts_a, pts_b = _synthetic_matches(rng)
    R, t_dir, _ = recover_relative_pose(pts_a, pts_b, K, D, K, D)

    refs = _refs_from_points(pts3, [(0, 10), (1, 20), (2, 30), (3, 40)])
    # Corrupt reference #2's measured distance (operator mis-measured / mis-clicked).
    bad = refs[2]
    refs[2] = ScaleReference(bad.p1_a, bad.p1_b, bad.p2_a, bad.p2_b, bad.distance_m * 1.5)

    est = estimate_metric_scale(refs, K, D, K, D, R, t_dir)
    assert 2 in est.outliers
    # The good references still pin the scale.
    metric_t = est.scale * (t_dir / np.linalg.norm(t_dir))
    assert np.linalg.norm(metric_t) == pytest.approx(np.linalg.norm(T_TRUE), rel=2e-2)


def test_single_reference_warns_but_works() -> None:
    rng = np.random.default_rng(3)
    pts3, pts_a, pts_b = _synthetic_matches(rng)
    R, t_dir, _ = recover_relative_pose(pts_a, pts_b, K, D, K, D)
    refs = _refs_from_points(pts3, [(0, 50)])
    with pytest.warns(UserWarning):
        est = estimate_metric_scale(refs, K, D, K, D, R, t_dir)
    assert est.n_references == 1


def test_no_reference_raises() -> None:
    with pytest.raises(ValueError):
        estimate_metric_scale([], K, D, K, D, np.eye(3), np.array([1.0, 0, 0]))


# --- full solve + BA ---------------------------------------------------------


def _full_solution(seed: int = 4):
    rng = np.random.default_rng(seed)
    pts3, pts_a, pts_b = _synthetic_matches(rng)
    matcher = FakeMatcher(pts_a=pts_a, pts_b=pts_b)
    refs = _refs_from_points(pts3, [(0, 10), (1, 20), (2, 30), (4, 44)])
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    return solve_feature_extrinsics(
        image_pairs=[(dummy, dummy)],
        matcher=matcher,
        K_a=K, D_a=D, K_b=K, D_b=D,
        references=refs,
        image_size_wh=IMAGE_SIZE,
    )


def test_full_solve_converges_low_rms_and_metric_pose() -> None:
    res = _full_solution()
    assert res.reprojection_rms_px < 0.5
    # cam_b rig-frame translation length equals the metric baseline (~0.8 m).
    cam_b = res.solution.cameras["cam_b"]
    # rig(=cam_a) ← cam_b translation is -R.T t; its norm equals |t|.
    assert np.linalg.norm(cam_b.t_in_rig) == pytest.approx(np.linalg.norm(T_TRUE), rel=2e-2)
    # Recovered R (cam_b ← cam_a) within ~1 deg of ground truth.
    import cv2

    ang = np.rad2deg(np.linalg.norm(cv2.Rodrigues(res.R @ R_TRUE.T)[0]))
    assert ang < 1.0


def test_solution_is_master_identity_for_cam_a() -> None:
    res = _full_solution()
    cam_a = res.solution.cameras["cam_a"]
    assert res.solution.master_camera == "cam_a"
    np.testing.assert_allclose(cam_a.R_in_rig, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(cam_a.t_in_rig, np.zeros(3), atol=1e-9)


# --- MultiCalSolution → calibration.json seam --------------------------------


def _lookdown_floor_anchor() -> FloorAnchor:
    """Stub anchor placing the cameras above a floor they look down onto.

    An identity anchor would put the floor plane (world Z=0) coincident with the
    camera centres (cam_a is the rig origin), making the floor homography
    singular — an artefact of the synthetic scene, not the pipeline. Instead we
    rotate the rig's forward axis (+Z) to world -Z (look-down) and lift the rig
    to 3 m, exactly as ``test_e2e_triangulation_synthetic`` does.
    """
    # Rig +Z (camera forward) → world -Z (down); rig +X → world +X.
    R_world_from_rig = np.diag([1.0, -1.0, -1.0])
    t_world_from_rig = np.array([0.0, 0.0, 3.0])
    return FloorAnchor(
        method="planefit",
        note="synthetic look-down floor anchor",
        R_world_from_rig=R_world_from_rig,
        t_world_from_rig=t_world_from_rig,
    )


def test_solution_assembles_into_valid_calibration(tmp_path) -> None:
    res = _full_solution()
    calib = assemble_calibration(res.solution, _lookdown_floor_anchor())
    assert isinstance(calib, CalibrationFile)
    assert set(calib.cameras) == {"cam_a", "cam_b"}

    # Round-trips to disk and reloads.
    out = tmp_path / "calibration.json"
    calib.write(out)
    reloaded = CalibrationFile.read(out)
    assert set(reloaded.cameras) == {"cam_a", "cam_b"}

    for cam_id in ("cam_a", "cam_b"):
        cam = reloaded.cameras[cam_id]
        H = cam.H_np()
        P = cam.P_np()
        assert H.shape == (3, 3)
        assert P.shape == (3, 4)
        assert np.isfinite(H).all()
        assert np.isfinite(P).all()


def test_assembled_H_and_P_are_geometrically_consistent() -> None:
    """A floor point (Z=0) projected via P must map back to (X,Y) via H."""
    res = _full_solution()
    calib = assemble_calibration(res.solution, _lookdown_floor_anchor())
    cam = calib.cameras["cam_a"]
    P = cam.P_np()
    H = cam.H_np()

    world_xy = np.array([[0.3, -0.2]])
    world_xyz = np.array([[0.3, -0.2, 0.0]])
    uv = project_world_to_pixel(world_xyz, P)
    back = pixel_to_floor(uv, H)
    np.testing.assert_allclose(back, world_xy, atol=1e-6)


# --- ONNX matcher missing-weights path (no network) --------------------------


def test_onnx_matcher_missing_weights_raises(tmp_path) -> None:
    with pytest.raises(MatcherWeightsMissing) as exc:
        OnnxSuperPointLightGlue(models_dir=tmp_path)
    msg = str(exc.value)
    assert "fabio-sim/LightGlue-ONNX" in msg
    assert "superpoint.onnx" in msg
