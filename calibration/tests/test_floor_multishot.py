"""Multi-placement ChArUco floor-anchor consensus plane-fit (hermetic).

No rig, no camera, no image rendering. We inject known board-in-rig poses (the
output of solvePnP + rig composition) by monkeypatching ``_board_pose_in_rig``,
so we can exercise the multi-placement consensus logic directly:

* several coplanar ChArUco placements on a KNOWN tilted floor plane (both cams)
  → the recovered world frame maps all board points to Z ~= 0, with a lower
  residual than a single noisy placement;
* single-placement input still reproduces the legacy world-IS-the-board frame;
* the multi-shot ``FloorAnchor`` flows through ``assemble_calibration`` into a
  valid ``calibration.json`` with finite H/P.
"""

from __future__ import annotations

import numpy as np
import pytest

import calibration.calibrate as cal
from calibration.calibrate import (
    assemble_calibration,
    estimate_floor_anchor_charuco,
)
from calibration.schema import CalibrationFile
from calibration.tests.test_feature_extrinsics import _full_solution

# A ChArUco board spec whose getChessboardCorners() gives us the board object
# points; only the geometry matters here (never rendered).
_BOARD = cal.CharucoBoardSpec(
    squares_x=5, squares_y=7, square_length_m=0.035, marker_length_m=0.026,
)


def _tilted_floor():
    """A known floor plane (rig frame): tilted normal + offset ~6.5 m ahead."""
    normal = np.array([0.06, 0.08, -1.0])
    normal /= np.linalg.norm(normal)
    plane_point = np.array([0.1, -0.2, 6.5])
    a = np.cross(normal, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    return normal, plane_point, a, b


def _board_pose_on_plane(a, b, normal, plane_point, u, v, theta):
    """A board pose (R_rig_board, t_rig_board) laid FLAT on the plane.

    The board's local +X/+Y span the plane (in-plane rotation ``theta``), +Z is the
    plane normal; the board origin sits at ``plane_point + u*a + v*b``.
    """
    x = np.cos(theta) * a + np.sin(theta) * b
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    R = np.column_stack([x, y, normal])
    t = plane_point + u * a + v * b
    return R, t


def _install_poses(monkeypatch, poses_by_shotpath):
    """Route ``_board_pose_in_rig(path, ...)`` to a pre-baked (R, t) per path."""
    def fake(shot_path, cam, board):
        return poses_by_shotpath[str(shot_path)]
    monkeypatch.setattr(cal, "_board_pose_in_rig", fake)


def _corners_to_z(anchor, R_board, t_board):
    """World-Z of the board's corner points under ``anchor``, for a board pose."""
    obj = _BOARD.board().getChessboardCorners().reshape(-1, 3)
    rig = obj @ R_board.T + t_board
    world = rig @ anchor.R_world_from_rig.T + anchor.t_world_from_rig
    return world[:, 2]


def test_multishot_maps_all_placements_to_zzero(monkeypatch):
    normal, plane_point, a, b = _tilted_floor()
    sol = _full_solution().solution

    placements = [(0.0, 0.0, 0.0), (0.8, -0.5, 0.4), (-0.6, 0.7, -0.3)]
    rng = np.random.default_rng(0)
    poses = {}
    truth = []
    shots = {"cam_a": [], "cam_b": []}
    for i, (u, v, th) in enumerate(placements):
        R, t = _board_pose_on_plane(a, b, normal, plane_point, u, v, th)
        truth.append((R, t))
        for cam in ("cam_a", "cam_b"):
            p = f"/floor/{cam}/{i:03d}.jpg"
            shots[cam].append(p)
            # tiny per-cam pose noise (independent solvePnP estimates)
            dt = rng.normal(scale=0.002, size=3)
            poses[p] = (R, t + dt)
    _install_poses(monkeypatch, poses)

    anchor = estimate_floor_anchor_charuco(shots, sol, _BOARD)
    assert anchor.method == "charuco_floor"
    # ALL placements' board corners land near world Z=0.
    for R, t in truth:
        z = _corners_to_z(anchor, R, t)
        assert np.abs(z).max() < 0.02


def test_multishot_lower_residual_than_single_noisy_shot(monkeypatch):
    normal, plane_point, a, b = _tilted_floor()
    sol = _full_solution().solution
    placements = [(0.0, 0.0, 0.0), (0.9, -0.4, 0.5), (-0.7, 0.6, -0.4), (0.3, 0.9, 0.2)]

    def residual(shots, poses):
        _install_poses(monkeypatch, poses)
        anchor = estimate_floor_anchor_charuco(shots, sol, _BOARD)
        allz = []
        for R, t in [poses[shots["cam_a"][i]] for i in range(len(shots["cam_a"]))]:
            allz.append(_corners_to_z(anchor, R, t))
        return float(np.sqrt(np.mean(np.concatenate(allz) ** 2)))

    rng = np.random.default_rng(3)
    truth = [_board_pose_on_plane(a, b, normal, plane_point, u, v, th)
             for (u, v, th) in placements]

    # Multi: 4 placements, each perturbed by out-of-plane tilt noise.
    multi_shots = {"cam_a": [], "cam_b": []}
    multi_poses = {}
    for i, (R, t) in enumerate(truth):
        rvec = rng.normal(scale=0.01, size=3)      # small pose tilt per placement
        import cv2
        dR = cv2.Rodrigues(rvec)[0]
        for cam in ("cam_a", "cam_b"):
            p = f"/m/{cam}/{i:03d}.jpg"
            multi_shots[cam].append(p)
            multi_poses[p] = (dR @ R, t)
    multi_res = residual(multi_shots, multi_poses)

    # Single: only the first, MORE tilted placement (a single bad shot).
    R0, t0 = truth[0]
    import cv2
    dR = cv2.Rodrigues(np.array([0.03, -0.02, 0.0]))[0]
    single_shots = {"cam_a": ["/s/cam_a/000.jpg"], "cam_b": ["/s/cam_b/000.jpg"]}
    single_poses = {"/s/cam_a/000.jpg": (dR @ R0, t0),
                    "/s/cam_b/000.jpg": (dR @ R0, t0)}
    # residual measured against the TRUE placements, not the single tilted one
    _install_poses(monkeypatch, single_poses)
    anchor_single = estimate_floor_anchor_charuco(single_shots, sol, _BOARD)
    allz = [_corners_to_z(anchor_single, R, t) for R, t in truth]
    single_res = float(np.sqrt(np.mean(np.concatenate(allz) ** 2)))

    assert multi_res < single_res


def test_single_shot_back_compat_world_is_the_board(monkeypatch):
    """One placement (list or bare Path) → world frame IS that board (Z=0 exact)."""
    normal, plane_point, a, b = _tilted_floor()
    sol = _full_solution().solution
    R, t = _board_pose_on_plane(a, b, normal, plane_point, 0.0, 0.0, 0.0)
    poses = {"/f/cam_a.jpg": (R, t), "/f/cam_b.jpg": (R, t)}
    _install_poses(monkeypatch, poses)

    # bare Path per camera (the legacy call shape)
    from pathlib import Path
    anchor = estimate_floor_anchor_charuco(
        {"cam_a": Path("/f/cam_a.jpg"), "cam_b": Path("/f/cam_b.jpg")}, sol, _BOARD)
    z = _corners_to_z(anchor, R, t)
    assert np.abs(z).max() < 1e-9            # world IS the board → corners exactly Z=0


def test_multishot_anchor_assembles_valid_calibration(monkeypatch, tmp_path):
    normal, plane_point, a, b = _tilted_floor()
    res = _full_solution()
    placements = [(0.0, 0.0, 0.0), (0.8, -0.5, 0.4), (-0.6, 0.7, -0.3)]
    poses, shots = {}, {"cam_a": [], "cam_b": []}
    for i, (u, v, th) in enumerate(placements):
        R, t = _board_pose_on_plane(a, b, normal, plane_point, u, v, th)
        for cam in ("cam_a", "cam_b"):
            p = f"/c/{cam}/{i:03d}.jpg"
            shots[cam].append(p)
            poses[p] = (R, t)
    _install_poses(monkeypatch, poses)

    anchor = estimate_floor_anchor_charuco(shots, res.solution, _BOARD)
    calib = assemble_calibration(res.solution, anchor)
    assert isinstance(calib, CalibrationFile)
    out = tmp_path / "calibration.json"
    calib.write(out)
    reloaded = CalibrationFile.read(out)
    for cam_id in ("cam_a", "cam_b"):
        cam = reloaded.cameras[cam_id]
        assert cam.H_np().shape == (3, 3) and cam.P_np().shape == (3, 4)
        assert np.isfinite(cam.H_np()).all() and np.isfinite(cam.P_np()).all()


def test_empty_floor_shots_raises():
    sol = _full_solution().solution
    with pytest.raises(RuntimeError):
        estimate_floor_anchor_charuco({}, sol, _BOARD)
