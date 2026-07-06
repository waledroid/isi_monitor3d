"""``multical_io.parse`` against hand-crafted Multical-shaped fixtures.

Real Multical runs are exercised separately via the dev rig; this module pins
the parser's behavior on the two key conventions Multical uses and on common
malformed inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from calibration.multical_io import (
    UNKNOWN_RMS_PX,
    MultiCalParseError,
    MultiCalSolution,
    from_dict,
    parse,
    parse_rms_from_log,
)

I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ZERO_T = [0.0, 0.0, 0.0]


def _camera_block(image_size=(1920, 1080)) -> dict:
    return {
        "model": "standard",
        "image_size": list(image_size),
        "K": [[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]],
        "dist": [-0.05, 0.02, 0.0, 0.0, 0.0],
    }


def _absolute_layout() -> dict:
    """Multical was exported with no `master` — absolute poses per camera."""
    return {
        "cameras": {"cam_a": _camera_block(), "cam_b": _camera_block()},
        "camera_poses": {
            "cam_a": {"R": I3, "T": ZERO_T},
            "cam_b": {"R": I3, "T": [4.0, 0.0, 0.0]},
        },
    }


def _master_relative_layout() -> dict:
    """Multical was exported with master=cam_a — others keyed `<x>_to_cam_a`."""
    return {
        "cameras": {"cam_a": _camera_block(), "cam_b": _camera_block()},
        "camera_poses": {
            "cam_a": {"R": I3, "T": ZERO_T},
            "cam_b_to_cam_a": {"R": I3, "T": [4.0, 0.0, 0.0]},
        },
    }


def test_absolute_layout_parses() -> None:
    sol = from_dict(_absolute_layout())
    assert set(sol.camera_ids) == {"cam_a", "cam_b"}
    assert sol.master_camera == "cam_a"  # deterministic pick: alphabetical
    cam_b = sol.cameras["cam_b"]
    assert cam_b.K.shape == (3, 3)
    assert cam_b.image_size_wh == (1920, 1080)
    # Multical's T is camera←rig; CameraInRig stores rig←camera (with R=I the
    # inversion is a sign flip). See the pose-convention note in multical_io.
    np.testing.assert_allclose(cam_b.t_in_rig, [-4.0, 0.0, 0.0])


def test_master_relative_layout_parses() -> None:
    sol = from_dict(_master_relative_layout())
    assert set(sol.camera_ids) == {"cam_a", "cam_b"}
    assert sol.master_camera == "cam_a"
    # master is identity in the rig frame
    np.testing.assert_allclose(sol.cameras["cam_a"].R_in_rig, np.eye(3))
    np.testing.assert_allclose(sol.cameras["cam_a"].t_in_rig, np.zeros(3))
    # camera←rig T=[4,0,0] inverts to a rig-frame camera center at [-4,0,0].
    np.testing.assert_allclose(sol.cameras["cam_b"].t_in_rig, [-4.0, 0.0, 0.0])


def test_pose_convention_inverted_from_multical() -> None:
    """THE convention pin: multical exports camera←rig extrinsics; CameraInRig
    must hold rig←camera (R_m.T, -R_m.T@T_m). Storing verbatim silently inverts
    every non-master camera — invisible to per-camera RMS, but the same
    physical point observed by two cameras then maps metres apart in world."""
    # Nontrivial rotation: 90° about Z, plus a translation.
    R_m = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T_m = np.array([1.0, 2.0, 3.0])
    layout = {
        "cameras": {"cam_a": _camera_block(), "cam_b": _camera_block()},
        "camera_poses": {
            "cam_a": {"R": I3, "T": ZERO_T},
            "cam_b_to_cam_a": {"R": R_m.tolist(), "T": T_m.tolist()},
        },
    }
    sol = from_dict(layout)
    cam_b = sol.cameras["cam_b"]
    np.testing.assert_allclose(cam_b.R_in_rig, R_m.T)
    np.testing.assert_allclose(cam_b.t_in_rig, -R_m.T @ T_m)
    # Round-trip: a rig point maps camera→rig→camera to itself.
    p_cam = np.array([0.5, -0.2, 4.0])
    p_rig = cam_b.R_in_rig @ p_cam + cam_b.t_in_rig
    np.testing.assert_allclose(R_m @ p_rig + T_m, p_cam, atol=1e-12)


def test_missing_cameras_block() -> None:
    with pytest.raises(MultiCalParseError, match="cameras"):
        from_dict({"camera_poses": {}})


def test_missing_camera_poses_block() -> None:
    with pytest.raises(MultiCalParseError, match="camera_poses"):
        from_dict({"cameras": {"cam_a": _camera_block()}})


def test_camera_without_pose_rejected() -> None:
    layout = _master_relative_layout()
    del layout["camera_poses"]["cam_b_to_cam_a"]
    with pytest.raises(MultiCalParseError, match="cam_b"):
        from_dict(layout)


def test_pose_with_bad_shape_rejected() -> None:
    layout = _absolute_layout()
    layout["camera_poses"]["cam_a"]["R"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(MultiCalParseError, match="R is"):
        from_dict(layout)


def test_camera_with_bad_K_shape_rejected() -> None:
    layout = _absolute_layout()
    layout["cameras"]["cam_a"]["K"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(MultiCalParseError, match="K is"):
        from_dict(layout)


def test_multiple_masters_rejected() -> None:
    layout = _master_relative_layout()
    layout["camera_poses"]["cam_c_to_cam_b"] = {"R": I3, "T": [1.0, 0.0, 0.0]}
    layout["cameras"]["cam_c"] = _camera_block()
    with pytest.raises(MultiCalParseError, match="multiple masters"):
        from_dict(layout)


def test_default_rms_sentinel() -> None:
    sol = from_dict(_absolute_layout())
    for cam in sol.cameras.values():
        assert cam.rms_px == UNKNOWN_RMS_PX


def test_with_rms_fills_in_values() -> None:
    sol = from_dict(_absolute_layout())
    updated = sol.with_rms({"cam_a": 0.32, "cam_b": 0.41})
    assert updated.cameras["cam_a"].rms_px == 0.32
    assert updated.cameras["cam_b"].rms_px == 0.41
    # Original solution is not mutated.
    assert sol.cameras["cam_a"].rms_px == UNKNOWN_RMS_PX


def test_with_rms_partial_keeps_sentinel() -> None:
    sol = from_dict(_absolute_layout())
    updated = sol.with_rms({"cam_a": 0.3})
    assert updated.cameras["cam_a"].rms_px == 0.3
    assert updated.cameras["cam_b"].rms_px == UNKNOWN_RMS_PX


def test_parse_file_roundtrip(tmp_path: Path) -> None:
    fixture = tmp_path / "calibration.json"
    fixture.write_text(json.dumps(_master_relative_layout()))
    sol = parse(fixture)
    assert isinstance(sol, MultiCalSolution)
    assert set(sol.camera_ids) == {"cam_a", "cam_b"}


# ---------- RMS log parsing ----------


def test_parse_rms_from_log_basic() -> None:
    log = """
    INFO - cam_a - RMS: 0.3142 quantiles: [0.0 0.1 0.2 0.3 0.9]
    INFO - cam_b - RMS: 0.4521 quantiles: [0.0 0.1 0.3 0.4 1.1]
    """
    rms = parse_rms_from_log(log, ("cam_a", "cam_b"))
    assert rms == {"cam_a": pytest.approx(0.3142), "cam_b": pytest.approx(0.4521)}


def test_parse_rms_keeps_final_value_per_camera() -> None:
    log = """
    cam_a - RMS: 1.000
    cam_a - RMS: 0.500
    cam_a - RMS: 0.314
    """
    rms = parse_rms_from_log(log, ("cam_a",))
    assert rms == {"cam_a": pytest.approx(0.314)}


def test_parse_rms_ignores_non_camera_names() -> None:
    log = """
    translation - RMS: 0.001
    angle(deg) - RMS: 0.05
    cam_a - RMS: 0.32
    """
    rms = parse_rms_from_log(log, ("cam_a",))
    assert rms == {"cam_a": pytest.approx(0.32)}
    assert "translation" not in rms


def test_parse_rms_empty_for_unmatched_log() -> None:
    rms = parse_rms_from_log("nothing relevant here", ("cam_a",))
    assert rms == {}


def test_cross_camera_board_pose_consistency() -> None:
    """End-to-end convention proof, no images: a synthetic 2-cam rig observes ONE
    board; each camera's exact camera←board pose is composed with the parsed rig
    pose. Both cameras must agree on the board's rig pose to machine precision.
    On the pre-fix (verbatim) parse this disagrees by ~2 m — the c1 bug."""
    from scipy.spatial.transform import Rotation

    from calibration.calibrate import _compose_board_in_rig

    # Ground truth: cam_a at rig origin; cam_b 2 m away, yawed 25°.
    R_rig_b = Rotation.from_euler("y", 25, degrees=True).as_matrix()
    C_b = np.array([2.0, 0.1, -0.3])
    # One board somewhere in front of both cameras (rig frame).
    R_rig_board = Rotation.from_euler("xyz", [5, -10, 40], degrees=True).as_matrix()
    t_rig_board = np.array([1.0, 0.4, 3.0])

    def cam_from_rig(R_rig_cam, C):
        # camera←rig extrinsic (what multical exports): p_cam = R@(p_rig) + T
        R = R_rig_cam.T
        return R, -R @ C

    # Exact board pose seen from each camera: T_cam←board = T_cam←rig ∘ T_rig←board.
    def board_in_cam(R_rig_cam, C):
        R_cr, t_cr = cam_from_rig(R_rig_cam, C)
        return R_cr @ R_rig_board, R_cr @ t_rig_board + t_cr

    Ra_cb, ta_cb = board_in_cam(np.eye(3), np.zeros(3))
    Rb_cb, tb_cb = board_in_cam(R_rig_b, C_b)

    # Export the rig in MULTICAL's format (cam_b entry = camera←rig of b vs a).
    R_m, T_m = cam_from_rig(R_rig_b, C_b)
    sol = from_dict({
        "cameras": {"cam_a": _camera_block(), "cam_b": _camera_block()},
        "camera_poses": {
            "cam_a": {"R": I3, "T": ZERO_T},
            "cam_b_to_cam_a": {"R": R_m.tolist(), "T": T_m.tolist()},
        },
    })

    Ra, ta = _compose_board_in_rig(sol.cameras["cam_a"], Ra_cb, ta_cb)
    Rb, tb = _compose_board_in_rig(sol.cameras["cam_b"], Rb_cb, tb_cb)

    np.testing.assert_allclose(ta, t_rig_board, atol=1e-9)
    np.testing.assert_allclose(tb, t_rig_board, atol=1e-9)
    np.testing.assert_allclose(Ra, R_rig_board, atol=1e-9)
    np.testing.assert_allclose(Rb, R_rig_board, atol=1e-9)
    # And the parsed camera center matches ground truth.
    np.testing.assert_allclose(sol.cameras["cam_b"].t_in_rig, C_b, atol=1e-9)
