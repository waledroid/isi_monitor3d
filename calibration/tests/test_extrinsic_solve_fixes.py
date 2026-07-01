"""Regression tests for the four bugs that blocked the isical extrinsic solve.

Before these fixes, ``run_extrinsic`` on a real 6x(1x2)-AprilGrid rig failed in
four distinct ways, each masking the next:

1. **Image staging** — captures are named ``<cam>_<NNN>.jpg``; Multical pairs
   synchronized frames by *identical basename* across camera dirs, so the
   ``cam_a_``/``cam_b_`` prefixes gave an empty intersection → "0 matching
   images" → ``IndexError``. Fixed by ``_stage_image_dirs(match_names=True)``.
2. **Pose gate** — Multical's AprilGrid defaults (``min_points=12, min_rows=2``)
   are unreachable for a 1-tag-wide, 2-tag board (max 8 corners, 1 column), so
   every board pose was dropped → no stereo overlap. Fixed by board-size-aware
   ``AprilGridBoardSpec.pose_gates()`` emitted into the boards YAML.
3. **Output location** — Multical 0.4.0 exports to ``image_path`` not
   ``output_path``; the extrinsic reader only checked ``work_dir``.
4. **RMS gate** — the joint BA logs one *overall* RMS (no per-camera lines), so
   the strict 0.5 px intrinsic gate raised "RMS not parsed". Fixed by
   ``parse_joint_rms_from_log`` + the KPI-aligned extrinsic limit.

(The numpy<1.24 pin for Multical's ``np.bool`` is enforced in
``setup_multical.sh`` and can't be unit-tested here.)
"""
from __future__ import annotations

import numpy as np
import pytest

from calibration.calibrate import (
    EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX,
    REPROJECTION_RMS_HARD_LIMIT_PX,
    AprilGridBoardSpec,
    FloorAnchor,
    _stage_image_dirs,
    assemble_calibration,
)
from calibration.multical_io import (
    CameraInRig,
    MultiCalSolution,
    parse_joint_rms_from_log,
)

# --------------------------------------------------------------------------
# Fix 2 — AprilGrid pose gates are board-size-aware and satisfiable
# --------------------------------------------------------------------------


def test_pose_gates_small_1x2_board_is_solvable():
    """A 1x2 board (2 tags, 8 corners) must get gates it can actually meet."""
    spec = AprilGridBoardSpec(tags_x=1, tags_y=2)
    min_rows, min_points = spec.pose_gates()
    # 8 corners max — the gate must be <= 8 and require >1 tag (so >4).
    assert 4 < min_points <= spec.tag_count() * 4
    # 1-wide board never has 2 unique columns → min_rows must be <= 1 here.
    assert min_rows <= min(spec.tags_x, spec.tags_y)


def test_pose_gates_large_board_requires_two_rows_two_tags():
    """A big grid keeps 2-row spread and caps the point gate at two full tags."""
    spec = AprilGridBoardSpec(tags_x=6, tags_y=6)
    assert spec.pose_gates() == (2, 8)


def test_pose_gates_emitted_into_yaml_block():
    spec = AprilGridBoardSpec(tags_x=1, tags_y=2)
    block = spec.multical_yaml_block("april_0")
    min_rows, min_points = spec.pose_gates()
    assert f"min_rows: {min_rows}" in block
    assert f"min_points: {min_points}" in block


def test_pose_gate_overrides_are_honoured():
    spec = AprilGridBoardSpec(tags_x=1, tags_y=2, min_rows=1, min_points=5)
    assert spec.pose_gates() == (1, 5)


# --------------------------------------------------------------------------
# Fix 1 — matched-name staging makes pairs share a basename
# --------------------------------------------------------------------------


def test_stage_image_dirs_match_names_strips_camera_prefix(tmp_path):
    src_a = tmp_path / "extrinsic" / "cam_a"
    src_b = tmp_path / "extrinsic" / "cam_b"
    src_a.mkdir(parents=True)
    src_b.mkdir(parents=True)
    for i in range(3):
        (src_a / f"cam_a_{i:03d}.jpg").write_bytes(b"a")
        (src_b / f"cam_b_{i:03d}.jpg").write_bytes(b"b")

    root = _stage_image_dirs(
        {"cam_a": src_a, "cam_b": src_b}, tmp_path / "work", match_names=True)

    names_a = {p.name for p in (root / "cam_a").glob("*.jpg")}
    names_b = {p.name for p in (root / "cam_b").glob("*.jpg")}
    # Multical intersects basenames across cameras — must be non-empty now.
    assert names_a == names_b == {"000.jpg", "001.jpg", "002.jpg"}
    # Symlinks still resolve to the real captures.
    assert (root / "cam_a" / "000.jpg").resolve().exists()


def test_stage_image_dirs_default_keeps_whole_dir_symlink(tmp_path):
    src_a = tmp_path / "intrinsic" / "cam_a"
    src_a.mkdir(parents=True)
    (src_a / "cam_a_000.jpg").write_bytes(b"a")
    root = _stage_image_dirs({"cam_a": src_a}, tmp_path / "work")
    # Per-camera intrinsic path: the dir itself is symlinked, names untouched.
    assert (root / "cam_a").is_symlink()
    assert {p.name for p in (root / "cam_a").glob("*.jpg")} == {"cam_a_000.jpg"}


# --------------------------------------------------------------------------
# Fix 4 — joint-RMS parsing + KPI-aligned extrinsic gate
# --------------------------------------------------------------------------


def test_parse_joint_rms_returns_final_value():
    log = (
        "INFO - Initialisation reprojection RMS=5.049, n=768\n"
        "INFO - Adjust_outliers 0: reprojection RMS=5.049, n=768\n"
        "INFO - Adjust_outliers end: reprojection RMS=1.621 (1.621), n=768\n"
    )
    assert parse_joint_rms_from_log(log) == pytest.approx(1.621)


def test_parse_joint_rms_none_without_joint_line():
    assert parse_joint_rms_from_log("INFO - cam_a - RMS: 0.31 quantiles: []") is None


def _minimal_solution(rms_px: float) -> MultiCalSolution:
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    D = np.zeros(5)
    cams = {
        "cam_a": CameraInRig("cam_a", (1920, 1080), K, D,
                             np.eye(3), np.zeros(3), rms_px=rms_px),
        "cam_b": CameraInRig("cam_b", (1920, 1080), K, D,
                             np.eye(3), np.array([0.5, 0, 0]), rms_px=rms_px),
    }
    return MultiCalSolution(master_camera="cam_a", cameras=cams)


def _lookdown_anchor() -> FloorAnchor:
    return FloorAnchor(
        method="charuco",
        note="test",
        R_world_from_rig=np.diag([1.0, -1.0, -1.0]),
        t_world_from_rig=np.array([0.0, 0.0, 3.0]),
    )


def test_extrinsic_limit_is_looser_than_intrinsic():
    assert EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX > REPROJECTION_RMS_HARD_LIMIT_PX


def test_joint_rms_1p6_rejected_by_strict_but_accepted_by_extrinsic_gate():
    sol = _minimal_solution(1.621)  # a realistic joint-BA RMS
    anchor = _lookdown_anchor()
    # Strict intrinsic gate (default 0.5 px) must refuse it.
    with pytest.raises(RuntimeError, match="exceeds"):
        assemble_calibration(sol, anchor)
    # The extrinsic KPI gate (<=2 px) accepts it and yields full K/D/R/t/H/P.
    calib = assemble_calibration(
        sol, anchor, rms_limit_px=EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX)
    assert set(calib.cameras) == {"cam_a", "cam_b"}
    for cam in calib.cameras.values():
        assert np.isfinite(cam.H_np()).all()
        assert np.isfinite(cam.P_np()).all()


def test_extrinsic_gate_still_rejects_truly_bad_rms():
    sol = _minimal_solution(9.0)  # way over even the 2 px KPI
    with pytest.raises(RuntimeError, match="exceeds"):
        assemble_calibration(
            sol, _lookdown_anchor(),
            rms_limit_px=EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX)
