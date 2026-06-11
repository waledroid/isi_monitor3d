"""``calibrate.py`` — CLI, floor-anchor math, assemble contract.

Multical is exercised by a stubbed binary fixture so the test suite stays
hermetic. Real Multical runs are covered by the dev rig, not unit tests.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

from calibration.calibrate import (
    REPROJECTION_RMS_HARD_LIMIT_PX,
    CharucoBoardSpec,
    FloorAnchor,
    IntrinsicsResult,
    _average_rotation,
    assemble_calibration,
    build_parser,
    calibrate_intrinsics,
    compose_camera_in_world,
    find_multical_binary,
    run_multical_calibration,
)
from calibration.multical_io import UNKNOWN_RMS_PX, CameraInRig, MultiCalSolution

# ---------- fixtures ----------


R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])
"""Realistic ceiling-mounted camera rotation: looking straight down at the floor."""


def _camera_in_rig(cam_id: str, t: np.ndarray, rms: float = 0.3) -> CameraInRig:
    return CameraInRig(
        camera_id=cam_id,
        image_size_wh=(1920, 1080),
        K=np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        R_in_rig=R_LOOK_DOWN,
        t_in_rig=t,
        rms_px=rms,
    )


def _solution(rms_a: float = 0.3, rms_b: float = 0.4) -> MultiCalSolution:
    return MultiCalSolution(
        master_camera="cam_a",
        cameras={
            "cam_a": _camera_in_rig("cam_a", np.array([0.0, 0.0, 3.5]), rms=rms_a),
            "cam_b": _camera_in_rig("cam_b", np.array([4.0, 0.0, 3.5]), rms=rms_b),
        },
    )


def _identity_anchor() -> FloorAnchor:
    return FloorAnchor(
        method="charuco_floor",
        note="identity (test only)",
        R_world_from_rig=np.eye(3),
        t_world_from_rig=np.zeros(3),
    )


# ---------- CLI parser ----------


def test_parser_calibrate_all_subcommand() -> None:
    parser = build_parser()
    ns = parser.parse_args(
        [
            "calibrate-all",
            "--camera-dir", "cam_a=/a",
            "--camera-dir", "cam_b=/b",
            "--floor-shot", "cam_a=/a/floor.jpg",
            "--floor-shot", "cam_b=/b/floor.jpg",
            "--work-dir", "/tmp/work",
            "--output", "/tmp/calibration.json",
        ]
    )
    assert ns.command == "calibrate-all"
    assert ns.camera_dir == ["cam_a=/a", "cam_b=/b"]
    assert ns.allow_unknown_rms is False


def test_parser_intrinsics_single_cam_subcommand() -> None:
    parser = build_parser()
    ns = parser.parse_args(
        [
            "intrinsics-single-cam",
            "--camera-id", "cam_a",
            "--images-dir", "/x",
            "--output", "/y.json",
        ]
    )
    assert ns.command == "intrinsics-single-cam"
    assert ns.camera_id == "cam_a"


# ---------- find_multical_binary ----------


def test_find_multical_binary_in_venv(tmp_path: Path) -> None:
    venv = tmp_path / ".venv-multical"
    (venv / "bin").mkdir(parents=True)
    bin_path = venv / "bin" / "multical"
    bin_path.touch()
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
    found = find_multical_binary(venv_dir=venv)
    assert found == bin_path


def test_find_multical_binary_missing_no_fallback(tmp_path: Path) -> None:
    venv = tmp_path / ".venv-multical"
    with pytest.raises(RuntimeError, match=r"setup_multical\.sh"):
        find_multical_binary(venv_dir=venv, allow_path_fallback=False)


# ---------- floor anchor math ----------


def test_compose_camera_in_world_identity_anchor() -> None:
    cam = _camera_in_rig("cam_a", np.array([1.0, 2.0, 3.0]))
    R_w, t_w = compose_camera_in_world(cam, _identity_anchor())
    np.testing.assert_allclose(R_w, R_LOOK_DOWN)
    np.testing.assert_allclose(t_w, [1.0, 2.0, 3.0])


def test_compose_camera_in_world_translation_anchor() -> None:
    """A world-from-rig translation must shift every camera by the same vector."""
    cam = _camera_in_rig("cam_a", np.array([1.0, 2.0, 3.0]))
    anchor = FloorAnchor(
        method="charuco_floor",
        note="t",
        R_world_from_rig=np.eye(3),
        t_world_from_rig=np.array([10.0, 0.0, 0.0]),
    )
    _R, t_w = compose_camera_in_world(cam, anchor)
    np.testing.assert_allclose(t_w, [11.0, 2.0, 3.0])


def test_average_rotation_identity_self() -> None:
    out = _average_rotation([np.eye(3), np.eye(3), np.eye(3)])
    np.testing.assert_allclose(out, np.eye(3), atol=1e-10)


def test_average_rotation_is_orthonormal() -> None:
    """Even with noisy inputs, the averaged rotation must be a valid SO(3) element."""
    rng = np.random.default_rng(0)
    base = np.eye(3)
    noisy = []
    for _ in range(5):
        delta = rng.normal(scale=0.01, size=(3, 3))
        noisy.append(base + delta)
    out = _average_rotation(noisy)
    np.testing.assert_allclose(out @ out.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(out) == pytest.approx(1.0, abs=1e-10)


# ---------- assemble_calibration ----------


def test_assemble_produces_valid_calibration() -> None:
    cal = assemble_calibration(_solution(), _identity_anchor())
    assert set(cal.cameras) == {"cam_a", "cam_b"}
    for cam in cal.cameras.values():
        assert np.asarray(cam.K).shape == (3, 3)
        assert np.asarray(cam.H).shape == (3, 3)
        assert np.asarray(cam.P).shape == (3, 4)
        assert cam.reprojection_rms_px <= REPROJECTION_RMS_HARD_LIMIT_PX


def test_assemble_refuses_bad_rms() -> None:
    sol = _solution(rms_a=0.8)  # over the 0.5 px hard limit
    with pytest.raises(RuntimeError, match="exceeds hard limit"):
        assemble_calibration(sol, _identity_anchor())


def test_assemble_refuses_unknown_rms_by_default() -> None:
    sol = MultiCalSolution(
        master_camera="cam_a",
        cameras={"cam_a": _camera_in_rig("cam_a", np.array([0.0, 0.0, 3.5]), rms=UNKNOWN_RMS_PX)},
    )
    with pytest.raises(RuntimeError, match="not parsed"):
        assemble_calibration(sol, _identity_anchor())


def test_assemble_allow_unknown_rms() -> None:
    sol = MultiCalSolution(
        master_camera="cam_a",
        cameras={"cam_a": _camera_in_rig("cam_a", np.array([0.0, 0.0, 3.5]), rms=UNKNOWN_RMS_PX)},
    )
    cal = assemble_calibration(sol, _identity_anchor(), allow_unknown_rms=True)
    assert cal.cameras["cam_a"].reprojection_rms_px == UNKNOWN_RMS_PX


def test_assembled_calibration_roundtrips_through_json() -> None:
    cal = assemble_calibration(_solution(), _identity_anchor())
    text = cal.to_json()
    parsed = json.loads(text)
    assert parsed["version"] == 1
    assert "cam_a" in parsed["cameras"]


# ---------- IntrinsicsResult debug API ----------


def test_charuco_board_spec_constructs_opencv_objects() -> None:
    spec = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.04, marker_length_m=0.03)
    assert spec.board() is not None
    assert spec.aruco_dict() is not None


def test_intrinsics_result_dataclass() -> None:
    r = IntrinsicsResult(K=np.eye(3), D=np.zeros(5), image_size_wh=(100, 100), reprojection_rms_px=0.2)
    assert r.reprojection_rms_px == 0.2


# ---------- run_multical_calibration with a stub binary ----------


def _write_stub_multical(tmp_path: Path) -> Path:
    """A fake `multical` binary that writes a fixture JSON and an RMS log."""
    fixture = {
        "cameras": {
            "cam_a": {
                "model": "standard",
                "image_size": [1920, 1080],
                "K": [[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]],
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "cam_b": {
                "model": "standard",
                "image_size": [1920, 1080],
                "K": [[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]],
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "camera_poses": {
            "cam_a": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "T": [0.0, 0.0, 0.0]},
            "cam_b_to_cam_a": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [4.0, 0.0, 0.0],
            },
        },
        "image_sets": {"rgb": []},
    }
    stub = tmp_path / "multical-stub.py"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"FIXTURE = {json.dumps(fixture)}\n"
        "args = sys.argv\n"
        'out_idx = args.index("--output_path") + 1\n'
        'name_idx = args.index("--name") + 1\n'
        "out = Path(args[out_idx]) / f'{args[name_idx]}.json'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(json.dumps(FIXTURE))\n"
        'print("INFO - cam_a - RMS: 0.32 quantiles: [0 0.1 0.2 0.3 0.9]")\n'
        'print("INFO - cam_b - RMS: 0.41 quantiles: [0 0.1 0.3 0.4 1.0]")\n'
    )
    stub.chmod(0o755)
    # Wrap with a bash launcher so `subprocess.run([stub])` works regardless of shebang resolution.
    launcher = tmp_path / "multical"
    launcher.write_text(f"#!/usr/bin/env bash\nexec {shutil.which('python3')} {stub} \"$@\"\n")
    launcher.chmod(0o755)
    return launcher


def test_run_multical_calibration_with_stub(tmp_path: Path) -> None:
    """Stub Multical → orchestrator parses JSON + RMS log → MultiCalSolution."""
    stub = _write_stub_multical(tmp_path)
    img_a = tmp_path / "cam_a_imgs"
    img_b = tmp_path / "cam_b_imgs"
    img_a.mkdir()
    img_b.mkdir()
    work = tmp_path / "work"

    board = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.04, marker_length_m=0.03)
    sol = run_multical_calibration(
        image_dirs_by_camera={"cam_a": img_a, "cam_b": img_b},
        board=board,
        work_dir=work,
        multical_binary=stub,
        name="calibration",
    )
    assert set(sol.camera_ids) == {"cam_a", "cam_b"}
    assert sol.cameras["cam_a"].rms_px == pytest.approx(0.32)
    assert sol.cameras["cam_b"].rms_px == pytest.approx(0.41)
    # boards.yaml and the symlinked image dirs should have been staged
    assert (work / "boards.yaml").exists()
    assert (work / "images" / "cam_a").exists()
    assert (work / "images" / "cam_b").exists()


def test_run_multical_calibration_propagates_subprocess_failure(tmp_path: Path) -> None:
    """A failing Multical subprocess must surface CalledProcessError, not silent fallback."""
    failing = tmp_path / "multical"
    failing.write_text("#!/usr/bin/env bash\nexit 17\n")
    failing.chmod(0o755)

    img = tmp_path / "imgs"
    img.mkdir()
    board = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.04, marker_length_m=0.03)
    with pytest.raises(subprocess.CalledProcessError):
        run_multical_calibration(
            image_dirs_by_camera={"cam_a": img},
            board=board,
            work_dir=tmp_path / "work",
            multical_binary=failing,
        )


# ---------- single-cam intrinsics still callable as debug ----------


def test_calibrate_intrinsics_rejects_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        calibrate_intrinsics(empty, CharucoBoardSpec(8, 6, 0.04, 0.03))
