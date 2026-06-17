"""AprilGrid extrinsic target + two-stage 2-camera calibration + board generation.

Hermetic: a stubbed `multical` binary handles the `intrinsic`/`calibrate`/`boards`
subcommands and records its argv, so we assert the right flags are passed without
a real Multical or rig. Real runs are covered on the dev rig.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from calibration.calibrate import (
    AprilGridBoardSpec,
    CharucoBoardSpec,
    build_parser,
    generate_board_images,
    make_aprilgrid_target,
    run_multical_extrinsics,
    run_multical_intrinsics,
    run_multical_vis,
    write_boards_yaml,
)

CHARUCO = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.04, marker_length_m=0.03)


# ---------- board specs + boards.yaml ----------


def test_aprilgrid_target_has_disjoint_start_ids():
    target = make_aprilgrid_target(6, AprilGridBoardSpec(tags_x=6, tags_y=6))
    assert list(target) == [f"april_{i}" for i in range(6)]
    # 6x6 = 36 tags/board → start_ids step by 36, never overlapping
    assert [b.start_id for b in target.values()] == [0, 36, 72, 108, 144, 180]


def test_aprilgrid_yaml_block_fields():
    block = AprilGridBoardSpec(tags_x=6, tags_y=6, tag_length_m=0.06,
                               tag_spacing=0.3, start_id=72).multical_yaml_block("april_2")
    assert "april_2:" in block
    assert "_type_: aprilgrid" in block          # Multical's key is _type_, not type
    assert "size: [6, 6]" in block
    assert "tag_length: 0.06" in block
    assert "tag_family: t36h11" in block
    assert "start_id: 72" in block
    assert "border_bits" not in block            # not in Multical's AprilConfig schema


def test_write_boards_yaml_is_valid_and_complete(tmp_path):
    boards = {"charuco_main": CHARUCO, **make_aprilgrid_target(6)}
    path = write_boards_yaml(boards, tmp_path / "boards.yaml")
    doc = yaml.safe_load(path.read_text())
    assert set(doc["boards"]) == {"charuco_main", *[f"april_{i}" for i in range(6)]}
    assert doc["boards"]["charuco_main"]["_type_"] == "charuco"
    # Multical re-adds the DICT_ prefix, so it must be stripped in the yaml
    assert doc["boards"]["charuco_main"]["aruco_dict"] == "5X5_50"
    assert doc["boards"]["april_3"]["_type_"] == "aprilgrid"
    assert doc["boards"]["april_3"]["start_id"] == 108


# ---------- stub multical ----------


def _write_stub_multical(tmp_path: Path) -> Path:
    """Fake `multical` for intrinsic/calibrate/boards. Records argv to
    <output_path or write dir>/_invoked_<subcommand>.json for assertions."""
    fixture = {
        "cameras": {
            "cam_a": {"image_size": [1920, 1080],
                      "K": [[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1.0]],
                      "dist": [0.0, 0, 0, 0, 0]},
            "cam_b": {"image_size": [1920, 1080],
                      "K": [[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1.0]],
                      "dist": [0.0, 0, 0, 0, 0]},
        },
        "camera_poses": {
            "cam_a": {"R": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "T": [0.0, 0, 0]},
            "cam_b_to_cam_a": {"R": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "T": [4.0, 0, 0]},
        },
    }
    stub = tmp_path / "multical-stub.py"
    stub.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"FIX = {json.dumps(fixture)}\n"
        "a = sys.argv\n"
        "cmd = a[1]\n"
        "def opt(k, d=None):\n"
        "    return a[a.index(k)+1] if k in a else d\n"
        "outp = opt('--output_path') or opt('--write') or '.'\n"
        "Path(outp).mkdir(parents=True, exist_ok=True)\n"
        "(Path(outp)/f'_invoked_{cmd}.json').write_text(json.dumps(a))\n"
        "if cmd == 'intrinsic':\n"
        "    (Path(outp)/'intrinsic.json').write_text(json.dumps(FIX['cameras']))\n"
        "elif cmd == 'calibrate':\n"
        "    name = opt('--name', 'calibration')\n"
        "    (Path(outp)/f'{name}.json').write_text(json.dumps(FIX))\n"
        "    print('INFO - cam_a - RMS: 0.30 quantiles: [0 0.1 0.2 0.3 0.9]')\n"
        "    print('INFO - cam_b - RMS: 0.38 quantiles: [0 0.1 0.3 0.4 1.0]')\n"
    )
    stub.chmod(0o755)
    launcher = tmp_path / "multical"
    launcher.write_text(f"#!/usr/bin/env bash\nexec {shutil.which('python3')} {stub} \"$@\"\n")
    launcher.chmod(0o755)
    return launcher


# ---------- generate_board_images ----------


def test_generate_board_images_invokes_multical(tmp_path):
    stub = _write_stub_multical(tmp_path)
    out = tmp_path / "boards_out"
    boards = {"charuco_main": CHARUCO, **make_aprilgrid_target(6)}
    generate_board_images(boards, out, paper_size="A4", pixels_mm=12, multical_binary=stub)
    assert (out / "boards.yaml").exists()
    argv = json.loads((out / "_invoked_boards.json").read_text())
    assert argv[1] == "boards"
    assert "--paper_size" in argv and argv[argv.index("--paper_size") + 1] == "A4"
    assert argv[argv.index("--pixels_mm") + 1] == "12"
    assert argv[argv.index("--margin_mm") + 1] == "20"          # default border


def test_generate_board_images_margin_override(tmp_path):
    stub = _write_stub_multical(tmp_path)
    out = tmp_path / "bo"
    # a board that nearly fills the sheet needs margin_mm=0 to satisfy Multical
    generate_board_images({"charuco_main": CHARUCO}, out, margin_mm=0, multical_binary=stub)
    argv = json.loads((out / "_invoked_boards.json").read_text())
    assert argv[argv.index("--margin_mm") + 1] == "0"


# ---------- two-stage Multical wrappers ----------


def test_run_multical_intrinsics_returns_intrinsic_json(tmp_path):
    stub = _write_stub_multical(tmp_path)
    a, b = tmp_path / "ia", tmp_path / "ib"
    a.mkdir()
    b.mkdir()
    work = tmp_path / "work"
    out = run_multical_intrinsics({"cam_a": a, "cam_b": b}, CHARUCO, work, multical_binary=stub)
    assert out.exists() and out.name == "intrinsic.json"
    argv = json.loads((work / "_invoked_intrinsic.json").read_text())
    assert argv[1] == "intrinsic"


def test_run_multical_extrinsics_fixes_intrinsics(tmp_path):
    stub = _write_stub_multical(tmp_path)
    a, b = tmp_path / "ea", tmp_path / "eb"
    a.mkdir()
    b.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    intr = work / "intrinsic.json"
    intr.write_text("{}")
    target = make_aprilgrid_target(6)
    sol = run_multical_extrinsics({"cam_a": a, "cam_b": b}, target, work, intr, multical_binary=stub)
    assert set(sol.camera_ids) == {"cam_a", "cam_b"}
    assert sol.cameras["cam_a"].rms_px == pytest.approx(0.30)
    argv = json.loads((work / "_invoked_calibrate.json").read_text())
    assert argv[argv.index("--fix_intrinsic") + 1] == "True"   # bool value, not bare flag
    assert argv[argv.index("--calibration") + 1] == str(intr)
    assert (work / "extrinsic_boards.yaml").exists()
    assert "--vis" not in argv                                 # off by default


def test_run_multical_extrinsics_vis_flag(tmp_path):
    stub = _write_stub_multical(tmp_path)
    a, b = tmp_path / "ea", tmp_path / "eb"
    a.mkdir()
    b.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    intr = work / "intrinsic.json"
    intr.write_text("{}")
    run_multical_extrinsics({"cam_a": a, "cam_b": b}, make_aprilgrid_target(6),
                            work, intr, multical_binary=stub, vis=True)
    argv = json.loads((work / "_invoked_calibrate.json").read_text())
    assert argv[argv.index("--vis") + 1] == "True"             # opens the 3D viewer


def test_run_multical_vis_builds_command(tmp_path, monkeypatch):
    """`vis` re-opens a saved workspace via `multical vis --workspace_file`,
    inheriting the terminal (no capture — the viewer is interactive)."""
    import calibration.calibrate as cal
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(cal.subprocess, "run", fake_run)
    ws = tmp_path / "work"
    run_multical_vis(ws, multical_binary=tmp_path / "multical")
    assert seen["cmd"] == [str(tmp_path / "multical"), "vis", "--workspace_file", str(ws)]
    assert "capture_output" not in seen["kwargs"]              # interactive → inherit terminal


# ---------- CLI parsing ----------


def test_cli_parses_gen_boards_and_calibrate_2cam():
    p = build_parser()
    g = p.parse_args(["gen-boards", "--output-dir", "/tmp/b", "--n-boards", "6"])
    assert g.command == "gen-boards" and g.n_boards == 6 and g.paper_size == "A4"
    c = p.parse_args([
        "calibrate-2cam",
        "--intrinsic-dir", "cam_a=/i/a", "--intrinsic-dir", "cam_b=/i/b",
        "--extrinsic-dir", "cam_a=/e/a", "--extrinsic-dir", "cam_b=/e/b",
        "--floor-shot", "cam_a=/f/a.jpg", "--floor-shot", "cam_b=/f/b.jpg",
        "--work-dir", "/tmp/w", "--output", "/tmp/calibration.json",
    ])
    assert c.command == "calibrate-2cam"
    assert c.intrinsic_dir == ["cam_a=/i/a", "cam_b=/i/b"]
    assert c.tag_family == "t36h11"
    assert c.vis is False                                       # default off


def test_cli_parses_vis_flags():
    p = build_parser()
    # --vis on both calibrate commands
    a = p.parse_args(["calibrate-all", "--camera-dir", "cam_a=/a", "--floor-shot",
                      "cam_a=/f.jpg", "--work-dir", "/w", "--output", "/o.json", "--vis"])
    assert a.vis is True
    # standalone vis command
    v = p.parse_args(["vis", "--workspace", "/tmp/work"])
    assert v.command == "vis" and v.workspace == "/tmp/work"
