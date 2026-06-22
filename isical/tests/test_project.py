"""Calibration project model + board adapters."""

from __future__ import annotations

import pytest

from isical.core.project import (
    BoardSpec,
    CameraSpec,
    aprilgrid_target,
    charuco_spec,
    create_project,
    delete_project,
    list_projects,
    load_project,
)


def test_create_load_list_delete(tmp_path):
    data = tmp_path / "data"
    cams = {"cam_a": CameraSpec(id="cam_a", type="rtsp", url="rtsp://x/a")}
    pdir = create_project(data, "rig1", cams)
    assert (pdir / "calib.yaml").exists()
    assert (pdir / "intrinsic" / "cam_a").is_dir()
    assert (pdir / "extrinsic" / "cam_a").is_dir()
    assert (pdir / "floor").is_dir() and (pdir / "work").is_dir()
    cfg = load_project(pdir)
    assert cfg.name == "rig1" and cfg.configured_cameras() == ["cam_a"]
    assert cfg.is_mode2() is False
    assert list_projects(data) == ["rig1"]
    with pytest.raises(FileExistsError):
        create_project(data, "rig1", cams)
    delete_project(data, "rig1")
    assert list_projects(data) == []


def test_mode2_detection(tmp_path):
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig2", cams)
    cfg = load_project(pdir)
    assert cfg.configured_cameras() == ["cam_a", "cam_b"] and cfg.is_mode2()


def test_empty_camb_is_mode1(tmp_path):
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="")}     # blank source
    pdir = create_project(tmp_path / "data", "rig3", cams)
    assert load_project(pdir).configured_cameras() == ["cam_a"]


def test_board_adapters_match_print():
    spec = charuco_spec(BoardSpec())
    assert (spec.squares_x, spec.squares_y) == (5, 7)
    assert spec.square_length_m == 0.035 and spec.dict_name == "DICT_5X5_50"
    target = aprilgrid_target(BoardSpec())
    assert len(target) == 6                              # 6 disjoint AprilGrids
    ids = sorted(b.start_id for b in target.values())
    assert len(set(ids)) == 6                            # disjoint start ids
