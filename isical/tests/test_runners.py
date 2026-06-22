"""Phase runners with the Multical backend monkeypatched."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from isical.core import runners
from isical.core.project import CameraSpec, create_project


def _proj(tmp_path, two=True):
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a")}
    if two:
        cams["cam_b"] = CameraSpec(id="cam_b", url="rtsp://x/b")
    return create_project(tmp_path / "data", "rig", cams)


def _jpgs(d, n):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"s{i:02d}.jpg"), np.full((40, 60, 3), 100, np.uint8))


def test_run_intrinsic(tmp_path, monkeypatch):
    import calibration.calibrate as cal
    pdir = _proj(tmp_path)
    _jpgs(pdir / "intrinsic" / "cam_a", 10)              # cam_a ready, cam_b not

    def _multical_intr(dirs, board, work, **k):
        assert list(dirs) == ["cam_a"]                  # only the ready camera
        out = work / "intrinsic.json"
        out.write_text("{}")
        return out

    class _Res:
        reprojection_rms_px = 0.21
    monkeypatch.setattr(cal, "run_multical_intrinsics", _multical_intr)
    monkeypatch.setattr(cal, "calibrate_intrinsics", lambda d, b: _Res())

    out = runners.run_intrinsic(pdir)
    assert out["cameras_solved"] == ["cam_a"]
    assert out["rms"]["cam_a"] == 0.21
    assert (pdir / "work" / "intrinsic.json").exists()


def test_run_intrinsic_needs_min_shots(tmp_path):
    pdir = _proj(tmp_path)
    _jpgs(pdir / "intrinsic" / "cam_a", 3)               # below the floor
    with pytest.raises(ValueError):
        runners.run_intrinsic(pdir)


def test_run_extrinsic_then_export_install(tmp_path, monkeypatch):
    import calibration.calibrate as cal
    pdir = _proj(tmp_path)
    (pdir / "work" / "intrinsic.json").write_text("{}")  # phase-1 artifact present
    for cid in ("cam_a", "cam_b"):
        _jpgs(pdir / "extrinsic" / cid, 5)
        cv2.imwrite(str(pdir / "floor" / f"{cid}.jpg"), np.full((40, 60, 3), 100, np.uint8))

    class _Cam:
        reprojection_rms_px = 0.33

    class _Calib:
        def __init__(self):
            self.cameras = {"cam_a": _Cam(), "cam_b": _Cam()}

        def write(self, path):
            path.write_text(json.dumps({"cameras": {
                "cam_a": {"reprojection_rms_px": 0.33},
                "cam_b": {"reprojection_rms_px": 0.33}}}))

    monkeypatch.setattr(cal, "run_multical_extrinsics", lambda *a, **k: object())
    monkeypatch.setattr(cal, "estimate_floor_anchor_charuco", lambda *a, **k: object())
    monkeypatch.setattr(cal, "assemble_calibration", lambda *a, **k: _Calib())

    out = runners.run_extrinsic(pdir)
    assert out["rms"] == {"cam_a": 0.33, "cam_b": 0.33}
    assert (pdir / "calibration.json").exists()

    # export + install → copies to the (tmp) mode2 path from env
    res = runners.run_export(pdir, install=True)
    assert res["installed"] is True
    from isical.config import Settings
    assert Settings().mode2_calibration_path.exists()


def test_run_extrinsic_requires_intrinsic(tmp_path):
    pdir = _proj(tmp_path)
    with pytest.raises(ValueError):
        runners.run_extrinsic(pdir)


def test_phase_status(tmp_path):
    pdir = _proj(tmp_path)
    _jpgs(pdir / "intrinsic" / "cam_a", 4)
    st = runners.phase_status(pdir)
    assert st["mode2"] is True
    assert st["intrinsic_counts"]["cam_a"] == 4
    assert st["intrinsic_done"] is False and st["extrinsic_done"] is False
