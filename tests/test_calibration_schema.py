"""Round-trip and version-gate tests for ``calibration/schema.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CalibrationVersionError,
)

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "calibration" / "calibration.example.json"


def test_example_file_parses() -> None:
    cal = CalibrationFile.read(EXAMPLE_PATH)
    assert cal.version == CALIBRATION_VERSION
    assert set(cal.cameras) == {"cam_a", "cam_b"}
    for cam in cal.cameras.values():
        assert cam.K_np().shape == (3, 3)
        assert cam.H_np().shape == (3, 3)
        assert cam.P_np().shape == (3, 4)
        assert cam.t_np().shape == (3,)
        assert cam.image_size_wh == (1920, 1080)


def test_roundtrip(tmp_path: Path) -> None:
    cal = CalibrationFile.read(EXAMPLE_PATH)
    out = tmp_path / "calibration.json"
    cal.write(out)
    reloaded = CalibrationFile.read(out)
    assert reloaded.version == cal.version
    assert set(reloaded.cameras) == set(cal.cameras)
    for cam_id, cam in cal.cameras.items():
        rcam = reloaded.cameras[cam_id]
        assert rcam.K == cam.K
        assert rcam.D == cam.D
        assert rcam.t == cam.t
        assert rcam.reprojection_rms_px == cam.reprojection_rms_px


def test_version_gate_rejects_older(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE_PATH.read_text())
    payload["version"] = 0
    bad = tmp_path / "old.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(CalibrationVersionError):
        CalibrationFile.read(bad)
