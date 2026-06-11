"""``CameraRig`` API surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backbone.shared import CameraRig

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "calibration" / "calibration.example.json"


def test_load_example_calibration() -> None:
    rig = CameraRig.from_file(EXAMPLE_PATH)
    assert set(rig.camera_ids) == {"cam_a", "cam_b"}
    assert "cam_a" in rig
    assert "ghost" not in rig


def test_camera_view_matrices_shapes_and_types() -> None:
    rig = CameraRig.from_file(EXAMPLE_PATH)
    cam = rig["cam_a"]
    assert cam.K.shape == (3, 3)
    assert cam.D.shape[0] >= 4
    assert cam.R.shape == (3, 3)
    assert cam.t.shape == (3,)
    assert cam.H.shape == (3, 3)
    assert cam.P.shape == (3, 4)
    assert cam.K.dtype == np.float64


def test_arrays_are_immutable() -> None:
    rig = CameraRig.from_file(EXAMPLE_PATH)
    cam = rig["cam_a"]
    with pytest.raises(ValueError):
        cam.K[0, 0] = 0.0


def test_unknown_camera_id_helpful_error() -> None:
    rig = CameraRig.from_file(EXAMPLE_PATH)
    with pytest.raises(KeyError) as exc:
        rig["cam_missing"]
    assert "cam_missing" in str(exc.value)
    assert "cam_a" in str(exc.value)
