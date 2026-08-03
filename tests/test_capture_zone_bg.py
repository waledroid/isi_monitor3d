"""Tests for tools/capture_zone_bg.py (hermetic — no cameras, no GPU)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "capture_zone_bg", _ROOT / "tools" / "capture_zone_bg.py")
czb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(czb)


def test_scale_box_matches_zone_scoped_detector():
    # Same formula as ZoneScopedDetector.detect (zone_scope.py:456-459):
    # int() floor on the min corner, ceil on the max corner, clamped.
    box = (100, 200, 900, 1000)
    fx0, fy0, fx1, fy1, sx, sy = czb.scale_box(box, (1920, 1080), (1280, 720))
    assert sx == 1280 / 1920 and sy == 720 / 1080
    assert fx0 == max(0, int(100 * sx))
    assert fy0 == max(0, int(200 * sy))
    assert fx1 == min(1280, math.ceil(900 * sx))
    assert fy1 == min(720, math.ceil(1000 * sy))


def test_scale_box_identity_when_sizes_match():
    assert czb.scale_box((10, 20, 30, 40), (640, 480), (640, 480))[:4] == (10, 20, 30, 40)


def test_fill_crop_grays_outside_polygon_and_copies():
    img = np.full((100, 100, 3), 200, np.uint8)
    poly = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float64)
    out = czb.fill_crop(img, (poly, 4.0), 1.0, 1.0, 0, 0)
    assert (out[0, 0] == czb._FILL_GRAY).all()      # corner: outside → gray
    assert (out[50, 50] == 200).all()               # center: inside → preserved
    assert (out[50, 72] == 200).all()               # dilation keeps the edge band
    assert (img[0, 0] == 200).all()                 # original frame untouched


def test_fill_crop_respects_crop_origin_offset():
    # Polygon at frame px (130..170); crop starts at fx0=100, fy0=100.
    img = np.full((100, 100, 3), 200, np.uint8)
    poly = np.array([[130, 130], [170, 130], [170, 170], [130, 170]], dtype=np.float64)
    out = czb.fill_crop(img, (poly, 4.0), 1.0, 1.0, 100, 100)
    assert (out[50, 50] == 200).all()               # inside shifted polygon
    assert (out[5, 5] == czb._FILL_GRAY).all()


def test_deduper_first_always_then_threshold():
    d = czb.CropDeduper(min_diff=4.0)
    flat = np.zeros((80, 80, 3), np.uint8)
    assert d.should_save("cam_a", "z1", flat) is True     # first crop always saves
    assert d.should_save("cam_a", "z1", flat) is False    # identical → skip
    brighter = np.full((80, 80, 3), 60, np.uint8)
    assert d.should_save("cam_a", "z1", brighter) is True # big change → save
    assert d.should_save("cam_a", "z2", flat) is True     # other zone independent
