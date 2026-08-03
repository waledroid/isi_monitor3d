"""Tests for tools/capture_zone_bg.py (hermetic — no cameras, no GPU)."""
from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path
from typing import ClassVar

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


def _frames_provider(frames):
    it = iter(frames)
    return lambda: next(it, None)


def test_capture_loop_saves_dedups_names_and_stops_when_idle(tmp_path):
    frames = [np.full((720, 1280, 3), v, np.uint8) for v in (10, 10, 200)]
    boxes = {"cam_a": [("Zone 1", (100, 100, 600, 600))]}
    tally = czb.capture_loop(
        {"cam_a": _frames_provider(frames)}, boxes, {"cam_a": {}},
        {"cam_a": (1280, 720)}, tmp_path,
        interval_s=0, count=10, min_diff=4.0, max_idle_polls=3)
    files = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert files == ["bg_cam_a_Zone-1_0000.jpg", "bg_cam_a_Zone-1_0001.jpg"]
    assert tally == {"cam_a/Zone 1": 2}          # middle frame dedup-skipped


def test_capture_loop_applies_fill(tmp_path):
    frames = [np.full((720, 1280, 3), 200, np.uint8)]
    boxes = {"cam_a": [("z", (0, 0, 1280, 720))]}
    poly = np.array([[300, 300], [900, 300], [900, 600], [300, 600]], dtype=np.float64)
    fills = {"cam_a": {"z": (poly, 4.0)}}
    czb.capture_loop(
        {"cam_a": _frames_provider(frames)}, boxes, fills,
        {"cam_a": (1280, 720)}, tmp_path,
        interval_s=0, count=1, max_idle_polls=2)
    img = cv2.imread(str(next(tmp_path.glob("*.jpg"))))
    assert abs(int(img[5, 5, 0]) - czb._FILL_GRAY) <= 3      # outside → gray (± JPEG)
    assert int(img[450, 640, 0]) > 180                        # inside preserved


def test_capture_loop_stops_at_count(tmp_path):
    frames = [np.full((720, 1280, 3), v, np.uint8) for v in (10, 200, 90, 250)]
    boxes = {"cam_a": [("z", (100, 100, 600, 600))]}
    tally = czb.capture_loop(
        {"cam_a": _frames_provider(frames)}, boxes, {"cam_a": {}},
        {"cam_a": (1280, 720)}, tmp_path,
        interval_s=0, count=2, max_idle_polls=3)
    assert sum(tally.values()) == 2


def test_slug_sanitizes_zone_names():
    assert czb._slug("Zone 1") == "Zone-1"
    assert czb._slug("étagère/2") == "tag-re-2"


def test_bus_provider_roundtrip(tmp_path):
    from backbone.shared.frame_shm import FrameShmWriter
    img = np.full((48, 64, 3), 37, np.uint8)
    writer = FrameShmWriter("camx", directory=str(tmp_path))
    try:
        writer.write(img, time.time())
        provider = czb.BusProvider("camx", directory=str(tmp_path))
        got = provider()
        assert got is not None and got.shape == (48, 64, 3) and (got == 37).all()
    finally:
        writer.close()


def test_bus_provider_none_when_bus_absent(tmp_path):
    assert czb.BusProvider("ghost", directory=str(tmp_path))() is None


def test_make_provider_prefers_bus(tmp_path):
    from backbone.shared.frame_shm import FrameShmWriter
    writer = FrameShmWriter("camy", directory=str(tmp_path))
    try:
        writer.write(np.zeros((8, 8, 3), np.uint8), time.time())
        p = czb.make_provider("camy", {"name": "rtsp", "url": "rtsp://x"},
                              bus_wait_s=1.0, directory=str(tmp_path))
        assert isinstance(p, czb.BusProvider)
    finally:
        writer.close()


def test_make_provider_none_without_bus_or_rtsp(tmp_path):
    p = czb.make_provider("ghost", {}, bus_wait_s=0.2, directory=str(tmp_path))
    assert p is None


def test_main_refuses_without_zones(tmp_path, monkeypatch):
    cfg = tmp_path / "backbone.yaml"
    cfg.write_text("calibration_path: /nonexistent.json\n")   # no zones_path
    monkeypatch.setattr(czb.CameraRig, "from_file",
                        staticmethod(lambda p: object()))
    rc = czb.main(["--config", str(cfg), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_main_exits_1_when_no_camera_delivers(tmp_path, monkeypatch):
    cfg = tmp_path / "backbone.yaml"
    cfg.write_text("calibration_path: /nonexistent.json\nzones_path: z.yaml\n")

    class _FakeView:
        image_size_wh = (1920, 1080)

    class _FakeRig:
        camera_ids: ClassVar = ["cam_a"]
        def __getitem__(self, k):
            return _FakeView()

    class _FakeZones:
        def __len__(self):
            return 1

    monkeypatch.setattr(czb.CameraRig, "from_file",
                        staticmethod(lambda p: _FakeRig()))
    monkeypatch.setattr(czb.ZoneRegistry, "load",
                        staticmethod(lambda p: _FakeZones()))
    monkeypatch.setattr(czb, "zone_crop_boxes",
                        lambda rig, zones, crop_height_m: {"cam_a": [("z", (0, 0, 100, 100))]})
    monkeypatch.setattr(czb, "zone_fill_polygons",
                        lambda rig, zones, crop_height_m: {"cam_a": {}})
    monkeypatch.setattr(czb, "make_provider", lambda *a, **k: None)
    rc = czb.main(["--config", str(cfg), "--out", str(tmp_path / "o")])
    assert rc == 1
