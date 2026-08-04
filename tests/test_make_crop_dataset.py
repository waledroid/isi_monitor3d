from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import numpy as np
import pytest

# Loader block: import tools/make_crop_dataset.py as mcd
_spec = spec_from_file_location("mcd", Path(__file__).parent.parent / "tools" / "make_crop_dataset.py")
mcd = module_from_spec(_spec)
_spec.loader.exec_module(mcd)


def test_parse_label_file_denormalizes(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("1 0.25 0.5 0.75 0.5 0.5 1.0\n")
    objs = mcd.parse_label_file(p, 200, 100)
    assert len(objs) == 1
    cls, poly = objs[0]
    assert cls == 1
    assert np.allclose(poly, [[50, 50], [150, 50], [100, 100]])


def test_parse_label_file_skips_malformed(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("garbage line\n0 0.1 0.1 0.9 0.1 0.5 0.9\n")
    objs = mcd.parse_label_file(p, 100, 100)
    assert len(objs) == 1 and objs[0][0] == 0


def test_format_label_lines_clamps_and_rounds():
    poly = np.array([[0.123456789, -0.01], [1.02, 0.5], [0.5, 0.999999]])
    out = mcd.format_label_lines([(2, poly)])
    assert out == "2 0.123457 0.000000 1.000000 0.500000 0.500000 0.999999\n"
    assert mcd.format_label_lines([]) == ""


def test_poly_bbox_and_area():
    sq = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]])
    assert mcd.poly_bbox(sq) == (10.0, 20.0, 110.0, 70.0)
    assert mcd.poly_area(sq) == pytest.approx(100 * 50)


def test_clip_polygon_keeps_inside_region():
    sq = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    clipped = mcd.clip_polygon(sq, 50, 0, 150, 100)
    assert clipped is not None
    assert mcd.poly_area(clipped) == pytest.approx(50 * 100)
    assert clipped[:, 0].min() >= 50 and clipped[:, 0].max() <= 100


def test_clip_polygon_outside_returns_none():
    sq = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert mcd.clip_polygon(sq, 50, 50, 100, 100) is None


def test_cluster_boxes_groups_overlapping():
    boxes = [(0, 0, 100, 100), (90, 0, 200, 100), (500, 500, 600, 600)]
    groups = mcd.cluster_boxes(boxes, expand_frac=0.1)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_cluster_boxes_far_apart_stay_separate():
    boxes = [(0, 0, 10, 10), (500, 500, 510, 510)]
    assert sorted(map(sorted, mcd.cluster_boxes(boxes, 0.25))) == [[0], [1]]


def test_crop_window_at_least_size_and_inside_image():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = mcd.crop_window((800.0, 500.0, 900.0, 560.0),
                                     (1920, 1080), 384, (0.10, 0.25), rng)
    assert x1 - x0 >= 384 and y1 - y0 >= 384
    assert x0 >= 0 and y0 >= 0 and x1 <= 1920 and y1 <= 1080
    # window covers the bbox
    assert x0 <= 800 and x1 >= 900 and y0 <= 500 and y1 >= 560


def test_crop_window_clamps_on_small_image():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = mcd.crop_window((10.0, 10.0, 40.0, 40.0),
                                     (200, 150), 384, (0.10, 0.25), rng)
    assert (x0, y0, x1, y1) == (0, 0, 200, 150)   # whole image, no overflow


def test_letterbox_downscales_and_pads():
    img = np.full((400, 800, 3), 200, np.uint8)
    canvas, scale, dx, dy = mcd.letterbox_to(img, 384)
    assert canvas.shape == (384, 384, 3)
    assert scale == pytest.approx(384 / 800)
    assert dx == 0 and dy == (384 - round(400 * scale)) // 2
    assert (canvas[0, 0] == mcd.GRAY).all()        # pad band
    assert (canvas[192, 192] == 200).all()         # content center


def test_letterbox_never_upscales():
    img = np.full((100, 100, 3), 50, np.uint8)
    canvas, scale, dx, dy = mcd.letterbox_to(img, 384)
    assert scale == 1.0 and dx == dy == (384 - 100) // 2
    assert (canvas[dy + 50, dx + 50] == 50).all()
    assert (canvas[0, 0] == mcd.GRAY).all()
