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
    assert out == "2 0.123457 0.000000 1.000000 0.500000 0.500000 1.000000\n"
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
