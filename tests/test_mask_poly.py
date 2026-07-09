"""``mask_to_polygon`` — instance masks travel the wire as simplified outlines."""

from __future__ import annotations

import numpy as np

from backbone.shared.mask_poly import mask_to_polygon


def test_rectangle_mask_yields_its_outline() -> None:
    m = np.zeros((100, 100), dtype=bool)
    m[20:60, 10:80] = True
    poly = mask_to_polygon(m)
    assert poly is not None and len(poly) >= 4
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert min(xs) == 10 and max(xs) == 79
    assert min(ys) == 20 and max(ys) == 59


def test_offset_shifts_to_frame_coords() -> None:
    m = np.zeros((50, 50), dtype=bool)
    m[10:40, 10:40] = True
    poly = mask_to_polygon(m, offset_xy=(300, 200))
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert min(xs) == 310 and min(ys) == 210


def test_degenerate_masks_return_none() -> None:
    assert mask_to_polygon(np.zeros((40, 40), dtype=bool)) is None       # empty
    dot = np.zeros((40, 40), dtype=bool)
    dot[5, 5] = True
    assert mask_to_polygon(dot) is None                                   # ~zero area


def test_largest_component_wins() -> None:
    m = np.zeros((100, 100), dtype=bool)
    m[5:10, 5:10] = True          # small blob
    m[40:90, 40:90] = True        # big blob
    poly = mask_to_polygon(m)
    xs = [p[0] for p in poly]
    assert min(xs) >= 40          # outline belongs to the big blob
