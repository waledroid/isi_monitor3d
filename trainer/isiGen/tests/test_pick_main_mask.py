"""Auto-fallback mask selection — pick the main subject, not the whole scene.

Hermetic: pure numpy, no SAM2/GPU. Pins the fix for the 'auto mask covers all
items in the scene' bug — promptless records must get the single main object,
not the union of every SAM2 automatic mask.
"""

import numpy as np
from src.core.runners import _pick_main_mask


def _box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_picks_central_object_over_background_and_specks():
    h, w = 100, 100
    bg = np.ones((h, w), bool)                 # whole frame (background)
    obj = _box(h, w, 35, 65, 35, 65)           # central ~9%
    speck = _box(h, w, 0, 3, 0, 3)             # noise
    corner = _box(h, w, 0, 40, 0, 10)          # off-center edge
    pick = _pick_main_mask([bg, obj, speck, corner], (h, w))
    assert np.array_equal(pick, obj)


def test_falls_back_to_largest_when_all_filtered():
    h, w = 50, 50
    bg = np.ones((h, w), bool)                 # full-frame (filtered)
    speck = _box(h, w, 0, 1, 0, 1)             # tiny (filtered)
    pick = _pick_main_mask([bg, speck], (h, w))
    assert np.array_equal(pick, bg)            # largest fallback


def test_never_unions_all_masks():
    """Two separate objects → exactly one is returned, not their union."""
    h, w = 100, 100
    a = _box(h, w, 40, 60, 40, 60)
    b = _box(h, w, 10, 20, 80, 90)
    pick = _pick_main_mask([a, b], (h, w))
    assert np.array_equal(pick, a) or np.array_equal(pick, b)
    assert not np.array_equal(pick, a | b)
