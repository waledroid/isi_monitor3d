"""Unified Mode-2 bird's-eye composite: shared world canvas + overlap blend.

Two synthetic cameras with pure scale+translate floor homographies:
  cam_a covers world X in [0,4] m, cam_b covers X in [2,6] m (overlap [2,4]).
Asserts the shared layout spans the union and the overlap is averaged.
"""

from __future__ import annotations

import numpy as np

from monitor_web.floor_rectify import composite_bev, shared_bev_layout

WH = (640, 480)
PPM_SRC = 160.0  # source px per metre in the synthetic homographies


def _H(x_off: float) -> np.ndarray:
    # pixel (u,v) → world (x_off + u/160, v/160). Floor maps cleanly (depth = 1).
    return np.array([[1.0 / PPM_SRC, 0.0, x_off],
                     [0.0, 1.0 / PPM_SRC, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def test_shared_layout_spans_union():
    cams = [(_H(0.0), WH, WH), (_H(2.0), WH, WH)]
    bounds, Ms = shared_bev_layout(cams, max_dim=1100)
    # cam_a → X[0,4], cam_b → X[2,6]  ⇒ union X[0,6], Y[0,3].
    assert bounds["x_min"] == 0.0
    assert abs(bounds["x_min"] + bounds["out_wh"][0] / bounds["px_per_m"] - 6.0) < 0.05
    assert abs(bounds["y_min"] + bounds["out_wh"][1] / bounds["px_per_m"] - 3.0) < 0.05
    assert len(Ms) == 2 and all(m is not None for m in Ms)


def test_overlap_is_blended():
    cams = [(_H(0.0), WH, WH), (_H(2.0), WH, WH)]
    bounds, Ms = shared_bev_layout(cams, max_dim=1100)
    ow, oh = bounds["out_wh"]
    K = np.eye(3)
    D = np.zeros(5)
    red = np.full((WH[1], WH[0], 3), (0, 0, 255), dtype=np.uint8)   # BGR red = cam_a
    blue = np.full((WH[1], WH[0], 3), (255, 0, 0), dtype=np.uint8)  # BGR blue = cam_b
    canvas = composite_bev([(red, K, D, Ms[0]), (blue, K, D, Ms[1])], (ow, oh))

    ppm = bounds["px_per_m"]
    vy = int(1.5 * ppm)                       # mid-height row
    px_left = int(1.0 * ppm)                  # X=1 → cam_a only
    px_overlap = int(3.0 * ppm)               # X=3 → both
    px_right = int(5.0 * ppm)                 # X=5 → cam_b only

    assert tuple(canvas[vy, px_left]) == (0, 0, 255)     # red only
    assert tuple(canvas[vy, px_right]) == (255, 0, 0)    # blue only
    # Overlap = average of red+blue → ~ (128, 0, 128) purple.
    b, g, r = (int(c) for c in canvas[vy, px_overlap])
    assert g == 0 and 100 < b < 160 and 100 < r < 160


def test_none_when_all_degenerate():
    # A homography whose corners all fall on/behind the horizon (depth ≤ 0).
    bad = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    assert shared_bev_layout([(bad, WH, WH)]) is None
