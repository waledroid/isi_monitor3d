"""Person↔pallet metric-distance overlay — the projection/distance math.

Hermetic: a synthetic calibrated `view` (K=I, D=0, scale homography) makes
pixel→metre mapping analytic, so the computed distances are exact. No model,
no camera, no drawing.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from monitor_web.detection_overlay import compute_person_pallet_distances


def _view(scale: float = 0.01, wh: tuple[int, int] = (640, 480)) -> SimpleNamespace:
    """A calibrated view where pixel (u,v) → world (scale*u, scale*v) metres."""
    return SimpleNamespace(
        K=np.eye(3),
        D=np.zeros(5),
        H=np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]),
        image_size_wh=wh,
    )


def test_distance_one_line_per_pallet() -> None:
    view = _view()                       # 0.01 m/px
    persons = [(100.0, 100.0)]           # → (1, 1) m
    pallets = [(300.0, 100.0),           # → (3, 1) m  → 2 m
               (100.0, 300.0)]           # → (1, 3) m  → 2 m
    pairs = compute_person_pallet_distances(persons, pallets, view, (640, 480), max_m=6.0)
    assert len(pairs) == 2               # one line per pallet
    assert sorted(round(d, 3) for _, _, d in pairs) == [2.0, 2.0]


def test_max_distance_gates_lines() -> None:
    view = _view()
    persons, pallets = [(100.0, 100.0)], [(300.0, 100.0)]   # 2 m apart
    assert compute_person_pallet_distances(persons, pallets, view, (640, 480), max_m=1.0) == []
    assert len(compute_person_pallet_distances(persons, pallets, view, (640, 480), max_m=3.0)) == 1


def test_frame_size_guard_rescales_h() -> None:
    # Calibrated at 640x480 but the live frame is 1280x960 (2x). The guard rescales
    # H so actual-frame pixels still map to the right metres.
    view = _view(wh=(640, 480))
    persons, pallets = [(200.0, 200.0)], [(600.0, 200.0)]   # in the 1280x960 frame
    pairs = compute_person_pallet_distances(persons, pallets, view, (1280, 960), max_m=6.0)
    assert len(pairs) == 1
    assert round(pairs[0][2], 3) == 2.0                     # (1,1)m ↔ (3,1)m


def test_empty_inputs_return_no_lines() -> None:
    view = _view()
    assert compute_person_pallet_distances([], [(1.0, 1.0)], view, (640, 480)) == []
    assert compute_person_pallet_distances([(1.0, 1.0)], [], view, (640, 480)) == []
