"""Overlap dedup in map_twin — an object seen by both cameras counts once."""

from monitor_web.api.routes_map import _dedupe_floor


def test_overlap_objects_merge_once():
    items = [
        {"cls": "carton", "conf": 0.9, "xy_m": [2.0, 3.0], "zone_id": "z1", "camera": "cam_a"},
        {"cls": "carton", "conf": 0.8, "xy_m": [2.05, 3.02], "zone_id": "z1", "camera": "cam_b"},
        {"cls": "carton", "conf": 0.7, "xy_m": [6.0, 1.0], "zone_id": "z1", "camera": "cam_b"},
    ]
    out = _dedupe_floor(items, key=("zone_id", "cls"))
    assert len(out) == 2
    merged = [o for o in out if len(o["cameras"]) == 2]
    assert merged and set(merged[0]["cameras"]) == {"cam_a", "cam_b"}
    assert merged[0]["conf"] == 0.9          # higher-confidence kept


def test_different_class_not_merged():
    items = [
        {"cls": "carton", "conf": 0.9, "xy_m": [2.0, 3.0], "zone_id": "z1", "camera": "cam_a"},
        {"cls": "polybag", "conf": 0.8, "xy_m": [2.01, 3.0], "zone_id": "z1", "camera": "cam_b"},
    ]
    assert len(_dedupe_floor(items, key=("zone_id", "cls"))) == 2   # different cls → kept


def test_people_merge_by_proximity():
    items = [
        {"conf": 0.9, "xy_m": [1.0, 1.0], "camera": "cam_a"},
        {"conf": 0.7, "xy_m": [1.1, 1.05], "camera": "cam_b"},
    ]
    assert len(_dedupe_floor(items, key=())) == 1                   # same person, both cams


def test_draw_unified_tracks_overlay():
    """Fused tracks overlay onto the rectified floor at their world (X,Y)."""
    import numpy as np
    from monitor_web.api.routes_video import _draw_unified_tracks

    bounds = {"px_per_m": 100.0, "x_min": 0.0, "y_min": 0.0, "out_wh": (400, 400)}

    class _T2:
        def __init__(self, tid, xy): self.track_id = tid; self.xy_m = xy; self.cls = "x"
    class _T3:
        def __init__(self, tid, xyz): self.track_id = tid; self.xyz_m = xyz; self.cls = "x"
    class _Snap:
        last_track2d_by_id = {1: _T2(1, (1.0, 1.0))}      # 2D-only @ (100,100)px
        last_track3d_by_id = {2: _T3(2, (2.0, 2.0, 0.0))}  # 3D @ (200,200)px
    class _Bus:
        def is_fresh(self, t): return True
        def snapshot(self): return _Snap()

    frame = np.zeros((400, 400, 3), np.uint8)
    _draw_unified_tracks(frame, bounds, _Bus())
    # both markers drew SOMETHING at their mapped pixels (non-black neighborhood)
    assert frame[100, 100].sum() > 0      # track #1
    assert frame[200, 200].sum() > 0      # track #2


def test_draw_unified_tracks_stale_noop():
    import numpy as np
    from monitor_web.api.routes_video import _draw_unified_tracks

    class _Bus:
        def is_fresh(self, t): return False        # stale → draw nothing
        def snapshot(self): raise AssertionError("should not snapshot when stale")
    frame = np.zeros((50, 50, 3), np.uint8)
    _draw_unified_tracks(frame, {"px_per_m": 1, "x_min": 0, "y_min": 0, "out_wh": (50, 50)}, _Bus())
    assert frame.sum() == 0
