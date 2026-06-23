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
