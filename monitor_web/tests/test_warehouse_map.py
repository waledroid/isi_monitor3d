"""warehouse_map — load / validate / write the layout YAML."""
from __future__ import annotations

import pytest

from monitor_web.warehouse_map import read_map, validate_map, write_map


def test_validate_accepts_a_rack():
    data = {"elements": [{
        "id": "rack_a1", "type": "rack", "shape": "rectangle",
        "footprint": [[2.1, 0.4], [3.6, 0.4], [3.6, 1.2], [2.1, 1.2]],
        "height_m": 2.5, "label": "Rack A1",
    }]}
    out = validate_map(data)
    assert out["elements"][0]["type"] == "rack"
    assert out["outline"] is None


def test_validate_rejects_bad_type():
    with pytest.raises(ValueError, match="type"):
        validate_map({"elements": [{"id": "x", "type": "spaceship",
                                    "footprint": [[0, 0], [1, 0], [1, 1]], "height_m": 1}]})


def test_validate_rejects_short_footprint():
    with pytest.raises(ValueError, match="footprint"):
        validate_map({"elements": [{"id": "x", "type": "wall",
                                    "footprint": [[0, 0], [1, 0]], "height_m": 1}]})


def test_round_trip(tmp_path):
    p = tmp_path / "warehouse_map.yaml"
    data = {"elements": [{"id": "w1", "type": "wall", "shape": "rectangle",
                          "footprint": [[0, 0], [6, 0], [6, 0.2], [0, 0.2]],
                          "height_m": 3.0, "label": ""}],
            "outline": {"footprint": [[0, 0], [12, 0], [12, 8], [0, 8]]}}
    write_map(p, data)
    loaded = read_map(p)
    assert loaded["elements"][0]["id"] == "w1"
    assert loaded["outline"]["footprint"][2] == [12, 8]


def test_read_missing_file_returns_empty(tmp_path):
    assert read_map(tmp_path / "nope.yaml") == {"elements": [], "outline": None}
