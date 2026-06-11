"""Zone-patch ROI store + crop-box math (pixel-space watch boxes)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.api.routes_video import _drop_persons, _zone_objects
from monitor_web.api.routes_zone_patches import patch_pixel_box
from monitor_web.app import create_app
from monitor_web.config import Settings

# ---- _zone_objects: strict zone-only object rule ---------------------------

class _Det:
    """Minimal Detection stand-in (cls / confidence / bbox_xyxy)."""

    def __init__(self, cls, conf, bbox):
        self.cls, self.confidence, self.bbox_xyxy = cls, conf, bbox


def test_zone_objects_drops_person_class():
    """On a zone-enabled cam the human is shown by pose only — person boxes are dropped."""
    dets = [_Det("palette", 0.9, (0, 0, 10, 10)), _Det("person", 0.8, (50, 50, 60, 80))]
    out = _zone_objects(dets)
    assert [d.cls for d in out] == ["palette"]


def test_zone_objects_dedupes_overlapping_zones():
    """The same object cached by two OVERLAPPING zones is drawn once (highest conf)."""
    a = _Det("palette", 0.7, (100, 100, 200, 200))
    b = _Det("palette", 0.95, (102, 101, 201, 199))   # ~same box, higher conf
    out = _zone_objects([a, b])
    assert len(out) == 1 and out[0].confidence == 0.95


def test_zone_objects_keeps_distinct_objects():
    """Different objects (low IoU, centres outside each other) are all kept — dedupe
    must not over-merge two pallets sitting side by side."""
    dets = [_Det("palette", 0.9, (0, 0, 50, 50)), _Det("palette", 0.9, (300, 300, 360, 360))]
    assert len(_zone_objects(dets)) == 2


def test_zone_objects_merges_partial_overlap_from_two_zones():
    """One object clipped differently by two OVERLAPPING zones → offset boxes with
    low IoU but mutual-centre containment → drawn once (highest conf wins)."""
    a = _Det("palette", 0.7, (100, 100, 200, 200))   # zone A's view of the pallet
    b = _Det("palette", 0.9, (140, 110, 240, 205))   # zone B's view — offset, IoU < 0.5
    out = _zone_objects([a, b])
    assert len(out) == 1 and out[0].confidence == 0.9


def test_zone_objects_merges_asymmetric_offset_twin():
    """Strengthened dedupe: a small clipped box whose centre is inside a larger box
    (only ONE centre contained) still merges — the old AND-rule would have kept both."""
    big = _Det("palette", 0.6, (100, 100, 260, 240))
    clip = _Det("palette", 0.85, (210, 150, 250, 200))   # inside big; big's centre NOT in clip
    out = _zone_objects([big, clip])
    assert len(out) == 1 and out[0].confidence == 0.85


def test_drop_persons_strips_all_human_classes():
    """STRICT: humans are never detected in a zone — every person-class variant is
    removed at the zone detector (panel + cache stay person-free); objects pass through."""
    dets = [_Det("person", 0.9, (0, 0, 1, 1)), _Det("human", 0.9, (0, 0, 1, 1)),
            _Det("pedestrian", 0.9, (0, 0, 1, 1)), _Det("palette", 0.8, (0, 0, 1, 1)),
            _Det("carton", 0.8, (0, 0, 1, 1))]
    assert sorted(d.cls for d in _drop_persons(dets)) == ["carton", "palette"]


# ---- patch_pixel_box (pure crop math) -------------------------------------

def test_box_basic_same_size():
    assert patch_pixel_box([10, 20, 110, 220], [640, 480], (640, 480)) == (10, 20, 110, 220)


def test_box_normalizes_reversed_corners():
    # x1<x0 / y1<y0 → sorted into a proper box.
    assert patch_pixel_box([110, 220, 10, 20], None, (640, 480)) == (10, 20, 110, 220)


def test_box_scales_when_frame_differs():
    # Drawn at 640x480, streamed at 1280x960 → doubled.
    assert patch_pixel_box([10, 20, 110, 220], [640, 480], (1280, 960)) == (20, 40, 220, 440)


def test_box_clamps_out_of_bounds():
    assert patch_pixel_box([-50, -50, 700, 500], [640, 480], (640, 480)) == (0, 0, 640, 480)


def test_box_none_when_degenerate():
    assert patch_pixel_box([100, 100, 101, 101], None, (640, 480)) is None


# ---- GET/POST round-trip --------------------------------------------------

@pytest.fixture
def app_cfg(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), tmp_path


def test_get_empty_then_post_then_get(app_cfg):
    app, _tmp_path = app_cfg
    with TestClient(app) as client:
        assert client.get("/api/zone-patches").json() == {"patches": []}

        roi = {"id": "z1", "name": "press north", "camera": "cam_a",
               "rect": [100, 80, 400, 300], "frame_wh": [1920, 1080]}
        res = client.post("/api/zone-patches", json={"patches": [roi]})
        assert res.json() == {"ok": True, "count": 1}

        # Persisted in the merged dashboard config, and read back.
        got = client.get("/api/zone-patches").json()["patches"]
        assert len(got) == 1
        assert got[0]["id"] == "z1" and got[0]["rect"] == [100, 80, 400, 300]


def test_stream_unknown_patch_404(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        assert client.get("/stream/zone/nope").status_code == 404
