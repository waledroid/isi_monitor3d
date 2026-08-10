"""Zone-patch ROI store + crop-box math (pixel-space watch boxes)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.api.routes_zone_patches import patch_pixel_box
from monitor_web.app import create_app
from monitor_web.config import Settings

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


def test_delete_leaves_other_patch_ids_and_names_unchanged(app_cfg):
    """Deleting one zone (re-POSTing the survivors, mirroring the UI) must not
    rename or reorder any other zone — every id AND label is preserved. This
    pins the fix for the positional-identity bug (no `length + 1` renumber)."""
    app, _ = app_cfg
    with TestClient(app) as client:
        rois = [
            {"id": "zp_a", "name": "Zone 1", "camera": "cam_a",
             "rect": [10, 10, 100, 100], "frame_wh": [1920, 1080]},
            {"id": "zp_b", "name": "Zone 2", "camera": "cam_a",
             "rect": [200, 10, 300, 100], "frame_wh": [1920, 1080]},
            {"id": "zp_c", "name": "custom label", "camera": "cam_a",
             "rect": [400, 10, 500, 100], "frame_wh": [1920, 1080]},
        ]
        client.post("/api/zone-patches", json={"patches": rois})
        # Delete "Zone 1" → the client re-POSTs the remaining two verbatim.
        client.post("/api/zone-patches", json={"patches": rois[1:]})
        got = {p["id"]: p["name"]
               for p in client.get("/api/zone-patches").json()["patches"]}
        assert got == {"zp_b": "Zone 2", "zp_c": "custom label"}
        # "Zone 2" was NOT renumbered down to "Zone 1"; the custom name survived.


def test_stream_unknown_patch_404(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        assert client.get("/stream/zone/nope").status_code == 404




# ---- cross-camera ghost projection (Mode-2 stereo UX) ----------------------


def _mode2_app(tmp_path: Path):
    """App with 2 cameras + a synthetic Mode-2 calibration (two look-down cams),
    ui_settings isolated in tmp so the calibration-path override never leaks."""
    import json

    import numpy as np
    from backbone.shared.geometry import (
        floor_homography_from_K_R_t,
        projection_from_K_R_t,
    )
    from calibration.schema import CALIBRATION_VERSION

    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])

    def cam(cam_id, x):
        t = np.array([x, 0.0, 3.0])
        return {
            "camera_id": cam_id, "image_size_wh": [1000, 1000],
            "K": K.tolist(), "D": [0.0] * 5, "R": R.tolist(), "t": t.tolist(),
            "H": floor_homography_from_K_R_t(K, R, t).tolist(),
            "P": projection_from_K_R_t(K, R, t).tolist(),
            "reprojection_rms_px": 0.1,
        }

    cal = tmp_path / "mode2" / "calibration.json"
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text(json.dumps({
        "version": CALIBRATION_VERSION, "created_at": "2026-07-02T00:00:00Z",
        "floor_anchor_method": "synthetic", "floor_origin_note": "test",
        "cameras": {"cam_a": cam("cam_a", 0.0), "cam_b": cam("cam_b", 2.0)},
    }))
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {}, "cam_b": {}},
        "calibration_path": str(cal),
        "metadata": {"sinks": []},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0,
                   ui_settings_path=tmp_path / "monitor_web_ui.yaml")
    return create_app(cfg)


def test_ghost_projected_into_other_camera(tmp_path: Path):
    """A patch on cam_a comes back with its cross-camera outline in cam_b —
    now as a detecting TWIN patch (occlusion persistence); the display-only
    ghost is omitted when a twin exists."""
    import numpy as np
    from backbone.shared.geometry import floor_to_pixel, pixel_to_floor

    app = _mode2_app(tmp_path)
    # Around cam_a pixel (833, 500) = world (1.0, 0) — the middle of the two
    # cameras' shared floor, inside cam_b's view as well.
    poly = [[783.0, 450.0], [883.0, 450.0], [883.0, 550.0], [783.0, 550.0]]
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "g1", "name": "ghosted", "camera": "cam_a",
             "polygon": poly, "frame_wh": [1000, 1000]},
        ]})
        patches_all = client.get("/api/zone-patches").json()["patches"]
        got = patches_all[0]

    # The cross-camera outline arrives as a TWIN (detects on cam_b).
    assert got.get("ghost") is None
    twin = next(p for p in patches_all if p.get("twin_of") == "g1")
    assert twin["camera"] == "cam_b"
    assert twin["frame_wh"] == [1000, 1000]
    ghost = {"polygon": twin["polygon"], "camera": twin["camera"],
             "image_wh": twin["frame_wh"]}

    # Expected: the DENSIFIED polygon round-tripped through the H matrices
    # (D=0 → undistort no-op and the distorted projection equals pinhole; the
    # synthetic rig has real K/R/t so the full-model path runs).
    from backbone.shared.geometry import densify_polygon

    from monitor_web.api.routes_projection import _load_rig_cached
    cal = tmp_path / "mode2" / "calibration.json"
    rig = _load_rig_cached(str(cal.resolve()), cal.stat().st_mtime_ns)
    dense = densify_polygon(np.asarray(poly), segments_per_edge=8)
    world = pixel_to_floor(dense, rig["cam_a"].H)
    expected = floor_to_pixel(world, rig["cam_b"].H)
    got = np.asarray(ghost["polygon"])
    assert got.shape == expected.shape          # densified: 4 edges x 8 samples
    assert np.allclose(got, expected, atol=1e-3)


def test_ghost_null_when_no_overlap(tmp_path: Path):
    """A patch whose floor footprint lies fully outside cam_b's view → ghost None."""
    app = _mode2_app(tmp_path)
    # cam_a's far-left edge maps to world x≈-1.5, well outside cam_b (at x=2).
    poly = [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "g2", "name": "far", "camera": "cam_a",
             "polygon": poly, "frame_wh": [1000, 1000]},
        ]})
        got = client.get("/api/zone-patches").json()["patches"][0]
    assert got.get("ghost") is None


def test_ghost_absent_without_calibration(app_cfg):
    """No calibration → response shape unchanged (no ghost key computation)."""
    app, _ = app_cfg
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "g3", "name": "ghostless", "camera": "cam_a",
             "rect": [10, 10, 60, 60], "frame_wh": [1000, 1000]},
        ]})
        got = client.get("/api/zone-patches").json()["patches"][0]
    assert "ghost" not in got or got["ghost"] is None


# ---- /api/zone-patches/state — live worker contents for the comms cards ----


class _Det:
    def __init__(self, cls, conf, bbox):
        self.cls = cls
        self.confidence = conf
        self.bbox_xyxy = bbox


class _FakeManager:
    """zone_manager stub: fixed per-zone detections + status."""

    def __init__(self, dets_by_zone, status="ok"):
        self._dets = dets_by_zone
        self._status = status

    def zone_status(self, zone_id):
        return self._status if zone_id in self._dets else ""

    def zone_dets(self, zone_id):
        return list(self._dets.get(zone_id, []))


def test_zone_patches_state_maps_occupancy(app_cfg):
    """A pallet with a carton on it → palette object with occupancy_state=full;
    the carton itself is listed too — same shape as the world-zone states."""
    app, _ = app_cfg
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_a",
             "rect": [0, 0, 400, 300], "frame_wh": [1920, 1080]},
            {"id": "z2", "name": "Zone 2", "camera": "cam_a",
             "rect": [500, 0, 900, 300], "frame_wh": [1920, 1080]},
        ]})
        # Zone 1: a pallet with a carton sitting on it (image overlap → full).
        client.app.state.zone_manager = _FakeManager({
            "z1": [
                _Det("palette", 0.9, (100.0, 200.0, 300.0, 300.0)),
                _Det("carton", 0.8, (150.0, 120.0, 250.0, 210.0)),
            ],
        })
        data = client.get("/api/zone-patches/state").json()

    assert set(data["states"]) == {"z1"}          # z2 has no coverage → absent (dim)
    objs = data["states"]["z1"]["objects"]
    pal = next(o for o in objs if o["cls"] == "palette")
    assert pal["occupancy_state"] == "full"
    # What the pallet carries, for the human-readable comms cards
    # ("Palette present — with carton").
    assert pal["occupancy_content"] == ["carton"]
    assert any(o["cls"] == "carton" for o in objs)
    assert data["states"]["z1"]["count"] == 2
    assert data["states"]["z1"]["status"] == "ok"


def test_zone_patches_state_empty_pallet_is_vide(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_a",
             "rect": [0, 0, 400, 300], "frame_wh": [1920, 1080]},
        ]})
        client.app.state.zone_manager = _FakeManager({
            "z1": [_Det("palette", 0.9, (100.0, 200.0, 300.0, 300.0))],
        })
        data = client.get("/api/zone-patches/state").json()
    pal = data["states"]["z1"]["objects"][0]
    assert pal["cls"] == "palette"
    assert pal["occupancy_state"] == "empty"
    assert "occupancy_content" not in pal         # empty pallet carries nothing


def test_zone_patches_state_no_manager(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        client.app.state.zone_manager = None
        data = client.get("/api/zone-patches/state").json()
    assert data == {"states": {}}


def test_zone_patches_state_conflicting_pallet_reads_resolve_to_loaded(app_cfg):
    """The same physical pallet double-detected as empty AND loaded in one
    zone must surface only the LOADED palette — mirrors the Backbone's MQTT
    zone_state rule so REST and MQTT agree."""
    app, _ = app_cfg
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_a",
             "rect": [0, 0, 900, 400], "frame_wh": [1920, 1080]},
        ]})
        client.app.state.zone_manager = _FakeManager({
            "z1": [
                # Loaded pallet (a carton sits on it)…
                _Det("palette", 0.9, (100.0, 200.0, 300.0, 300.0)),
                _Det("carton", 0.8, (150.0, 120.0, 250.0, 210.0)),
                # …and a second, overlapping read of the SAME pallet, empty.
                _Det("palette", 0.6, (420.0, 200.0, 620.0, 300.0)),
            ],
        })
        data = client.get("/api/zone-patches/state").json()
    objs = data["states"]["z1"]["objects"]
    pals = [o for o in objs if o["cls"] == "palette"]
    assert len(pals) == 1
    assert pals[0]["occupancy_state"] == "full"
    assert pals[0]["occupancy_content"] == ["carton"]
    assert data["states"]["z1"]["count"] == len(objs)


# ---- compulsory + unique zone names (2026-07-22) ----


def _roi(pid, name, x0=10):
    return {"id": pid, "name": name, "camera": "cam_a",
            "rect": [x0, 10, x0 + 90, 100], "frame_wh": [1920, 1080]}


def test_post_rejects_empty_zone_name(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        res = client.post("/api/zone-patches",
                          json={"patches": [_roi("z1", "  ")]})
        assert res.status_code == 422
        assert "name" in res.json()["detail"].lower()
        assert client.get("/api/zone-patches").json()["patches"] == []


def test_post_rejects_duplicate_zone_names_case_insensitive(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        res = client.post("/api/zone-patches", json={"patches": [
            _roi("z1", "Dock A"), _roi("z2", "dock a", x0=200)]})
        assert res.status_code == 422
        assert "unique" in res.json()["detail"].lower()
        assert client.get("/api/zone-patches").json()["patches"] == []


def test_post_accepts_distinct_names(app_cfg):
    app, _ = app_cfg
    with TestClient(app) as client:
        res = client.post("/api/zone-patches", json={"patches": [
            _roi("z1", "Dock A"), _roi("z2", "Dock B", x0=200)]})
        assert res.status_code == 200, res.text


# ---- twin staleness signature covers zones.yaml ----------------------------


def test_calibration_sig_changes_on_zones_yaml_edit(tmp_path: Path):
    """A zone edit (polygon redraw, base-height change) moves the twin
    geometry exactly like a calibration switch — the staleness signature must
    change when zones.yaml changes so ensure_twins_current regenerates."""
    import time

    from monitor_web.api.routes_zone_patches import _calibration_sig

    backbone_yaml = tmp_path / "backbone.yaml"
    zones_path = tmp_path / "zones.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {}, "metadata": {"sinks": []},
        "zones_path": str(zones_path),
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)

    zones_path.write_text(yaml.safe_dump({"zones": []}))
    sig1 = _calibration_sig(cfg)
    time.sleep(0.01)
    zones_path.write_text(yaml.safe_dump(
        {"zones": [{"id": "zp_1", "name": "Z", "polygon": [[0, 0], [1, 0], [1, 1]],
                    "z_base_m": 0.304}]}))
    sig2 = _calibration_sig(cfg)
    assert sig1 != sig2
