"""Floor zones derived from camera zone patches (`floor_zone_sync`).

One drawing: the operator's camera-pixel patches become the Backbone's floor
zones (same names) so the COMMUNICATION cards and the MQTT ``zone_state`` /
zone-scoped detection run from the same physical zones.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import floor_homography_from_K_R_t, projection_from_K_R_t

from monitor_web.floor_zone_sync import sync_floor_zones_from_patches

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R = np.diag([1.0, -1.0, -1.0])          # look-down rig from the backbone tests
T = np.array([0.0, 0.0, 3.0])


def _rig(tmp_path: Path) -> CameraRig:
    cal = {
        "version": 1, "created_at": "2026-07-06T00:00:00Z",
        "floor_anchor_method": "synthetic", "floor_origin_note": "test",
        "cameras": {"cam_a": {
            "camera_id": "cam_a", "image_size_wh": [1000, 1000],
            "K": K.tolist(), "D": [0.0] * 5, "R": R.tolist(), "t": T.tolist(),
            "H": floor_homography_from_K_R_t(K, R, T).tolist(),
            "P": projection_from_K_R_t(K, R, T).tolist(),
            "reprojection_rms_px": 0.1,
        }},
    }
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(cal))
    return CameraRig.from_file(p)


def _cfg(tmp_path: Path) -> SimpleNamespace:
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump({"zones_path": str(tmp_path / "zones.yaml")}))
    return SimpleNamespace(backbone_config_path=bb,
                           ui_settings_path=tmp_path / "ui.yaml")


def test_patch_polygon_becomes_floor_zone(tmp_path) -> None:
    """Pixel (833.33, 500) back-projects to world (1, 0) on this rig — the
    derived floor polygon must land there, under the patch's name."""
    cfg, rig = _cfg(tmp_path), _rig(tmp_path)
    patches = [{
        "id": "zp_1", "name": "Zone 1", "camera": "cam_a",
        "polygon": [[766.67, 433.33], [900.0, 433.33], [900.0, 566.67],
                    [766.67, 566.67]],
        "frame_wh": [1000, 1000],
    }]
    n = sync_floor_zones_from_patches(cfg, patches=patches, rig=rig)
    assert n == 1
    zones = yaml.safe_load((tmp_path / "zones.yaml").read_text())["zones"]
    (z,) = zones
    assert z["name"] == "Zone 1" and z["derived_from"] == "zone_patch"
    # Corners: u = 1000*X/3 + 500 → X = 3*(u-500)/1000. (766.67 → 0.8, 900 → 1.2)
    xs = [pt[0] for pt in z["polygon"]]
    ys = [pt[1] for pt in z["polygon"]]
    assert abs(min(xs) - 0.8) < 0.01 and abs(max(xs) - 1.2) < 0.01
    assert abs(min(ys) + 0.2) < 0.01 and abs(max(ys) - 0.2) < 0.01


def test_twins_skipped_and_downscaled_frame_rescaled(tmp_path) -> None:
    """A twin is the same physical zone — one floor zone. Patches drawn on a
    downscaled stream (frame_wh 500²) scale up to the calibration frame."""
    cfg, rig = _cfg(tmp_path), _rig(tmp_path)
    patches = [
        {"id": "zp_1", "name": "Zone 1", "camera": "cam_a",
         "polygon": [[383.3, 216.7], [450.0, 216.7], [450.0, 283.3], [383.3, 283.3]],
         "frame_wh": [500, 500]},                    # half-size drawing surface
        {"id": "zp_1_twin", "name": "Zone 1", "camera": "cam_b",
         "twin_of": "zp_1",
         "polygon": [[1, 1], [2, 1], [2, 2]], "frame_wh": [1000, 1000]},
    ]
    n = sync_floor_zones_from_patches(cfg, patches=patches, rig=rig)
    assert n == 1                                    # twin produced no zone
    (z,) = yaml.safe_load((tmp_path / "zones.yaml").read_text())["zones"]
    xs = [pt[0] for pt in z["polygon"]]
    assert abs(min(xs) - 0.8) < 0.02 and abs(max(xs) - 1.2) < 0.02


def test_manual_zones_preserved_and_deleted_patches_cleared(tmp_path) -> None:
    cfg, rig = _cfg(tmp_path), _rig(tmp_path)
    (tmp_path / "zones.yaml").write_text(yaml.safe_dump({"zones": [
        {"name": "hand_drawn", "type": "danger", "kind": "danger",
         "polygon": [[0, 0], [1, 0], [1, 1]]},
        {"name": "old_derived", "type": "palette", "kind": "palette",
         "polygon": [[5, 5], [6, 5], [6, 6]], "derived_from": "zone_patch"},
    ]}))
    n = sync_floor_zones_from_patches(cfg, patches=[], rig=rig)
    assert n == 0
    zones = yaml.safe_load((tmp_path / "zones.yaml").read_text())["zones"]
    assert [z["name"] for z in zones] == ["hand_drawn"]   # derived cleared, manual kept


def test_uncalibrated_camera_skipped_no_calibration_leaves_file(tmp_path) -> None:
    cfg, rig = _cfg(tmp_path), _rig(tmp_path)
    patches = [{"id": "zp_9", "name": "Z", "camera": "cam_zzz",
                "polygon": [[1, 1], [2, 1], [2, 2]], "frame_wh": [1000, 1000]}]
    assert sync_floor_zones_from_patches(cfg, patches=patches, rig=rig) == 0
    # No rig at all → untouched (don't clear zones on a transient calib miss).
    (tmp_path / "zones.yaml").write_text(yaml.safe_dump(
        {"zones": [{"name": "x", "type": "palette",
                    "polygon": [[0, 0], [1, 0], [1, 1]],
                    "derived_from": "zone_patch"}]}))
    assert sync_floor_zones_from_patches(cfg, patches=patches, rig=None) == 0
    assert yaml.safe_load((tmp_path / "zones.yaml").read_text())["zones"]


def test_rect_only_patch_uses_rect_corners(tmp_path) -> None:
    cfg, rig = _cfg(tmp_path), _rig(tmp_path)
    patches = [{"id": "zp_r", "name": "R", "camera": "cam_a",
                "rect": [766.67, 433.33, 900.0, 566.67], "frame_wh": [1000, 1000]}]
    assert sync_floor_zones_from_patches(cfg, patches=patches, rig=rig) == 1
