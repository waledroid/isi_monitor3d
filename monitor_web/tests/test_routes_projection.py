"""S17: pixel ↔ floor projection endpoints.

Hermetic — builds a Mode 1 ``calibration.json`` fixture with a known
homography so we can round-trip points exactly without spinning up OpenCV's
ChArUco / Multical pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


def _write_mode1_calibration(path: Path, H: np.ndarray, image_size=(1920, 1080),
                              camera_id: str = "cam_a") -> None:
    """Write a single-cam-4pt calibration.json with the supplied H.

    K=I, D=0, R=I, t=0 (Mode 1 placeholders). P = K @ [R|t] = [I | 0].
    """
    K = np.eye(3).tolist()
    D = [0.0, 0.0, 0.0, 0.0, 0.0]
    R = np.eye(3).tolist()
    t = [0.0, 0.0, 0.0]
    P = np.hstack([np.eye(3), np.zeros((3, 1))]).tolist()
    data = {
        "version": 1,
        "created_at": "2026-05-28T00:00:00+00:00",
        "floor_anchor_method": "4pt_floor",
        "floor_origin_note": "Test fixture",
        "calibration_mode": "single_cam_4pt",
        "cameras": {
            camera_id: {
                "camera_id": camera_id,
                "image_size_wh": list(image_size),
                "K": K, "D": D, "R": R, "t": t,
                "H": H.tolist(), "P": P,
                "reprojection_rms_px": 0.0,
            },
        },
    }
    path.write_text(json.dumps(data))


def _build_app_with_calibration(tmp_path: Path, H: np.ndarray, **extra):
    """An app whose backbone.yaml points at the fixture calibration."""
    cal_path = tmp_path / "calibration.json"
    _write_mode1_calibration(cal_path, H, **extra)
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
        "calibration_path": str(cal_path),
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), cal_path


# ---- pixel-to-floor ----


def test_pixel_to_floor_identity_homography(tmp_path: Path) -> None:
    """With H = I, pixel (u, v) maps to world (u, v) — sanity test."""
    app, _ = _build_app_with_calibration(tmp_path, np.eye(3))
    with TestClient(app) as client:
        res = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a",
            "points": [[100.0, 200.0], [500.0, 800.0]],
        })
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["camera_id"] == "cam_a"
        # Identity H: floor coords == pixel coords (mod undistortion which is
        # also identity since D=0).
        assert np.allclose(data["points"][0], [100.0, 200.0])
        assert np.allclose(data["points"][1], [500.0, 800.0])


def test_pixel_to_floor_known_scale_homography(tmp_path: Path) -> None:
    """A H that scales pixels by 0.01 should send pixel 500 to 5.0 m."""
    H = np.diag([0.01, 0.01, 1.0])
    app, _ = _build_app_with_calibration(tmp_path, H)
    with TestClient(app) as client:
        res = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a",
            "points": [[500.0, 300.0]],
        })
        assert res.status_code == 200
        assert np.allclose(res.json()["points"][0], [5.0, 3.0])


def test_pixel_to_floor_unknown_camera_404(tmp_path: Path) -> None:
    app, _ = _build_app_with_calibration(tmp_path, np.eye(3))
    with TestClient(app) as client:
        res = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_z", "points": [[0, 0]],
        })
        assert res.status_code == 404
        assert "cam_z" in res.json()["detail"]


def test_pixel_to_floor_no_calibration_503(tmp_path: Path) -> None:
    """No calibration.json on disk → 503 with operator-friendly message."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}}))   # no calibration_path
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[0, 0]],
        })
        assert res.status_code == 503


# ---- floor-to-pixel ----


def test_floor_to_pixel_identity_homography(tmp_path: Path) -> None:
    app, _ = _build_app_with_calibration(tmp_path, np.eye(3))
    with TestClient(app) as client:
        res = client.post("/api/project/floor-to-pixel", json={
            "camera_id": "cam_a",
            "polygon": [[100.0, 200.0], [500.0, 800.0]],
        })
        assert res.status_code == 200
        data = res.json()
        assert data["image_size"] == [1920, 1080]
        assert np.allclose(data["points"][0], [100.0, 200.0])
        assert np.allclose(data["points"][1], [500.0, 800.0])


def test_floor_to_pixel_round_trips_with_pixel_to_floor(tmp_path: Path) -> None:
    """A non-trivial H: world(0.01 px) — pixel→floor→pixel should be the original."""
    H = np.diag([0.01, 0.01, 1.0])
    app, _ = _build_app_with_calibration(tmp_path, H)
    pixels = [[400.0, 250.0], [1200.0, 900.0], [1800.0, 50.0]]
    with TestClient(app) as client:
        floor = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": pixels,
        }).json()["points"]
        back = client.post("/api/project/floor-to-pixel", json={
            "camera_id": "cam_a", "polygon": floor,
        }).json()["points"]
    for orig, recovered in zip(pixels, back, strict=True):
        assert np.allclose(orig, recovered, atol=1e-6), f"{orig} vs {recovered}"


# ---- cameras list ----


def test_cameras_list_returns_calibrated_cameras(tmp_path: Path) -> None:
    app, _ = _build_app_with_calibration(tmp_path, np.eye(3))
    with TestClient(app) as client:
        res = client.get("/api/project/cameras")
        assert res.status_code == 200
        data = res.json()
        assert data["cameras"] == ["cam_a"]
        assert data["mode"] == "single_cam_4pt"
        assert data["image_sizes"]["cam_a"] == [1920, 1080]


def test_cameras_list_empty_when_no_calibration(tmp_path: Path) -> None:
    """Graceful 200 with empty cameras so the dashboard can hide CAM-draw cleanly."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}}))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/project/cameras")
        assert res.status_code == 200
        data = res.json()
        assert data["cameras"] == []
        assert "error" in data


# ---- calibration cache invalidates on file change ----


def test_calibration_cache_invalidates_on_mtime_change(tmp_path: Path) -> None:
    """Re-saving calibration.json with a different H is reflected on the next call.

    The endpoint caches by (path, mtime_ns); a write changes mtime so the cache
    is bypassed and the new H is applied without needing a process restart.
    """
    import time

    app, cal_path = _build_app_with_calibration(tmp_path, np.eye(3))
    with TestClient(app) as client:
        # First call with H=I → pixel 500 stays 500.
        first = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[500.0, 500.0]],
        }).json()["points"][0]
        assert np.allclose(first, [500.0, 500.0])

        # Overwrite the calibration with H=0.01*I → pixel 500 becomes 5.0.
        # Ensure mtime advances (Linux mtime can be sub-second).
        time.sleep(0.01)
        _write_mode1_calibration(cal_path, np.diag([0.01, 0.01, 1.0]))

        second = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[500.0, 500.0]],
        }).json()["points"][0]
        assert np.allclose(second, [5.0, 5.0])


# ---- plane decode (z_m — raised platform/shelf zones) ----


def _write_mode2_calibration(path: Path, camera_id: str = "cam_a") -> None:
    """A metric look-down camera at (0, 0, 3): f=1000, c=(500, 500), no
    distortion. Pose (world←camera) R=diag(1,-1,-1), t=(0,0,3). A world
    point (X, Y, z) projects to u = 1000*X/(3-z) + 500, v = 500 - 1000*Y/(3-z);
    H is that mapping folded for z=0."""
    R_pose = np.diag([1.0, -1.0, -1.0])
    t_pose = np.array([0.0, 0.0, 3.0])
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
    R_cw = R_pose.T
    t_cw = -R_cw @ t_pose
    P = K @ np.hstack([R_cw, t_cw.reshape(3, 1)])
    H = np.array([[3 / 1000.0, 0.0, -1.5], [0.0, -3 / 1000.0, 1.5], [0.0, 0.0, 1.0]])
    data = {
        "version": 1,
        "created_at": "2026-08-10T00:00:00+00:00",
        "floor_anchor_method": "charuco_floor",
        "floor_origin_note": "Test fixture (Mode 2)",
        "calibration_mode": "multical_full",
        "cameras": {
            camera_id: {
                "camera_id": camera_id,
                "image_size_wh": [1000, 1000],
                "K": K.tolist(), "D": [0.0] * 5,
                "R": R_pose.tolist(), "t": t_pose.tolist(),
                "H": H.tolist(), "P": P.tolist(),
                "reprojection_rms_px": 0.0,
            },
        },
    }
    path.write_text(json.dumps(data))


def _build_app_mode2(tmp_path: Path):
    cal_path = tmp_path / "calibration.json"
    _write_mode2_calibration(cal_path)
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
        "calibration_path": str(cal_path),
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg)


def test_pixel_to_plane_decodes_platform_edge_true_footprint(tmp_path: Path) -> None:
    """A click on a raised edge decoded at z_m recovers the TRUE world point;
    the same click decoded at the floor stores the displaced shadow."""
    app = _build_app_mode2(tmp_path)
    z = 0.304
    # pixel of world (1, 1, 0.304) through the fixture camera
    u = 1000.0 * 1.0 / (3.0 - z) + 500.0
    v = 500.0 - 1000.0 * 1.0 / (3.0 - z)
    with TestClient(app) as client:
        on_plane = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[u, v]], "z_m": z,
        }).json()["points"][0]
        assert np.allclose(on_plane, [1.0, 1.0], atol=1e-6)

        shadow = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[u, v]],
        }).json()["points"][0]
        expected_shadow = 3.0 / (3.0 - z)          # ray continued to the floor
        assert np.allclose(shadow, [expected_shadow, expected_shadow], atol=1e-6)
        assert not np.allclose(shadow, on_plane, atol=0.05)


def test_pixel_to_plane_mode1_falls_back_to_floor(tmp_path: Path) -> None:
    """Mode-1 placeholder extrinsics can't lift a ray — z_m is ignored."""
    app, _ = _build_app_with_calibration(tmp_path, np.eye(3))
    with TestClient(app) as client:
        got = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[10.0, 20.0]], "z_m": 0.5,
        }).json()["points"][0]
        assert np.allclose(got, [10.0, 20.0])


def test_pixel_to_plane_z_out_of_range_422(tmp_path: Path) -> None:
    app = _build_app_mode2(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/project/pixel-to-floor", json={
            "camera_id": "cam_a", "points": [[500.0, 500.0]], "z_m": 9.0,
        })
        assert res.status_code == 422
