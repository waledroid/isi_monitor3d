"""S18.A: in-dashboard calibration endpoints.

* ``GET /api/calibrate/status`` — colour-state driver for the toolbar button.
* ``POST /api/calibrate/single-cam`` — pallet 4-corner flow.

Hermetic — uses synthetic pixel ↔ world correspondences computed from a
chosen H, so we never need a real camera or ChArUco / Multical / OpenCV
ChArUco runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from calibration.schema import CALIBRATION_VERSION
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings

# ---- helpers ----


def _mode_cal(tmp_path: Path, mode: int) -> Path:
    """Per-mode calibration file path (mode{N}/calibration.json), parent created."""
    p = tmp_path / f"mode{mode}" / "calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _pallet_corner_pixels(H_world_to_pixel: np.ndarray, w: float, h: float):
    """Given a world→pixel H (3x3), return the 4 pallet corners in pixel
    coords (TL, TR, BR, BL), matching the API's ordering."""
    world = np.array([
        [0.0, 0.0, 1.0],
        [w,   0.0, 1.0],
        [w,   h,   1.0],
        [0.0, h,   1.0],
    ])
    pix_h = (H_world_to_pixel @ world.T).T
    return [(float(p[0] / p[2]), float(p[1] / p[2])) for p in pix_h]


def _build_app(tmp_path: Path, *, cameras: dict | None = None, calibration_path: Path | None = None):
    """An app whose backbone.yaml lists the supplied cameras."""
    bb_yaml = tmp_path / "backbone.yaml"
    body = {"cameras": cameras if cameras is not None else {}}
    if calibration_path is not None:
        body["calibration_path"] = str(calibration_path)
    bb_yaml.write_text(yaml.safe_dump(body))
    cfg = Settings(backbone_config_path=bb_yaml, udp_port=0, port=0)
    return create_app(cfg), bb_yaml


# ---- /api/calibrate/status ----


def test_status_no_calibration_white_state(tmp_path: Path) -> None:
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    with TestClient(app) as client:
        res = client.get("/api/calibrate/status")
        assert res.status_code == 200
        data = res.json()
        assert data["calibrated_cameras"] == []
        assert data["configured_cameras"] == ["cam_a"]
        assert data["is_fully_calibrated"] is False
        assert data["calibration_mode"] is None


def test_status_full_coverage_green_state(tmp_path: Path) -> None:
    """Two cameras configured + a Mode-2 calibration covering both → fully calibrated.

    Status reads the CURRENT-mode file (2 cams → mode2/calibration.json)."""
    # Synthetic H (diag scale 0.01) — exact value doesn't matter for the status check.
    H = np.diag([0.01, 0.01, 1.0]).tolist()
    cal_path = _mode_cal(tmp_path, 2)
    cal_path.write_text(json.dumps({
        "version": CALIBRATION_VERSION,
        "created_at": "2026-05-28T00:00:00+00:00",
        "floor_anchor_method": "4pt_floor",
        "floor_origin_note": "test",
        "calibration_mode": "single_cam_4pt",
        "cameras": {
            cam: {
                "camera_id": cam,
                "image_size_wh": [1920, 1080],
                "K": np.eye(3).tolist(),
                "D": [0.0] * 5,
                "R": np.eye(3).tolist(),
                "t": [0.0, 0.0, 0.0],
                "H": H,
                "P": np.hstack([np.eye(3), np.zeros((3, 1))]).tolist(),
                "reprojection_rms_px": 0.01,
            }
            for cam in ("cam_a", "cam_b")
        },
    }))
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}, "cam_b": {}},
                        calibration_path=cal_path)
    with TestClient(app) as client:
        data = client.get("/api/calibrate/status").json()
        assert set(data["calibrated_cameras"]) == {"cam_a", "cam_b"}
        assert data["is_fully_calibrated"] is True
        assert data["calibration_mode"] == "single_cam_4pt"


def test_status_partial_coverage_not_fully(tmp_path: Path) -> None:
    """Two configured but only cam_a in the Mode-2 file → not fully calibrated."""
    H = np.diag([0.01, 0.01, 1.0]).tolist()
    cal_path = _mode_cal(tmp_path, 2)
    cal_path.write_text(json.dumps({
        "version": CALIBRATION_VERSION,
        "created_at": "2026-05-28T00:00:00+00:00",
        "floor_anchor_method": "4pt_floor",
        "floor_origin_note": "test",
        "calibration_mode": "single_cam_4pt",
        "cameras": {
            "cam_a": {
                "camera_id": "cam_a",
                "image_size_wh": [1920, 1080],
                "K": np.eye(3).tolist(), "D": [0.0] * 5,
                "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
                "H": H,
                "P": np.hstack([np.eye(3), np.zeros((3, 1))]).tolist(),
                "reprojection_rms_px": 0.01,
            },
        },
    }))
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}, "cam_b": {}},
                        calibration_path=cal_path)
    with TestClient(app) as client:
        data = client.get("/api/calibrate/status").json()
        assert data["calibrated_cameras"] == ["cam_a"]
        assert set(data["configured_cameras"]) == {"cam_a", "cam_b"}
        assert data["is_fully_calibrated"] is False


# ---- /api/calibrate/single-cam ----


def test_pallet_post_writes_new_calibration_json(tmp_path: Path) -> None:
    """No prior calibration.json → endpoint creates one beside backbone.yaml."""
    app, bb_yaml = _build_app(tmp_path, cameras={"cam_a": {}})

    # Generate 4 corner pixels from a known H (world→pixel diag 100): so
    # a pallet of (1.2, 0.8) at world coords (0,0)..(1.2,0.8) projects to
    # pixel (0,0)..(120,80).
    H_world_to_pixel = np.diag([100.0, 100.0, 1.0])
    corners = _pallet_corner_pixels(H_world_to_pixel, 1.2, 0.8)

    payload = {
        "camera_id": "cam_a",
        "image_size": [1920, 1080],
        "pallet_width_m": 1.2,
        "pallet_height_m": 0.8,
        "corners_uv": corners,
    }
    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json=payload)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["ok"] is True
        assert body["calibration_mode"] == "single_cam_4pt"
        assert body["calibrated_cameras"] == ["cam_a"]
        assert body["max_residual_m"] < 1e-6   # synthetic — no noise

    # File written beside backbone.yaml.
    cal_path = Path(body["calibration_path"])
    assert cal_path.exists()
    data = json.loads(cal_path.read_text())
    assert data["calibration_mode"] == "single_cam_4pt"
    assert "cam_a" in data["cameras"]

    # backbone.yaml was stamped with calibration_path so the orchestrator
    # picks it up on next start.
    bb = yaml.safe_load(bb_yaml.read_text())
    assert bb["calibration_path"] == str(cal_path)


def test_pallet_post_preserves_other_camera_entries(tmp_path: Path) -> None:
    """A single-cam POST must merge, not drop, the other camera's existing entry.

    Single-cam always writes the Mode-1 file, so seed cam_a there."""
    H = np.diag([0.01, 0.01, 1.0]).tolist()
    cal_path = _mode_cal(tmp_path, 1)
    cal_path.write_text(json.dumps({
        "version": CALIBRATION_VERSION,
        "created_at": "2026-05-28T00:00:00+00:00",
        "floor_anchor_method": "4pt_floor",
        "floor_origin_note": "seed",
        "calibration_mode": "single_cam_4pt",
        "cameras": {
            "cam_a": {
                "camera_id": "cam_a",
                "image_size_wh": [1920, 1080],
                "K": np.eye(3).tolist(), "D": [0.0] * 5,
                "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
                "H": H,
                "P": np.hstack([np.eye(3), np.zeros((3, 1))]).tolist(),
                "reprojection_rms_px": 0.0,
            },
        },
    }))
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}, "cam_b": {}},
                        calibration_path=cal_path)

    # POST cam_b — synthetic pixels for a different H.
    H_world_to_pixel = np.diag([200.0, 200.0, 1.0])
    corners = _pallet_corner_pixels(H_world_to_pixel, 1.2, 0.8)

    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_b",
            "image_size": [1280, 720],
            "pallet_width_m": 1.2,
            "pallet_height_m": 0.8,
            "corners_uv": corners,
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert set(body["calibrated_cameras"]) == {"cam_a", "cam_b"}

    # Both cameras present on disk after the second write.
    data = json.loads(cal_path.read_text())
    assert "cam_a" in data["cameras"]
    assert "cam_b" in data["cameras"]
    # cam_a's original H is unchanged.
    np.testing.assert_allclose(data["cameras"]["cam_a"]["H"], H)


def test_pallet_post_rejects_high_residual(tmp_path: Path) -> None:
    """Garbage corner picks → residual exceeds the threshold → 422."""
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})

    # 4 points that fit ONE H, then deliberately shift the 4th by a huge
    # offset so the residual gate fires. We use exactly 4 input pairs so
    # the gate has to catch error via the residual, not under-determination.
    # Trick: pass 5 pairs (4 consistent + 1 outlier) so the fit is overdetermined.
    H_world_to_pixel = np.diag([100.0, 100.0, 1.0])
    consistent = _pallet_corner_pixels(H_world_to_pixel, 1.2, 0.8)
    # We can't feed 5 corners via this endpoint (corners_uv is len=4). So
    # produce 4 valid corners but pass world dimensions that LIE about the
    # pallet's geometry — that mis-anchors the world points so the fit
    # can't reach all 4 with sub-threshold residual.
    # With exactly 4 corners the fit is exactly-determined and the residual
    # is always near zero. To force a >threshold residual we shift one
    # CORNER far away in pixel space — but that still satisfies a 4-pt fit
    # since there's an H that maps any 4 non-collinear pixels to the 4
    # nominal world points. So pre-checking that this actually trips the
    # gate is hard without changing the residual_threshold_m. Use a tiny
    # threshold and a nearly-collinear point to force a degenerate solve.
    corners = list(consistent)
    corners[3] = (corners[3][0] + 1e-7, corners[0][1])  # nearly collinear with 3 others

    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_a",
            "image_size": [1920, 1080],
            "pallet_width_m": 1.2,
            "pallet_height_m": 0.8,
            "corners_uv": corners,
            "residual_threshold_m": 0.0001,   # 0.1 mm — anything real-world fails
        })
        # cv2.findHomography may degenerate → 422 from our SingleCamCalibrationError,
        # OR it may succeed with a borderline H whose residual exceeds 0.1 mm → also 422.
        assert res.status_code == 422


def _world_to_pixel(H_wp: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H_wp @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def test_pallet_post_five_points_succeeds(tmp_path: Path) -> None:
    """4 pallet corners + 1 extra tape-measured floor point (consistent) → 200,
    overdetermined fit, sub-px residual."""
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    H_wp = np.diag([100.0, 100.0, 1.0])
    corners = _pallet_corner_pixels(H_wp, 1.2, 0.8)
    extra_world = (0.6, 0.4)                       # mid-pallet floor point
    extra_px = _world_to_pixel(H_wp, *extra_world)
    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_a",
            "image_size": [1920, 1080],
            "pallet_width_m": 1.2,
            "pallet_height_m": 0.8,
            "corners_uv": [*corners, extra_px],
            "extra_world_xy": [list(extra_world)],
        })
        assert res.status_code == 200, res.text
        assert res.json()["max_residual_m"] < 1e-5


def test_pallet_post_five_points_bad_extra_rejected(tmp_path: Path) -> None:
    """5th point whose declared world is wildly inconsistent with its pixel →
    the overdetermined residual gate fires (the whole point of 5+ points)."""
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    H_wp = np.diag([100.0, 100.0, 1.0])
    corners = _pallet_corner_pixels(H_wp, 1.2, 0.8)
    extra_px = _world_to_pixel(H_wp, 0.6, 0.4)     # pixel for (0.6, 0.4)…
    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_a",
            "image_size": [1920, 1080],
            "pallet_width_m": 1.2,
            "pallet_height_m": 0.8,
            "corners_uv": [*corners, extra_px],
            "extra_world_xy": [[5.0, 5.0]],        # …but declared 5 m away → huge residual
        })
        assert res.status_code == 422


def test_extra_points_without_world_coords_rejected(tmp_path: Path) -> None:
    """>4 corners but no matching extra_world_xy → pydantic validation 422."""
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    H_wp = np.diag([100.0, 100.0, 1.0])
    corners = _pallet_corner_pixels(H_wp, 1.2, 0.8)
    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_a",
            "image_size": [1920, 1080],
            "pallet_width_m": 1.2,
            "pallet_height_m": 0.8,
            "corners_uv": [*corners, (500.0, 500.0)],   # 5 corners, no extra_world_xy
        })
        assert res.status_code == 422


def test_pallet_post_rejects_bad_pallet_dimensions(tmp_path: Path) -> None:
    """Pydantic guards: pallet_width_m must be > 0."""
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    H_world_to_pixel = np.diag([100.0, 100.0, 1.0])
    corners = _pallet_corner_pixels(H_world_to_pixel, 1.2, 0.8)
    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_a",
            "image_size": [1920, 1080],
            "pallet_width_m": 0,    # invalid
            "pallet_height_m": 0.8,
            "corners_uv": corners,
        })
        assert res.status_code == 422


def test_pallet_round_trip_homography_recovers_world_points(tmp_path: Path) -> None:
    """Synthetic round-trip: pixels we send map back to the world coords we
    declared (within sub-mm residual). Validates the math wiring end-to-end."""
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    # Pick a non-trivial world→pixel H — translation + scale + slight shear.
    H_wp = np.array([
        [100.0,   2.0, 500.0],
        [  3.0, 110.0, 300.0],
        [  0.001, 0.0,   1.0],
    ])
    corners = _pallet_corner_pixels(H_wp, 1.2, 0.8)
    with TestClient(app) as client:
        res = client.post("/api/calibrate/single-cam", json={
            "camera_id": "cam_a",
            "image_size": [1920, 1080],
            "pallet_width_m": 1.2,
            "pallet_height_m": 0.8,
            "corners_uv": corners,
        }).json()
    assert res["max_residual_m"] < 1e-6

    # Verify the saved H actually projects pixels back to the declared world.
    data = json.loads(Path(res["calibration_path"]).read_text())
    H_saved = np.asarray(data["cameras"]["cam_a"]["H"])
    pixels = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pixels, H_saved).reshape(-1, 2)
    expected = np.array([[0, 0], [1.2, 0], [1.2, 0.8], [0, 0.8]])
    np.testing.assert_allclose(world, expected, atol=1e-6)


# ---- per-mode storage + clear (calibration-driven auto-warp) ----


def _mode1_doc(cam_id: str) -> str:
    """A minimal valid calibration file JSON holding one 4pt camera."""
    H = np.diag([0.01, 0.01, 1.0]).tolist()
    return json.dumps({
        "version": CALIBRATION_VERSION,
        "created_at": "2026-05-30T00:00:00+00:00",
        "floor_anchor_method": "4pt_floor",
        "floor_origin_note": "test",
        "calibration_mode": "single_cam_4pt",
        "cameras": {cam_id: {
            "camera_id": cam_id, "image_size_wh": [1920, 1080],
            "K": np.eye(3).tolist(), "D": [0.0] * 5,
            "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
            "H": H, "P": np.hstack([np.eye(3), np.zeros((3, 1))]).tolist(),
            "reprojection_rms_px": 0.0,
        }},
    })


def test_status_reads_current_mode_file(tmp_path: Path) -> None:
    """Mode is decided by camera count; status reads that mode's file.

    A Mode-1 file present but TWO cameras configured (Mode 2) → not calibrated,
    because Mode 2 looks at mode2/calibration.json (absent)."""
    (_mode_cal(tmp_path, 1)).write_text(_mode1_doc("cam_a"))
    # One camera → Mode 1 → sees the mode1 file → green.
    app1, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    with TestClient(app1) as c:
        d = c.get("/api/calibrate/status").json()
        assert d["mode"] == 1
        assert d["is_fully_calibrated"] is True
    # Two cameras → Mode 2 → mode2 file absent → white, despite the mode1 file.
    app2, _ = _build_app(tmp_path, cameras={"cam_a": {}, "cam_b": {}})
    with TestClient(app2) as c:
        d = c.get("/api/calibrate/status").json()
        assert d["mode"] == 2
        assert d["is_fully_calibrated"] is False


def test_clear_removes_current_mode_only(tmp_path: Path) -> None:
    """POST /api/calibrate/clear deletes the current mode's file, leaving the
    other mode's saved calibration intact."""
    (_mode_cal(tmp_path, 1)).write_text(_mode1_doc("cam_a"))
    (_mode_cal(tmp_path, 2)).write_text(_mode1_doc("cam_a"))
    # Mode 1 (one camera): clear removes mode1 file, keeps mode2.
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    with TestClient(app) as c:
        res = c.post("/api/calibrate/clear")
        assert res.status_code == 200
        assert res.json()["removed"] is True
        # Button now white.
        assert c.get("/api/calibrate/status").json()["is_fully_calibrated"] is False
    assert not (_mode_cal(tmp_path, 1)).exists()
    assert (_mode_cal(tmp_path, 2)).exists()   # other mode untouched


def test_clear_is_idempotent_when_absent(tmp_path: Path) -> None:
    app, _ = _build_app(tmp_path, cameras={"cam_a": {}})
    with TestClient(app) as c:
        res = c.post("/api/calibrate/clear")
        assert res.status_code == 200
        assert res.json()["removed"] is False
