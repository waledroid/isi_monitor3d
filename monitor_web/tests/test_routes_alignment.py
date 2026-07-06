"""Alignment fine-tune endpoints — fit / toggle / refit / staleness.

Synthetic Mode-2 rig where cam_b carries a KNOWN rigid floor error; operator
point-pairs are generated from the true geometry (what a careful operator
would click). Fitting must produce a refined calibration that collapses the
error; the toggle must flip the calibration path both ways; refit must work
from the stored pairs after a 'new solve'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    pixel_to_floor,
    projection_from_K_R_t,
    undistort_points,
)
from calibration.schema import CALIBRATION_VERSION
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
R_DOWN = np.diag([1.0, -1.0, -1.0])
# The physical floor points the operator clicks (well spread).
POINTS = np.array([[0.4, -0.6], [1.6, -0.6], [1.6, 0.6], [0.4, 0.6]])
# cam_b's injected rigid floor error (what the tool must correct).
ERR_YAW_DEG, ERR_T = 2.0, (0.10, -0.06)


def _cam(cid: str, center, *, yaw_deg=0.0, t_extra=(0.0, 0.0)) -> dict:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = T @ R_DOWN
    t = T @ np.array([center[0], center[1], 2.5]) + [t_extra[0], t_extra[1], 0.0]
    return {
        "camera_id": cid, "image_size_wh": [1920, 1080],
        "K": K.tolist(), "D": [0.0] * 5, "R": R.tolist(), "t": t.tolist(),
        "H": floor_homography_from_K_R_t(K, R, t).tolist(),
        "P": projection_from_K_R_t(K, R, t).tolist(),
        "reprojection_rms_px": 0.5,
    }


def _true_pixels(cid: str, world_xy: np.ndarray) -> list[list[float]]:
    """Where the physical points ACTUALLY appear on each camera (the true,
    uncorrupted geometry — cam_b physically sits at (2,0))."""
    cam = _cam(cid, (0.0, 0.0) if cid == "cam_a" else (2.0, 0.0))
    P = np.asarray(cam["P"])
    w = np.hstack([world_xy, np.zeros((len(world_xy), 1)), np.ones((len(world_xy), 1))])
    pix = (P @ w.T).T
    pix = pix[:, :2] / pix[:, 2:3]
    return [[float(u), float(v)] for u, v in pix]


def _write_cal(path: Path, *, corrupt_b: bool) -> None:
    cams = {"cam_a": _cam("cam_a", (0.0, 0.0))}
    cams["cam_b"] = _cam("cam_b", (2.0, 0.0),
                         yaw_deg=ERR_YAW_DEG if corrupt_b else 0.0,
                         t_extra=ERR_T if corrupt_b else (0.0, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": CALIBRATION_VERSION, "created_at": "2026-07-03T00:00:00Z",
        "floor_anchor_method": "synthetic", "floor_origin_note": "test",
        "calibration_mode": "multical_full", "cameras": cams,
    }))


def _build_app(tmp_path: Path):
    cal = tmp_path / "mode2" / "calibration.json"
    _write_cal(cal, corrupt_b=True)
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {}, "cam_b": {}},
        "calibration_path": str(cal),
        "metadata": {"sinks": []},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0,
                   ui_settings_path=tmp_path / "monitor_web_ui.yaml")
    return create_app(cfg), cal, backbone_yaml


def _pairs() -> list[dict]:
    a = _true_pixels("cam_a", POINTS)
    b = _true_pixels("cam_b", POINTS)
    return [{"cam_a": pa, "cam_b": pb} for pa, pb in zip(a, b, strict=True)]


def _cross_error(cal_path: Path) -> float:
    """Max cross-camera floor disagreement of the physical points through a
    calibration file (the symptom metric)."""
    data = json.loads(Path(cal_path).read_text())
    out = {}
    for cid in ("cam_a", "cam_b"):
        cam = data["cameras"][cid]
        pix = np.asarray(_true_pixels(cid, POINTS))
        out[cid] = pixel_to_floor(
            undistort_points(pix, np.asarray(cam["K"]), np.asarray(cam["D"])),
            np.asarray(cam["H"]))
    return float(np.linalg.norm(out["cam_a"] - out["cam_b"], axis=1).max())


def test_fit_collapses_cross_camera_error_and_writes_refined(tmp_path: Path) -> None:
    app, cal, _bb = _build_app(tmp_path)
    assert _cross_error(cal) > 0.05                       # corrupted to start
    with TestClient(app) as client:
        r = client.post("/api/alignment/fit", json={"pairs": _pairs()})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["fit"]["max_residual_m"] < 0.001
        assert max(data["before_error_m"]) > 0.05         # honest before-number
        refined = Path(data["refined_path"])
    assert refined.exists()
    assert _cross_error(refined) < 1e-6                   # collapsed
    assert _cross_error(cal) > 0.05                       # base untouched


def test_enable_toggles_calibration_path_both_ways(tmp_path: Path) -> None:
    app, cal, bb = _build_app(tmp_path)
    with TestClient(app) as client:
        client.post("/api/alignment/fit", json={"pairs": _pairs()})
        r = client.post("/api/alignment/enable", json={"enabled": True})
        assert r.status_code == 200
        assert yaml.safe_load(bb.read_text())["calibration_path"].endswith(
            "calibration_refined.json")
        assert client.get("/api/alignment").json()["enabled"] is True

        client.post("/api/alignment/enable", json={"enabled": False})
        assert yaml.safe_load(bb.read_text())["calibration_path"] == str(cal)
        assert client.get("/api/alignment").json()["enabled"] is False


def test_enable_without_fit_rejected(tmp_path: Path) -> None:
    app, _cal, _bb = _build_app(tmp_path)
    with TestClient(app) as client:
        r = client.post("/api/alignment/enable", json={"enabled": True})
    assert r.status_code == 422


def test_bad_pairs_gated_nothing_changed(tmp_path: Path) -> None:
    """A wrong correspondence must be refused — no refined file, no repoint."""
    app, _cal, bb = _build_app(tmp_path)
    pairs = _pairs()
    pairs[2]["cam_b"] = [pairs[2]["cam_b"][0] + 400.0, pairs[2]["cam_b"][1]]
    with TestClient(app) as client:
        r = client.post("/api/alignment/fit", json={"pairs": pairs})
        assert r.status_code == 422
        assert "don't correspond" in r.json()["detail"]
        assert client.get("/api/alignment").json()["fit"] is None
    assert not (Path(bb).parent / "mode2" / "calibration_refined.json").exists()


def test_refit_after_new_solve_uses_stored_pairs(tmp_path: Path) -> None:
    """'Remake on new calibration': the stored PIXEL pairs are re-authored
    through the new base; staleness is reported until the refit."""
    app, cal, _bb = _build_app(tmp_path)
    with TestClient(app) as client:
        client.post("/api/alignment/fit", json={"pairs": _pairs()})
        client.post("/api/alignment/enable", json={"enabled": True})

        # A NEW solve lands (different corruption → different correction needed).
        import time as _t
        _t.sleep(0.01)
        _write_cal(cal, corrupt_b=True)
        st = client.get("/api/alignment").json()
        assert st["stale"] is True

        r = client.post("/api/alignment/refit")
        assert r.status_code == 200
        assert r.json()["fit"]["max_residual_m"] < 0.001
        st = client.get("/api/alignment").json()
        assert st["stale"] is False
        assert st["enabled"] is True                      # stayed on, refreshed
    refined = Path(st["refined_path"])
    assert _cross_error(refined) < 1e-6


def test_clear_repoints_base(tmp_path: Path) -> None:
    app, cal, bb = _build_app(tmp_path)
    with TestClient(app) as client:
        client.post("/api/alignment/fit", json={"pairs": _pairs()})
        client.post("/api/alignment/enable", json={"enabled": True})
        client.delete("/api/alignment")
        assert client.get("/api/alignment").json()["pairs"] == []
    assert yaml.safe_load(bb.read_text())["calibration_path"] == str(cal)


def test_enable_regenerates_zone_twins_so_outlines_move(tmp_path: Path) -> None:
    """The user-visible effect: a zone's cross-camera TWIN is stored at save
    time — toggling the refined calibration must REGENERATE it, or the
    outlines don't move and the fine-tune looks like a no-op."""
    app, _cal, _bb = _build_app(tmp_path)
    zone_px = _true_pixels("cam_b", POINTS * 0.6 + [0.5, 0.0])  # a zone on cam_b
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_b",
             "polygon": zone_px, "frame_wh": [1920, 1080]},
        ]})
        twin_before = next(p for p in client.get("/api/zone-patches").json()["patches"]
                           if p.get("twin_of") == "z1")

        client.post("/api/alignment/fit", json={"pairs": _pairs()})
        client.post("/api/alignment/enable", json={"enabled": True})
        twin_after = next(p for p in client.get("/api/zone-patches").json()["patches"]
                          if p.get("twin_of") == "z1")

    a = np.asarray(twin_before["polygon"], dtype=np.float64)
    b = np.asarray(twin_after["polygon"], dtype=np.float64)
    shift = np.linalg.norm(a - b, axis=1)
    # The injected 2° + (10, -6) cm error corresponds to a clearly visible
    # pixel shift of the corrected twin.
    assert shift.max() > 15.0, f"twin did not move (max shift {shift.max():.1f} px)"

    # Toggling OFF regenerates against the base again → back to the original.
    with TestClient(app) as client:
        client.post("/api/alignment/enable", json={"enabled": False})
        twin_off = next(p for p in client.get("/api/zone-patches").json()["patches"]
                        if p.get("twin_of") == "z1")
    c = np.asarray(twin_off["polygon"], dtype=np.float64)
    assert np.linalg.norm(a - c, axis=1).max() < 1e-6
