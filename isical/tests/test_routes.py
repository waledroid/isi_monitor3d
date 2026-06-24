"""Studio routes via TestClient (hermetic — no camera/Multical)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from isical.app import create_app
from isical.config import Settings


def _client():
    return TestClient(create_app(Settings()))


def test_pages_and_project_crud():
    with _client() as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/projects").json() == {"projects": []}
        body = {"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}}
        assert c.post("/api/projects", json=body).json()["ok"] is True
        assert c.post("/api/projects", json=body).status_code == 409          # dup
        names = [p["name"] for p in c.get("/api/projects").json()["projects"]]
        assert names == ["rig"]
        assert c.get("/p/rig").status_code == 200
        assert c.get("/p/rig/capture/intrinsic").status_code == 200
        assert c.get("/p/nope").status_code == 404


def test_create_requires_cam_a():
    with _client() as c:
        r = c.post("/api/projects", json={"name": "x", "cam_a": {"type": "rtsp", "url": ""}})
        assert r.status_code == 422


def test_status_and_capture_status():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        st = c.get("/api/p/rig/status").json()
        assert st["cameras"] == ["cam_a"] and st["intrinsic_done"] is False
        cap = c.get("/api/p/rig/capture/status").json()
        assert cap["active"] is False


def test_extrinsic_capture_needs_two_cameras():
    with _client() as c:
        c.post("/api/projects", json={"name": "solo", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        r = c.post("/api/p/solo/capture/extrinsic/start")
        assert r.status_code == 422                       # extrinsic needs both cams


def test_run_unknown_phase_404():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.post("/api/p/rig/run/bogus").status_code == 404


def test_edit_cameras_add_cam_b():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.get("/api/p/rig/status").json()["mode2"] is False
        # GET pre-fill
        cams = c.get("/api/p/rig/cameras").json()
        assert cams["cam_a"]["url"] == "rtsp://x/a" and cams["cam_b"] is None
        # add cam_b
        r = c.put("/api/p/rig/cameras", json={
            "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
            "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        assert r.json()["mode2"] is True
        assert c.get("/api/p/rig/status").json()["cameras"] == ["cam_a", "cam_b"]


def test_floor_shot_unknown_cam_404():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.post("/api/p/rig/floor/cam_b").status_code == 404   # cam_b not configured


def test_intrinsic_start_single_camera_and_restart():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        # single-camera intrinsic start (cam param)
        r = c.post("/api/p/rig/capture/intrinsic/start?cam=cam_a")
        assert r.status_code == 200
        assert list(r.json()["status"]["cameras"]) == ["cam_a"]
        # restart wipes + starts (no files yet → removed 0)
        rr = c.post("/api/p/rig/capture/intrinsic/restart?cam=cam_a")
        assert rr.status_code == 200 and rr.json()["removed"] == 0
        c.post("/api/p/rig/capture/intrinsic/stop")


def test_intrinsic_start_unknown_cam_404():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.post("/api/p/rig/capture/intrinsic/start?cam=cam_b").status_code == 404


def test_status_captured_flags():
    from isical.config import Settings
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        st = c.get("/api/p/rig/status").json()
        assert st["intrinsic_captured"] is False
        assert st["extrinsic_captured"] is False
        # fill cam_a's intrinsic dir up to target
        d = Settings().data_dir / "rig" / "intrinsic" / "cam_a"
        target = st["targets"]["intrinsic"]
        for i in range(target):
            (d / f"cam_a_{i:03d}.jpg").write_bytes(b"x")
        st2 = c.get("/api/p/rig/status").json()
        assert st2["intrinsic_captured"] is True       # capture complete
        assert st2["intrinsic_done"] is False           # but not solved


def test_list_shots_and_serve_with_backfill():
    import cv2
    import numpy as np

    from isical.config import Settings
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        d = Settings().data_dir / "rig" / "intrinsic" / "cam_a"
        cv2.imwrite(str(d / "cam_a_000.jpg"), np.zeros((48, 64, 3), np.uint8))
        assert not (d / "cam_a_000.json").exists()              # no sidecar yet
        r = c.get("/api/p/rig/shots/intrinsic/cam_a").json()
        assert r["count"] == 1
        assert r["shots"][0]["file"] == "cam_a_000.jpg"
        assert "corners" in r["shots"][0] and "blur_var" in r["shots"][0]
        assert "blur_min_var" in r
        assert (d / "cam_a_000.json").exists()                   # backfilled + cached
        assert c.get("/shots/rig/intrinsic/cam_a/cam_a_000.jpg").status_code == 200


def test_shot_serve_path_guarded():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.get("/shots/rig/intrinsic/cam_a/evil.png").status_code == 404      # not .jpg
        assert c.get("/shots/rig/intrinsic/cam_a/missing.jpg").status_code == 404   # absent


def test_list_shots_unknown_cam_404():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.get("/api/p/rig/shots/intrinsic/cam_b").status_code == 404


def test_intrinsic_capture_page_has_tabs_and_gallery_markup():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        html = c.get("/p/rig/capture/intrinsic").text
        assert "cam-tab" in html          # per-camera tab buttons
        assert "shot-gallery" in html      # gallery container per figure


# ---- intrinsic-summary endpoint ----

def test_intrinsic_summary_empty_when_no_intrinsic_json():
    """Endpoint returns empty cameras dict when work/intrinsic.json is absent."""
    with _client() as c:
        c.post("/api/projects", json={"name": "rig",
                                      "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        r = c.get("/api/p/rig/intrinsic-summary")
        assert r.status_code == 200
        body = r.json()
        assert body["cameras"] == {}


def test_intrinsic_summary_shape_with_intrinsic_json_and_rms():
    """With both work/intrinsic.json and work/intrinsic_rms.json present the endpoint
    returns per-camera K, dist, image_size, rms and the rms_gate_px."""
    import json as _json
    with _client() as c:
        c.post("/api/projects", json={"name": "rig",
                                      "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        d = Settings().data_dir / "rig" / "work"
        d.mkdir(parents=True, exist_ok=True)
        intr = {
            "cameras": {
                "cam_a": {
                    "model": "standard",
                    "image_size": [1920, 1080],
                    "K": [[1387.16, 0.0, 942.99],
                           [0.0, 1389.26, 548.75],
                           [0.0, 0.0, 1.0]],
                    "dist": [[-0.4525, 0.2886, 0.0006, 0.0009, -0.1396]],
                },
                "cam_b": {
                    "model": "standard",
                    "image_size": [1920, 1080],
                    "K": [[1076.55, 0.0, 961.15],
                           [0.0, 1076.70, 550.00],
                           [0.0, 0.0, 1.0]],
                    "dist": [[-0.3603, 0.1617, -0.0033, -0.0010, -0.0686]],
                },
            }
        }
        (d / "intrinsic.json").write_text(_json.dumps(intr))
        (d / "intrinsic_rms.json").write_text(_json.dumps({"cam_a": 0.8712, "cam_b": 1.2345}))

        r = c.get("/api/p/rig/intrinsic-summary")
        assert r.status_code == 200
        body = r.json()

        assert "rms_gate_px" in body
        assert set(body["cameras"].keys()) == {"cam_a", "cam_b"}

        ca = body["cameras"]["cam_a"]
        assert ca["image_size"] == [1920, 1080]
        assert ca["fx"] == pytest.approx(1387.16, abs=0.01)
        assert ca["fy"] == pytest.approx(1389.26, abs=0.01)
        assert ca["cx"] == pytest.approx(942.99, abs=0.01)
        assert ca["cy"] == pytest.approx(548.75, abs=0.01)
        assert len(ca["K"]) == 3 and len(ca["K"][0]) == 3
        assert len(ca["dist"]) == 5
        assert ca["rms"] == pytest.approx(0.8712, abs=1e-4)

        cb = body["cameras"]["cam_b"]
        assert cb["rms"] == pytest.approx(1.2345, abs=1e-4)


def test_intrinsic_summary_rms_null_when_no_rms_sidecar():
    """rms is null per camera when intrinsic_rms.json is absent (old solve)."""
    import json as _json
    with _client() as c:
        c.post("/api/projects", json={"name": "rig",
                                      "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        d = Settings().data_dir / "rig" / "work"
        d.mkdir(parents=True, exist_ok=True)
        intr = {
            "cameras": {
                "cam_a": {
                    "model": "standard",
                    "image_size": [1920, 1080],
                    "K": [[1000.0, 0.0, 960.0],
                           [0.0, 1000.0, 540.0],
                           [0.0, 0.0, 1.0]],
                    "dist": [[0.1, -0.05, 0.0, 0.0, 0.0]],
                }
            }
        }
        (d / "intrinsic.json").write_text(_json.dumps(intr))
        # no intrinsic_rms.json written

        r = c.get("/api/p/rig/intrinsic-summary")
        assert r.status_code == 200
        body = r.json()
        assert body["cameras"]["cam_a"]["rms"] is None


def test_intrinsic_summary_404_for_unknown_project():
    with _client() as c:
        assert c.get("/api/p/nope/intrinsic-summary").status_code == 404


def test_phases_board_page_200_with_intrinsic_done(tmp_path):
    """Board page still returns 200 when intrinsic.json exists (panel visibility check)."""
    import json as _json
    with _client() as c:
        c.post("/api/projects", json={"name": "rig",
                                      "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        d = Settings().data_dir / "rig" / "work"
        d.mkdir(parents=True, exist_ok=True)
        (d / "intrinsic.json").write_text(_json.dumps({"cameras": {}}))
        assert c.get("/p/rig").status_code == 200
        html = c.get("/p/rig").text
        assert "intrinsic-results-card" in html   # panel present in DOM
