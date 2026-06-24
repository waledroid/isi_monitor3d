"""Studio routes via TestClient (hermetic — no camera/Multical)."""

from __future__ import annotations

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
