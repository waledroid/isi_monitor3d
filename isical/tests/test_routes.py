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


def test_capture_config_get_and_put_with_floor():
    from isical.core.project import EXTRINSIC_TARGET_MIN, load_project
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        cfg = c.get("/api/p/rig/capture-config").json()
        assert cfg["extrinsic_target"] == 10                 # new lower default
        assert cfg["extrinsic_target_min"] == EXTRINSIC_TARGET_MIN
        # set a custom (valid) target → persisted to calib.yaml
        r = c.put("/api/p/rig/capture-config", json={"extrinsic_target": 8})
        assert r.status_code == 200 and r.json()["extrinsic_target"] == 8
        assert load_project(Settings().data_dir / "rig").capture.extrinsic_target == 8
        # below the floor is rejected at the schema (422)
        assert c.put("/api/p/rig/capture-config",
                     json={"extrinsic_target": 1}).status_code == 422


def test_board_config_get_and_put_roundtrip():
    from isical.core.project import load_project
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        # fresh project reverses to the 18 cm / 5.4 cm defaults
        bc = c.get("/api/p/rig/board-config").json()
        assert bc["tag_length_cm"] == pytest.approx(18)
        assert bc["tag_gap_cm"] == pytest.approx(5.4)
        assert bc["tag_length_m"] == pytest.approx(0.18)
        assert bc["tag_spacing"] == pytest.approx(0.3)
        # operator measures a 20 cm tag with a 4 cm gap → 0.20 / 0.20 persisted
        r = c.put("/api/p/rig/board-config", json={"tag_length_cm": 20, "tag_gap_cm": 4})
        assert r.status_code == 200
        assert r.json()["tag_length_m"] == pytest.approx(0.20)
        assert r.json()["tag_spacing"] == pytest.approx(0.20)
        board = load_project(Settings().data_dir / "rig").board
        assert board.tag_length_m == pytest.approx(0.20)
        assert board.tag_spacing == pytest.approx(0.20)
        # and it round-trips back to cm
        bc2 = c.get("/api/p/rig/board-config").json()
        assert bc2["tag_length_cm"] == pytest.approx(20)
        assert bc2["tag_gap_cm"] == pytest.approx(4)


def test_board_config_rejects_bad_input():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        # tag length 0 / out of range → 422
        assert c.put("/api/p/rig/board-config",
                     json={"tag_length_cm": 0, "tag_gap_cm": 4}).status_code == 422
        assert c.put("/api/p/rig/board-config",
                     json={"tag_length_cm": 200, "tag_gap_cm": 4}).status_code == 422
        assert c.put("/api/p/rig/board-config",
                     json={"tag_length_cm": 18, "tag_gap_cm": 60}).status_code == 422


def test_extrinsic_capture_page_has_board_measure_inputs():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        html = c.get("/p/rig/capture/extrinsic").text
        assert 'id="tag-length-cm"' in html
        assert 'id="tag-gap-cm"' in html
        assert 'value="18' in html and 'value="5.4"' in html


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


def test_floor_preview_route_targets_camera(monkeypatch):
    """POST /floor/{cam}/preview opens a single-camera preview; the manager exposes
    it ONLY for that camera, and grabbing reuses the preview's open source."""
    from isical.capture import session as sess_mod
    from isical.tests.test_session import _StubCharuco, _StubSource

    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    monkeypatch.setattr(sess_mod, "_open_source", lambda spec, cid: _StubSource(80))
    with _client() as c:
        c.post("/api/projects", json={"name": "rig",
                                      "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        assert c.post("/api/p/rig/floor/cam_b/preview").status_code == 200
        mgr = c.app.state.capture
        # only cam_b is being previewed (floor cam_b shows cam_b, not cam_a)
        assert mgr.floor("rig", "cam_b") is not None
        assert mgr.floor("rig", "cam_a") is None
        # unknown camera rejected
        assert c.post("/api/p/rig/floor/nope/preview").status_code == 404
        # grab reuses the live preview source (no 409, even though a session is "busy")
        import time as _t
        for _ in range(200):
            if mgr.floor("rig", "cam_b") and mgr.floor("rig", "cam_b")._latest_good:
                break
            _t.sleep(0.01)
        r = c.post("/api/p/rig/floor/cam_b")
        assert r.status_code == 200 and r.json()["corners"] >= 4
        c.post("/api/p/rig/floor/cam_b/preview/stop")
        assert mgr.floor("rig", "cam_b") is None


def test_floor_shot_409_when_full_capture_active(monkeypatch):
    """With a full capture session holding the cameras and NO floor preview, the
    floor grab still 409s (cameras busy) — the original safety is preserved."""
    from isical.capture import session as sess_mod
    from isical.tests.test_session import _StubCharuco, _StubSource

    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    monkeypatch.setattr(sess_mod, "_open_source", lambda spec, cid: _StubSource(200))
    with _client() as c:
        c.post("/api/projects", json={"name": "rig",
                                      "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        c.post("/api/p/rig/capture/intrinsic/start?cam=cam_a")
        try:
            assert c.post("/api/p/rig/floor/cam_a").status_code == 409
        finally:
            c.post("/api/p/rig/capture/intrinsic/stop")


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


def test_list_extrinsic_shots_both_cameras():
    """The Boards notebook's Extrinsic-pairs cell reads the same shot-listing route
    for phase=extrinsic across both cameras (read-only)."""
    import cv2
    import numpy as np

    from isical.config import Settings
    with _client() as c:
        c.post("/api/projects", json={
            "name": "rig",
            "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
            "cam_b": {"type": "rtsp", "url": "rtsp://x/b"},
        })
        data = Settings().data_dir / "rig"
        for cam, n in (("cam_a", 2), ("cam_b", 1)):
            d = data / "extrinsic" / cam
            for i in range(n):
                cv2.imwrite(str(d / f"{cam}_{i:03d}.jpg"), np.zeros((48, 64, 3), np.uint8))
        ra = c.get("/api/p/rig/shots/extrinsic/cam_a").json()
        rb = c.get("/api/p/rig/shots/extrinsic/cam_b").json()
        assert ra["count"] == 2 and rb["count"] == 1
        assert ra["shots"][0]["file"] == "cam_a_000.jpg"
        # bytes serve through the existing shot route for the extrinsic sub-dir
        assert c.get("/shots/rig/extrinsic/cam_a/cam_a_000.jpg").status_code == 200


def test_floor_shots_empty_then_present():
    """floor-shots reports per-camera presence; empty gracefully, then serves bytes."""
    import cv2
    import numpy as np

    from isical.config import Settings
    with _client() as c:
        c.post("/api/projects", json={
            "name": "rig",
            "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
            "cam_b": {"type": "rtsp", "url": "rtsp://x/b"},
        })
        r = c.get("/api/p/rig/floor-shots").json()
        assert set(r["cameras"]) == {"cam_a", "cam_b"}
        assert r["cameras"]["cam_a"]["present"] is False
        assert r["cameras"]["cam_a"]["file"] is None

        floor = Settings().data_dir / "rig" / "floor"
        floor.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(floor / "cam_a.jpg"), np.zeros((48, 64, 3), np.uint8))
        r = c.get("/api/p/rig/floor-shots").json()
        assert r["cameras"]["cam_a"] == {"present": True, "file": "cam_a.jpg"}
        assert r["cameras"]["cam_b"]["present"] is False
        assert c.get("/floor-shot/rig/cam_a.jpg").status_code == 200


def test_floor_shot_serve_path_guarded():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.get("/floor-shot/rig/cam_a.png").status_code == 404    # not .jpg
        assert c.get("/floor-shot/rig/cam_a.jpg").status_code == 404    # absent


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
