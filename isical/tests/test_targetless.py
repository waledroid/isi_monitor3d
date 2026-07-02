"""isical targetless-extrinsics wiring (hermetic — no camera / ONNX / Multical).

Covers the Stage-3 backend surface the frontend drives: the method selector routes
correctly, scale-references persist + validate, the extrinsic job dispatches to the
targetless runner when selected, the report/stage-image endpoints behave, and the
runner surfaces a clear ONNX-weights-missing error on the (hermetic) no-weights path.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from isical.app import create_app
from isical.config import Settings


def _client():
    return TestClient(create_app(Settings()))


def _mk_rig(c, name="rig"):
    c.post("/api/projects", json={
        "name": name,
        "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
        "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})


# --- method selector ---------------------------------------------------------


def test_extrinsic_method_defaults_aprilgrid_and_persists():
    from isical.core.project import load_project
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/extrinsic-method").json()["method"] == "aprilgrid"
        r = c.put("/api/p/rig/extrinsic-method", json={"method": "targetless"})
        assert r.json() == {"ok": True, "method": "targetless"}
        assert c.get("/api/p/rig/extrinsic-method").json()["method"] == "targetless"
        # persisted to calib.yaml
        data_dir = c.app.state.settings.data_dir
        assert load_project(data_dir / "rig").extrinsic_method == "targetless"


def test_extrinsic_method_rejects_unknown():
    with _client() as c:
        _mk_rig(c)
        assert c.put("/api/p/rig/extrinsic-method", json={"method": "nope"}).status_code == 422


# --- scale references --------------------------------------------------------


def _refs(n=3):
    return [{"p1_a": [10.0, 20.0], "p1_b": [12.0, 20.0],
             "p2_a": [50.0, 60.0], "p2_b": [52.0, 60.0],
             "distance_m": 1.0 + i} for i in range(n)]


def test_scale_references_roundtrip():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/scale-references").json() == {"references": [], "count": 0}
        r = c.put("/api/p/rig/scale-references", json={"references": _refs(3)})
        assert r.json()["ok"] is True and r.json()["enough"] is True
        got = c.get("/api/p/rig/scale-references").json()
        assert got["count"] == 3
        # persisted on disk where the runner reads it
        p = c.app.state.settings.data_dir / "rig" / "work" / "scale_references.json"
        assert len(json.loads(p.read_text())) == 3


def test_scale_references_under_three_flags_not_enough():
    with _client() as c:
        _mk_rig(c)
        r = c.put("/api/p/rig/scale-references", json={"references": _refs(2)})
        assert r.json()["enough"] is False


def test_scale_reference_rejects_nonpositive_distance():
    with _client() as c:
        _mk_rig(c)
        bad = _refs(1)
        bad[0]["distance_m"] = 0.0
        assert c.put("/api/p/rig/scale-references", json={"references": bad}).status_code == 422


# --- feature matches (snap-assist source) ------------------------------------


def test_feature_matches_empty_without_pair_or_weights():
    """No captured pair (and no weights) → count:0 with a reason; UI falls back to
    manual marking. Never raises."""
    with _client() as c:
        _mk_rig(c)
        r = c.get("/api/p/rig/feature-matches").json()
        assert r["count"] == 0
        assert r["matches"] == []
        assert isinstance(r.get("reason"), str) and r["reason"]


def test_feature_matches_serves_stored_synthetic_set():
    """A stored work/feature_matches.json is served verbatim (hermetic snap demo).

    The returned coordinates are exactly the ones snap would auto-fill the
    ScaleReference b-points from — a round-trip of the snap-relevant coords."""
    with _client() as c:
        _mk_rig(c)
        data_dir = c.app.state.settings.data_dir
        work = data_dir / "rig" / "work"
        work.mkdir(parents=True, exist_ok=True)
        synthetic = {"matches": [
            {"a": [100.0, 200.0], "b": [110.0, 205.0], "score": 0.9},
            {"a": [300.0, 400.0], "b": [312.0, 402.0], "score": 0.8},
        ]}
        (work / "feature_matches.json").write_text(json.dumps(synthetic))
        r = c.get("/api/p/rig/feature-matches").json()
        assert r["count"] == 2
        assert r["matches"][0]["a"] == [100.0, 200.0]
        assert r["matches"][0]["b"] == [110.0, 205.0]
        assert r["matches"][1]["b"] == [312.0, 402.0]


def test_feature_matches_404_for_unknown_project():
    with _client() as c:
        assert c.get("/api/p/nope/feature-matches").status_code == 404


def test_scale_reference_snapped_flag_roundtrips():
    """The display-only `snapped` flag persists but doesn't alter the solve-relevant
    shape (p1_a,p1_b,p2_a,p2_b,distance_m)."""
    with _client() as c:
        _mk_rig(c)
        refs = _refs(3)
        refs[0]["snapped"] = True
        r = c.put("/api/p/rig/scale-references", json={"references": refs})
        assert r.json()["ok"] is True
        got = c.get("/api/p/rig/scale-references").json()["references"]
        assert got[0]["snapped"] is True
        assert got[1]["snapped"] is False
        # solve-relevant coordinates untouched
        assert got[0]["p1_a"] == [10.0, 20.0]


# --- stage images + report ---------------------------------------------------


def test_targetless_stage_and_report_empty_before_solve():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/targetless-stages").json() == {"stages": []}
        assert c.get("/api/p/rig/targetless-report").json() == {"report": None}
        assert c.get("/targetless-stage/rig/pair").status_code == 404


def test_stage_image_path_traversal_guarded():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/targetless-stage/rig/..%2f..%2fcalib").status_code == 404


# --- result matrices (Result cell) ------------------------------------------


def test_calibration_matrices_null_before_solve():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/calibration-matrices").json() == {"matrices": None}


def test_calibration_matrices_after_solve():
    """calibration.json on disk → per-camera R/t/RMS + anchor exposed for the
    notebook Result cell."""
    with _client() as c:
        _mk_rig(c)
        data_dir = c.app.state.settings.data_dir
        calib = {
            "version": 1, "created_at": "2026-01-01T00:00:00",
            "floor_anchor_method": "planefit", "floor_origin_note": "",
            "calibration_mode": "multical_full",
            "cameras": {
                "cam_a": {"camera_id": "cam_a", "image_size_wh": [1280, 960],
                          "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "D": [0, 0, 0, 0, 0],
                          "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0],
                          "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          "P": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                          "reprojection_rms_px": 0.9},
                "cam_b": {"camera_id": "cam_b", "image_size_wh": [1280, 960],
                          "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "D": [0, 0, 0, 0, 0],
                          "R": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "t": [0.5, 0, 0.1],
                          "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          "P": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                          "reprojection_rms_px": 1.3},
            },
        }
        (data_dir / "rig" / "calibration.json").write_text(json.dumps(calib))
        m = c.get("/api/p/rig/calibration-matrices").json()["matrices"]
        assert m["floor_anchor_method"] == "planefit"
        assert m["calibration_mode"] == "multical_full"
        assert set(m["cameras"]) == {"cam_a", "cam_b"}
        assert m["cameras"]["cam_b"]["t"] == [0.5, 0, 0.1]
        assert m["cameras"]["cam_b"]["R"] == [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert m["cameras"]["cam_a"]["reprojection_rms_px"] == 0.9


def test_calibration_matrices_tolerates_garbage_json():
    with _client() as c:
        _mk_rig(c)
        data_dir = c.app.state.settings.data_dir
        (data_dir / "rig" / "calibration.json").write_text("{ not json")
        assert c.get("/api/p/rig/calibration-matrices").json() == {"matrices": None}


# --- extrinsic job dispatch --------------------------------------------------


def test_extrinsic_job_dispatches_to_targetless_runner(monkeypatch):
    """When method=targetless, POST /run/extrinsic must call the targetless runner."""
    called = {}

    def _fake_targetless(d):
        called["targetless"] = str(d)
        return {"ok": True, "method": "targetless"}

    monkeypatch.setattr("isical.api.routes_jobs.run_extrinsic_targetless", _fake_targetless)
    with _client() as c:
        _mk_rig(c)
        c.put("/api/p/rig/extrinsic-method", json={"method": "targetless"})
        r = c.post("/api/p/rig/run/extrinsic")
        assert r.status_code == 200
        # let the single-worker JobRunner drain
        import time
        for _ in range(50):
            jobs = c.get("/api/jobs").json()["jobs"]
            if jobs and jobs[0]["state"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert called.get("targetless") is not None


def test_targetless_runner_missing_scale_refs_errors(tiny_project):
    """No scale references marked → a clear ValueError (before touching ONNX)."""
    from isical.core.runners import run_extrinsic_targetless
    pdir, _cfg = tiny_project
    # intrinsic.json + pairs so the failure is specifically the missing refs
    (pdir / "work").mkdir(parents=True, exist_ok=True)
    (pdir / "work" / "intrinsic.json").write_text(json.dumps({"cameras": {
        "cam_a": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                  "image_size": [1280, 960]},
        "cam_b": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                  "image_size": [1280, 960]}}}))
    import cv2
    import numpy as np
    # targetless reads its OWN captures (targetless/), not the board extrinsic/ ones
    for cid in ("cam_a", "cam_b"):
        d = pdir / "targetless" / cid
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{cid}_000.jpg"), np.zeros((960, 1280, 3), np.uint8))
    with pytest.raises(ValueError, match="scale references"):
        run_extrinsic_targetless(pdir)
