"""Per-pair feature-match DIAGNOSTIC preview (targetless cell ②).

Hermetic — a FakeMatcher injects deterministic correspondences so no ONNX/weights
are needed. Covers: the runner renders one match + one keypoints image per captured
pair into work/targetless_diag/ with the right counts, the list route returns them,
the image server serves them (path-guarded), the caching skips fresh pairs, the job
wiring dispatches, the weights-missing path surfaces a clean error, and the whole
thing is ADDITIVE — it never solves, never writes calibration_targetless.json, and
never touches board files.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
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


def _seed_project(pdir, n_pairs=2):
    """intrinsic.json + n captured targetless pairs (real jpgs)."""
    (pdir / "work").mkdir(parents=True, exist_ok=True)
    (pdir / "work" / "intrinsic.json").write_text(json.dumps({"cameras": {
        "cam_a": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]],
                  "dist": [0, 0, 0, 0, 0], "image_size": [1280, 960]},
        "cam_b": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]],
                  "dist": [0, 0, 0, 0, 0], "image_size": [1280, 960]}}}))
    rng = np.random.default_rng(0)
    for cid in ("cam_a", "cam_b"):
        d = pdir / "targetless" / cid
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_pairs):
            img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{cid}_{i:03d}.jpg"), img)


class _FakeMatcher:
    """Deterministic, geometry-plausible correspondences (a horizontal shift) so
    findEssentialMat yields a real inlier mask — no ONNX/weights."""

    def __init__(self, models_dir=None, **kw):
        pass

    def match(self, img_a, img_b):
        rng = np.random.default_rng(1)
        pa = rng.uniform([50, 50], [270, 190], size=(40, 2))
        pb = pa + np.array([8.0, 0.0])
        scores = np.ones(len(pa))
        return pa, pb, scores


def test_runner_renders_one_diag_per_pair(tiny_project, monkeypatch):
    from isical.core import runners
    pdir, _cfg = tiny_project
    _seed_project(pdir, n_pairs=2)
    monkeypatch.setattr(runners, "_build_diag_matcher", lambda md: _FakeMatcher())
    out = runners.preview_targetless_matches(pdir)
    assert out["count"] == 2
    diag = pdir / "work" / "targetless_diag"
    assert (diag / "pair_000_matches.jpg").is_file()
    assert (diag / "pair_000_keypoints.jpg").is_file()
    assert (diag / "pair_001_matches.jpg").is_file()
    for p in out["pairs"]:
        assert p["n_matches"] == 40
        # a horizontal shift is a valid epipolar geometry → most matches are inliers
        assert p["n_inliers"] >= 5


def test_runner_caches_fresh_pairs(tiny_project, monkeypatch):
    from isical.core import runners
    pdir, _cfg = tiny_project
    _seed_project(pdir, n_pairs=1)
    calls = {"n": 0}

    class _Counting(_FakeMatcher):
        def match(self, a, b):
            calls["n"] += 1
            return super().match(a, b)

    monkeypatch.setattr(runners, "_build_diag_matcher", lambda md: _Counting())
    runners.preview_targetless_matches(pdir)
    assert calls["n"] == 1
    # second run: images are fresh (newer than captures) → matcher never invoked
    out = runners.preview_targetless_matches(pdir)
    assert calls["n"] == 1
    assert out["pairs"][0]["cached"] is True


def test_runner_no_pairs_errors(tiny_project):
    from isical.core.runners import preview_targetless_matches
    pdir, _cfg = tiny_project
    (pdir / "work").mkdir(parents=True, exist_ok=True)
    (pdir / "work" / "intrinsic.json").write_text(json.dumps({"cameras": {
        "cam_a": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5,
                  "image_size": [320, 240]},
        "cam_b": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5,
                  "image_size": [320, 240]}}}))
    with pytest.raises(ValueError, match="no targetless stereo pairs"):
        preview_targetless_matches(pdir)


def test_runner_weights_missing_surfaces_clean_error(tiny_project):
    """No ONNX weights → MatcherWeightsMissing, not a bare crash."""
    from calibration.feature_extrinsics import MatcherWeightsMissing
    from isical.core import runners
    pdir, _cfg = tiny_project
    _seed_project(pdir, n_pairs=1)
    # point the matcher builder at an empty models dir
    real = runners._build_diag_matcher
    with pytest.raises(MatcherWeightsMissing):
        # exercise the real builder against a dir with no weights
        real(pdir / "no_models")


def test_list_route_returns_rendered_pairs(tiny_project, monkeypatch):
    from isical.core import runners
    with _client() as c:
        _mk_rig(c)
        pdir = c.app.state.settings.data_dir / "rig"
        _seed_project(pdir, n_pairs=2)
        monkeypatch.setattr(runners, "_build_diag_matcher", lambda md: _FakeMatcher())
        runners.preview_targetless_matches(pdir)
        r = c.get("/api/p/rig/targetless-match-preview").json()
        assert r["count"] == 2
        assert r["pairs"][0]["matches"] == "pair_000_matches.jpg"
        assert r["pairs"][0]["keypoints"] == "pair_000_keypoints.jpg"
        assert r["pairs"][0]["n_matches"] == 40
        # image server serves both, path-guarded
        assert c.get("/targetless-diag/rig/pair_000_matches.jpg").status_code == 200
        assert c.get("/targetless-diag/rig/pair_000_keypoints.jpg").status_code == 200


def test_list_route_empty_before_preview():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/targetless-match-preview").json() == {"count": 0, "pairs": []}


def test_diag_image_path_guarded():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/targetless-diag/rig/..%2f..%2fcalib.yaml").status_code == 404
        assert c.get("/targetless-diag/rig/nope.jpg").status_code == 404


def test_diag_job_dispatches(monkeypatch):
    """POST /run/targetless-diag calls the preview runner via the JobRunner."""
    import time
    called = {}

    def _fake(d):
        called["d"] = str(d)
        return {"ok": True, "count": 0, "pairs": []}

    monkeypatch.setattr("isical.api.routes_jobs.preview_targetless_matches", _fake)
    with _client() as c:
        _mk_rig(c)
        r = c.post("/api/p/rig/run/targetless-diag")
        assert r.status_code == 200
        for _ in range(50):
            jobs = c.get("/api/jobs").json()["jobs"]
            if jobs and jobs[0]["state"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert called.get("d") is not None


def test_preview_never_writes_calibration_or_refs(tiny_project, monkeypatch):
    """The diagnostic is additive: no calibration_targetless.json, no scale_references."""
    from isical.core import runners
    pdir, _cfg = tiny_project
    _seed_project(pdir, n_pairs=1)
    monkeypatch.setattr(runners, "_build_diag_matcher", lambda md: _FakeMatcher())
    runners.preview_targetless_matches(pdir)
    assert not (pdir / "calibration_targetless.json").exists()
    assert not (pdir / "calibration.json").exists()
    assert not (pdir / "work" / "scale_references.json").exists()
    # everything it wrote is under work/targetless_diag/
    assert (pdir / "work" / "targetless_diag").is_dir()
