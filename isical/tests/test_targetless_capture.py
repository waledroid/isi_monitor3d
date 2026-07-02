"""Targetless capture flow — self-contained textured-scene stereo-pair capture.

Hermetic (no cameras, no ONNX, no Multical). Proves the targetless path is fully
independent of the board/AprilGrid method:

  * texture_score is a cheap per-frame feature/sharpness readout (no ONNX);
  * TargetlessSession shows BOTH cameras live and captures SYNCHRONIZED stereo
    pairs on a MANUAL trigger into targetless/{cam}/ (NOT extrinsic/, NO board gate);
  * the routes start/stop/capture-pair/status/shots-list behave;
  * run_extrinsic_targetless reads targetless/ and writes calibration_targetless.json
    (never the board calibration.json);
  * the board capture phase + its extrinsic/ dir are untouched.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from isical.capture import session as sess_mod


class _StubFrame:
    def __init__(self, img):
        self.image = img


class _StubSource:
    def __init__(self, n=200):
        self._n = n

    def start(self):
        pass

    def frames(self):
        for i in range(self._n):
            yield _StubFrame(np.full((120, 160, 3), i % 255, np.uint8))
            time.sleep(0.002)

    def stop(self):
        pass


def _two_cam_project(tmp_path):
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    return pdir, load_project(pdir)


# --- texture quality readout (cheap, no ONNX) --------------------------------


def test_texture_score_more_features_scores_higher():
    from isical.capture.session import texture_score
    flat = np.full((120, 160, 3), 127, np.uint8)          # no texture
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)   # rich texture
    s_flat = texture_score(flat)
    s_rich = texture_score(noisy)
    assert s_rich["features"] > s_flat["features"]
    assert s_rich["blur_var"] > s_flat["blur_var"]
    assert isinstance(s_flat["ok"], bool)


# --- targetless capture session ----------------------------------------------


def test_targetless_capture_pair_writes_to_targetless_dir(tmp_path):
    pdir, cfg = _two_cam_project(tmp_path)
    s = sess_mod.TargetlessSession(pdir, cfg,
                                   source_factory=lambda spec, cid: _StubSource(400))
    s.start()
    # wait until both live workers have a frame staged
    for _ in range(300):
        if all(w.latest_jpeg() is not None for w in s.workers.values()):
            break
        time.sleep(0.01)
    res1 = s.capture_pair()
    res2 = s.capture_pair()
    s.stop()
    assert res1["pair_count"] == 1 and res2["pair_count"] == 2
    a = sorted((pdir / "targetless" / "cam_a").glob("*.jpg"))
    b = sorted((pdir / "targetless" / "cam_b").glob("*.jpg"))
    assert len(a) == 2 and len(b) == 2
    # NOTHING written under the board extrinsic/ dir
    assert not (pdir / "extrinsic").exists() or not list(
        (pdir / "extrinsic").rglob("*.jpg"))


def test_targetless_capture_pair_before_frames_returns_reason(tmp_path):
    """Manual trigger before both cams have a frame → no pair, a clear reason."""
    pdir, cfg = _two_cam_project(tmp_path)
    s = sess_mod.TargetlessSession(pdir, cfg,
                                   source_factory=lambda spec, cid: _StubSource(0))
    # do not start the workers → no staged frames
    res = s.capture_pair()
    assert res["captured"] is False
    assert res["pair_count"] == 0
    assert isinstance(res.get("reason"), str) and res["reason"]


def test_targetless_status_reports_texture(tmp_path):
    pdir, cfg = _two_cam_project(tmp_path)
    s = sess_mod.TargetlessSession(pdir, cfg,
                                   source_factory=lambda spec, cid: _StubSource(400))
    s.start()
    for _ in range(300):
        st = s.status()
        if all("texture" in cam for cam in st["cameras"].values()):
            break
        time.sleep(0.01)
    st = s.status()
    s.stop()
    assert st["phase"] == "targetless"
    assert set(st["cameras"]) == {"cam_a", "cam_b"}
    for cam in st["cameras"].values():
        assert "texture" in cam and "features" in cam["texture"]


def test_targetless_manager_lifecycle(tmp_path):
    pdir, cfg = _two_cam_project(tmp_path)
    mgr = sess_mod.CaptureManager()
    mgr.start_targetless("rig", pdir, cfg,
                         source_factory=lambda spec, cid: _StubSource(400))
    try:
        assert mgr.targetless("rig") is not None
        assert mgr.targetless("other") is None
    finally:
        mgr.stop_targetless()
    assert mgr.targetless("rig") is None


# --- board isolation: the board capture phase is untouched -------------------


def test_board_extrinsic_phase_still_uses_extrinsic_dir(tmp_path, monkeypatch):
    """A regression guard: the board 'extrinsic' capture phase writes to extrinsic/
    (not targetless/) exactly as before."""
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    cfg = load_project(pdir)
    s = sess_mod.CaptureSession(pdir, cfg, "extrinsic",
                                source_factory=lambda spec, cid: _StubSource(0))
    assert s.workers["cam_a"].out_dir == pdir / "extrinsic" / "cam_a"


# --- solve reads targetless/ + writes calibration_targetless.json ------------


class _FakeMatcher:
    """A stand-in matcher (never actually invoked — solve is monkeypatched)."""


def _intrinsic_json(pdir):
    (pdir / "work").mkdir(parents=True, exist_ok=True)
    (pdir / "work" / "intrinsic.json").write_text(json.dumps({"cameras": {
        "cam_a": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]],
                  "dist": [0, 0, 0, 0, 0], "image_size": [1280, 960]},
        "cam_b": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]],
                  "dist": [0, 0, 0, 0, 0], "image_size": [1280, 960]}}}))


def test_targetless_solve_reads_targetless_dir_and_writes_own_output(tmp_path, monkeypatch):
    import cv2

    from isical.core import runners
    pdir, _cfg = _two_cam_project(tmp_path)
    _intrinsic_json(pdir)
    # scale refs
    (pdir / "work" / "scale_references.json").write_text(json.dumps([
        {"p1_a": [10, 20], "p1_b": [12, 20], "p2_a": [50, 60], "p2_b": [52, 60],
         "distance_m": 1.0 + i} for i in range(3)]))
    # targetless captures ONLY (no extrinsic/)
    for cid in ("cam_a", "cam_b"):
        d = pdir / "targetless" / cid
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{cid}_000.jpg"), np.zeros((960, 1280, 3), np.uint8))

    seen = {}

    class _Cam:
        def __init__(self, rms):
            self.reprojection_rms_px = rms

    class _Calib:
        def __init__(self):
            self.cameras = {"cam_a": _Cam(0.7), "cam_b": _Cam(0.9)}

    def _fake_run_targetless(**kw):
        seen.update(kw)
        # emulate the runner writing the calibration to output_path
        from pathlib import Path
        Path(kw["output_path"]).write_text(json.dumps({"cameras": {
            "cam_a": {"reprojection_rms_px": 0.7},
            "cam_b": {"reprojection_rms_px": 0.9}}}))
        return _Calib()

    monkeypatch.setattr(runners, "_import_run_targetless",
                        lambda: _fake_run_targetless)
    res = runners.run_extrinsic_targetless(pdir)
    # reads targetless/, not extrinsic/
    assert seen["pair_dir_a"] == pdir / "targetless" / "cam_a"
    assert seen["pair_dir_b"] == pdir / "targetless" / "cam_b"
    # writes its OWN output — NOT the board calibration.json
    assert str(seen["output_path"]).endswith("calibration_targetless.json")
    assert (pdir / "calibration_targetless.json").exists()
    assert not (pdir / "calibration.json").exists()
    assert res["method"] == "targetless"
    assert res["calibration_json"].endswith("calibration_targetless.json")


def test_targetless_solve_missing_pairs_errors(tmp_path):
    from isical.core.runners import run_extrinsic_targetless
    pdir, _cfg = _two_cam_project(tmp_path)
    _intrinsic_json(pdir)
    (pdir / "work" / "scale_references.json").write_text(json.dumps([
        {"p1_a": [10, 20], "p1_b": [12, 20], "p2_a": [50, 60], "p2_b": [52, 60],
         "distance_m": 1.0} for _ in range(3)]))
    with pytest.raises(ValueError, match="targetless"):
        run_extrinsic_targetless(pdir)
