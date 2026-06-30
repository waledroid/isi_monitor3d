"""CaptureSession auto-snap with a stub frame source + stub detector."""

from __future__ import annotations

import time

import numpy as np

from isical.capture import session as sess_mod
from isical.capture.detect import Detection


class _StubFrame:
    def __init__(self, img):
        self.image = img


class _StubSource:
    """Yields N frames then ends; mimics RtspFrameSource start()/frames()/stop()."""
    def __init__(self, n=40):
        self._n = n

    def start(self):
        pass

    def frames(self):
        for i in range(self._n):
            yield _StubFrame(np.full((120, 160, 3), i % 255, np.uint8))
            time.sleep(0.002)

    def stop(self):
        pass


class _StubCharuco:
    """Detector that returns a snap-worthy, steady, pose-shifting Detection.

    Centroid shifts every 4 frames so the novelty gate lets one snap per pose;
    a fixed corner set keeps motion ~0 (steady) between frames of the same pose.
    """
    def __init__(self, *_a, **_k):
        self._i = 0

    def detect(self, frame):
        pose = self._i // 4
        self._i += 1
        c = (0.1 + 0.08 * pose, 0.5)
        pts = np.full((20, 2), 50.0, np.float32) + pose      # steady within a pose
        return Detection(n=20, centroid=c, corners_px=pts, blur_var=300.0, coverage=1.0)

    def annotate(self, frame, det):
        return frame


def _project(tmp_path, target=3):
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    cfg = load_project(pdir)
    cfg.capture.target_per_camera = target
    return pdir, cfg


def test_intrinsic_autosnaps_to_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    pdir, cfg = _project(tmp_path, target=3)
    s = sess_mod.CaptureSession(pdir, cfg, "intrinsic",
                                source_factory=lambda spec, cid: _StubSource(40))
    s.start()
    # wait for the worker thread to hit its target
    for _ in range(200):
        if s.workers["cam_a"].count >= 3:
            break
        time.sleep(0.02)
    s.stop()
    saved = list((pdir / "intrinsic" / "cam_a").glob("*.jpg"))
    assert len(saved) == 3                               # exactly target, novel poses
    st = s.status()
    assert st["cameras"]["cam_a"]["count"] == 3


def test_camera_open_failure_surfaces(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    pdir, cfg = _project(tmp_path)

    def _boom(spec, cid):
        raise RuntimeError("no such camera")

    s = sess_mod.CaptureSession(pdir, cfg, "intrinsic", source_factory=_boom)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert "camera error" in s.workers["cam_a"].status


def test_grab_floor_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    pdir, cfg = _project(tmp_path)
    res = sess_mod.grab_floor_shot(pdir, cfg, "cam_a",
                                   source_factory=lambda spec, cid: _StubSource(40),
                                   settle_frames=2)
    assert res["camera"] == "cam_a" and res["corners"] >= 4
    assert (pdir / "floor" / "cam_a.jpg").exists()


def test_grab_floor_shot_no_board(tmp_path, monkeypatch):
    class _Blank:
        def __init__(self, *_a, **_k): pass
        def detect(self, frame):
            from isical.capture.detect import Detection
            return Detection(n=0)
        def annotate(self, frame, det): return frame
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _Blank)
    pdir, cfg = _project(tmp_path)
    import pytest
    with pytest.raises(ValueError):
        sess_mod.grab_floor_shot(pdir, cfg, "cam_a",
                                 source_factory=lambda spec, cid: _StubSource(80),
                                 settle_frames=2)


def test_floor_preview_grabs_from_live_source(tmp_path, monkeypatch):
    """FloorPreview opens the camera, keeps a live JPEG, and grab() writes
    floor/<cam>.jpg from the SAME source (the well-detected latest frame)."""
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    pdir, cfg = _project(tmp_path)
    fp = sess_mod.FloorPreview(pdir, cfg, "cam_a",
                               source_factory=lambda spec, cid: _StubSource(60))
    fp.start()
    for _ in range(200):
        if fp.latest_jpeg() is not None and fp._latest_good is not None:
            break
        time.sleep(0.01)
    assert fp.latest_jpeg() is not None                     # live MJPEG frame ready
    res = fp.grab()
    fp.stop()
    assert res["camera"] == "cam_a" and res["corners"] >= 4
    assert (pdir / "floor" / "cam_a.jpg").exists()


def test_floor_preview_grab_without_board_raises(tmp_path, monkeypatch):
    class _Blank:
        def __init__(self, *_a, **_k): pass
        def detect(self, frame):
            from isical.capture.detect import Detection
            return Detection(n=0)
        def annotate(self, frame, det): return frame
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _Blank)
    pdir, cfg = _project(tmp_path)
    fp = sess_mod.FloorPreview(pdir, cfg, "cam_a",
                               source_factory=lambda spec, cid: _StubSource(10))
    fp.start()
    time.sleep(0.15)
    fp.stop()
    import pytest
    with pytest.raises(ValueError):
        fp.grab()


def test_manager_floor_targets_requested_camera(tmp_path, monkeypatch):
    """start_floor for cam_b yields a preview only for cam_b; floor(project, cam)
    returns it only for the matching camera (cam_a aiming must not show cam_b)."""
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    cfg = load_project(pdir)
    mgr = sess_mod.CaptureManager()
    mgr.start_floor("rig", pdir, cfg, "cam_b",
                    source_factory=lambda spec, cid: _StubSource(20))
    try:
        assert mgr.floor("rig", "cam_b") is not None
        assert mgr.floor("rig", "cam_a") is None           # not the targeted camera
        assert mgr.floor("other", "cam_b") is None         # not this project
        assert mgr.floor("rig", "cam_b").camera_id == "cam_b"
    finally:
        mgr.stop_floor()
    assert mgr.floor("rig", "cam_b") is None


def test_extrinsic_disk_image_is_clahe_grayscale(tmp_path, monkeypatch):
    """Extrinsic shots are written with the SAME CLAHE/grayscale preprocessing the
    gate detects on, so Multical re-detects tags on identical pixels."""
    monkeypatch.setattr(sess_mod, "AprilTagDetector", _StubCharuco)
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    cfg = load_project(pdir)
    s = sess_mod.CaptureSession(pdir, cfg, "extrinsic",
                                source_factory=lambda spec, cid: _StubSource(0))
    w = s.workers["cam_a"]
    # a colour gradient frame → disk image must be effectively single-channel (gray)
    raw = np.dstack([
        np.tile(np.arange(160, dtype=np.uint8), (120, 1)),
        np.full((120, 160), 30, np.uint8),
        np.full((120, 160), 200, np.uint8),
    ])
    out = w._image_for_disk(raw)
    assert out.ndim == 2                       # grayscale written to disk
    assert out.shape == raw.shape[:2]          # geometry preserved (corners intact)


def test_intrinsic_disk_image_stays_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    pdir, cfg = _project(tmp_path)
    s = sess_mod.CaptureSession(pdir, cfg, "intrinsic",
                                source_factory=lambda spec, cid: _StubSource(0))
    raw = np.full((120, 160, 3), 80, np.uint8)
    out = s.workers["cam_a"]._image_for_disk(raw)
    assert out.ndim == 3 and out is raw        # intrinsic untouched (raw BGR)


def test_wipe_phase_captures(tmp_path):
    from isical.capture.session import wipe_phase_captures
    pdir, _cfg = _project(tmp_path)
    d = pdir / "intrinsic" / "cam_a"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / f"x{i}.jpg").write_bytes(b"x")
    removed = wipe_phase_captures(pdir, "intrinsic", ["cam_a"])
    assert removed == 3 and not list(d.glob("*.jpg"))


def test_session_camera_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "CharucoBoardDetector", _StubCharuco)
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    cfg = load_project(pdir)
    s = sess_mod.CaptureSession(pdir, cfg, "intrinsic", cameras=["cam_b"],
                                source_factory=lambda spec, cid: _StubSource(4))
    assert list(s.workers) == ["cam_b"]                 # only the selected camera


def test_probe_streams_skew(tmp_path):
    from isical.capture import probe as probe_mod
    from isical.core.project import CameraSpec, create_project, load_project
    cams = {"cam_a": CameraSpec(id="cam_a", url="rtsp://x/a"),
            "cam_b": CameraSpec(id="cam_b", url="rtsp://x/b")}
    pdir = create_project(tmp_path / "data", "rig", cams)
    cfg = load_project(pdir)

    import time as _t

    class _TsSource:
        """Yields frames at ~50 fps; cam_b offset by a fixed lag to exercise skew."""
        def __init__(self, lag_s):
            self._lag = lag_s
        def start(self): pass
        def frames(self):
            import numpy as np
            for _ in range(40):
                f = type("F", (), {})()
                f.image = np.zeros((4, 4, 3), "uint8")
                f.capture_ts = _t.time() + self._lag
                yield f
                _t.sleep(0.005)
        def stop(self): pass

    def factory(spec, cid):
        return _TsSource(0.0 if cid == "cam_a" else 0.02)   # cam_b lags 20 ms

    r = probe_mod.probe_streams(pdir, cfg, seconds=1.0, source_factory=factory)
    assert set(r["cameras"]) == {"cam_a", "cam_b"}
    assert r["cameras"]["cam_a"]["fps"] > 0
    assert "sync" in r and r["sync"]["pairs"] > 0
    assert r["sync"]["mean_skew_ms"] >= 0    # measured a skew
