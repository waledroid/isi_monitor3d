"""Board detection + the auto-snap gate (synthetic, OpenCV-only)."""

from __future__ import annotations

import cv2
import numpy as np

from isical.capture.detect import (
    AprilTagDetector,
    CharucoBoardDetector,
    Detection,
    SnapGate,
    preprocess_for_tags,
)
from isical.core.project import BoardSpec, charuco_spec


def _render_charuco(spec, px=900) -> np.ndarray:
    """Render the project's ChArUco board to a BGR image (white border)."""
    img = spec.board().generateImage((px, px))
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def test_charuco_detector_finds_corners():
    spec = charuco_spec(BoardSpec())
    det = CharucoBoardDetector(spec)
    d = det.detect(_render_charuco(spec))
    assert d.n >= 12 and d.corners_px is not None
    assert d.centroid is not None and 0.0 <= d.centroid[0] <= 1.0
    assert d.blur_var > 0


def test_charuco_detector_blank_is_empty():
    spec = charuco_spec(BoardSpec())
    d = CharucoBoardDetector(spec).detect(np.zeros((480, 640, 3), np.uint8))
    assert d.n == 0 and d.corners_px is None


def test_charuco_detector_runs_clahe_preprocess(monkeypatch):
    """The ChArUco detector must route its input through the CLAHE preprocess."""
    import isical.capture.detect as detect_mod

    spec = charuco_spec(BoardSpec())
    calls: list[dict] = []
    real = detect_mod.preprocess_for_tags

    def spy(frame, *, clahe, clip, grid):
        calls.append({"clahe": clahe, "clip": clip, "grid": grid})
        return real(frame, clahe=clahe, clip=clip, grid=grid)

    monkeypatch.setattr(detect_mod, "preprocess_for_tags", spy)
    det = CharucoBoardDetector(spec, clahe=True, clahe_clip=3.0, clahe_grid=4)
    det.detect(_render_charuco(spec))
    assert calls and calls[0] == {"clahe": True, "clip": 3.0, "grid": 4}


def test_charuco_detector_clahe_default_on_and_still_detects():
    """CLAHE defaults ON and a clean board still detects (corner-preserving)."""
    spec = charuco_spec(BoardSpec())
    det = CharucoBoardDetector(spec)              # default clahe=True
    assert det._clahe is True
    d = det.detect(_render_charuco(spec))
    assert d.n >= 12 and d.corners_px is not None


def _render_apriltag(px=240) -> np.ndarray:
    """Render one 36h11 AprilTag marker centred on a low-contrast gray field."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, 0, px)
    bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    canvas = np.full((px + 120, px + 120, 3), 150, np.uint8)   # gray surround
    canvas[60:60 + px, 60:60 + px] = bgr
    return canvas


def test_preprocess_for_tags_is_corner_preserving():
    bgr = _render_apriltag()
    g = preprocess_for_tags(bgr, clahe=True, clip=2.0, grid=8)
    assert g.ndim == 2 and g.shape == bgr.shape[:2]    # same geometry, single channel
    assert g.dtype == np.uint8
    # CLAHE without clahe flag is plain grayscale (still single channel, same shape)
    g0 = preprocess_for_tags(bgr, clahe=False)
    assert g0.ndim == 2 and g0.shape == bgr.shape[:2]


def test_apriltag_detector_finds_tag():
    det = AprilTagDetector(quad_decimate=1.0, clahe=True)
    d = det.detect(_render_apriltag())
    assert d.n >= 1 and d.corners_px is not None
    assert d.centroid is not None


def test_apriltag_detector_blank_is_empty():
    d = AprilTagDetector().detect(np.zeros((240, 320, 3), np.uint8))
    assert d.n == 0 and d.corners_px is None


def _det(n=20, centroid=(0.5, 0.5), blur=200.0):
    pts = np.full((n, 2), 100.0, np.float32)
    return Detection(n=n, centroid=centroid, corners_px=pts, blur_var=blur, coverage=1.0)


def test_gate_needs_two_steady_frames_then_snaps():
    g = SnapGate(min_detections=12, blur_min_var=80, steady_max_motion=2.5, novelty_min_dist=0.06)
    # first frame: prev is None → motion huge → not steady yet
    assert g.evaluate(_det()).snap is False
    # second identical frame: motion 0 → snaps
    v = g.evaluate(_det())
    assert v.snap is True


def test_gate_rejects_blur_and_low_count():
    g = SnapGate(min_detections=12, blur_min_var=80, steady_max_motion=2.5, novelty_min_dist=0.06)
    g.evaluate(_det())                                   # prime prev
    assert g.evaluate(_det(blur=10.0)).snap is False     # blurry
    assert g.evaluate(_det(n=5)).snap is False           # too few corners


def test_gate_novelty_blocks_duplicate_pose():
    g = SnapGate(min_detections=12, blur_min_var=80, steady_max_motion=2.5, novelty_min_dist=0.10)
    g.evaluate(_det(centroid=(0.5, 0.5)))                # prime
    v = g.evaluate(_det(centroid=(0.5, 0.5)))
    assert v.snap is True
    g.note_kept(v.detection)                             # remember this pose
    g.evaluate(_det(centroid=(0.5, 0.5)))                # prime
    assert g.evaluate(_det(centroid=(0.5, 0.5))).snap is False   # same pose → blocked
    g.evaluate(_det(centroid=(0.9, 0.9)))                # prime new pose
    assert g.evaluate(_det(centroid=(0.9, 0.9))).snap is True    # novel pose → snaps
