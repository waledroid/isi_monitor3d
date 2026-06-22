"""Live board detection + quality scoring for auto-snap (OpenCV 4.13, monitor3d).

Two detectors, both pure cv2.aruco (no Multical, no GPU):
  * ChArUco (intrinsic phase) — cv2.aruco.CharucoDetector → corner count + coverage.
  * AprilTag (extrinsic phase) — cv2.aruco.ArucoDetector(DICT_APRILTAG_36h11) →
    tag count. This is the AUTO-SNAP TRIGGER only; the authoritative AprilGrid pose
    solve is Multical's at solve time.

A frame is snap-worthy when it is well-detected, sharp (Laplacian variance), steady
(low corner motion vs the previous frame), and a NOVEL pose (board centroid moved
enough vs already-kept shots) — the last gate spreads coverage + prunes near-dupes.
Annotation draws the detections + a HUD for the live MJPEG view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# OpenCV aruco AprilTag dictionary for the 36h11 family (live tag-count trigger).
_APRILTAG_DICT = "DICT_APRILTAG_36h11"


@dataclass
class Detection:
    """One frame's detection result + quality scores (normalized 0..1 coords)."""
    n: int = 0                              # detected corners (charuco) or tags (april)
    centroid: tuple[float, float] | None = None   # board centre, normalized [0,1]
    corners_px: np.ndarray | None = None    # (N,2) detected points in pixels (for motion)
    blur_var: float = 0.0                   # Laplacian variance (sharpness)
    coverage: float = 0.0                   # detected / expected (charuco only)


@dataclass
class SnapVerdict:
    snap: bool
    reason: str
    detection: Detection


class CharucoBoardDetector:
    """ChArUco detection + coverage for the intrinsic phase."""

    def __init__(self, charuco_spec) -> None:
        self._board = charuco_spec.board()
        self._detector = cv2.aruco.CharucoDetector(self._board)
        self._expected = max(1, (charuco_spec.squares_x - 1) * (charuco_spec.squares_y - 1))

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        ch_corners, ch_ids, _, _ = self._detector.detectBoard(gray)
        det = Detection(blur_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if ch_corners is None or ch_ids is None or len(ch_ids) == 0:
            return det
        pts = ch_corners.reshape(-1, 2).astype(np.float32)
        det.n = len(ch_ids)
        det.corners_px = pts
        det.coverage = det.n / float(self._expected)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        det.centroid = (cx / w, cy / h)
        return det

    def annotate(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        out = frame_bgr.copy()
        if det.corners_px is not None:
            for x, y in det.corners_px:
                cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
        return out


class AprilTagDetector:
    """AprilTag (36h11) tag-count detection for the extrinsic-phase auto-snap trigger."""

    def __init__(self) -> None:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _APRILTAG_DICT))
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(d, params)

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        corners, ids, _ = self._detector.detectMarkers(gray)
        det = Detection(blur_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if ids is None or len(ids) == 0:
            return det
        pts = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0).astype(np.float32)
        det.n = len(ids)
        det.corners_px = pts
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        det.centroid = (cx / w, cy / h)
        return det

    def annotate(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        out = frame_bgr.copy()
        if det.corners_px is not None:
            for x, y in det.corners_px:
                cv2.circle(out, (int(x), int(y)), 3, (0, 200, 255), -1)
        return out


def _mean_motion(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean nearest-point displacement between two corner sets (px). Big = moving."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 1e9
    m = min(len(a), len(b))
    return float(np.linalg.norm(a[:m] - b[:m], axis=1).mean())


@dataclass
class SnapGate:
    """Stateful auto-snap decision: detection + blur + steadiness + pose-novelty.

    Holds the previous frame's corners (for motion) and the list of already-kept
    normalized centroids (for novelty). `evaluate` returns whether to snap THIS frame.
    """
    min_detections: int
    blur_min_var: float
    steady_max_motion: float
    novelty_min_dist: float
    _prev_corners: np.ndarray | None = field(default=None, repr=False)
    _kept_centroids: list[tuple[float, float]] = field(default_factory=list)

    def reset_kept(self) -> None:
        self._kept_centroids.clear()

    def note_kept(self, det: Detection) -> None:
        if det.centroid is not None:
            self._kept_centroids.append(det.centroid)

    def evaluate(self, det: Detection) -> SnapVerdict:
        prev = self._prev_corners
        self._prev_corners = det.corners_px
        if det.n < self.min_detections:
            return SnapVerdict(False, f"need ≥{self.min_detections} (have {det.n})", det)
        if det.blur_var < self.blur_min_var:
            return SnapVerdict(False, f"blurry ({det.blur_var:.0f}<{self.blur_min_var:.0f})", det)
        motion = _mean_motion(det.corners_px, prev)
        if motion > self.steady_max_motion:
            return SnapVerdict(False, f"hold steady (motion {motion:.1f}px)", det)
        if det.centroid is not None and self._kept_centroids:
            cx, cy = det.centroid
            nearest = min(((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
                          for kx, ky in self._kept_centroids)
            if nearest < self.novelty_min_dist:
                return SnapVerdict(False, "move the board (too similar)", det)
        return SnapVerdict(True, "snap", det)


def draw_hud(frame_bgr: np.ndarray, *, count: int, target: int, status: str,
             ok: bool) -> np.ndarray:
    """Overlay a capture HUD (count/target + status banner) on the annotated frame."""
    out = frame_bgr
    h, w = out.shape[:2]
    bar = max(28, h // 18)
    cv2.rectangle(out, (0, 0), (w, bar), (0, 0, 0), -1)
    color = (80, 230, 120) if ok else (80, 170, 255)
    cv2.putText(out, f"{count}/{target}   {status}", (10, int(bar * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1400), color,
                max(1, round(w / 900)), cv2.LINE_AA)
    return out
