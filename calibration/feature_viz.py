"""Key-stage visualizations for the targetless stereo extrinsic flow (Stage 3).

Every key pipeline stage of the targetless method (:mod:`feature_extrinsics` +
:mod:`floor_planefit`) returns an **annotated BGR image** so the isical operator
verifies progress and catches problems (weak matches, mis-clicked scale points, a
skewed floor) instead of trusting a black box. Each function here draws with
``cv2`` on the captured stereo pair and returns a ``uint8`` ``(H, W, 3)`` array the
isical UI serves over the existing MJPEG/JPEG transport (the same machinery the
floor live-preview uses).

The five stages (mirroring the plan's "operator sees each key stage"):

1. :func:`draw_stereo_pair`            — the cam_a/cam_b frames actually used.
2. :func:`draw_feature_matches`        — keypoints + match lines, inliers vs RANSAC
                                          outliers colour-coded, with a match count.
3. :func:`draw_scale_references`       — the operator-clicked ≥3 floor point-pairs
                                          + measured metres, drawn on the pair.
4. :func:`draw_triangulation_floor`    — triangulated 3D points projected back onto
                                          cam_a, floor-plane inliers highlighted, a
                                          projected floor grid drawn.
5. :func:`draw_result_overlay`         — reprojection overlay + the report numbers.

These functions are pure (no I/O, no rig, no ONNX) and are covered by hermetic
tests with synthetic inputs.
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR colours (OpenCV order).
_GREEN = (60, 200, 60)     # inliers / good
_RED = (60, 60, 220)       # outliers / bad
_YELLOW = (40, 210, 235)   # scale marks
_CYAN = (220, 210, 40)     # floor grid
_WHITE = (245, 245, 245)
_BLACK = (0, 0, 0)


def _as_bgr(img: np.ndarray) -> np.ndarray:
    """Coerce any image to a contiguous 3-channel uint8 BGR array (a copy)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.astype(np.uint8)
    else:
        raise ValueError(f"expected HxW or HxWx3 image, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _hstack_pair(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, int]:
    """Side-by-side canvas of two BGR images. Returns (canvas, x_offset_of_b)."""
    a = _as_bgr(img_a)
    b = _as_bgr(img_b)
    h = max(a.shape[0], b.shape[0])
    pa = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=_BLACK)
    pb = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=_BLACK)
    canvas = np.hstack([pa, pb])
    return canvas, pa.shape[1]


def _banner(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a small caption block top-left (dark background for legibility)."""
    y = 8
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (6, y), (6 + tw + 8, y + th + 8), _BLACK, -1)
        cv2.putText(img, line, (10, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, _WHITE, 1, cv2.LINE_AA)
        y += th + 12
    return img


# ---------------------------------------------------------------------------
# Stage 1 — captured stereo pair
# ---------------------------------------------------------------------------


def draw_stereo_pair(
    img_a: np.ndarray, img_b: np.ndarray, *, pair_index: int | None = None
) -> np.ndarray:
    """Side-by-side view of the captured cam_a / cam_b frames actually used."""
    canvas, xb = _hstack_pair(img_a, img_b)
    cv2.putText(canvas, "cam_a", (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "cam_b", (xb + 10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA)
    label = "captured stereo pair"
    if pair_index is not None:
        label += f" #{pair_index}"
    return _banner(canvas, [label])


# ---------------------------------------------------------------------------
# Stage 2 — feature matches (inliers vs RANSAC outliers)
# ---------------------------------------------------------------------------


def draw_feature_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    max_lines: int = 200,
) -> np.ndarray:
    """Keypoints + LightGlue match lines, colour-coded inliers vs RANSAC outliers.

    ``pts_a`` / ``pts_b`` are ``(N, 2)`` matched pixels; ``inlier_mask`` an
    ``(N,)`` bool from ``recover_relative_pose``. Inliers are green, RANSAC-rejected
    outliers red. Down-samples the drawn lines to ``max_lines`` for legibility but
    reports the full counts.
    """
    canvas, xb = _hstack_pair(img_a, img_b)
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    n = min(len(pa), len(pb), len(mask))
    n_in = int(mask[:n].sum())
    n_out = int(n - n_in)

    idx = np.arange(n)
    if n > max_lines:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_lines, replace=False)
    for i in idx:
        ax, ay = round(pa[i, 0]), round(pa[i, 1])
        bx, by = round(pb[i, 0]) + xb, round(pb[i, 1])
        color = _GREEN if mask[i] else _RED
        cv2.circle(canvas, (ax, ay), 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (ax, ay), (bx, by), color, 1, cv2.LINE_AA)

    return _banner(canvas, [
        f"matches: {n}  (inliers {n_in} green / outliers {n_out} red)",
    ])


# ---------------------------------------------------------------------------
# Stage 3 — interactive scale-reference marking
# ---------------------------------------------------------------------------


def draw_scale_references(
    img_a: np.ndarray,
    img_b: np.ndarray,
    references,
    *,
    outliers: list[int] | None = None,
) -> np.ndarray:
    """Draw the operator-clicked floor point-pairs + measured metres on the pair.

    ``references`` is a list of :class:`~calibration.feature_extrinsics.ScaleReference`.
    Each reference's two landmarks are drawn (a labelled segment) on both views with
    its measured distance; references in ``outliers`` (cross-validation-flagged) are
    drawn red instead of yellow.
    """
    canvas, xb = _hstack_pair(img_a, img_b)
    outliers = set(outliers or [])
    for i, ref in enumerate(references):
        color = _RED if i in outliers else _YELLOW
        p1a = (round(ref.p1_a[0]), round(ref.p1_a[1]))
        p2a = (round(ref.p2_a[0]), round(ref.p2_a[1]))
        p1b = (round(ref.p1_b[0]) + xb, round(ref.p1_b[1]))
        p2b = (round(ref.p2_b[0]) + xb, round(ref.p2_b[1]))
        for (q1, q2) in ((p1a, p2a), (p1b, p2b)):
            cv2.line(canvas, q1, q2, color, 2, cv2.LINE_AA)
            for q in (q1, q2):
                cv2.circle(canvas, q, 5, color, 2, cv2.LINE_AA)
            mid = ((q1[0] + q2[0]) // 2, (q1[1] + q2[1]) // 2)
            cv2.putText(canvas, f"#{i}:{ref.distance_m:.2f}m", (mid[0] + 6, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return _banner(canvas, [
        f"scale references: {len(references)} "
        f"({len(outliers)} flagged)" if outliers else
        f"scale references: {len(references)}",
    ])


# ---------------------------------------------------------------------------
# Stage 4 — triangulated points + floor fit
# ---------------------------------------------------------------------------


def _project_cam_a(pts3: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project cam_a-frame 3D points (identity pose) into cam_a pixels."""
    pts = np.asarray(pts3, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = (K @ (pts / z).T).T
    return uv[:, :2]


def draw_triangulation_floor(
    img_a: np.ndarray,
    points_3d: np.ndarray,
    K_a: np.ndarray,
    *,
    floor_inlier_mask: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
    plane_offset: float | None = None,
    grid_half_m: float = 2.0,
    grid_step_m: float = 0.5,
) -> np.ndarray:
    """Triangulated 3D points projected back onto cam_a, floor fit highlighted.

    Off-floor points are drawn small/grey, floor-plane RANSAC inliers highlighted
    green. When a plane (``plane_normal`` + ``plane_offset``) is supplied a projected
    floor grid (cyan) is overlaid so the operator sees the recovered world frame is
    sensible.
    """
    canvas = _as_bgr(img_a)
    K = np.asarray(K_a, dtype=np.float64)
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    uv = _project_cam_a(pts, K)
    mask = (np.asarray(floor_inlier_mask).reshape(-1).astype(bool)
            if floor_inlier_mask is not None else np.zeros(len(pts), dtype=bool))
    n_in = int(mask.sum()) if floor_inlier_mask is not None else 0
    for i, (u, v) in enumerate(uv):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        p = (round(u), round(v))
        if i < len(mask) and mask[i]:
            cv2.circle(canvas, p, 4, _GREEN, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, p, 2, (150, 150, 150), -1, cv2.LINE_AA)

    if plane_normal is not None and plane_offset is not None:
        _draw_floor_grid(canvas, K, np.asarray(plane_normal, float),
                         float(plane_offset), pts[mask] if n_in else pts,
                         grid_half_m, grid_step_m)

    return _banner(canvas, [
        f"triangulated {len(pts)} pts  |  floor inliers {n_in}",
    ])


def _draw_floor_grid(
    canvas: np.ndarray,
    K: np.ndarray,
    normal: np.ndarray,
    offset: float,
    on_plane_pts: np.ndarray,
    half_m: float,
    step_m: float,
) -> None:
    """Overlay a projected grid lying on the fitted plane (in the cam_a frame)."""
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # A plane point (centroid of provided points projected onto the plane).
    if len(on_plane_pts):
        c = on_plane_pts.mean(axis=0)
    else:
        c = -offset * normal
    c = c - (normal @ c + offset) * normal
    # In-plane basis.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(normal @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ex = ref - (ref @ normal) * normal
    ex = ex / (np.linalg.norm(ex) + 1e-12)
    ey = np.cross(normal, ex)

    ticks = np.arange(-half_m, half_m + 1e-6, step_m)

    def line3(p1: np.ndarray, p2: np.ndarray) -> None:
        uv = _project_cam_a(np.vstack([p1, p2]), K)
        if not np.isfinite(uv).all():
            return
        cv2.line(canvas, tuple(np.round(uv[0]).astype(int)),
                 tuple(np.round(uv[1]).astype(int)), _CYAN, 1, cv2.LINE_AA)

    for t in ticks:
        line3(c + t * ex - half_m * ey, c + t * ex + half_m * ey)
        line3(c + t * ey - half_m * ex, c + t * ey + half_m * ex)


# ---------------------------------------------------------------------------
# Stage 5 — result / validation overlay
# ---------------------------------------------------------------------------


def draw_result_overlay(
    img_a: np.ndarray,
    report_lines: list[str],
    *,
    passed: bool | None = None,
) -> np.ndarray:
    """Result banner: the report numbers + a pass/pending/fail tint.

    ``report_lines`` are short strings (RMS, baseline, metric errors...) produced by
    the validation report; ``passed`` tints the banner green/red/amber.
    """
    canvas = _as_bgr(img_a)
    header = "RESULT"
    if passed is True:
        header += " — PASS"
    elif passed is False:
        header += " — FAIL"
    else:
        header += " — PENDING (metric acceptance requires on-rig measurements)"
    return _banner(canvas, [header, *report_lines])
