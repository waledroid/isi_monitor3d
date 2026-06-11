"""Bird's-eye floor rectification — a calibration *verification* view.

The metric Backbone never warps the camera image; it only maps each detection's
foot point through the homography ``H`` to floor metres. But warping the *whole*
frame through the same ``H`` is the clearest way for an operator to confirm a
4-point floor calibration is correct: a correctly-calibrated floor flattens so
that the pallet (and any floor tape) becomes an axis-aligned rectangle.

Pipeline mirrors the projection one exactly so the view matches the maths the
Backbone uses: ``undistort(K, D)`` → homography ``H`` (pixel → metres) → scale
metres to output pixels. ``M = S · H`` is precomputed once per stream.
"""

from __future__ import annotations

import cv2
import numpy as np

_DEFAULT_PX_PER_M = 120.0
_DEFAULT_OUT_WH = (720, 720)


def floor_world_center(H: np.ndarray, source_wh: tuple[int, int]) -> tuple[float, float]:
    """World ``(X, Y)`` metres that the *source-image centre* projects to.

    Used to centre the bird's-eye view on whatever the camera is actually
    pointed at, so the rectified pallet/floor sits in the middle of the canvas
    instead of being pushed to an edge. Falls back to the world origin if the
    projection is degenerate (point on the horizon line)."""
    w, h = source_wh
    c = np.asarray(H, dtype=np.float64) @ np.array([w / 2.0, h / 2.0, 1.0])
    if abs(c[2]) < 1e-9:
        return (0.0, 0.0)
    return (float(c[0] / c[2]), float(c[1] / c[2]))


def fit_rectify_bounds(
    H: np.ndarray,
    source_wh: tuple[int, int],
    *,
    max_dim: int = 820,
    max_extent_m: float = 30.0,
) -> dict | None:
    """Metric placement of the auto-fit bird's-eye image for this camera.

    Returns ``{"px_per_m", "x_min", "y_min", "out_wh"}`` such that rectified pixel
    ``(u, v)`` <-> world metres ``X = x_min + u/px_per_m``, ``Y = y_min + v/px_per_m``.
    Returns ``None`` when most of the frame is beyond the floor horizon (degenerate).
    """
    H = np.asarray(H, dtype=np.float64)
    w, h = source_wh
    img_corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    wc = H @ img_corners                       # 3x4 homogeneous world
    depth = wc[2]
    valid = depth > 1e-6
    if int(valid.sum()) < 3:                   # most of the frame is beyond the horizon
        return None
    xy = (wc[:2, valid] / depth[valid]).T      # (k, 2) world XY metres
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    half = max_extent_m / 2.0
    xs = np.clip(xy[:, 0], cx - half, cx + half)
    ys = np.clip(xy[:, 1], cy - half, cy + half)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    ext_x = max(x_max - x_min, 1e-3)
    ext_y = max(y_max - y_min, 1e-3)
    px_per_m = max_dim / max(ext_x, ext_y)
    out_w = max(1, min(max_dim, round(ext_x * px_per_m)))
    out_h = max(1, min(max_dim, round(ext_y * px_per_m)))
    return {"px_per_m": px_per_m, "x_min": x_min, "y_min": y_min, "out_wh": (out_w, out_h)}


def rectify_params_for_frame(
    H: np.ndarray,
    calibration_wh: tuple[int, int],
    frame_wh: tuple[int, int],
    *,
    max_dim: int = 820,
    max_extent_m: float = 30.0,
) -> dict | None:
    """``{M, out_wh, bounds}`` to rectify a frame of size ``frame_wh`` through the
    calibration homography ``H`` (which was calibrated at ``calibration_wh``).

    When the live frame size differs from the calibration size, ``H`` is rescaled
    so it maps *actual-frame* pixels → world metres (the frame-size guard). This is
    the SINGLE SOURCE OF TRUTH shared by the live CAM warp and the MAP tracing
    snapshot, so the two rectified images are pixel-for-pixel identical. Returns
    ``None`` when the rectification is degenerate (most of the frame past horizon)."""
    H = np.asarray(H, dtype=np.float64)
    iw, ih = frame_wh
    cal_w, cal_h = calibration_wh
    if (iw, ih) != (cal_w, cal_h):
        H = H @ np.diag([cal_w / iw, cal_h / ih, 1.0])   # actual-frame px → world
    bounds = fit_rectify_bounds(H, frame_wh, max_dim=max_dim, max_extent_m=max_extent_m)
    if bounds is None:
        return None
    M, out_wh = build_fit_rectify_matrix(H, frame_wh, max_dim=max_dim, max_extent_m=max_extent_m)
    return {"M": M, "out_wh": out_wh, "bounds": bounds}


def world_rect_to_pixel_box(bounds: dict, crop_world) -> tuple | None:
    """Pixel box ``(u0, v0, u1, v1)`` in a rectified image for a world rectangle.

    ``bounds`` is the ``{px_per_m, x_min, y_min, out_wh}`` placement of the rectified
    image (from :func:`fit_rectify_bounds`); ``crop_world`` is ``(x0, y0, x1, y1)`` in
    world metres. The rectified image maps pixel ``(u, v)`` → world ``X = x_min + u/ppm``,
    ``Y = y_min + v/ppm`` (``build_fit_rectify_matrix``, +Y down), so the world rectangle
    is a straight sub-box. Clamped to the image; returns ``None`` if the box is empty."""
    ppm = bounds["px_per_m"]
    x_min, y_min = bounds["x_min"], bounds["y_min"]
    ow, oh = bounds["out_wh"]
    x0, y0, x1, y1 = crop_world
    u0 = round((min(x0, x1) - x_min) * ppm)
    u1 = round((max(x0, x1) - x_min) * ppm)
    v0 = round((min(y0, y1) - y_min) * ppm)
    v1 = round((max(y0, y1) - y_min) * ppm)
    u0 = max(0, min(ow, u0))
    u1 = max(0, min(ow, u1))
    v0 = max(0, min(oh, v0))
    v1 = max(0, min(oh, v1))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    return (u0, v0, u1, v1)


def cropped_bounds(bounds: dict, box: tuple) -> dict:
    """The ``{px_per_m, x_min, y_min, out_wh}`` of a rectified image after cropping it
    to pixel box ``(u0, v0, u1, v1)`` — same scale, shifted origin, smaller canvas."""
    u0, v0, u1, v1 = box
    ppm = bounds["px_per_m"]
    return {
        "px_per_m": ppm,
        "x_min": bounds["x_min"] + u0 / ppm,
        "y_min": bounds["y_min"] + v0 / ppm,
        "out_wh": (u1 - u0, v1 - v0),
    }


def build_fit_rectify_matrix(
    H: np.ndarray,
    source_wh: tuple[int, int],
    *,
    max_dim: int = 820,
    max_extent_m: float = 30.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Auto-fit ``(M, out_wh)`` so the **entire** warped source frame is visible.

    The fixed-canvas render crops whatever falls outside its window and leaves a
    big black margin. Instead we project the four *source-image corners* to world
    metres, size the output canvas to their bounding box, and pick the scale so
    it fills ``max_dim`` — regaining all the camera content and tightening the
    canvas to the warped shape (so the only black left is the perspective
    "keystone" wedges, which are geometrically unavoidable). Renders **+Y down**.

    Horizon protection: a tilted camera maps its top rows toward the floor's
    vanishing line (homogeneous depth → 0 ⇒ world → ∞). Corners with non-positive
    depth are dropped, and the extent is clamped to ``max_extent_m`` about the
    median so a near-horizon view can't blow the canvas up to infinity. Degenerate
    cases fall back to a centred fixed view."""
    b = fit_rectify_bounds(H, source_wh, max_dim=max_dim, max_extent_m=max_extent_m)
    if b is None:                              # most of the frame is beyond the horizon
        center = floor_world_center(H, source_wh)
        return build_rectify_matrix(H, _DEFAULT_PX_PER_M, _DEFAULT_OUT_WH, center), _DEFAULT_OUT_WH
    px_per_m = b["px_per_m"]
    x_min, y_min = b["x_min"], b["y_min"]
    out_w, out_h = b["out_wh"]
    S = np.array(
        [[px_per_m, 0.0, -px_per_m * x_min],
         [0.0, px_per_m, -px_per_m * y_min],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return S @ np.asarray(H, dtype=np.float64), (out_w, out_h)


def shared_bev_layout(cameras, *, max_dim: int = 1100, max_extent_m: float = 40.0):
    """Common bird's-eye layout spanning the UNION of several cameras' floor
    coverage, so each camera warps into the SAME metric canvas — the basis of the
    Mode-2 unified view.

    ``cameras`` = list of ``(H, calibration_wh, frame_wh)``. Each ``H`` is rescaled
    for its actual frame size (the frame-size guard), its floor extent is measured
    via :func:`fit_rectify_bounds`, and the union of all extents sets one shared
    ``px_per_m``/origin/canvas. Returns ``({px_per_m, x_min, y_min, out_wh}, [M, …])``
    where ``M`` per camera maps its undistorted pixels → the shared canvas (``None``
    for a degenerate camera), or ``None`` if no camera yields a valid floor extent.
    """
    rescaled: list = []
    xs: list[float] = []
    ys: list[float] = []
    for H, cal_wh, frame_wh in cameras:
        H = np.asarray(H, dtype=np.float64)
        if tuple(frame_wh) != tuple(cal_wh):
            H = H @ np.diag([cal_wh[0] / frame_wh[0], cal_wh[1] / frame_wh[1], 1.0])
        b = fit_rectify_bounds(H, frame_wh, max_extent_m=max_extent_m)
        if b is None:
            rescaled.append(None)
            continue
        rescaled.append(H)
        x0, y0 = b["x_min"], b["y_min"]
        xs += [x0, x0 + b["out_wh"][0] / b["px_per_m"]]
        ys += [y0, y0 + b["out_wh"][1] / b["px_per_m"]]
    if not xs:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    ext_x = max(x_max - x_min, 1e-3)
    ext_y = max(y_max - y_min, 1e-3)
    ppm = max_dim / max(ext_x, ext_y)
    out_w = max(1, min(max_dim, round(ext_x * ppm)))
    out_h = max(1, min(max_dim, round(ext_y * ppm)))
    S = np.array(
        [[ppm, 0.0, -ppm * x_min],
         [0.0, ppm, -ppm * y_min],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Ms = [None if H is None else S @ H for H in rescaled]
    bounds = {"px_per_m": ppm, "x_min": x_min, "y_min": y_min, "out_wh": (out_w, out_h)}
    return bounds, Ms


def composite_bev(layers, out_wh: tuple[int, int]) -> np.ndarray:
    """Blend several warped bird's-eye layers into one BGR canvas.

    ``layers`` = list of ``(frame_bgr, K, D, M)``. Each frame is undistorted (when
    ``D`` ≠ 0) and warped through its ``M`` into ``out_wh``; OVERLAPPING regions are
    averaged. (Height ghosts in the overlap — inherent to a floor homography; the
    composite is exact only on the ground plane.)"""
    ow, oh = out_wh
    acc = np.zeros((oh, ow, 3), dtype=np.float32)
    cnt = np.zeros((oh, ow), dtype=np.float32)
    for frame, K, D, M in layers:
        if M is None:
            continue
        src = frame
        if _needs_undistort(D):
            src = cv2.undistort(frame, np.asarray(K, dtype=np.float64),
                                np.asarray(D, dtype=np.float64))
        warped = cv2.warpPerspective(src, M, out_wh, flags=cv2.INTER_LINEAR)
        ones = np.full(src.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpPerspective(ones, M, out_wh, flags=cv2.INTER_NEAREST) > 0
        acc[mask] += warped[mask].astype(np.float32)
        cnt[mask] += 1.0
    canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
    valid = cnt > 0
    canvas[valid] = (acc[valid] / cnt[valid, None]).astype(np.uint8)
    return canvas


def build_rectify_matrix(
    H: np.ndarray,
    px_per_m: float,
    out_wh: tuple[int, int],
    center_world: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """``M = S · H`` mapping undistorted source pixels → output (bird's-eye) pixels.

    ``H`` maps pixel → world metres ``(X, Y)``. The Mode-1 floor frame shares the
    image's handedness — ``X`` right and ``Y`` **down** (TL→(0,0), BL→(0,h)) — so
    the render uses **+Y down** to keep the view right-side-up; rendering +Y up
    mirrors it vertically (the "flipped camera" bug). ``center_world`` (metres) is
    placed at the output centre::

        u_out = out_w/2 + px_per_m * (X - cx)
        v_out = out_h/2 + px_per_m * (Y - cy)
    """
    out_w, out_h = out_wh
    cx, cy = center_world
    S = np.array(
        [[px_per_m, 0.0, out_w / 2.0 - px_per_m * cx],
         [0.0, px_per_m, out_h / 2.0 - px_per_m * cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return S @ np.asarray(H, dtype=np.float64)


def _needs_undistort(D: np.ndarray) -> bool:
    return bool(np.any(np.abs(np.asarray(D, dtype=np.float64)) > 1e-9))


def rectify_frame(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    *,
    px_per_m: float = _DEFAULT_PX_PER_M,
    out_wh: tuple[int, int] = _DEFAULT_OUT_WH,
    center_world: tuple[float, float] = (0.0, 0.0),
    M: np.ndarray | None = None,
) -> np.ndarray:
    """Return the bird's-eye rectified frame. Pass a precomputed ``M`` to skip the
    per-frame matrix build (the homography is static for a stream)."""
    src = image
    if _needs_undistort(D):
        # Real intrinsics (Mode 2 / Multical). For Mode 1 the placeholders are
        # K=I, D=0 → no-op, so we skip the remap entirely.
        src = cv2.undistort(image, np.asarray(K, dtype=np.float64),
                            np.asarray(D, dtype=np.float64))
    if M is None:
        M = build_rectify_matrix(H, px_per_m, out_wh, center_world)
    return cv2.warpPerspective(src, M, out_wh, flags=cv2.INTER_LINEAR)
