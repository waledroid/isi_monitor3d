"""Bird's-eye floor rectification — the calibration verification view.

The key property: warping a frame through the calibration homography flattens a
floor pallet (a trapezoid in the image) into an axis-aligned rectangle whose size
matches the pallet's real metres at the chosen px/m scale.
"""

from __future__ import annotations

import cv2
import numpy as np
import yaml
from calibration.calibrate_single_cam import PointPair, build_single_camera_calibration

from monitor_web.api.routes_video import _warp_camera
from monitor_web.config import Settings
from monitor_web.floor_rectify import (
    build_fit_rectify_matrix,
    build_rectify_matrix,
    cropped_bounds,
    rectify_frame,
    world_rect_to_pixel_box,
)


def test_world_rect_to_pixel_box_maps_metres_to_pixels():
    # 800x600 rectified image at 100 px/m, world origin (2.0, 1.0).
    bounds = {"px_per_m": 100.0, "x_min": 2.0, "y_min": 1.0, "out_wh": (800, 600)}
    # Work area X in [4,7], Y in [2,5] → 3x3 m → 300x300 px starting at (200,100).
    assert world_rect_to_pixel_box(bounds, (4.0, 2.0, 7.0, 5.0)) == (200, 100, 500, 400)


def test_world_rect_to_pixel_box_clamps_and_rejects_degenerate():
    bounds = {"px_per_m": 100.0, "x_min": 2.0, "y_min": 1.0, "out_wh": (800, 600)}
    # Oversized crop clamps to the full image.
    assert world_rect_to_pixel_box(bounds, (-9, -9, 99, 99)) == (0, 0, 800, 600)
    # A sub-pixel crop is rejected.
    assert world_rect_to_pixel_box(bounds, (4.0, 2.0, 4.001, 2.001)) is None


def test_cropped_bounds_shifts_origin_keeps_scale():
    bounds = {"px_per_m": 100.0, "x_min": 2.0, "y_min": 1.0, "out_wh": (800, 600)}
    cb = cropped_bounds(bounds, (200, 100, 500, 400))
    assert cb["px_per_m"] == 100.0
    assert cb["x_min"] == 4.0 and cb["y_min"] == 2.0
    assert cb["out_wh"] == (300, 300)

# A 1.2 x 0.8 m pallet seen as a perspective trapezoid in a 1920x1080 frame.
_CORNERS = [(800.0, 600.0), (1120.0, 610.0), (1180.0, 820.0), (740.0, 810.0)]  # TL,TR,BR,BL
_WORLD = [(0.0, 0.0), (1.2, 0.0), (1.2, 0.8), (0.0, 0.8)]
_PX_PER_M = 120.0


def _cam():
    pairs = [PointPair(pixel_uv=uv, world_xy_m=xy)
             for uv, xy in zip(_CORNERS, _WORLD, strict=True)]
    cal = build_single_camera_calibration(
        camera_id="cam_a", image_size_wh=(1920, 1080), pairs=pairs,
        floor_origin_note="test", residual_threshold_m=0.5,
    )
    return cal.cameras["cam_a"]


def test_pallet_flattens_to_axis_aligned_rectangle() -> None:
    cam = _cam()
    M = build_rectify_matrix(cam.H_np(), _PX_PER_M, (720, 720))
    pts = cv2.perspectiveTransform(np.array([_CORNERS], np.float64), M)[0]
    tl, tr, br, bl = pts
    # Top + bottom edges horizontal, left + right edges vertical.
    assert abs(tl[1] - tr[1]) < 1.0
    assert abs(bl[1] - br[1]) < 1.0
    assert abs(tl[0] - bl[0]) < 1.0
    assert abs(tr[0] - br[0]) < 1.0
    # Dimensions match real metres at the chosen scale.
    assert abs(abs(tr[0] - tl[0]) - 1.2 * _PX_PER_M) < 1.0
    assert abs(abs(bl[1] - tl[1]) - 0.8 * _PX_PER_M) < 1.0


def test_rectified_view_is_not_vertically_flipped() -> None:
    """The Mode-1 floor frame is X-right / Y-down (TL=(0,0), BL=(0,h)), so the
    bird's-eye render must keep the camera's vertical sense: the pallet's TOP
    corners (TL/TR) stay ABOVE its bottom corners (BL/BR) in the output. A +Y-up
    render would mirror this — the 'flipped camera' bug."""
    cam = _cam()
    M = build_rectify_matrix(cam.H_np(), _PX_PER_M, (720, 720))
    tl, tr, br, bl = cv2.perspectiveTransform(np.array([_CORNERS], np.float64), M)[0]
    assert tl[1] < bl[1]   # TL above BL (smaller v = higher in the image)
    assert tr[1] < br[1]   # TR above BR
    assert tl[0] < tr[0]   # TL left of TR (no horizontal mirror either)


# A calibration that spans most of the frame (4 x 3 m floor rectangle), so the
# full-frame extrapolation stays well inside the horizon clamp — the realistic
# on-site case, unlike _cam()'s tiny clustered pallet.
_CORNERS_WIDE = [(300.0, 400.0), (1600.0, 410.0), (1750.0, 1000.0), (180.0, 980.0)]
_WORLD_WIDE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]


def _cam_wide():
    pairs = [PointPair(pixel_uv=uv, world_xy_m=xy)
             for uv, xy in zip(_CORNERS_WIDE, _WORLD_WIDE, strict=True)]
    return build_single_camera_calibration(
        camera_id="cam_a", image_size_wh=(1920, 1080), pairs=pairs,
        floor_origin_note="wide", residual_threshold_m=0.5,
    ).cameras["cam_a"]


def test_autofit_is_bounded_and_upright() -> None:
    """Auto-fit always produces a sane, bounded, non-degenerate canvas (a
    near-horizon corner can't blow it up to infinity) and stays right-side-up."""
    cam = _cam()
    M, (out_w, out_h) = build_fit_rectify_matrix(cam.H_np(), (1920, 1080))
    assert 50 < out_w <= 820 and 50 < out_h <= 820
    pal = cv2.perspectiveTransform(np.array([_CORNERS], np.float64), M)[0]
    assert pal[0][1] < pal[3][1]          # pallet TL above BL (upright, not flipped)
    assert pal[0][0] < pal[1][0]          # TL left of TR (no horizontal mirror)


def test_autofit_keeps_all_source_content_on_canvas() -> None:
    """For a realistic floor-spanning calibration (no horizon clamp), every
    source-image corner lands inside the auto-fit canvas — nothing is cropped."""
    cam = _cam_wide()
    M, (out_w, out_h) = build_fit_rectify_matrix(cam.H_np(), (1920, 1080))
    img_corners = np.array([[(0, 0), (1920, 0), (1920, 1080), (0, 1080)]], np.float64)
    pts = cv2.perspectiveTransform(img_corners, M)[0]
    assert pts[:, 0].min() >= -1.0 and pts[:, 0].max() <= out_w + 1.0
    assert pts[:, 1].min() >= -1.0 and pts[:, 1].max() <= out_h + 1.0


def test_fit_rectify_bounds_inverts_a_known_point() -> None:
    """The bounds (px_per_m, x_min, y_min) let a rectified pixel be mapped back to
    world metres: X = x_min + u/px, Y = y_min + v/px. Round-tripping a world point
    through the rectify matrix and back via the bounds must return the original."""
    from monitor_web.floor_rectify import build_fit_rectify_matrix, fit_rectify_bounds
    cam = _cam_wide()
    b = fit_rectify_bounds(cam.H_np(), (1920, 1080))
    assert b is not None
    M, out_wh = build_fit_rectify_matrix(cam.H_np(), (1920, 1080))
    assert out_wh == b["out_wh"]
    # a source pixel inside the frame → rectified pixel via M → world via bounds
    src = np.array([[[960.0, 540.0]]], np.float64)        # image centre
    u, v = cv2.perspectiveTransform(src, M)[0][0]
    X = b["x_min"] + u / b["px_per_m"]
    Y = b["y_min"] + v / b["px_per_m"]
    # …must equal the floor point H maps that source pixel to
    wx = cam.H_np() @ np.array([960.0, 540.0, 1.0])
    assert abs(X - wx[0] / wx[2]) < 1e-3
    assert abs(Y - wx[1] / wx[2]) < 1e-3


def test_rectify_frame_produces_warped_image() -> None:
    cam = _cam()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    cv2.fillConvexPoly(frame, np.array(_CORNERS, np.int32), (255, 255, 255))
    out = rectify_frame(frame, cam.K_np(), cam.D_np(), cam.H_np(),
                        px_per_m=_PX_PER_M, out_wh=(720, 720))
    assert out.shape == (720, 720, 3)
    # The pallet's filled area survives the warp (≈ 1.2*120 x 0.8*120 px rectangle).
    assert int((out.sum(axis=2) > 200).sum()) > 0.5 * (1.2 * _PX_PER_M) * (0.8 * _PX_PER_M)


def _write_mode1_calibration(tmp_path) -> Settings:
    """A 1-camera backbone.yaml + a Mode-1 calibration file with cam_a."""
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump(
        {"cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x/y"}}}}
    ))
    cal = _cam_calibration_file()
    (tmp_path / "mode1").mkdir(exist_ok=True) or (tmp_path / "mode1" / "calibration.json").write_text(cal.to_json())
    return Settings(backbone_config_path=bb, udp_port=0, port=0)


def _cam_calibration_file():
    pairs = [PointPair(pixel_uv=uv, world_xy_m=xy)
             for uv, xy in zip(_CORNERS, _WORLD, strict=True)]
    return build_single_camera_calibration(
        camera_id="cam_a", image_size_wh=(1920, 1080), pairs=pairs,
        floor_origin_note="test", residual_threshold_m=0.5,
    )


def test_warp_camera_none_when_uncalibrated(tmp_path) -> None:
    """Auto-warp is best-effort: an uncalibrated camera yields None (the feed
    falls back to raw+detect), NOT an error — the stream must never break."""
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump(
        {"cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x/y"}}}}
    ))
    cfg = Settings(backbone_config_path=bb, udp_port=0, port=0)
    assert _warp_camera(cfg, "cam_a") is None


def test_warp_camera_returns_view_when_calibrated(tmp_path) -> None:
    """When the current-mode file has the camera, _warp_camera returns a view
    with a usable homography for the warp."""
    cfg = _write_mode1_calibration(tmp_path)
    cam = _warp_camera(cfg, "cam_a")
    assert cam is not None
    assert cam.H.shape == (3, 3)   # rig view exposes K/D/H as numpy arrays
    # An unknown camera is still None even with a calibration present.
    assert _warp_camera(cfg, "cam_b") is None
