"""Single-camera 4-point floor calibration — the Mode 1 path.

The operator:

1. Picks 4 non-collinear points on the warehouse floor (typically the corners
   of a rectangle bracketing the working area).
2. Measures each point's metric ``(X, Y)`` with a tape measure.
3. Identifies each point's pixel ``(u, v)`` in a still frame from the camera.
4. Feeds the 4 pairs into this tool, which fits the homography
   ``H : (u, v) → (X, Y)`` via ``cv2.findHomography`` and writes a
   ``calibration.json`` with ``calibration_mode = "single_cam_4pt"``.

Mode 1 stores ``K`` as identity, ``D`` as zeros, ``R`` as identity, ``t`` as
zeros, and ``P`` as ``K @ [I | 0]``. These are placeholders — only ``H`` is
real. Downstream homography (``pixel_to_floor``) works untouched because
``undistort_points`` with ``D=0`` is the identity transform (pinned by
``tests/test_geometry.py::test_undistort_zero_distortion_is_identity``).
Triangulation is unavailable in Mode 1 and the orchestrator skips it.

Sanity gate: after the fit, every input pixel is back-projected through
``H`` and the distance to its declared world point is checked. Anything
above ``residual_threshold_m`` (default 0.10 m = 10 cm) refuses to write.

**Recommended: provide 5+ points, not just 4.** With exactly 4 points the
homography is *exactly-determined* — ``cv2.findHomography`` fits all 4
pixel→world pairs perfectly regardless of their accuracy, so the sanity
gate cannot detect operator error. With 5+ points the system is
overdetermined, residuals are observable, and a mistyped tape measurement
or a wrong pixel pick will be caught by the gate.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration.schema import (
    CALIBRATION_MODE_SINGLE_CAM_4PT,
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)


@dataclass(slots=True)
class PointPair:
    """One (pixel, world) correspondence for the 4-point fit."""

    pixel_uv: tuple[float, float]
    world_xy_m: tuple[float, float]


class SingleCamCalibrationError(RuntimeError):
    """The 4-point fit could not produce a calibration we trust."""


def fit_single_camera_homography(
    pairs: list[PointPair],
    *,
    residual_threshold_m: float = 0.10,
) -> tuple[np.ndarray, float]:
    """Solve ``H`` such that ``H @ (u, v, 1)`` ≈ ``(X, Y, 1)``.

    Args:
        pairs: at least 4 ``PointPair``s. Must not be collinear.
        residual_threshold_m: max allowed pixel→world disagreement at any
            input point after the fit (sanity check, defaults to 10 cm).

    Returns:
        ``(H, max_residual_m)``.

    Raises:
        SingleCamCalibrationError: degenerate point set, or residual exceeds
            ``residual_threshold_m``.
    """
    if len(pairs) < 4:
        raise SingleCamCalibrationError(
            f"need at least 4 (pixel, world) pairs, got {len(pairs)}"
        )
    pixels = np.asarray([p.pixel_uv for p in pairs], dtype=np.float64)
    world = np.asarray([p.world_xy_m for p in pairs], dtype=np.float64)

    H, _mask = cv2.findHomography(pixels, world, method=0)
    if H is None:
        raise SingleCamCalibrationError(
            "cv2.findHomography returned None — points are likely collinear or duplicated"
        )

    # Sanity: project each input pixel through H and check against truth.
    homog_pixels = np.hstack([pixels, np.ones((len(pixels), 1))])
    projected = (H @ homog_pixels.T).T            # (N, 3) homogeneous world
    projected_xy = projected[:, :2] / projected[:, 2:3]
    residuals = np.linalg.norm(projected_xy - world, axis=1)
    max_residual = float(residuals.max())
    if max_residual > residual_threshold_m:
        raise SingleCamCalibrationError(
            f"4-point fit residual {max_residual:.3f} m exceeds threshold "
            f"{residual_threshold_m} m — re-check pixel picks or tape measurements. "
            f"Per-point residuals (m): {residuals.tolist()}"
        )
    return H, max_residual


def build_single_camera_calibration(
    *,
    camera_id: str,
    image_size_wh: tuple[int, int],
    pairs: list[PointPair],
    floor_origin_note: str = "",
    residual_threshold_m: float = 0.10,
) -> CalibrationFile:
    """Run the 4-point fit and produce a complete ``CalibrationFile``.

    Refuses to return if the fit's max residual exceeds the threshold (the
    architecture's "fail honestly" principle for calibration).
    """
    H, max_residual = fit_single_camera_homography(
        pairs, residual_threshold_m=residual_threshold_m,
    )

    K_identity = np.eye(3, dtype=np.float64)
    D_zeros = np.zeros(5, dtype=np.float64)
    R_identity = np.eye(3, dtype=np.float64)
    t_zeros = np.zeros(3, dtype=np.float64)
    # P = K @ [R | t] = [I | 0]. Not meaningful for triangulation; placeholder
    # so the schema's required field is populated and CameraRig doesn't crash.
    P_placeholder = np.hstack([R_identity, t_zeros.reshape(3, 1)])

    cam = CameraCalibration(
        camera_id=camera_id,
        image_size_wh=image_size_wh,
        K=K_identity.tolist(),
        D=D_zeros.tolist(),
        R=R_identity.tolist(),
        t=t_zeros.tolist(),
        H=H.tolist(),
        P=P_placeholder.tolist(),
        reprojection_rms_px=float(max_residual),  # repurposed: max world-residual in meters
    )
    return CalibrationFile(
        version=CALIBRATION_VERSION,
        created_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        floor_anchor_method="4pt_floor",
        floor_origin_note=floor_origin_note,
        cameras={camera_id: cam},
        calibration_mode=CALIBRATION_MODE_SINGLE_CAM_4PT,
    )


def _parse_pair_arg(spec: str) -> PointPair:
    """Parse a CLI ``--pair u,v,X,Y`` string into a :class:`PointPair`."""
    parts = spec.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"--pair expects 'u,v,X,Y' (4 floats), got {spec!r}"
        )
    try:
        u, v, x, y = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--pair: could not parse {spec!r} as 4 floats") from exc
    return PointPair(pixel_uv=(u, v), world_xy_m=(x, y))


def add_single_cam_subparser(subparsers) -> None:
    """Register the ``single-cam`` CLI under ``calibrate.py``."""
    p = subparsers.add_parser(
        "single-cam",
        help=(
            "Mode 1: fit a homography from 4 tape-measured floor points "
            "for a single camera (no Multical, no intrinsics)."
        ),
    )
    p.add_argument("--camera-id", required=True)
    p.add_argument(
        "--image-size", nargs=2, type=int, metavar=("W", "H"), required=True,
    )
    p.add_argument(
        "--pair", action="append", required=True,
        help="One pixel→world pair as 'u,v,X,Y' (meters). Repeat ≥4 times.",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--floor-origin-note", default="",
        help="Free-text description of how the 4 points were chosen.",
    )
    p.add_argument(
        "--residual-threshold-m", type=float, default=0.10,
        help="Refuse the fit if max world-residual exceeds this (meters).",
    )
    p.set_defaults(func=_cmd_single_cam)


def _cmd_single_cam(args) -> int:
    pairs = [_parse_pair_arg(spec) for spec in args.pair]
    cal = build_single_camera_calibration(
        camera_id=args.camera_id,
        image_size_wh=(int(args.image_size[0]), int(args.image_size[1])),
        pairs=pairs,
        floor_origin_note=args.floor_origin_note,
        residual_threshold_m=args.residual_threshold_m,
    )
    Path(args.output).write_text(cal.to_json())
    cam = cal.cameras[args.camera_id]
    print(
        f"[single-cam:{args.camera_id}] H fit; "
        f"max world-residual={cam.reprojection_rms_px*1000:.1f} mm; "
        f"wrote {args.output}",
        flush=True,
    )
    return 0
