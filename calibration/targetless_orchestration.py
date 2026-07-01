"""Targetless stereo extrinsics — full end-to-end orchestration (Stage 3).

Chains the committed Stage-1 (:mod:`feature_extrinsics`) and Stage-2
(:mod:`floor_planefit`) into one flow that produces the same ``calibration.json``
the AprilGrid path writes, **plus** a per-stage annotated visualization
(:mod:`feature_viz`) and a three-level validation report
(:mod:`targetless_validation`):

    synchronized pairs + Multical K,D + operator scale-references + floor points
        ↓  solve_feature_extrinsics   (Stage 1: E → recoverPose → scale → BA)
        ↓  estimate_floor_anchor_planefit   (Stage 2: floor-plane world frame)
        ↓  assemble_calibration        (unchanged — K,D,R,t,H,P → calibration.json)
        ↓  build_validation_report     (geometric + metric + end-to-end)

The isical Studio calls :func:`solve_targetless` (via a JobRunner phase) and a CLI
subcommand ``calibrate-targetless`` wraps it too. The heavy ONNX matcher is
injected (``matcher``), so the flow is fully hermetic with a ``FakeMatcher``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from calibration.feature_extrinsics import (
    FeatureExtrinsicsResult,
    ScaleReference,
    StereoMatcher,
    solve_feature_extrinsics,
)
from calibration.floor_planefit import (
    FloorPointFilter,
    estimate_floor_anchor_planefit,
)


@dataclass(slots=True)
class TargetlessResult:
    """Everything the targetless flow produces for the UI + downstream."""

    calibration: Any                       # calibration.schema.CalibrationFile
    feature: FeatureExtrinsicsResult
    floor_anchor: Any                      # calibrate.FloorAnchor
    floor_result: Any                      # floor_planefit.PlaneFitResult
    report: Any                            # targetless_validation.ValidationReport
    stage_images: dict[str, np.ndarray] = field(default_factory=dict)


def _floor_points_from_references(
    feature: FeatureExtrinsicsResult,
    references: list[ScaleReference],
    K_a: np.ndarray,
    D_a: np.ndarray,
    K_b: np.ndarray,
    D_b: np.ndarray,
) -> np.ndarray:
    """Triangulate the operator's scale-reference landmarks as designated floor pts.

    The ≥3 measured floor references lie on the floor (the plan's world-frame
    anchor), so their triangulated 3D positions (in the refined metric pose) are the
    Stage-2 plane-fit input. Uses the refined ``feature.R`` / ``feature.t`` so the
    points are consistent with the assembled extrinsics.
    """
    from calibration.feature_extrinsics import (
        _triangulate_normalized,
        _undistort_to_normalized,
    )

    pts: list[np.ndarray] = []
    for ref in references:
        a = np.array([ref.p1_a, ref.p2_a], dtype=np.float64)
        b = np.array([ref.p1_b, ref.p2_b], dtype=np.float64)
        na = _undistort_to_normalized(a, K_a, D_a)
        nb = _undistort_to_normalized(b, K_b, D_b)
        p3 = _triangulate_normalized(na, nb, feature.R, feature.t)
        pts.append(p3)
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


def solve_targetless(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    matcher: StereoMatcher,
    K_a: np.ndarray,
    D_a: np.ndarray,
    K_b: np.ndarray,
    D_b: np.ndarray,
    references: list[ScaleReference],
    image_size_wh: tuple[int, int],
    *,
    floor_points: np.ndarray | None = None,
    floor_filter: FloorPointFilter | None = None,
    plane_threshold_m: float = 0.03,
    reference_calib: dict | None = None,
    measurements: list | None = None,
    end_to_end_result: dict | None = None,
    render_stage_images: bool = True,
    allow_unknown_rms: bool = False,
) -> TargetlessResult:
    """Run Stage-1 + Stage-2 + assemble, returning calibration + viz + report.

    ``floor_points`` (rig-frame ``(N,3)``) are the operator-designated floor points
    for the plane fit; when ``None`` they default to the triangulated scale-reference
    landmarks (which lie on the floor). ``reference_calib`` (the AprilGrid result on
    the same rig) + ``measurements`` (tape-measured metric ground truth) feed the
    validation report — both optional (the metric level reports pending without them).
    """
    from calibration.calibrate import assemble_calibration
    from calibration.targetless_validation import build_validation_report

    K_a = np.asarray(K_a, float)
    D_a = np.asarray(D_a, float).reshape(-1)
    K_b = np.asarray(K_b, float)
    D_b = np.asarray(D_b, float).reshape(-1)

    feature = solve_feature_extrinsics(
        image_pairs=image_pairs, matcher=matcher,
        K_a=K_a, D_a=D_a, K_b=K_b, D_b=D_b,
        references=references, image_size_wh=image_size_wh,
    )

    if floor_points is None:
        floor_points = _floor_points_from_references(
            feature, references, K_a, D_a, K_b, D_b)
    floor_points = np.asarray(floor_points, float).reshape(-1, 3)

    anchor, floor_result = estimate_floor_anchor_planefit(
        floor_points, filt=floor_filter, threshold_m=plane_threshold_m,
        return_result=True,
    )

    calibration = assemble_calibration(
        feature.solution, anchor, allow_unknown_rms=allow_unknown_rms)

    report = build_validation_report(
        _calib_dict(calibration),
        reference_calib=reference_calib,
        reprojection_rms_px=feature.reprojection_rms_px,
        measurements=measurements,
        end_to_end_result=end_to_end_result,
    )

    stage_images: dict[str, np.ndarray] = {}
    if render_stage_images and image_pairs:
        stage_images = _render_all_stages(
            image_pairs, feature, references, K_a, floor_result, report)

    return TargetlessResult(
        calibration=calibration, feature=feature, floor_anchor=anchor,
        floor_result=floor_result, report=report, stage_images=stage_images,
    )


def _calib_dict(calibration) -> dict:
    """CalibrationFile → plain dict (schema exposes .cameras with K/R/t)."""
    cams = {}
    for cid, c in calibration.cameras.items():
        cams[cid] = {
            "K": c.K, "D": c.D, "R": c.R, "t": c.t,
            "image_size_wh": list(c.image_size_wh),
            "reprojection_rms_px": c.reprojection_rms_px,
        }
    return {
        "cameras": cams,
        "floor_anchor_method": calibration.floor_anchor_method,
    }


def _render_all_stages(
    image_pairs, feature, references, K_a, floor_result, report,
) -> dict[str, np.ndarray]:
    """Render the 5 key-stage images from the solved result."""
    from calibration import feature_viz

    img_a, img_b = image_pairs[0]
    imgs: dict[str, np.ndarray] = {}
    imgs["pair"] = feature_viz.draw_stereo_pair(img_a, img_b, pair_index=0)
    # Stage 1 retains the concatenated matched pixels + inlier mask; the match viz
    # draws the whole-solve inlier/outlier split (a faithful confidence check).
    imgs["matches"] = feature_viz.draw_feature_matches(
        img_a, img_b, feature.matched_pts_a, feature.matched_pts_b,
        feature.inlier_mask)
    imgs["scale_refs"] = feature_viz.draw_scale_references(
        img_a, img_b, references, outliers=feature.scale.outliers)
    imgs["triangulation"] = feature_viz.draw_triangulation_floor(
        img_a, feature.points_3d, K_a,
        floor_inlier_mask=None,
        plane_normal=floor_result.normal, plane_offset=floor_result.offset)
    imgs["result"] = feature_viz.draw_result_overlay(
        img_a, report.summary_lines, passed=report.accepted)
    return imgs
