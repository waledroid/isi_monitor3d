"""Floor-plane world frame from triangulated floor points (Stage 2).

An **optional, targetless** replacement for the ChArUco floor anchor
(:func:`calibration.calibrate.estimate_floor_anchor_charuco`). Instead of
solving a physical board's pose, it fits a plane to the triangulated **floor**
3D points recovered by the Stage-1 feature-extrinsics pipeline and defines the
warehouse world frame from that plane:

    floor 3D points (rig frame) + per-point quality signals
        ↓  pre-filter (reproj error / confidence / depth)   →  clean floor set
        ↓  RANSAC plane fit (SVD refine on inliers)          →  plane (normal, offset)
        ↓  build orthonormal world basis, Z=0 on the plane   →  R, t (rig → world)
        ↓  FloorAnchor(method="planefit")                    →  compose_camera_in_world

The output is the **same** :class:`~calibration.calibrate.FloorAnchor` shape as
the ChArUco anchor (``method``, ``note``, ``R_world_from_rig``,
``t_world_from_rig``, all meaning *rig → world*), so
``compose_camera_in_world`` + ``assemble_calibration`` +
``backbone.shared.geometry``'s H/P derivation reuse it unchanged.

**Caller responsibility.** Stage 2 does **not** auto-classify floor vs. walls.
The Stage-3 UI designates which triangulated points are floor (the ≥3 measured
floor scale-references are on the floor, plus any operator-marked floor region);
this module consumes that provided set and only *refines* it by dropping points
that are individually untrustworthy (bad reprojection / low confidence / an
implausible depth) before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ``FloorAnchor`` lives in ``calibrate.py``; import lazily inside the function to
# avoid importing the (heavier) calibrate module — and any OpenCV/aruco surface —
# at module import time. The dataclass is tiny and cheap once resolved.


@dataclass(slots=True)
class FloorPointFilter:
    """Per-point quality gates applied *before* the plane fit.

    Every gate is optional: leave a signal as ``None`` in
    :func:`estimate_floor_anchor_planefit` and its threshold is ignored. A point
    survives only if it passes *every* gate for which a signal was supplied.
    """

    max_reproj_error_px: float = 3.0     # drop points reprojecting worse than this
    min_confidence: float = 0.2          # drop matches / stereo below this score
    min_depth_m: float = 0.1             # drop points closer than this (behind/at cam)
    max_depth_m: float = 50.0            # drop points farther than this (unstable)


@dataclass(slots=True)
class PlaneFitResult:
    """Diagnostics from the RANSAC plane fit (for the UI + validation)."""

    normal: np.ndarray          # (3,) unit plane normal in the rig frame
    offset: float               # plane: normal · x + offset = 0
    centroid: np.ndarray        # (3,) inlier centroid, projected onto the plane
    inlier_mask: np.ndarray     # (N_filtered,) bool over the post-filter points
    n_input: int                # points handed to this module
    n_after_filter: int         # survivors of the quality pre-filter
    n_dropped_filter: int       # dropped by the pre-filter
    n_inliers: int              # RANSAC plane inliers
    n_outliers: int             # RANSAC plane outliers (off-plane, kept off the fit)
    rms_inlier_dist_m: float    # RMS point-to-plane distance of the inliers


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------


def prefilter_floor_points(
    points: np.ndarray,
    *,
    reproj_error_px: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    filt: FloorPointFilter | None = None,
) -> np.ndarray:
    """Boolean keep-mask over ``points`` from the per-point quality signals.

    ``points`` is ``(N, 3)`` in the rig frame; depth is taken as the point's
    distance along the rig's optical axis (``+Z``, the cam_a forward axis used by
    Stage 1). A point is kept only if it passes every *supplied* gate — an
    unsupplied signal (``None``) never rejects anything.
    """
    filt = filt or FloorPointFilter()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)

    if reproj_error_px is not None:
        err = np.asarray(reproj_error_px, dtype=np.float64).reshape(-1)
        keep &= err <= filt.max_reproj_error_px

    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
        keep &= conf >= filt.min_confidence

    depth = pts[:, 2]
    keep &= np.isfinite(pts).all(axis=1)
    keep &= depth >= filt.min_depth_m
    keep &= depth <= filt.max_depth_m

    return keep


# ---------------------------------------------------------------------------
# RANSAC plane fit
# ---------------------------------------------------------------------------


def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares plane through ``pts`` (>=3) via SVD.

    Returns ``(normal, offset, centroid)`` for the plane
    ``normal · x + offset = 0`` with ``normal`` unit-length. The normal is the
    least-significant singular direction of the mean-centred points.
    """
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    offset = float(-normal @ centroid)
    return normal, offset, centroid


def _point_plane_distance(pts: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return np.abs(pts @ normal + offset)


def ransac_plane_fit(
    points: np.ndarray,
    *,
    threshold_m: float = 0.03,
    iterations: int = 200,
    min_inlier_frac: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """RANSAC plane fit: sample 3 points, score inliers, refine on the best set.

    Returns ``(normal, offset, centroid, inlier_mask)``. ``threshold_m`` is the
    point-to-plane inlier band in metres. Falls back to a plain SVD fit on all
    points when there are too few (<3) to sample.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        raise ValueError(f"plane fit needs >=3 points, got {n}")

    rng = rng or np.random.default_rng(0)

    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(int(iterations)):
        idx = rng.choice(n, size=3, replace=False)
        sample = pts[idx]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue  # collinear sample
        normal = normal / nn
        offset = float(-normal @ sample[0])
        dist = _point_plane_distance(pts, normal, offset)
        inliers = dist <= threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < max(3, int(min_inlier_frac * n)):
        # Degenerate scene (all collinear / no consensus) — fall back to a plain
        # SVD fit over every point so the caller still gets a usable plane.
        normal, offset, centroid = _fit_plane_svd(pts)
        inliers = _point_plane_distance(pts, normal, offset) <= threshold_m
        return normal, offset, centroid, inliers

    # Refine on the winning inlier set (least-squares SVD), then re-score once so
    # the reported inliers are consistent with the refined plane.
    normal, offset, centroid = _fit_plane_svd(pts[best_inliers])
    inliers = _point_plane_distance(pts, normal, offset) <= threshold_m
    if int(inliers.sum()) >= 3:
        normal, offset, centroid = _fit_plane_svd(pts[inliers])
        inliers = _point_plane_distance(pts, normal, offset) <= threshold_m
    return normal, offset, centroid, inliers


# ---------------------------------------------------------------------------
# World-frame construction
# ---------------------------------------------------------------------------


def _orient_normal_up(normal: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Choose the plane-normal sign so world +Z points *up* toward the cameras.

    The rig origin is cam_a (Stage 1's master). The floor lies in front of and
    below the cameras, so the floor-to-camera direction is the "up" we want. We
    orient the normal to point from the floor centroid back toward the rig
    origin (the camera). Deterministic: if that vector is (numerically)
    perpendicular to the normal, fall back to preferring a negative rig-``Y``
    component (rig +Y points down in the usual image convention).
    """
    to_camera = -centroid  # rig origin (0,0,0) less the floor centroid
    proj = float(normal @ to_camera)
    if abs(proj) > 1e-6:
        return normal if proj > 0 else -normal
    # Degenerate: normal perpendicular to (origin - centroid). Prefer -Y (image up).
    return normal if normal[1] <= 0 else -normal


def _inplane_basis(normal: np.ndarray, x_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal in-plane (X, Y) basis given the plane normal.

    ``x_hint`` (a world-X preference, e.g. the rig +X axis) is projected onto the
    plane to fix world +X; world +Y = normal cross X completes a right-handed frame.
    If the hint is (near) parallel to the normal, fall back to the rig +Y axis.
    """
    z = normal / (np.linalg.norm(normal) + 1e-12)
    hint = np.asarray(x_hint, dtype=np.float64)
    x = hint - (hint @ z) * z
    if np.linalg.norm(x) < 1e-6:
        alt = np.array([0.0, 1.0, 0.0])
        x = alt - (alt @ z) * z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    # Re-orthogonalise x against y,z (guards accumulated float error).
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    return x, y


def estimate_floor_anchor_planefit(
    floor_points: np.ndarray,
    *,
    reproj_error_px: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    filt: FloorPointFilter | None = None,
    threshold_m: float = 0.03,
    iterations: int = 200,
    x_hint: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    return_result: bool = False,
):
    """Estimate the rig → world floor anchor by plane-fitting floor 3D points.

    Parameters
    ----------
    floor_points:
        ``(N, 3)`` triangulated points *known to lie on the floor* (rig frame),
        as designated by the Stage-3 UI. This module does not classify floor vs.
        walls — it trusts the provided set and only drops individually
        untrustworthy points via the pre-filter.
    reproj_error_px, confidence:
        Optional per-point ``(N,)`` quality signals used by the pre-filter.
    filt:
        Pre-filter thresholds (:class:`FloorPointFilter`).
    threshold_m:
        RANSAC point-to-plane inlier band, metres.
    x_hint:
        World-X preference (default: rig +X, ``[1, 0, 0]``). Projected onto the
        plane to make the in-plane basis deterministic.
    return_result:
        When ``True`` also return the :class:`PlaneFitResult` diagnostics.

    Returns
    -------
    ``FloorAnchor`` (``method="planefit"``), or ``(FloorAnchor, PlaneFitResult)``
    when ``return_result=True``. The anchor's world frame places **Z=0 on the
    fitted floor plane**, world **+Z along the plane normal oriented up** toward
    the cameras, the origin at the inlier centroid projected onto the plane, and
    an orthonormal in-plane X/Y basis.
    """
    from calibration.calibrate import FloorAnchor

    pts = np.asarray(floor_points, dtype=np.float64).reshape(-1, 3)
    n_input = pts.shape[0]
    if n_input < 3:
        raise ValueError(f"floor plane-fit needs >=3 points, got {n_input}")

    keep = prefilter_floor_points(
        pts, reproj_error_px=reproj_error_px, confidence=confidence, filt=filt,
    )
    kept = pts[keep]
    n_after = kept.shape[0]
    if n_after < 3:
        raise RuntimeError(
            f"floor plane-fit: only {n_after} of {n_input} points survived the "
            "quality pre-filter (need >=3). Loosen the filter or mark more floor."
        )

    normal, offset, centroid, inliers = ransac_plane_fit(
        kept, threshold_m=threshold_m, iterations=iterations, rng=rng,
    )
    normal = _orient_normal_up(normal, centroid)
    # Keep offset consistent with the (possibly sign-flipped) normal.
    offset = float(-normal @ centroid)

    # Origin: inlier centroid projected onto the plane (drops any residual
    # off-plane component so the world origin sits exactly on Z=0).
    signed = float(normal @ centroid + offset)
    origin = centroid - signed * normal

    z_axis = normal
    hint = np.array([1.0, 0.0, 0.0]) if x_hint is None else np.asarray(x_hint, float)
    x_axis, y_axis = _inplane_basis(z_axis, hint)

    # R_world_from_rig maps rig vectors into world coords: its rows are the world
    # basis axes expressed in rig coordinates. t_world_from_rig places the origin.
    R_world_from_rig = np.vstack([x_axis, y_axis, z_axis])
    if np.linalg.det(R_world_from_rig) < 0:  # guarantee a proper rotation
        y_axis = -y_axis
        R_world_from_rig = np.vstack([x_axis, y_axis, z_axis])
    t_world_from_rig = -R_world_from_rig @ origin

    inlier_dist = _point_plane_distance(kept[inliers], normal, offset)
    rms = float(np.sqrt(np.mean(inlier_dist**2))) if inlier_dist.size else 0.0

    result = PlaneFitResult(
        normal=normal,
        offset=offset,
        centroid=origin,
        inlier_mask=inliers,
        n_input=n_input,
        n_after_filter=n_after,
        n_dropped_filter=int(n_input - n_after),
        n_inliers=int(inliers.sum()),
        n_outliers=int(n_after - int(inliers.sum())),
        rms_inlier_dist_m=rms,
    )

    anchor = FloorAnchor(
        method="planefit",
        note=(
            f"RANSAC floor plane fit: {result.n_inliers}/{n_after} inliers "
            f"({result.n_dropped_filter} pre-filtered of {n_input}), "
            f"plane RMS {rms * 1000:.1f} mm"
        ),
        R_world_from_rig=R_world_from_rig,
        t_world_from_rig=t_world_from_rig,
    )

    if return_result:
        return anchor, result
    return anchor
