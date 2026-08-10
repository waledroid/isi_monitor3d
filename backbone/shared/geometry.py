"""Pure geometric helpers used by both the homography and triangulation layers.

Stateless, NumPy/OpenCV in, NumPy out. No I/O, no logging — these functions
are called per detection at high frequency and any sneaky side effect would
show up in the latency budget.

Conventions:
    * Pixel coordinates: (u, v) in pixels, origin top-left, +u right, +v down.
    * World coordinates: (X, Y, Z) in meters; the floor plane is Z = 0.
    * "Floor (X, Y)" means dropping Z from the world frame.
    * Distortion model: OpenCV plumb-bob (k1, k2, p1, p2[, k3, ...]).
"""

from __future__ import annotations

import cv2
import numpy as np


def undistort_points(
    points_uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Correct pixel coordinates for lens distortion.

    Skipping this step costs several centimeters of metric accuracy at frame
    edges — non-negotiable for the ≤2 px reprojection KPI.

    Args:
        points_uv: ``(N, 2)`` array of pixel coordinates.
        K: 3x3 intrinsic matrix.
        D: distortion coefficients (k1, k2, p1, p2, k3, ...).

    Returns:
        ``(N, 2)`` array of undistorted pixel coordinates (still in the same
        pixel frame, i.e. multiplied back by ``K``).
    """
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, D, P=K)
    return out.reshape(-1, 2)


def undistort_points_checked(
    points_uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    *,
    max_roundtrip_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`undistort_points` plus a validity mask.

    ``cv2.undistortPoints``'s iterative inversion DIVERGES near/beyond the edge
    of a strong barrel lens (k1 ~ -0.4): a border pixel can "undistort" to
    coordinates a thousand pixels off, which then authors an absurd floor
    point (observed: cam_b pixel x=75 -> undistorted x=-1412 -> a 4.5 m outlier
    that spiked a projected zone outline). Validity is checked by pushing the
    undistorted point back through the FORWARD distortion model — a genuine
    inversion returns to the original pixel within ``max_roundtrip_px``.

    Returns ``(undistorted_points, valid_mask)``.
    """
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    und = undistort_points(pts, K, D)
    # Forward model: normalized coords of the undistorted pixel, re-distorted.
    norm = np.hstack([
        ((und[:, 0] - K[0, 2]) / K[0, 0]).reshape(-1, 1),
        ((und[:, 1] - K[1, 2]) / K[1, 1]).reshape(-1, 1),
        np.ones((len(und), 1)),
    ])
    redist, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), K, D)
    err = np.linalg.norm(redist.reshape(-1, 2) - pts, axis=1)
    return und, np.isfinite(err) & (err <= max_roundtrip_px)


def pixel_to_floor(
    points_uv: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Project pixels to floor coordinates (X, Y) in meters via the homography.

    Expects ``points_uv`` to be already undistorted — apply :func:`undistort_points`
    first. The pipeline is: detection → bottom-center of bbox → undistort → here.

    Args:
        points_uv: ``(N, 2)`` undistorted pixel coordinates.
        H: 3x3 homography matrix, pixel → world (meters).

    Returns:
        ``(N, 2)`` floor coordinates in meters.
    """
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def pixel_to_plane(
    points_uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    z_m: float,
    *,
    min_ray_z_component: float = 1e-9,
) -> np.ndarray | None:
    """Project pixels onto a horizontal plane ``Z = z_m`` via ray/plane intersection.

    Generalizes :func:`pixel_to_floor` (restricted to ``Z = 0`` and driven by
    the baked-in homography ``H``) to an arbitrary height — the plane a raised
    zone (a loading platform, a shelf) actually sits on. For ``z_m = 0`` this
    agrees with ``pixel_to_floor(undistort_points(points_uv, K, D), H)`` to
    <1e-6 m (pinned by test) — ``H`` IS this same ray/plane intersection,
    specialized to ``Z = 0`` and folded into a 3x3 matrix at calibration time.

    Unlike :func:`pixel_to_floor`, which expects already-undistorted input,
    ``points_uv`` here are RAW (distorted) pixel coordinates — undistortion
    happens internally, once, since there's no plane-specific homography to
    fold it into.

    ``R, t`` follow this module's camera-POSE convention (world←camera, see
    :func:`projection_from_K_R_t`): the camera center in world coordinates is
    ``t``, and a camera-frame ray direction rotates into world with ``R``
    alone (no translation — directions, not points).

    Args:
        points_uv: ``(N, 2)`` RAW (distorted) pixel coordinates.
        K: 3x3 intrinsic matrix.
        D: distortion coefficients (k1, k2, p1, p2, k3, ...).
        R: 3x3 rotation, world←camera (camera pose).
        t: camera center in world coordinates, meters.
        z_m: height (meters) of the horizontal plane to intersect.
        min_ray_z_component: a ray whose world-frame Z-direction magnitude is
            below this (parallel to the plane) is degenerate.

    Returns:
        ``(N, 2)`` array of world ``(X, Y)`` meters on the plane ``Z = z_m``.
        Degenerate points (ray parallel to the plane, or the plane lies
        behind the camera along the ray) come back as a NaN row so callers
        processing a batch can locate exactly which inputs failed. If EVERY
        point in the call is degenerate, the whole call returns ``None``
        instead — the common case is a single-point call (e.g. one detection's
        foot pixel), where "some rows NaN" and "totally degenerate" coincide
        and a plain ``None`` is the more useful signal for the caller.
    """
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    und = undistort_points(pts, K, D)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dir_cam = np.column_stack([
        (und[:, 0] - cx) / fx,
        (und[:, 1] - cy) / fy,
        np.ones(len(und)),
    ])
    dir_world = dir_cam @ R.T   # row i = R @ dir_cam[i] (rotate direction only)

    dz = dir_world[:, 2]
    valid = np.abs(dz) > min_ray_z_component
    s = np.full(len(pts), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s[valid] = (z_m - t[2]) / dz[valid]
    valid &= s > 0.0   # plane must be ahead of the camera along the ray

    if not valid.any():
        return None

    out = np.full((len(pts), 2), np.nan)
    world = t[np.newaxis, :] + s[:, np.newaxis] * dir_world
    out[valid] = world[valid, :2]
    return out


def floor_to_pixel(
    points_xy_m: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Project floor (X, Y) meters back to pixels. Used by the dashboard overlay."""
    H_inv = np.linalg.inv(H)
    pts = np.asarray(points_xy_m, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_inv)
    return out.reshape(-1, 2)


def has_metric_camera_model(K, R, t) -> bool:
    """False for Mode-1 placeholder extrinsics (``K=I, R=I, t=0`` — only ``H``
    is real). Consumers must then use the H-based :func:`floor_to_pixel`;
    the full-model :func:`floor_to_pixel_distorted` would emit garbage."""
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    return not (np.allclose(K, np.eye(3)) and np.allclose(R, np.eye(3))
                and np.allclose(t, 0.0))


def floor_to_pixel_distorted(
    points_xy_m: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size_wh: tuple[int, int],
    *,
    margin_frac: float = 0.25,
) -> np.ndarray:
    """Project floor (X, Y) metres to RAW (distorted) image pixels.

    :func:`floor_to_pixel` returns PINHOLE pixels — drawing those over the live
    (distorted) camera frame misplaces overlays by 50-150+ px near the frame
    edges on a strong barrel lens (k1 ≈ -0.45). This variant applies the full
    ``cv2.projectPoints`` model so overlays hug the real image.

    Divergence guard: the distortion polynomial is only valid inside the
    calibrated field — points far outside explode to absurd coordinates. A
    point whose PINHOLE projection lies beyond ``margin_frac`` of the image
    bounds keeps its pinhole coordinates instead (it is off-screen either way;
    consumers clip). Points behind the camera also keep pinhole coordinates.
    """
    out, _pinhole, _distorted_mask = _floor_to_pixel_distorted_impl(
        points_xy_m, K, D, R, t, image_size_wh, margin_frac=margin_frac)
    return out


def _floor_to_pixel_distorted_impl(
    points_xy_m: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size_wh: tuple[int, int],
    *,
    margin_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core of :func:`floor_to_pixel_distorted`; also returns the pinhole
    baseline and the mask of points that took the DISTORTED path (the rest
    hold the pinhole fallback)."""
    pts = np.asarray(points_xy_m, dtype=np.float64).reshape(-1, 2)
    world3 = np.hstack([pts, np.zeros((len(pts), 1))])
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    # Pinhole baseline (also the fallback for out-of-field points).
    P = projection_from_K_R_t(np.asarray(K, dtype=np.float64), R, t)
    pinhole = project_world_to_pixel(world3, P)

    # (R, t) are world←camera in this codebase (the camera POSE — see
    # projection_from_K_R_t); cv2.projectPoints wants the camera←world
    # extrinsic, i.e. the inverse.
    R_cw = R.T
    t_cw = -R_cw @ t

    w, h = float(image_size_wh[0]), float(image_size_wh[1])
    mx, my = w * margin_frac, h * margin_frac
    cam_z = (R_cw @ world3.T).T[:, 2] + t_cw[2]
    ok = (
        (cam_z > 1e-6)
        & (pinhole[:, 0] >= -mx) & (pinhole[:, 0] < w + mx)
        & (pinhole[:, 1] >= -my) & (pinhole[:, 1] < h + my)
    )
    out = pinhole.copy()
    if ok.any():
        rvec, _ = cv2.Rodrigues(R_cw)
        duv, _ = cv2.projectPoints(world3[ok], rvec, t_cw, np.asarray(K, dtype=np.float64),
                                   np.asarray(D, dtype=np.float64))
        out[ok] = duv.reshape(-1, 2)
    return out, pinhole, ok


def floor_to_pixel_distorted_checked(
    points_xy_m: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size_wh: tuple[int, int],
    *,
    max_roundtrip_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`floor_to_pixel_distorted` plus a validity mask.

    The FORWARD distortion polynomial of a strong lens folds back beyond a
    critical radius (the site cam_b lens folds at normalized r = 1.11 — just
    outside its own frame corners at r = 1.03): a floor point slightly outside
    the view gets projected back INSIDE the frame at a wrong, folded position,
    silently warping projected outlines. Validity is checked by undistorting
    the projected pixel and comparing against the point's ideal PINHOLE
    projection — a folded/non-invertible projection can't round-trip.

    Returns ``(pixels, valid_mask)``. Pinhole-fallback points (out of field)
    are always invalid — their coordinates are clippable but not trustworthy
    distorted pixels.
    """
    out, pinhole, distorted = _floor_to_pixel_distorted_impl(
        points_xy_m, K, D, R, t, image_size_wh)
    try:
        und = undistort_points(out, K, D)
        err = np.linalg.norm(und - pinhole, axis=1)
        valid = distorted & np.isfinite(err) & (err <= max_roundtrip_px)
    except Exception:
        valid = np.zeros(len(out), dtype=bool)
    return out, valid


def radial_fold_radius(D) -> float | None:
    """The normalized radius where the FORWARD radial polynomial stops being
    monotonic (``d/dr [r(1 + k1 r² + k2 r⁴ + k3 r⁶)] = 0``), or ``None`` for a
    monotone lens. Beyond it the projection folds — two world rays map to the
    same pixel — so nothing outside this radius is trustworthy. Tangential and
    higher-order terms are ignored (the radial terms dominate the fold)."""
    D = np.asarray(D, dtype=np.float64).ravel()
    k1 = D[0] if len(D) > 0 else 0.0
    k2 = D[1] if len(D) > 1 else 0.0
    k3 = D[4] if len(D) > 4 else 0.0
    r = np.linspace(1e-3, 5.0, 5000)
    deriv = 1.0 + 3.0 * k1 * r**2 + 5.0 * k2 * r**4 + 7.0 * k3 * r**6
    bad = np.flatnonzero(deriv <= 0.0)
    return float(r[bad[0]]) if len(bad) else None


def clip_polygon_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: ``subject`` ∩ ``clip`` (``clip`` must be convex).
    Returns the intersection polygon (possibly empty)."""
    clip = np.asarray(clip, dtype=np.float64).reshape(-1, 2)
    # Ensure counter-clockwise clip orientation (positive signed area).
    area2 = float(np.sum(clip[:, 0] * np.roll(clip[:, 1], -1)
                         - np.roll(clip[:, 0], -1) * clip[:, 1]))
    if area2 < 0:
        clip = clip[::-1]
    out = [p for p in np.asarray(subject, dtype=np.float64).reshape(-1, 2)]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if not out:
            break
        inp, out = out, []
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        def cross_at(p, q, a=a, e=e):
            d = q - p
            denom = e[0] * d[1] - e[1] * d[0]
            s = (e[0] * (a[1] - p[1]) - e[1] * (a[0] - p[0])) / denom
            return p + s * d

        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            sp, sq = side(p), side(q)
            if sq >= 0.0:
                if sp < 0.0:
                    out.append(cross_at(p, q))
                out.append(q)
            elif sp >= 0.0:
                out.append(cross_at(p, q))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def project_floor_polygon_distorted(
    polygon_xy_m: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size_wh: tuple[int, int],
    *,
    margin_frac: float = 0.25,
    fold_safety: float = 0.9,
    circle_segments: int = 96,
) -> np.ndarray | None:
    """Project a floor POLYGON into RAW (distorted) pixels, clipped to the
    camera's reliably-projectable field.

    Point-wise projection cannot represent a zone that spills past the
    camera's field: samples beyond the lens's fold radius (see
    :func:`radial_fold_radius`) project back INSIDE the frame at folded
    positions (warped outlines), and merely dropping them collapses the
    outline to a sliver — the visible overlap is bounded by the FIELD RIM,
    not by the zone's own boundary. So the polygon is clipped, in normalized
    pinhole coordinates, against the projection margin rectangle ∩ the fold
    disk, and only then distorted. Points behind the camera are dropped first.

    Returns the clipped, distorted pixel polygon, or ``None`` when the zone
    doesn't meaningfully overlap the field.
    """
    pts = np.asarray(polygon_xy_m, dtype=np.float64).reshape(-1, 2)
    world3 = np.hstack([pts, np.zeros((len(pts), 1))])
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    R_cw = R.T
    t_cw = -R_cw @ t
    camp = (R_cw @ world3.T).T + t_cw
    camp = camp[camp[:, 2] > 1e-6]
    if len(camp) < 3:
        return None
    norm = camp[:, :2] / camp[:, 2:3]

    w, h = float(image_size_wh[0]), float(image_size_wh[1])
    mx, my = w * margin_frac, h * margin_frac
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rect = np.array([
        [(-mx - cx) / fx, (-my - cy) / fy],
        [(w + mx - cx) / fx, (-my - cy) / fy],
        [(w + mx - cx) / fx, (h + my - cy) / fy],
        [(-mx - cx) / fx, (h + my - cy) / fy],
    ])
    poly = clip_polygon_convex(norm, rect)
    fold = radial_fold_radius(D)
    if fold is not None and len(poly):
        rv = fold * fold_safety
        ang = np.linspace(0.0, 2.0 * np.pi, circle_segments, endpoint=False)
        disk = np.stack([rv * np.cos(ang), rv * np.sin(ang)], axis=1)
        poly = clip_polygon_convex(poly, disk)
    if len(poly) < 3:
        return None
    obj = np.hstack([poly, np.ones((len(poly), 1))])
    duv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K,
                               np.asarray(D, dtype=np.float64))
    return duv.reshape(-1, 2)


def densify_polygon(polygon: np.ndarray, segments_per_edge: int = 8) -> np.ndarray:
    """Subdivide each polygon edge — a straight image line is CURVED after a
    cross-camera floor round-trip (homography + lens distortion), so overlays
    must be sampled, not just vertex-mapped."""
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    n = len(poly)
    ts = np.linspace(0.0, 1.0, segments_per_edge, endpoint=False)
    out = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        out.extend(a + (b - a) * s for s in ts)
    return np.asarray(out)


def project_world_to_pixel(
    points_xyz_m: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to pixels with the full projection matrix.

    Used by the reprojection-error gate after triangulation. The gate is the
    only thing standing between bad input and silent bad 3D output.

    Args:
        points_xyz_m: ``(N, 3)`` world points in meters.
        P: 3x4 projection matrix.

    Returns:
        ``(N, 2)`` pixel coordinates.
    """
    pts = np.asarray(points_xyz_m, dtype=np.float64)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = homog @ P.T
    return proj[:, :2] / proj[:, 2:3]


def reprojection_error_px(
    point_xyz_m: np.ndarray,
    observations_uv: dict[str, np.ndarray],
    P_by_camera: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-camera reprojection error, in pixels, for a single 3D point.

    Returns a dict keyed by camera_id. The reprojection gate rejects the
    triangulation if ``max(values()) > threshold``.
    """
    errors: dict[str, float] = {}
    for cam_id, obs_uv in observations_uv.items():
        P = P_by_camera[cam_id]
        proj = project_world_to_pixel(point_xyz_m.reshape(1, 3), P).reshape(2)
        errors[cam_id] = float(np.linalg.norm(np.asarray(obs_uv, dtype=np.float64) - proj))
    return errors


def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Compute the fundamental matrix F such that x2.T @ F @ x1 = 0.

    Derived directly from the two 3x4 projection matrices — no need for the
    eight-point algorithm or live observations. Used by the keypoint
    associator's epipolar fallback (left/right limb disambiguation).
    """

    def _drop_row(M: np.ndarray, row: int) -> np.ndarray:
        return np.delete(M, row, axis=0)

    F = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M = np.vstack([_drop_row(P1, i), _drop_row(P2, j)])
            F[j, i] = ((-1) ** (i + j)) * np.linalg.det(M)
    return F


def epipolar_line(point_uv: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Epipolar line ``(a, b, c)`` in the second view such that ``a u + b v + c = 0``.

    ``point_uv`` is in the first view.
    """
    u, v = float(point_uv[0]), float(point_uv[1])
    line = F @ np.array([u, v, 1.0])
    return line


def point_line_distance_px(point_uv: np.ndarray, line_abc: np.ndarray) -> float:
    """Perpendicular pixel distance from a point to an epipolar line."""
    a, b, c = line_abc
    u, v = float(point_uv[0]), float(point_uv[1])
    denom = float(np.hypot(a, b))
    if denom == 0.0:
        return float("inf")
    return abs(a * u + b * v + c) / denom


def projection_from_K_R_t(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose ``P = K @ [R | t]`` — used during calibration export.

    Note: ``R, t`` are world ← camera in this codebase. The projection matrix
    that maps world points to pixels needs the inverse: ``R^T, -R^T t``.
    """
    R_cw = R.T
    t_cw = -R_cw @ t.reshape(3)
    Rt = np.hstack([R_cw, t_cw.reshape(3, 1)])
    return K @ Rt


def floor_homography_from_K_R_t(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose the floor-plane homography (pixel → world meters at Z=0).

    Given camera projection columns ``[r1 r2 r3 | t]`` of ``[R_cw | t_cw]``,
    the floor homography (world → pixel) is ``K @ [r1 r2 t_cw]``. Inverting
    it yields pixel → world. Returned in the pixel→world direction so it
    drops directly into :func:`pixel_to_floor`.
    """
    R_cw = R.T
    t_cw = -R_cw @ t.reshape(3)
    H_world_to_pixel = K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], t_cw])
    return np.linalg.inv(H_world_to_pixel)
