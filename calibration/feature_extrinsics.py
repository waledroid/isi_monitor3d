"""Targetless stereo extrinsics via feature matching (Stage 1).

An **optional, experimental** alternative to the AprilGrid extrinsic phase. It
reuses Multical's intrinsics (``K``, ``D`` per camera) and a set of synchronized
stereo image pairs, and recovers the stereo pose (``R``, ``t``) from **feature
correspondences** instead of a physical calibration target:

    matcher (SuperPoint + LightGlue, ONNX)  →  pixel correspondences
        ↓  undistortPoints (K, D)            →  normalized rays
        ↓  findEssentialMat (RANSAC)         →  inlier mask + E
        ↓  recoverPose                       →  R_rel, unit T_dir
        ↓  metric scale from ≥3 references   →  T = s · T_dir  (cross-validated)
        ↓  triangulatePoints                 →  metric 3D points
        ↓  bundle adjustment (least_squares) →  refined R, t, scale, points
        ↓  MultiCalSolution                  →  assemble_calibration (unchanged)

The pose is emitted as a :class:`~calibration.multical_io.MultiCalSolution` with
the master camera (``cam_a``) as the rig frame identity and the second camera
(``cam_b``) carrying the refined pose, so the existing floor-anchor +
``assemble_calibration`` path (``calibrate.py``) reuses this output unchanged.

**Conventions.** ``CameraInRig.(R_in_rig, t_in_rig)`` is *rig ← camera*: it maps
a point from that camera's frame into the rig frame
(``p_rig = R_in_rig @ p_cam + t_in_rig``). ``cv2.recoverPose`` returns
``(R, t)`` mapping cam_a (= rig) coordinates *into* cam_b's frame — the *cam_b ←
rig* transform. We therefore **invert** it for cam_b's rig-frame pose:
``R_in_rig = R.T``, ``t_in_rig = -R.T @ t``.

**Runtime.** SuperPoint + LightGlue run as ONNX in the ``monitor3d`` env
(onnxruntime + CUDAExecutionProvider — torch-free). The real ONNX path
(:class:`OnnxSuperPointLightGlue`) is written but **not exercised** by the
hermetic tests; a :class:`FakeMatcher` injects deterministic correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from calibration.multical_io import CameraInRig, MultiCalSolution

# ---------------------------------------------------------------------------
# Matcher interface + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class StereoMatcher(Protocol):
    """Produces pixel-level correspondences between two views.

    ``match`` returns ``(pts_a, pts_b, scores)`` where ``pts_a`` / ``pts_b`` are
    ``(N, 2)`` float arrays of matched pixel coordinates in view A / view B, and
    ``scores`` is an ``(N,)`` array of per-match confidences in ``[0, 1]``.
    """

    def match(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class MatcherWeightsMissing(RuntimeError):
    """Raised when the ONNX matcher's weight files are not vendored into ``models/``."""


@dataclass(slots=True)
class FakeMatcher:
    """Deterministic matcher for hermetic tests / dependency injection.

    Ignores the image content and returns a fixed set of correspondences. Build
    it from a synthetic scene (project a 3D cloud to two cameras) so the full
    Essential → pose → scale → BA pipeline is exercised without ONNX weights.
    """

    pts_a: np.ndarray
    pts_b: np.ndarray
    scores: np.ndarray | None = None

    def match(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(self.pts_a, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(self.pts_b, dtype=np.float64).reshape(-1, 2)
        if a.shape != b.shape:
            raise ValueError(
                f"FakeMatcher pts_a {a.shape} and pts_b {b.shape} must match"
            )
        if self.scores is None:
            s = np.ones(a.shape[0], dtype=np.float64)
        else:
            s = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        return a, b, s


# Expected ONNX weight filenames, vendored from fabio-sim/LightGlue-ONNX.
_SUPERPOINT_ONNX = "superpoint.onnx"
_LIGHTGLUE_ONNX = "superpoint_lightglue.onnx"


@dataclass(slots=True)
class OnnxSuperPointLightGlue:
    """SuperPoint keypoints + LightGlue matching, as ONNX (onnxruntime).

    Weights are **not** downloaded — vendor them from
    `fabio-sim/LightGlue-ONNX <https://github.com/fabio-sim/LightGlue-ONNX>`_
    into ``models/`` (``superpoint.onnx`` + ``superpoint_lightglue.onnx``). If
    they are absent, construction raises :class:`MatcherWeightsMissing` with a
    clear operator message rather than attempting any network access.

    This implementation is the on-rig path; the hermetic test suite never loads
    it (it exercises :class:`FakeMatcher` instead), so the ONNX inference glue
    is intentionally minimal and validated only on the real rig.
    """

    models_dir: Path
    max_keypoints: int = 1024
    match_threshold: float = 0.2
    _superpoint: object = field(default=None, init=False, repr=False)
    _lightglue: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        models_dir = Path(self.models_dir)
        sp = models_dir / _SUPERPOINT_ONNX
        lg = models_dir / _LIGHTGLUE_ONNX
        missing = [p.name for p in (sp, lg) if not p.exists()]
        if missing:
            raise MatcherWeightsMissing(
                "targetless matcher ONNX weights not found in "
                f"{models_dir}: missing {missing}. Vendor the SuperPoint + "
                "LightGlue ONNX weights from fabio-sim/LightGlue-ONNX into "
                f"{models_dir} (expected files: {_SUPERPOINT_ONNX!r}, "
                f"{_LIGHTGLUE_ONNX!r}). Weights are NOT downloaded automatically."
            )

        import onnxruntime as ort  # local: keep module import torch-/ort-free

        providers = ort.get_available_providers()
        want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
        self._superpoint = ort.InferenceSession(str(sp), providers=want)
        self._lightglue = ort.InferenceSession(str(lg), providers=want)

    @staticmethod
    def _to_gray_tensor(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.float32)[None, None] / 255.0

    def match(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # On-rig path — not exercised hermetically. SuperPoint extracts
        # keypoints/descriptors per view; LightGlue matches them. Output tensor
        # names vary across the LightGlue-ONNX export revisions, so we resolve
        # them by role (keypoints / matches / scores) at call time.
        ta = self._to_gray_tensor(img_a)
        tb = self._to_gray_tensor(img_b)

        def _run_superpoint(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            sess = self._superpoint
            outs = sess.run(None, {sess.get_inputs()[0].name: t})
            names = [o.name for o in sess.get_outputs()]
            by = dict(zip(names, outs, strict=False))
            kpts = by.get("keypoints", outs[0])
            desc = by.get("descriptors", outs[-1])
            return np.asarray(kpts), np.asarray(desc)

        kpts_a, desc_a = _run_superpoint(ta)
        kpts_b, desc_b = _run_superpoint(tb)

        sess = self._lightglue
        feed = {}
        for inp in sess.get_inputs():
            n = inp.name
            if "kpts0" in n or "keypoints0" in n:
                feed[n] = kpts_a.astype(np.float32)
            elif "kpts1" in n or "keypoints1" in n:
                feed[n] = kpts_b.astype(np.float32)
            elif "desc0" in n or "descriptors0" in n:
                feed[n] = desc_a.astype(np.float32)
            elif "desc1" in n or "descriptors1" in n:
                feed[n] = desc_b.astype(np.float32)
        outs = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()]
        by = dict(zip(names, outs, strict=False))
        matches = np.asarray(by.get("matches", by.get("matches0", outs[0])))
        mscores = by.get("scores", by.get("mscores0"))

        kpts_a2 = kpts_a.reshape(-1, 2)
        kpts_b2 = kpts_b.reshape(-1, 2)
        matches = matches.reshape(-1, 2)
        pts_a = kpts_a2[matches[:, 0].astype(int)]
        pts_b = kpts_b2[matches[:, 1].astype(int)]
        if mscores is None:
            scores = np.ones(pts_a.shape[0], dtype=np.float64)
        else:
            scores = np.asarray(mscores, dtype=np.float64).reshape(-1)
        keep = scores >= self.match_threshold
        return (
            pts_a[keep].astype(np.float64),
            pts_b[keep].astype(np.float64),
            scores[keep],
        )


# ---------------------------------------------------------------------------
# Scale references
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScaleReference:
    """One measured floor distance between two clicked correspondences.

    ``p1_a`` / ``p1_b`` are the first landmark's pixel coordinates in view A / B;
    ``p2_a`` / ``p2_b`` the second landmark's. ``distance_m`` is the tape-measured
    distance between the two 3D landmarks, in metres.
    """

    p1_a: tuple[float, float]
    p1_b: tuple[float, float]
    p2_a: tuple[float, float]
    p2_b: tuple[float, float]
    distance_m: float


@dataclass(slots=True)
class ScaleEstimate:
    """Result of metric-scale recovery across references."""

    scale: float                       # metres per unit of the unit-baseline reconstruction
    per_reference_scale: list[float]   # scale each reference alone would imply
    per_reference_residual: list[float]  # |implied - median| / median, per reference
    outliers: list[int]                # indices flagged as inconsistent
    n_references: int


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureExtrinsicsResult:
    """Everything Stage 1 produces, for the UI + downstream assembly."""

    solution: MultiCalSolution
    R: np.ndarray                 # cam_b ← cam_a (recoverPose convention), refined
    t: np.ndarray                 # metric, cam_b ← cam_a
    scale: ScaleEstimate
    reprojection_rms_px: float
    n_matches: int
    n_inliers: int
    inlier_mask: np.ndarray       # (N,) bool over the input matches
    points_3d: np.ndarray         # (M, 3) triangulated inliers, cam_a frame, metric


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _undistort_to_normalized(
    pts: np.ndarray, K: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """Undistort pixels to normalized image coordinates (P=None)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, K, D)
    return out.reshape(-1, 2)


def recover_relative_pose(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    K_a: np.ndarray,
    D_a: np.ndarray,
    K_b: np.ndarray,
    D_b: np.ndarray,
    *,
    ransac_threshold_px: float = 1.0,
    ransac_prob: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Essential matrix + relative pose from correspondences.

    Undistorts to normalized coordinates, runs ``findEssentialMat`` with RANSAC
    (identity intrinsics, threshold expressed in the normalized frame), then
    ``recoverPose``. Returns ``(R, t_dir, inlier_mask)`` where ``R`` / ``t_dir``
    are cam_b ← cam_a with ``t_dir`` a **unit** direction.
    """
    na = _undistort_to_normalized(pts_a, K_a, D_a)
    nb = _undistort_to_normalized(pts_b, K_b, D_b)

    focal = 0.5 * (float(K_a[0, 0]) + float(K_a[1, 1]))
    thresh_norm = ransac_threshold_px / focal
    E, mask = cv2.findEssentialMat(
        na, nb, np.eye(3),
        method=cv2.RANSAC, prob=ransac_prob, threshold=thresh_norm,
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed to estimate an essential matrix")
    # recoverPose can pick among the 3x3 stacked E hypotheses; keep the first.
    if E.shape[0] > 3:
        E = E[:3]
    mask = mask.reshape(-1).astype(bool)
    _n, R, t, mask_pose = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask.astype(np.uint8))
    inliers = mask_pose.reshape(-1).astype(bool)
    return R, t.reshape(3), inliers


def _triangulate_normalized(
    na: np.ndarray, nb: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Triangulate normalized correspondences → 3D points in cam_a frame.

    cam_a is the identity ([I|0]); cam_b is [R|t]. Points returned metric only
    if ``t`` is metric; with a unit ``t`` they are in unit-baseline units.
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(P1, P2, na.T, nb.T)
    pts3 = (pts4[:3] / pts4[3]).T
    return pts3


def estimate_metric_scale(
    references: list[ScaleReference],
    K_a: np.ndarray,
    D_a: np.ndarray,
    K_b: np.ndarray,
    D_b: np.ndarray,
    R: np.ndarray,
    t_dir: np.ndarray,
    *,
    outlier_rel_threshold: float = 0.10,
) -> ScaleEstimate:
    """Recover the metric scale from ≥1 measured reference distances.

    Each reference is triangulated in the unit-baseline reconstruction; its
    reconstructed length is compared to the measured metres to imply a scale.
    The final scale is the **median** across references (robust to a mis-clicked
    or mis-measured one); each reference's residual against the median is
    reported and references beyond ``outlier_rel_threshold`` are flagged.

    Requires ≥1 reference; warns (does not fail) with <3 — a single reference is
    sensitive to click + triangulation noise, as the plan notes.
    """
    if not references:
        raise ValueError("metric scale needs at least one reference distance")
    if len(references) < 3:
        import warnings

        warnings.warn(
            f"metric scale from only {len(references)} reference(s); ≥3 are "
            "recommended for cross-validation (a single one is noise-sensitive)",
            stacklevel=2,
        )

    per_ref: list[float] = []
    for ref in references:
        a = np.array([ref.p1_a, ref.p2_a], dtype=np.float64)
        b = np.array([ref.p1_b, ref.p2_b], dtype=np.float64)
        na = _undistort_to_normalized(a, K_a, D_a)
        nb = _undistort_to_normalized(b, K_b, D_b)
        pts = _triangulate_normalized(na, nb, R, t_dir)
        recon_len = float(np.linalg.norm(pts[0] - pts[1]))
        if recon_len <= 1e-12:
            per_ref.append(np.nan)
            continue
        per_ref.append(ref.distance_m / recon_len)

    per_ref_arr = np.asarray(per_ref, dtype=np.float64)
    valid = per_ref_arr[np.isfinite(per_ref_arr)]
    if valid.size == 0:
        raise RuntimeError("no reference produced a usable triangulated length")
    median_scale = float(np.median(valid))

    residuals: list[float] = []
    outliers: list[int] = []
    for i, s in enumerate(per_ref):
        if not np.isfinite(s):
            residuals.append(float("inf"))
            outliers.append(i)
            continue
        rel = abs(s - median_scale) / median_scale
        residuals.append(rel)
        if rel > outlier_rel_threshold:
            outliers.append(i)

    # Final scale from the inlier references only (fall back to all if all flagged).
    inlier_scales = [per_ref[i] for i in range(len(per_ref))
                     if i not in outliers and np.isfinite(per_ref[i])]
    scale = float(np.median(inlier_scales)) if inlier_scales else median_scale

    return ScaleEstimate(
        scale=scale,
        per_reference_scale=per_ref,
        per_reference_residual=residuals,
        outliers=outliers,
        n_references=len(references),
    )


# ---------------------------------------------------------------------------
# Bundle adjustment (jointly refine R, t, scale, and 3D points)
# ---------------------------------------------------------------------------


def _project_normalized(pts3: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D points (cam_a frame) into cam_b normalized coords via [R|t]."""
    cam = pts3 @ R.T + t.reshape(1, 3)
    return cam[:, :2] / cam[:, 2:3]


def _bundle_adjust(
    na: np.ndarray,
    nb: np.ndarray,
    R0: np.ndarray,
    t0_dir: np.ndarray,
    scale0: float,
    focal_px: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Joint BA over rotation vector, unit-translation direction, scale, points.

    Minimizes reprojection residuals (in normalized coords) of the inlier
    correspondences across both views. cam_a is fixed at [I|0]; cam_b is
    ``[R | scale · t_dir]``. Returns ``(R, t_metric, scale, points_3d, rms_px)``.
    """
    from scipy.optimize import least_squares

    rvec0, _ = cv2.Rodrigues(R0)
    rvec0 = rvec0.reshape(3)
    tdir0 = t0_dir / (np.linalg.norm(t0_dir) + 1e-12)

    # Initial 3D points from the scaled reconstruction (metric).
    pts0 = _triangulate_normalized(na, nb, R0, scale0 * tdir0)

    # Params: [rvec(3), tdir(3), scale(1), points(3n)].
    x0 = np.concatenate([rvec0, tdir0, [scale0], pts0.reshape(-1)])

    def unpack(x: np.ndarray):
        rvec = x[0:3]
        tdir = x[3:6]
        tdir = tdir / (np.linalg.norm(tdir) + 1e-12)
        s = x[6]
        pts = x[7:].reshape(-1, 3)
        R, _ = cv2.Rodrigues(rvec)
        t = s * tdir
        return R, t, s, pts

    def residuals(x: np.ndarray) -> np.ndarray:
        R, t, _s, pts = unpack(x)
        proj_a = pts[:, :2] / pts[:, 2:3]        # cam_a = [I|0]
        proj_b = _project_normalized(pts, R, t)
        res_a = (proj_a - na).reshape(-1)
        res_b = (proj_b - nb).reshape(-1)
        return np.concatenate([res_a, res_b])

    sol = least_squares(residuals, x0, method="lm", max_nfev=200)
    R, t, s, pts = unpack(sol.x)

    # RMS in pixels: normalized residual · focal length.
    res = residuals(sol.x).reshape(-1, 2)
    rms_norm = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))
    rms_px = rms_norm * focal_px
    return R, t, float(s), pts, rms_px


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def solve_feature_extrinsics(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    matcher: StereoMatcher,
    K_a: np.ndarray,
    D_a: np.ndarray,
    K_b: np.ndarray,
    D_b: np.ndarray,
    references: list[ScaleReference],
    image_size_wh: tuple[int, int],
    *,
    cam_a_id: str = "cam_a",
    cam_b_id: str = "cam_b",
    ransac_threshold_px: float = 1.0,
) -> FeatureExtrinsicsResult:
    """End-to-end targetless stereo extrinsic solve → ``MultiCalSolution``.

    For Stage 1 the matches from all ``image_pairs`` are concatenated and solved
    as one Essential/pose problem (a single rigid stereo baseline). ``references``
    supply the metric scale. Emits ``cam_a`` as the rig-frame identity master and
    ``cam_b`` with the refined pose.
    """
    if not image_pairs:
        raise ValueError("need at least one stereo image pair")

    K_a = np.asarray(K_a, dtype=np.float64)
    D_a = np.asarray(D_a, dtype=np.float64).reshape(-1)
    K_b = np.asarray(K_b, dtype=np.float64)
    D_b = np.asarray(D_b, dtype=np.float64).reshape(-1)

    all_a: list[np.ndarray] = []
    all_b: list[np.ndarray] = []
    for img_a, img_b in image_pairs:
        pa, pb, _scores = matcher.match(img_a, img_b)
        all_a.append(np.asarray(pa, dtype=np.float64).reshape(-1, 2))
        all_b.append(np.asarray(pb, dtype=np.float64).reshape(-1, 2))
    pts_a = np.concatenate(all_a, axis=0)
    pts_b = np.concatenate(all_b, axis=0)
    n_matches = pts_a.shape[0]
    if n_matches < 5:
        raise RuntimeError(
            f"only {n_matches} correspondences; need ≥5 for the essential matrix"
        )

    R_rel, t_dir, inliers = recover_relative_pose(
        pts_a, pts_b, K_a, D_a, K_b, D_b,
        ransac_threshold_px=ransac_threshold_px,
    )
    n_inliers = int(inliers.sum())

    scale = estimate_metric_scale(references, K_a, D_a, K_b, D_b, R_rel, t_dir)

    # Refine on the inlier correspondences only.
    na = _undistort_to_normalized(pts_a[inliers], K_a, D_a)
    nb = _undistort_to_normalized(pts_b[inliers], K_b, D_b)
    focal_px = 0.5 * (float(K_a[0, 0]) + float(K_a[1, 1]))
    R_ref, t_ref, scale_ref, pts3, rms_px = _bundle_adjust(
        na, nb, R_rel, t_dir, scale.scale, focal_px,
    )

    # recoverPose (R, t) is cam_b ← cam_a. CameraInRig wants rig(=cam_a) ← cam_b.
    R_in_rig_b = R_ref.T
    t_in_rig_b = -R_ref.T @ t_ref.reshape(3)

    cam_a = CameraInRig(
        camera_id=cam_a_id,
        image_size_wh=(int(image_size_wh[0]), int(image_size_wh[1])),
        K=K_a,
        D=D_a,
        R_in_rig=np.eye(3),
        t_in_rig=np.zeros(3),
        rms_px=rms_px,
    )
    cam_b = CameraInRig(
        camera_id=cam_b_id,
        image_size_wh=(int(image_size_wh[0]), int(image_size_wh[1])),
        K=K_b,
        D=D_b,
        R_in_rig=R_in_rig_b,
        t_in_rig=t_in_rig_b,
        rms_px=rms_px,
    )
    solution = MultiCalSolution(
        master_camera=cam_a_id,
        cameras={cam_a_id: cam_a, cam_b_id: cam_b},
    )

    return FeatureExtrinsicsResult(
        solution=solution,
        R=R_ref,
        t=t_ref,
        scale=ScaleEstimate(
            scale=scale_ref,
            per_reference_scale=scale.per_reference_scale,
            per_reference_residual=scale.per_reference_residual,
            outliers=scale.outliers,
            n_references=scale.n_references,
        ),
        reprojection_rms_px=rms_px,
        n_matches=n_matches,
        n_inliers=n_inliers,
        inlier_mask=inliers,
        points_3d=pts3,
    )
