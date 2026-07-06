"""Post-calibration floor-alignment refinement (operator fine-tune).

After the board solve, a small rigid mismatch can remain between the two
cameras' floor frames (residual extrinsic rotation, nudged mounts). The
operator clicks N >= 3 corresponding FLOOR points in both cameras; mapping
each pair through the current calibration yields two world positions that
should coincide. This module fits the rigid 2D transform (rotation about Z +
XY translation — deliberately NO scale, that is anchored by the measured
board) aligning the target camera's floor frame onto the reference camera's,
and bakes it into the target camera's extrinsics so ``R, t, H, P`` all move
together — the "one calibration, two queries" rule stays intact and the
correction improves display, fusion and triangulation alike.

The refined calibration is a DERIVED artifact: callers write it to a separate
file and repoint consumers; the base solve is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import CalibrationFile, CameraCalibration


@dataclass(frozen=True)
class FloorAlignment:
    """A fitted rigid floor correction: ``X_ref ≈ Rz(theta) @ X_target + t``."""

    theta_rad: float
    tx_m: float
    ty_m: float
    residuals_m: tuple[float, ...]   # per-pair error AFTER the fit

    @property
    def max_residual_m(self) -> float:
        return max(self.residuals_m) if self.residuals_m else 0.0

    def as_dict(self) -> dict:
        return {
            "theta_deg": float(np.degrees(self.theta_rad)),
            "tx_m": self.tx_m,
            "ty_m": self.ty_m,
            "residuals_m": [round(float(r), 4) for r in self.residuals_m],
            "max_residual_m": round(self.max_residual_m, 4),
        }


def fit_rigid_floor_alignment(
    target_xy: np.ndarray,
    ref_xy: np.ndarray,
) -> FloorAlignment:
    """Least-squares rigid 2D fit (Kabsch, no scale): ``ref ≈ Rz @ target + t``.

    ``target_xy`` are the floor points as seen through the TARGET camera (the
    one being corrected), ``ref_xy`` the same physical points through the
    reference camera. Needs >= 2 pairs; use >= 3 (4 recommended) so the
    residuals are meaningful.
    """
    A = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    B = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    if A.shape != B.shape or len(A) < 2:
        raise ValueError(f"need >= 2 matched pairs, got {len(A)} vs {len(B)}")

    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:            # guard against a reflection solution
        Vt2 = Vt.copy()
        Vt2[-1] *= -1
        R = Vt2.T @ U.T
    t = cb - R @ ca
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    residuals = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return FloorAlignment(
        theta_rad=theta, tx_m=float(t[0]), ty_m=float(t[1]),
        residuals_m=tuple(float(r) for r in residuals),
    )


def apply_floor_alignment(
    base: CalibrationFile,
    target_camera: str,
    alignment: FloorAlignment,
    *,
    note: str | None = None,
) -> CalibrationFile:
    """Bake a floor alignment into ``target_camera``'s extrinsics.

    The 2D correction lifts exactly to a 3D world-frame transform (rotation
    about Z + XY translation). The stored per-camera ``(R, t)`` is the camera
    POSE (world <- camera), so the corrected pose is ``R' = T·R, t' = T·t + d``
    — and ``H``/``P`` are RE-DERIVED from the corrected pose so both geometric
    queries stay consistent. Returns a NEW ``CalibrationFile``; the base is
    untouched.
    """
    from backbone.shared.geometry import (
        floor_homography_from_K_R_t,
        projection_from_K_R_t,
    )

    if target_camera not in base.cameras:
        raise ValueError(f"camera {target_camera!r} not in calibration "
                         f"(have {sorted(base.cameras)})")

    c, s = float(np.cos(alignment.theta_rad)), float(np.sin(alignment.theta_rad))
    T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    d = np.array([alignment.tx_m, alignment.ty_m, 0.0])

    cam = base.cameras[target_camera]
    K = cam.K_np()
    R_new = T @ cam.R_np()
    t_new = T @ cam.t_np().reshape(3) + d

    refined_cam = CameraCalibration(
        camera_id=cam.camera_id,
        image_size_wh=tuple(cam.image_size_wh),
        K=cam.K,
        D=cam.D,
        R=R_new.tolist(),
        t=t_new.tolist(),
        H=floor_homography_from_K_R_t(K, R_new, t_new).tolist(),
        P=projection_from_K_R_t(K, R_new, t_new).tolist(),
        reprojection_rms_px=cam.reprojection_rms_px,
    )
    cameras = dict(base.cameras)
    cameras[target_camera] = refined_cam
    return CalibrationFile(
        version=base.version,
        created_at=base.created_at,
        floor_anchor_method=base.floor_anchor_method,
        floor_origin_note=(note or f"{base.floor_origin_note} "
                           f"[+ operator floor alignment on {target_camera}]"),
        cameras=cameras,
        calibration_mode=base.calibration_mode,
    )
