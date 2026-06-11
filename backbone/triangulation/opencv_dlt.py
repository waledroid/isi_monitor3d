"""``OpencvDltTriangulator`` — closed-form 2-camera triangulation via OpenCV.

For 2-camera nodes (the v1 production rig), ``cv2.triangulatePoints`` is the
canonical linear DLT triangulator. The plan stubs the N-camera (≥3) path via
``aniposelib`` for a future S5.5+ deployment; we explicitly raise so that an
accidentally-3-camera scene doesn't silently degrade.

Identity flow:

    The triangulator receives **already-associated** observations
    (``{cam_id: (u, v)}``) from the orchestrator — it does not solve the
    correspondence problem. The architecture's "one identity space" is
    preserved by KeypointAssociator + the 2D tracker; this stage is pure
    geometry.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from backbone.core.interfaces import Triangulator, triangulator_registry
from backbone.shared.camera_rig import CameraRig


@triangulator_registry.register("opencv_dlt")
class OpencvDltTriangulator(Triangulator):
    """2-camera linear DLT via ``cv2.triangulatePoints``."""

    def __init__(self, rig: CameraRig) -> None:
        self._rig = rig

    def triangulate_point(
        self,
        observations: Mapping[str, tuple[float, float]] | dict[str, tuple[float, float]],
    ) -> np.ndarray | None:
        """Recover XYZ in meters from per-camera pixel observations.

        Returns ``None`` when fewer than 2 cameras are present (degenerate
        geometry — caller should skip publication for this frame).
        """
        cam_ids = sorted(observations.keys())
        if len(cam_ids) < 2:
            return None
        if len(cam_ids) > 2:
            raise NotImplementedError(
                "opencv_dlt: 3+ cameras not supported in v1. "
                "Switch to an aniposelib-backed Triangulator (S5.5+)."
            )

        cam_a, cam_b = cam_ids
        P_a = self._rig[cam_a].P
        P_b = self._rig[cam_b].P

        # cv2.triangulatePoints expects (2, N) arrays per camera.
        pts_a = np.asarray(observations[cam_a], dtype=np.float64).reshape(2, 1)
        pts_b = np.asarray(observations[cam_b], dtype=np.float64).reshape(2, 1)

        homog = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)   # (4, 1)
        if homog.shape != (4, 1):
            return None
        w = float(homog[3, 0])
        if not np.isfinite(w) or abs(w) < 1e-9:
            return None
        return np.array([
            float(homog[0, 0]) / w,
            float(homog[1, 0]) / w,
            float(homog[2, 0]) / w,
        ])
