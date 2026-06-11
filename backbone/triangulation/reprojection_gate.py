"""``ReprojectionGate`` — the load-bearing safety check on triangulated 3D.

After triangulation produces an ``(X, Y, Z)``, the gate re-projects it into
**every** contributing camera and rejects the lift if the max per-view error
exceeds a threshold (default 5 px, plan-allowed range 5-8 px). This is the
"fail honestly" principle made non-negotiable: without this gate, a bad
correspondence silently produces a bad 3D point that downstream consumers
would trust blindly.

Reuses ``backbone.shared.geometry.reprojection_error_px`` — the same helper
proven correct by S1's geometry tests.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import reprojection_error_px


class ReprojectionGate:
    """Reject triangulations whose per-view reprojection error exceeds threshold."""

    def __init__(self, rig: CameraRig, *, max_error_px: float = 5.0) -> None:
        if max_error_px <= 0.0:
            raise ValueError(f"max_error_px must be positive, got {max_error_px}")
        self._rig = rig
        self._max_error_px = float(max_error_px)
        self._rejected_count = 0
        self._last_max_error_px: float = 0.0

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    @property
    def max_error_px(self) -> float:
        return self._max_error_px

    @property
    def last_max_error_px(self) -> float:
        return self._last_max_error_px

    def check(
        self,
        xyz_m: np.ndarray,
        observations_uv: Mapping[str, tuple[float, float]] | dict[str, tuple[float, float]],
    ) -> bool:
        """Return ``True`` when the 3D point reprojects within tolerance.

        Sets ``self.last_max_error_px`` so the caller can attach the measured
        error to the emitted ``Track3D`` (the field exists in
        ``backbone.core.types.Track3D``).
        """
        P_by_camera = {
            cam_id: self._rig[cam_id].P for cam_id in observations_uv
        }
        # geometry.reprojection_error_px wants per-cam observations as np arrays.
        obs_np = {
            cam_id: np.asarray(uv, dtype=np.float64)
            for cam_id, uv in observations_uv.items()
        }
        errors = reprojection_error_px(np.asarray(xyz_m, dtype=np.float64), obs_np, P_by_camera)
        max_err = max(errors.values()) if errors else float("inf")
        self._last_max_error_px = float(max_err)
        if max_err > self._max_error_px:
            self._rejected_count += 1
            return False
        return True
