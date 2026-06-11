"""``FootProjector`` — Detection → metric floor coordinates.

Per Detection: undistort the foot pixel with the camera's K/D, then apply the
homography H to land on the shared floor frame in meters. Concrete,
single-implementation — there is one sensible way to do this, and an ABC here
would be ceremony.

The foot point is the bottom-centre of the detection's bbox (set by the
detector in S3: ``Detection.foot_uv``). For a person standing upright on the
floor, this is the contact point — which is the ONLY pixel whose homography
projection is geometrically valid (the rest of the bbox is in 3D above the
plane and would produce a biased X/Y on the floor plane).
"""

from __future__ import annotations

import numpy as np

from backbone.core.types import Detection
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import pixel_to_floor, undistort_points


class FootProjector:
    """Project a Detection's foot pixel to floor (X, Y) meters.

    Args:
        rig: loaded calibration with per-camera K, D, H.
    """

    def __init__(self, rig: CameraRig) -> None:
        self._rig = rig

    def project(self, det: Detection) -> tuple[float, float]:
        """Return the detection's floor position in meters.

        Raises:
            KeyError: if the detection's camera_id isn't in the calibration.
        """
        cam = self._rig[det.camera_id]
        foot_pixel = np.asarray(det.foot_uv, dtype=np.float64).reshape(1, 2)
        undistorted = undistort_points(foot_pixel, cam.K, cam.D)
        xy = pixel_to_floor(undistorted, cam.H).reshape(2)
        return float(xy[0]), float(xy[1])

    def project_batch(
        self,
        detections: list[Detection],
    ) -> list[tuple[Detection, tuple[float, float]]]:
        """Convenience: project a whole list, preserving original Detection objects.

        The output pairs each Detection with its floor (X, Y). Useful as the
        feed into ``CrossCamFusion.fuse``.
        """
        return [(d, self.project(d)) for d in detections]
