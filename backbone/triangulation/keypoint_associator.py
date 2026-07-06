"""``KeypointAssociator`` — resolve a ``Track2D`` to its per-camera 2D pixels.

Even though the name talks about keypoints, **v1 only resolves the foot point**
(centroid of the bbox's bottom edge). The keypoint case is a pose-mode
extension scoped to S5.5.

Why an "associator" at all? The S4 tracker emits ``Track2D`` objects with a
``cameras_seeing`` field, but it doesn't carry the per-camera pixel
coordinates that drove the fusion. Triangulation needs those pixels. Rather
than threading rich observations through the tracker ABC, we re-resolve from
the same-frame detections by nearest floor-projection match. That keeps the
``Tracker`` ABC minimal and the lookup is O(N·M) for tiny N (cameras) and M
(detections per camera) — irrelevant cost.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from backbone.core.types import Detection, Track2D
from backbone.homography.foot_projector import FootProjector
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import undistort_points

logger = logging.getLogger(__name__)


class KeypointAssociator:
    """Map a ``Track2D`` back to per-camera foot pixels for triangulation."""

    def __init__(
        self,
        rig: CameraRig,
        foot_projector: FootProjector,
        *,
        max_match_distance_m: float = 1.0,
    ) -> None:
        self._rig = rig
        self._projector = foot_projector
        self._max_distance_m = float(max_match_distance_m)

    def resolve_foot_uv(
        self,
        track: Track2D,
        detections_by_camera: Mapping[str, list[Detection]],
    ) -> dict[str, tuple[float, float]]:
        """Return ``{cam_id: (u, v)}`` for the foot pixel matching this track.

        Iterates over the track's ``cameras_seeing``. For each camera, picks
        the Detection of the same ``cls`` whose floor projection is closest
        to the track's ``xy_m`` (and within ``max_match_distance_m``). If no
        candidate qualifies, that camera is simply omitted from the result —
        the caller decides whether two cameras remain (triangulation requires
        ≥2 contributors).

        The returned pixels are **undistorted** (lens correction applied with
        the camera's ``K, D``): the downstream DLT and reprojection gate
        consume them against the pinhole ``P = K[R|t]``, which knows nothing
        about distortion. Feeding raw pixels would bias the 3D solve by
        several centimeters on a real lens (the same reason
        ``FootProjector`` undistorts before ``H``).
        """
        result: dict[str, tuple[float, float]] = {}
        track_xy = np.asarray(track.xy_m, dtype=np.float64)
        for cam_id in track.cameras_seeing:
            candidates = detections_by_camera.get(cam_id, [])
            best: Detection | None = None
            best_dist = self._max_distance_m
            for det in candidates:
                if det.cls != track.cls:
                    continue
                floor_xy = np.asarray(self._projector.project(det), dtype=np.float64)
                dist = float(np.linalg.norm(floor_xy - track_xy))
                if dist <= best_dist:
                    best = det
                    best_dist = dist
            if best is not None:
                cam = self._rig[cam_id]
                uv = undistort_points(
                    np.asarray(best.foot_uv, dtype=np.float64), cam.K, cam.D
                )[0]
                result[cam_id] = (float(uv[0]), float(uv[1]))
            else:
                logger.debug(
                    "associator: track %d on %s — no detection within %.2f m",
                    track.track_id, cam_id, self._max_distance_m,
                )
        return result
