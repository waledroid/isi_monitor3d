"""``DisagreementGate`` — second-pass quality check on multi-camera fusions.

Cross-camera fusion is permissive — it pairs detections within a wide
``match_distance_m`` to handle calibration drift and detector noise. The
disagreement gate is conservative: of the matched pairs, only those whose
per-camera positions agree within a tighter ``agreement_distance_m`` are kept
as multi-camera observations. Otherwise the fusion is "rejected" — the
higher-confidence camera's observation is kept as a single-cam observation,
and the lower-confidence camera's contribution is dropped.

This is the "fail honestly" principle for the 2D path: if two cameras
disagree about where an object is, don't average them into a number that's
wrong; trust the cleaner observation, and let the tracker decide if the
second camera's observation should spawn a new track.

Single-camera observations bypass the gate entirely.
"""

from __future__ import annotations

import numpy as np

from .cross_cam_fusion import FusedObservation

DEFAULT_AGREEMENT_DISTANCE_M: dict[str, float] = {
    "person": 0.4,
    "forklift": 0.8,
    "pallet": 0.4,
}


class DisagreementGate:
    """Post-fusion check that multi-cam observations actually agree."""

    def __init__(
        self,
        agreement_distance_m: dict[str, float] | None = None,
        default_distance_m: float = 0.4,
    ) -> None:
        self._thresholds = dict(DEFAULT_AGREEMENT_DISTANCE_M)
        if agreement_distance_m:
            self._thresholds.update(agreement_distance_m)
        self._default_distance = default_distance_m
        self._rejected_count = 0

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def _threshold_for(self, cls: str) -> float:
        return self._thresholds.get(cls, self._default_distance)

    def check(self, observations: list[FusedObservation]) -> list[FusedObservation]:
        """Return a list with rejected fusions replaced by their cleaner half."""
        out: list[FusedObservation] = []
        for obs in observations:
            if len(obs.cameras_seeing) < 2:
                out.append(obs)
                continue
            if self._fusion_agrees(obs):
                out.append(obs)
            else:
                self._rejected_count += 1
                out.append(_demote_to_cleaner(obs))
        return out

    def _fusion_agrees(self, obs: FusedObservation) -> bool:
        """Max pairwise position distance ≤ threshold ⇒ keep fused."""
        positions = list(obs.per_camera_positions.values())
        if len(positions) < 2:
            return True
        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = float(np.hypot(
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                ))
                if d > max_dist:
                    max_dist = d
        return max_dist <= self._threshold_for(obs.cls)


def _demote_to_cleaner(obs: FusedObservation) -> FusedObservation:
    """Pick the highest-confidence camera and return it as a single-cam observation."""
    cleanest_cam = max(obs.per_camera_confidence, key=obs.per_camera_confidence.get)
    xy = obs.per_camera_positions[cleanest_cam]
    return FusedObservation(
        cls=obs.cls,
        xy_m=xy,
        confidence=obs.per_camera_confidence[cleanest_cam],
        cameras_seeing=(cleanest_cam,),
        capture_ts=obs.capture_ts,
        per_camera_positions={cleanest_cam: xy},
        per_camera_confidence={cleanest_cam: obs.per_camera_confidence[cleanest_cam]},
    )
