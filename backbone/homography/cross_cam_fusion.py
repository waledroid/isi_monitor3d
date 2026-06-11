"""``CrossCamFusion`` — pair per-camera floor observations across cameras.

For each class and camera-pair, build a Euclidean cost matrix in metric
space, solve Hungarian assignment, accept pairs within ``match_distance_m``.
Matched pairs become multi-camera ``FusedObservation``s with an averaged
position; unmatched detections become single-camera ``FusedObservation``s
(``cameras_seeing`` of length 1).

Single-camera observations are first-class — sometimes a camera sees something
the other doesn't (occlusion). The tracker must be robust to single-cam input.

The disagreement gate (next stage) re-checks fused pairs with a tighter
``agreement_distance_m`` and may reject fusions even after they pass matching.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from backbone.core.types import Detection

DEFAULT_MATCH_DISTANCE_M: dict[str, float] = {
    "person": 0.8,
    "forklift": 1.6,
    "pallet": 0.8,
}


@dataclass(slots=True)
class FusedObservation:
    """One real-world object's observation this frame.

    ``per_camera_positions`` keeps each contributor's raw floor position for
    the disagreement gate (and diagnostics). ``xy_m`` is the averaged
    position used as the tracker measurement.
    """

    cls: str
    xy_m: tuple[float, float]
    confidence: float
    cameras_seeing: tuple[str, ...]
    capture_ts: float
    per_camera_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_camera_confidence: dict[str, float] = field(default_factory=dict)


class CrossCamFusion:
    """Hungarian assignment of per-camera observations into ``FusedObservation``s."""

    def __init__(
        self,
        match_distance_m: dict[str, float] | None = None,
        default_distance_m: float = 0.8,
    ) -> None:
        self._thresholds = dict(DEFAULT_MATCH_DISTANCE_M)
        if match_distance_m:
            self._thresholds.update(match_distance_m)
        self._default_distance = default_distance_m

    def _threshold_for(self, cls: str) -> float:
        return self._thresholds.get(cls, self._default_distance)

    def fuse(
        self,
        detections: Iterable[tuple[Detection, tuple[float, float]]],
    ) -> list[FusedObservation]:
        """Cluster (Detection, floor_xy) pairs across cameras → FusedObservations.

        Args:
            detections: iterable of ``(Detection, (X_m, Y_m))`` pairs, typically
                produced by ``FootProjector.project_batch``.

        Returns:
            One ``FusedObservation`` per real-world object detected this frame.
        """
        # Group by class so we never match across classes.
        by_class: dict[str, list[tuple[Detection, tuple[float, float]]]] = defaultdict(list)
        for det, xy in detections:
            by_class[det.cls].append((det, xy))

        fused: list[FusedObservation] = []
        for cls, items in by_class.items():
            fused.extend(self._fuse_one_class(cls, items))
        return fused

    def _fuse_one_class(
        self,
        cls: str,
        items: list[tuple[Detection, tuple[float, float]]],
    ) -> list[FusedObservation]:
        """Pair within one class across cameras."""
        if not items:
            return []

        threshold = self._threshold_for(cls)

        # Bucket by camera_id.
        by_cam: dict[str, list[tuple[Detection, tuple[float, float]]]] = defaultdict(list)
        for det, xy in items:
            by_cam[det.camera_id].append((det, xy))

        cameras = sorted(by_cam)
        if len(cameras) == 1:
            # No cross-camera fusion possible; every detection is single-cam.
            return [_single_cam_obs(cls, det, xy) for det, xy in items]

        # v1 production rigs are 2-cam. N≥3 needs aniposelib-style iterative
        # pairwise fusion — landing with the triangulation N-cam path (S5.5+).
        assert len(cameras) == 2, (
            f"CrossCamFusion: v1 supports exactly 2 cameras, got {sorted(cameras)}. "
            f"N≥3 cross-cam fusion lands with the aniposelib triangulation path."
        )

        cam_a, cam_b = cameras
        a_items = by_cam[cam_a]
        b_items = by_cam[cam_b]

        n_a, n_b = len(a_items), len(b_items)
        if n_a == 0 or n_b == 0:
            return [_single_cam_obs(cls, det, xy) for det, xy in (a_items + b_items)]

        # Cost matrix — Euclidean distance in meters.
        cost = np.zeros((n_a, n_b), dtype=np.float64)
        for i, (_, xy_a) in enumerate(a_items):
            for j, (_, xy_b) in enumerate(b_items):
                cost[i, j] = float(np.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1]))

        # Solve Hungarian — but mask out pairs beyond threshold so they don't
        # force a bad assignment. We give beyond-threshold pairs a large cost
        # so the solver never picks them when better options exist; afterwards
        # we re-check and discard any matched pair that exceeds threshold.
        large = float(cost.max() + threshold * 10.0 + 1.0)
        cost_masked = np.where(cost <= threshold, cost, large)

        rows, cols = linear_sum_assignment(cost_masked)

        matched_a: set[int] = set()
        matched_b: set[int] = set()
        results: list[FusedObservation] = []

        for i, j in zip(rows, cols, strict=True):
            if cost[i, j] > threshold:
                continue  # unmatched: the masking pushed this into the "no-match" zone
            det_a, xy_a = a_items[i]
            det_b, xy_b = b_items[j]
            matched_a.add(i)
            matched_b.add(j)
            results.append(_pair_fused(cls, det_a, xy_a, det_b, xy_b))

        # Anything left over is single-cam.
        for i, (det, xy) in enumerate(a_items):
            if i not in matched_a:
                results.append(_single_cam_obs(cls, det, xy))
        for j, (det, xy) in enumerate(b_items):
            if j not in matched_b:
                results.append(_single_cam_obs(cls, det, xy))

        return results


def _single_cam_obs(
    cls: str,
    det: Detection,
    xy: tuple[float, float],
) -> FusedObservation:
    return FusedObservation(
        cls=cls,
        xy_m=xy,
        confidence=det.confidence,
        cameras_seeing=(det.camera_id,),
        capture_ts=det.capture_ts,
        per_camera_positions={det.camera_id: xy},
        per_camera_confidence={det.camera_id: det.confidence},
    )


def _pair_fused(
    cls: str,
    det_a: Detection,
    xy_a: tuple[float, float],
    det_b: Detection,
    xy_b: tuple[float, float],
) -> FusedObservation:
    avg = ((xy_a[0] + xy_b[0]) / 2.0, (xy_a[1] + xy_b[1]) / 2.0)
    # Sort camera ids for a deterministic tuple — useful for downstream tests.
    cams = tuple(sorted([det_a.camera_id, det_b.camera_id]))
    return FusedObservation(
        cls=cls,
        xy_m=avg,
        confidence=max(det_a.confidence, det_b.confidence),
        cameras_seeing=cams,
        capture_ts=det_a.capture_ts,
        per_camera_positions={det_a.camera_id: xy_a, det_b.camera_id: xy_b},
        per_camera_confidence={
            det_a.camera_id: det_a.confidence,
            det_b.camera_id: det_b.confidence,
        },
    )
