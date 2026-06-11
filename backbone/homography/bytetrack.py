"""``ByteTrackMeters`` — ByteTrack adapted to operate in metric (meters) space.

Algorithm (per ``Tracker.update`` call):

1.  Compute ``dt`` from the previous capture_ts; predict all active tracks
    forward by that dt.
2.  Split incoming observations into high-confidence (``>= conf_high``) and
    low-confidence (``conf_low <= c < conf_high``). Observations below
    ``conf_low`` are dropped entirely.
3.  First-pass match: TRACKED tracks ↔ high-conf observations via Hungarian
    on Mahalanobis distance (using each track's Kalman covariance). Pairs
    above ``match_distance_m`` are not allowed.
4.  Second-pass match: still-unmatched TRACKED + LOST tracks ↔ low-conf
    observations via Hungarian (same metric).
5.  Unmatched high-conf observations spawn NEW tracks.
6.  Unmatched tracks call ``mark_missed`` (and may transition LOST/REMOVED).
7.  REMOVED tracks are dropped from the internal list.
8.  Emit a ``Track2D`` for every publishable (TRACKED) track.

Identity stability is the whole point — track_ids are assigned once on
creation and never reassigned. The single identity space the architecture
talks about is owned here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

from backbone.core.interfaces import Tracker, tracker_registry
from backbone.core.types import Track2D

from .track import InternalTrack, TrackConfig, TrackState

logger = logging.getLogger(__name__)

DEFAULT_MATCH_DISTANCE_M: dict[str, float] = {
    "person": 1.0,
    "forklift": 2.0,
    "pallet": 1.0,
}
"""Gating distance used inside ByteTrack matching — wider than the fusion
gate, since the Kalman prediction may not be exact under fast motion."""


@tracker_registry.register("bytetrack")
class ByteTrackMeters(Tracker):
    """Two-stage Hungarian + Kalman tracking in metric space."""

    def __init__(
        self,
        *,
        conf_high: float = 0.5,
        conf_low: float = 0.1,
        match_distance_m: dict[str, float] | None = None,
        default_match_distance_m: float = 1.0,
        track_config: TrackConfig | None = None,
    ) -> None:
        if conf_low > conf_high:
            raise ValueError("conf_low must be ≤ conf_high")
        self._conf_high = float(conf_high)
        self._conf_low = float(conf_low)
        self._match_distances = dict(DEFAULT_MATCH_DISTANCE_M)
        if match_distance_m:
            self._match_distances.update(match_distance_m)
        self._default_distance = float(default_match_distance_m)
        self._track_config = track_config or TrackConfig()

        self._tracks: list[InternalTrack] = []
        self._id_gen = count(1)
        self._last_ts: float | None = None

    @property
    def active_tracks(self) -> tuple[InternalTrack, ...]:
        return tuple(t for t in self._tracks if t.is_active)

    def update(
        self,
        capture_ts: float,
        observations: list[tuple[str, tuple[float, float], float, tuple[str, ...]]],
    ) -> list[Track2D]:
        """Standard Tracker ABC entry point."""
        dt = 0.0 if self._last_ts is None else max(0.0, capture_ts - self._last_ts)
        self._last_ts = capture_ts

        # Predict every active track forward.
        for t in self._tracks:
            if t.state in (TrackState.NEW, TrackState.TRACKED, TrackState.LOST):
                t.predict(dt)

        high, low = self._split_by_confidence(observations)

        unmatched_tracks = [t for t in self._tracks if t.is_active]
        unmatched_high = list(range(len(high)))
        unmatched_low = list(range(len(low)))

        # First pass: TRACKED + NEW + LOST tracks ↔ high-conf observations.
        pairs_first = self._hungarian_match(
            tracks=unmatched_tracks,
            observations=high,
        )
        matched_track_ids = set()
        for ti, oi in pairs_first:
            track = unmatched_tracks[ti]
            cls_label, xy, conf, cams = high[oi]
            track.update_with(
                xy_m=xy, cls_label=cls_label, confidence=conf,
                cameras_seeing=cams, capture_ts=capture_ts,
            )
            matched_track_ids.add(track.track_id)
            unmatched_high.remove(oi)

        # Second pass: still-unmatched tracks ↔ low-conf observations.
        leftover_tracks = [t for t in unmatched_tracks if t.track_id not in matched_track_ids]
        pairs_second = self._hungarian_match(
            tracks=leftover_tracks,
            observations=low,
        )
        for ti, oi in pairs_second:
            track = leftover_tracks[ti]
            cls_label, xy, conf, cams = low[oi]
            track.update_with(
                xy_m=xy, cls_label=cls_label, confidence=conf,
                cameras_seeing=cams, capture_ts=capture_ts,
            )
            matched_track_ids.add(track.track_id)
            unmatched_low.remove(oi)

        # Tracks that got no match this frame.
        for t in unmatched_tracks:
            if t.track_id not in matched_track_ids:
                t.mark_missed()

        # Spawn NEW tracks for unmatched high-conf observations.
        for oi in unmatched_high:
            cls_label, xy, conf, cams = high[oi]
            new_track = InternalTrack.create(
                track_id=next(self._id_gen),
                cls_label=cls_label,
                xy_m=xy,
                confidence=conf,
                cameras_seeing=cams,
                capture_ts=capture_ts,
                cfg=self._track_config,
            )
            self._tracks.append(new_track)
        # Low-conf observations that didn't match anything are dropped — they
        # are not strong enough to start a new track on their own.

        # Garbage-collect REMOVED tracks.
        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]

        # Emit Track2D for every publishable track. The temporal stabilizer
        # may further filter / re-label these before they hit the bus.
        return [
            self._to_track2d(t) for t in self._tracks
            if t.is_publishable
        ]

    # ----- Internals -----

    def _split_by_confidence(
        self,
        observations: Iterable[tuple[str, tuple[float, float], float, tuple[str, ...]]],
    ) -> tuple[
        list[tuple[str, tuple[float, float], float, tuple[str, ...]]],
        list[tuple[str, tuple[float, float], float, tuple[str, ...]]],
    ]:
        high: list = []
        low: list = []
        for obs in observations:
            conf = obs[2]
            if conf >= self._conf_high:
                high.append(obs)
            elif conf >= self._conf_low:
                low.append(obs)
            # else dropped
        return high, low

    def _hungarian_match(
        self,
        tracks: list[InternalTrack],
        observations: list[tuple[str, tuple[float, float], float, tuple[str, ...]]],
    ) -> list[tuple[int, int]]:
        """Solve a (tracks x observations) Mahalanobis-distance assignment.

        Returns pairs ``(track_index, observation_index)``. Pairs beyond the
        per-class gating distance are pruned post-solve.
        """
        if not tracks or not observations:
            return []

        n_t, n_o = len(tracks), len(observations)
        cost = np.full((n_t, n_o), fill_value=np.inf, dtype=np.float64)

        for ti, track in enumerate(tracks):
            track_xy = np.asarray(track.xy(), dtype=np.float64)
            pos_cov = track.position_covariance()
            # Numerically-stable inverse with a tiny diagonal regulariser —
            # the Kalman covariance can be near-singular at NEW tracks.
            try:
                pos_cov_inv = np.linalg.inv(pos_cov + np.eye(2) * 1e-6)
            except np.linalg.LinAlgError:
                pos_cov_inv = np.eye(2)
            for oi, obs in enumerate(observations):
                obs_xy = np.asarray(obs[1], dtype=np.float64)
                diff = obs_xy - track_xy
                # Mahalanobis distance squared.
                mahal2 = float(diff @ pos_cov_inv @ diff)
                # We also enforce a hard Euclidean gate per class — that prevents
                # absurd matches when the Kalman covariance has gone wide.
                euc = float(np.hypot(diff[0], diff[1]))
                gate = self._match_distances.get(obs[0], self._default_distance)
                if euc > gate:
                    cost[ti, oi] = np.inf
                else:
                    cost[ti, oi] = mahal2

        # Replace inf with a large finite cost for the solver, then prune.
        big = 1e9
        cost_for_solver = np.where(np.isinf(cost), big, cost)
        if cost_for_solver.size == 0:
            return []
        rows, cols = linear_sum_assignment(cost_for_solver)
        pairs: list[tuple[int, int]] = []
        for ti, oi in zip(rows, cols, strict=True):
            if np.isfinite(cost[ti, oi]):
                pairs.append((int(ti), int(oi)))
        return pairs

    def _to_track2d(self, track: InternalTrack) -> Track2D:
        return Track2D(
            track_id=track.track_id,
            cls=track.cls,
            capture_ts=track.capture_ts,
            xy_m=track.xy(),
            vxy_m=track.vxy(),
            confidence=track.confidence,
            cameras_seeing=track.cameras_seeing,
        )
