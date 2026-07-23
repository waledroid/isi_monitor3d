"""Per-camera, per-person, per-keypoint One Euro smoothing of pose skeletons.

The wire's person observations are anonymous (no track id crosses process
boundaries), so the smoother owns a tiny frame-to-frame association per
camera: greedy nearest-neighbor on the RAW foot point, gated so a big jump
starts a fresh slot instead of dragging another person's filter state.

Hard constraint: only ``keypoints_uv`` is filtered — ``foot_uv`` feeds the
metric engine (undistort → homography → fusion → ByteTrack, which has its own
Kalman) and must stay raw, as must every non-person detection. Timestamps are
the frames' ``capture_ts``; the motion gate's cached re-emissions never reach
this module (it runs only on real pose inferences), so frozen periods inject
no phantom velocity.

One Euro filter: Casiez, Roussel & Vogel, CHI 2012.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from backbone.core.types import Detection

# A new foot position farther than this from every live slot is a new person,
# not a moved one (mirrors the dashboard overlay's snap gate).
_ASSOC_GATE_PX = 120.0


class OneEuro:
    """Scalar One Euro filter: jitter-free at rest, low-lag in motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


@dataclass
class _Slot:
    foot: tuple[float, float]
    last_t: float
    filters: list[tuple[OneEuro, OneEuro]] = field(default_factory=list)


class PoseSmoother:
    """Smooths person keypoints in-place per camera; everything else raw."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01,
                 stale_s: float = 2.0) -> None:
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._stale_s = float(stale_s)
        self._slots: dict[str, list[_Slot]] = {}

    def _slot_for(self, cam_slots: list[_Slot], foot: tuple[float, float],
                  used: set[int]) -> _Slot | None:
        best, best_d = None, _ASSOC_GATE_PX
        for i, s in enumerate(cam_slots):
            if i in used:
                continue
            d = math.hypot(foot[0] - s.foot[0], foot[1] - s.foot[1])
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return None
        used.add(best)
        return cam_slots[best]

    def smooth(self, cam_id: str, dets: list[Detection],
               t: float) -> list[Detection]:
        cam_slots = self._slots.setdefault(cam_id, [])
        cam_slots[:] = [s for s in cam_slots if t - s.last_t <= self._stale_s]

        used: set[int] = set()
        out: list[Detection] = []
        # highest-confidence persons claim their slots first
        order = sorted(
            range(len(dets)),
            key=lambda i: -float(dets[i].confidence)
            if dets[i].cls == "person" else 1.0,
        )
        smoothed: dict[int, Detection] = {}
        for i in order:
            d = dets[i]
            if d.cls != "person" or d.keypoints_uv is None:
                continue
            kps = np.asarray(d.keypoints_uv, dtype=np.float32).reshape(-1, 3)
            foot = (float(d.foot_uv[0]), float(d.foot_uv[1]))
            slot = self._slot_for(cam_slots, foot, used)
            if slot is None or len(slot.filters) != len(kps):
                slot = _Slot(foot=foot, last_t=t, filters=[
                    (OneEuro(self._min_cutoff, self._beta),
                     OneEuro(self._min_cutoff, self._beta))
                    for _ in range(len(kps))
                ])
                cam_slots.append(slot)
                used.add(len(cam_slots) - 1)
            slot.foot, slot.last_t = foot, t
            sm = kps.copy()
            for k, (fx, fy) in enumerate(slot.filters):
                sm[k, 0] = fx.filter(float(kps[k, 0]), t)
                sm[k, 1] = fy.filter(float(kps[k, 1]), t)
            smoothed[i] = replace(d, keypoints_uv=sm)
        for i, d in enumerate(dets):
            out.append(smoothed.get(i, d))
        return out
