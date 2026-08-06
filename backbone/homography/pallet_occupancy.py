"""Pallet occupancy — empty / carton / polybag — via a redundant two-estimator fusion.

For every detected pallet, decide whether it holds a load and what type, by
associating the carton/polybag detections (already produced by the 3-class seg
model, whole-frame) to pallets. **Two independent estimators** vote per
(object, pallet):

  A — image overlap : the object's base inside the pallet's "occupancy box"
      (its bbox extended upward). Strong always; **blind only when two pallets
      overlap in image depth**.
  B — metric margin : the object's foot vs the pallet's foot on the floor plane
      (homography). Strong always; **blind only at the rectangle edge** (the
      ~15 cm elevation drift band).

Their blind spots are complementary, so a small gate fuses them (agree → use;
disagree → defer to whichever is in the other's blind spot; one signal missing →
the other decides alone) — the same shape as ``DisagreementGate``. Both
estimators associate WITHIN one camera; a third, cross-camera pass in
``enrich`` (the metric occupancy fallback) pools load objects from every
camera against the FUSED track position, so a pallet only cam_a saw still
reads "full" from the carton only cam_b saw. The ``OccupancyStabilizer`` then
majority-votes the per-pallet state over each track's recent window so the
published empty/full state doesn't flicker.

This is the cahier-des-charges "pallet empty/full" KPI deliverable. Pure +
unit-testable (only the floor projection touches calibration).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from backbone.core.types import Detection, Track2D

PALLET_CLASSES = frozenset({"palette", "pallet", "palette_vide"})
OBJECT_CLASSES = frozenset({"carton", "polybag"})


def _load_area(d: Detection) -> float:
    """Mask area if present (precise), else bbox area — for the dominant-content pick."""
    if d.mask is not None:
        return float(d.mask.sum())
    x1, y1, x2, y2 = d.bbox_xyxy
    return max(0.0, (x2 - x1) * (y2 - y1))


@dataclass
class _Verdict:
    """One estimator's call for an object: which pallet (or None) + how sure."""
    pallet_idx: int | None
    score: float                 # A: overlap fraction [0,1]; B: confidence [0,1]
    ambiguous: bool = False      # A only: ≥2 pallets compete (can't separate)
    uncertain: bool = False      # B only: within the drift band (near an edge)
    available: bool = True        # False ⇒ this estimator had nothing to go on


class OccupancyStabilizer:
    """Vote the empty/full state (and content) per pallet ``track_id`` over a
    sliding window, with FLIP HYSTERESIS: the first observation establishes
    the state immediately, but once held it only flips when the challenger
    state wins at least ``flip_ratio`` of the window. A plain 5-frame
    majority flapped the published state every few seconds live (2026-08-06:
    a carton hovering at both estimators' thresholds alternated the
    per-frame verdict ~50/50); a real load/unload still flips within ~one
    window because sustained verdicts reach the supermajority fast."""

    def __init__(self, window: int = 15, flip_ratio: float = 0.7) -> None:
        self._hist: dict[int, deque] = {}
        self._window = int(window)
        self._flip_ratio = float(flip_ratio)
        self._held: dict[int, str] = {}

    def _voted(self, hist: deque, fallback_content: str | None, held: str | None):
        counts = Counter(s for s, _ in hist)
        if held is None:
            state = counts.most_common(1)[0][0]
        else:
            challenger = "empty" if held == "full" else "full"
            need = math.ceil(self._flip_ratio * len(hist))
            state = challenger if counts.get(challenger, 0) >= need else held
        if state != "full":
            return "empty", None
        contents = Counter(c for s, c in hist if s == "full" and c)
        return "full", (contents.most_common(1)[0][0] if contents else fallback_content)

    def vote(self, track_id: int, state: str, content: str | None):
        hist = self._hist.setdefault(track_id, deque(maxlen=self._window))
        hist.append((state, content))
        voted = self._voted(hist, content, self._held.get(track_id))
        self._held[track_id] = voted[0]
        return voted

    def last(self, track_id: int):
        """Current voted (state, content) without a new observation, or None."""
        hist = self._hist.get(track_id)
        return (self._voted(hist, None, self._held.get(track_id))
                if hist else None)


class PalletOccupancy:
    """Associate load objects to pallets (A+B fusion) and vote a stable state."""

    def __init__(
        self,
        projector,
        *,
        occupancy_box_k: float = 1.5,
        a_overlap_min: float = 0.2,
        metric_radius_m: float = 0.7,
        drift_band_m: float = 0.2,
        track_match_distance_m: float = 0.8,
        window: int = 15,
        flip_ratio: float = 0.7,
    ) -> None:
        self._projector = projector
        self._k = float(occupancy_box_k)
        self._a_min = float(a_overlap_min)
        self._radius = float(metric_radius_m)
        self._band = float(drift_band_m)
        self._track_match = float(track_match_distance_m)
        self._stabilizer = OccupancyStabilizer(window=window, flip_ratio=flip_ratio)

    # ---- A: image overlap ----

    def _a_overlap(self, obj: Detection, pal: Detection) -> float:
        ox1, _oy1, ox2, oy2 = obj.bbox_xyxy
        px1, py1, px2, py2 = pal.bbox_xyxy
        h = max(1e-6, py2 - py1)
        top = py1 - self._k * h                       # occupancy box: pallet bbox extended up
        if not (top <= oy2 <= py2 + 0.5 * h):         # object base must sit in the band over the pallet
            return 0.0
        overlap_x = max(0.0, min(ox2, px2) - max(ox1, px1))
        return overlap_x / max(1e-6, ox2 - ox1)       # horizontal-alignment fraction

    def _a_verdict(self, obj: Detection, pallets: list[Detection]) -> _Verdict:
        best_idx, best = None, 0.0
        competing = 0
        for i, pal in enumerate(pallets):
            s = self._a_overlap(obj, pal)
            if s >= self._a_min:
                competing += 1
            if s > best:
                best, best_idx = s, i
        if best_idx is None or best < self._a_min:
            return _Verdict(None, 0.0)
        return _Verdict(best_idx, best, ambiguous=competing >= 2)

    # ---- B: metric margin ----

    def _b_verdict(self, obj_m, pallets_m: list) -> _Verdict:
        if obj_m is None or not any(pm is not None for pm in pallets_m):
            return _Verdict(None, 0.0, available=False)
        best_idx, best_margin = None, -1e9
        for i, pm in enumerate(pallets_m):
            if pm is None:
                continue
            dist = float(np.hypot(obj_m[0] - pm[0], obj_m[1] - pm[1]))
            margin = self._radius - dist              # >0 inside the disc
            if margin > best_margin:
                best_margin, best_idx = margin, i
        if best_idx is None or best_margin <= -self._band:   # clearly outside every pallet
            return _Verdict(None, 0.0, available=True)
        conf = float(np.clip(best_margin / self._radius, 0.0, 1.0))
        return _Verdict(best_idx, conf, uncertain=abs(best_margin) < self._band, available=True)

    # ---- fuse one object's two verdicts → pallet idx | None ----

    def _fuse(self, a: _Verdict, b: _Verdict) -> int | None:
        if a.pallet_idx is not None and b.pallet_idx is not None:
            if a.pallet_idx == b.pallet_idx:
                return a.pallet_idx
            if b.uncertain:                 # B near an edge (its blind spot) → trust A
                return a.pallet_idx
            if a.ambiguous:                 # A can't split overlapping pallets → trust B
                return b.pallet_idx
            return a.pallet_idx if a.score >= b.score else b.pallet_idx
        if a.pallet_idx is not None:        # only A has a verdict (B unavailable/outside)
            return a.pallet_idx
        if b.pallet_idx is not None:        # only B has a verdict (no A overlap)
            return b.pallet_idx
        return None

    def _project(self, d: Detection):
        try:
            return self._projector.project(d)
        except Exception:
            return None

    # ---- per-camera per-frame pallet states ----

    def _frame_states(self, dets: list[Detection]) -> list:
        pallets = [d for d in dets if str(d.cls).lower() in PALLET_CLASSES]
        objects = [d for d in dets if str(d.cls).lower() in OBJECT_CLASSES]
        pallets_m = [self._project(p) for p in pallets]
        loads: dict[int, list[Detection]] = {i: [] for i in range(len(pallets))}
        for obj in objects:
            idx = self._fuse(self._a_verdict(obj, pallets),
                             self._b_verdict(self._project(obj), pallets_m))
            if idx is not None:
                loads[idx].append(obj)
        states = []
        for i, _pal in enumerate(pallets):
            objs = loads[i]
            if not objs:
                states.append((pallets_m[i], "empty", None, 0.6))
            else:
                dom = max(objs, key=_load_area)
                states.append((pallets_m[i], "full", str(dom.cls).lower(), float(dom.confidence)))
        return states

    def frame_states(self, dets: list[Detection]) -> list:
        """Public per-camera classification: ``[(pallet_xy_m|None, state,
        content|None, confidence)]`` aligned with the pallet-class detections
        in ``dets`` (same order). Used by the observations publisher to attach
        occupancy hints to the wire without re-deriving the association."""
        return self._frame_states(dets)

    # ---- public: enrich pallet tracks with a voted occupancy state ----

    def enrich(self, tracks_2d: list[Track2D], detections_by_camera: dict) -> list[Track2D]:
        states = []
        for dets in detections_by_camera.values():
            states.extend(self._frame_states(dets))
        pallet_tracks = [t for t in tracks_2d
                         if str(t.cls).lower() in PALLET_CLASSES]
        # Per-camera pass (primary): nearest same-camera A+B state per track.
        chosen: dict[int, tuple | None] = {}
        for t in pallet_tracks:
            best, best_d = None, self._track_match
            for (pm, st, content, conf) in states:
                if pm is None:
                    continue
                d = float(np.hypot(t.xy_m[0] - pm[0], t.xy_m[1] - pm[1]))
                if d < best_d:
                    best_d, best = d, (st, content, conf)
            chosen[t.track_id] = best
        # Metric occupancy fallback — cross-camera fusion for the DECISION:
        # per-camera association is structurally blind to the split view
        # (cam_a sees only the pallet, cam_b only the carton on it). The
        # track's xy_m IS the fused cross-camera pallet position, so pool the
        # load objects from EVERY camera and attach each to its nearest pallet
        # track within the metric radius (winner-takes-all keeps one object on
        # one pallet, mirroring the per-camera exclusivity). Positive evidence
        # from any camera outranks one viewpoint's "empty" — an "empty" only
        # claims absence from that angle; a per-camera "full" is never
        # overridden.
        loads: dict[int, list[Detection]] = {}
        if pallet_tracks:
            for dets in detections_by_camera.values():
                for o in dets:
                    if str(o.cls).lower() not in OBJECT_CLASSES:
                        continue
                    om = self._project(o)
                    if om is None:
                        continue
                    best_t, best_d = None, self._radius
                    for t in pallet_tracks:
                        d = float(np.hypot(om[0] - t.xy_m[0], om[1] - t.xy_m[1]))
                        if d <= best_d:
                            best_d, best_t = d, t
                    if best_t is not None:
                        loads.setdefault(best_t.track_id, []).append(o)
        for t in pallet_tracks:
            best = chosen[t.track_id]
            st, content, conf = best if best is not None else (None, None, None)
            if st != "full" and t.track_id in loads:
                dom = max(loads[t.track_id], key=_load_area)
                st, content = "full", str(dom.cls).lower()
                conf = float(dom.confidence)
            if st is not None:
                t.occupancy_state, t.occupancy_content = self._stabilizer.vote(
                    t.track_id, st, content)
                t.occupancy_confidence = conf
            else:
                last = self._stabilizer.last(t.track_id)   # occluded this frame → carry last vote
                if last is not None:
                    t.occupancy_state, t.occupancy_content = last
        return tracks_2d
