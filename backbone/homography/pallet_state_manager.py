"""``PalletStateManager`` — the single decision path for zone communication.

Consolidates every "what do we tell the AGV about this zone" call into one
class, decided from per-camera DETECTION EVIDENCE — tracks are never
consulted. Two independent signals feed each zone's decision:

* **Presence** (decision 2 of the design doc): a zone holds a class if ANY
  camera's detection of that class projects (via ``FootProjector``) inside
  the zone polygon, within ``tol_m`` — the same tolerant 5-point-cross
  containment ``backbone.detection.zone_scope.build_zone_membership_filter``
  uses, reused here rather than reinvented. Cross-camera OR: one camera's
  positive detection is proof, no other camera need agree.
* **Occupancy** (decision 3): reuses ``PalletOccupancy.frame_states`` per
  camera (the existing A+B image-overlap / metric-margin fusion, computed
  independently within each camera's own detections), bucketed into zones by
  each pallet's floor position. A "full" verdict from ANY camera OUTRANKS
  another camera's "empty" for the same zone (an "empty" only claims absence
  from that angle — same rule as ``PalletOccupancy.enrich``'s cross-camera
  fallback).

Both signals are stabilized against per-frame flicker before they reach the
published enum:

* Presence goes through a ``ZoneMembershipHysteresis`` per class (2
  consecutive evidence frames to ENTER, 15 consecutive absent frames to
  EXIT — the debounce ``ZoneMembershipHysteresis`` already implements for
  per-track zone membership). **Hysteresis keying choice:** rather than
  synthesize composite string keys or hash zone/class pairs, this class
  instantiates ONE ``ZoneMembershipHysteresis`` PER DETECTED CLASS and feeds
  it a single pseudo-track id (``0``) whose "raw membership" each frame is
  the set of zone ids where that class had fresh evidence. This is the
  simplest correct reuse of the existing debounce semantics — no new state
  machine, no key-collision risk (whereas ``hash()`` on strings is
  PYTHONHASHSEED-randomized and a manual ``f"{zone_id}:{cls}"`` key just
  re-derives the same (zone, cls) pair `ZoneMembershipHysteresis` already
  keys on structurally via one instance per class).
* Occupancy's per-frame full-wins fold is then voted through an
  ``OccupancyStabilizer`` keyed by ZONE ID (a plain string — its dict keys
  are opaque to type, verified by ``pallet_occupancy.OccupancyStabilizer``
  using ``dict[int, deque]`` only as a type *hint*) instead of pallet
  ``track_id``, so a zone's loaded/empty read is stable even though this
  class never looks at tracks.

``no_data`` is reserved for "the manager has never stepped" — it is not a
per-zone runtime state reachable from ``step()``; use ``initial()`` to seed
UI/wire state before the first detection batch arrives. Once ``step()`` has
run at least once, a zone with no held presence reads ``no_palette``
(evidence WAS observed — the palette class in particular just wasn't found),
never ``no_data``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from backbone.core.types import Detection
from backbone.homography.pallet_occupancy import (
    PALLET_CLASSES,
    OccupancyStabilizer,
    PalletOccupancy,
)
from backbone.shared.zones import Zone, ZoneMembershipHysteresis, ZoneRegistry

# The pseudo-track id every per-class ZoneMembershipHysteresis is fed under —
# there is exactly one "track" per class (the class's zone-membership set).
_PSEUDO_TRACK = 0


@dataclass(frozen=True)
class ZoneDecision:
    """One zone's published communication state for this step."""

    zone_id: str
    zone_name: str
    palette_state: str          # no_data|no_palette|palette_empty|palette_loaded
    content: tuple[str, ...] = ()      # e.g. ("carton",) when loaded
    counts: dict[str, int] = field(default_factory=dict)  # per detected class, max-across-cameras


def _norm_cls(cls: str) -> str:
    """Fold pallet-class synonyms (``pallet``/``palette_vide``) to ``"palette"``
    so presence, counts, and the enum all key on one canonical name."""
    c = str(cls).lower()
    return "palette" if c in PALLET_CLASSES else c


def _in_zone(zone: Zone, xy: tuple[float, float], tol: float) -> bool:
    """Tolerant containment: a 5-point cross (center ± tol on each axis),
    the same mechanic ``zone_scope.build_zone_membership_filter`` samples so
    a detection straddling the boundary by projection error is still kept."""
    x, y = xy
    for dx, dy in ((0.0, 0.0), (tol, 0.0), (-tol, 0.0), (0.0, tol), (0.0, -tol)):
        if zone.contains((x + dx, y + dy)):
            return True
    return False


class PalletStateManager:
    """Decide every zone's palette communication state from detection
    evidence alone. See the module docstring for the presence/occupancy
    signals and the hysteresis-keying design choice.
    """

    def __init__(
        self,
        zones: ZoneRegistry,
        projector,
        occupancy: PalletOccupancy,
        *,
        tol_m: float = 0.15,
        enter_after: int = 2,
        exit_after: int = 15,
    ) -> None:
        self._zones = zones
        self._projector = projector
        self._occupancy = occupancy
        self._tol = float(tol_m)
        self._enter_after = int(enter_after)
        self._exit_after = int(exit_after)
        self._hyst: dict[str, ZoneMembershipHysteresis] = {}
        self._occ_stabilizer = OccupancyStabilizer()

    def initial(self) -> list[ZoneDecision]:
        """Decisions before ``step`` has ever run: every zone reads
        ``no_data`` (no evidence observed yet — distinct from ``no_palette``,
        which means evidence WAS observed and simply found no palette)."""
        return [
            ZoneDecision(zone_id=zid, zone_name=self._zone_name(zid),
                        palette_state="no_data")
            for zid in self._zones.ids
        ]

    def step(self, detections_by_camera: Mapping[str, list[Detection]]) -> list[ZoneDecision]:
        zone_ids = self._zones.ids

        # ---- per-zone per-class counts (raw, max-across-cameras) + this
        # frame's evidence set per class (union across cameras — OR).
        counts: dict[str, dict[str, int]] = {zid: {} for zid in zone_ids}
        evidence: dict[str, set[str]] = {}

        for cam_dets in detections_by_camera.values():
            cam_counts: dict[str, dict[str, int]] = {zid: {} for zid in zone_ids}
            for det in cam_dets:
                xy = self._project(det)
                if xy is None:
                    continue
                cls = _norm_cls(det.cls)
                for zid in zone_ids:
                    zone = self._zones.by_id(zid)
                    if zone is not None and _in_zone(zone, xy, self._tol):
                        cam_counts[zid][cls] = cam_counts[zid].get(cls, 0) + 1
            for zid, cls_counts in cam_counts.items():
                for cls, n in cls_counts.items():
                    if n > counts[zid].get(cls, 0):
                        counts[zid][cls] = n
                    if n > 0:
                        evidence.setdefault(cls, set()).add(zid)

        # ---- presence hysteresis, one ZoneMembershipHysteresis per class.
        present: dict[str, set[str]] = {}
        for cls in set(evidence) | set(self._hyst):
            hyst = self._hyst.setdefault(
                cls, ZoneMembershipHysteresis(exit_after=self._exit_after,
                                              enter_after=self._enter_after))
            raw = tuple(sorted(evidence.get(cls, ())))
            present[cls] = set(hyst.update(_PSEUDO_TRACK, raw))

        # ---- occupancy: per-camera A+B classification, bucketed into zones
        # by pallet floor position, full-wins-across-cameras this frame.
        frame_occ: dict[str, tuple[str, str | None]] = {}
        for cam_dets in detections_by_camera.values():
            for pallet_xy, state, content, _conf in self._occupancy.frame_states(cam_dets):
                if pallet_xy is None:
                    continue
                for zid in zone_ids:
                    zone = self._zones.by_id(zid)
                    if zone is None or not _in_zone(zone, pallet_xy, self._tol):
                        continue
                    cur = frame_occ.get(zid)
                    if cur is None or (state == "full" and cur[0] != "full"):
                        frame_occ[zid] = (state, content)

        # ---- temporal vote of the fused occupancy, keyed by ZONE id.
        zone_occ: dict[str, tuple[str, str | None] | None] = {}
        for zid in zone_ids:
            if zid in frame_occ:
                zone_occ[zid] = self._occ_stabilizer.vote(zid, *frame_occ[zid])
            else:
                zone_occ[zid] = self._occ_stabilizer.last(zid)

        # ---- assemble the published enum per zone.
        decisions = []
        for zid in zone_ids:
            palette_present = zid in present.get("palette", set())
            occ = zone_occ.get(zid)
            if not palette_present:
                palette_state, content = "no_palette", ()
            elif occ is not None and occ[0] == "full":
                palette_state = "palette_loaded"
                content = (occ[1],) if occ[1] else ()
            else:
                palette_state, content = "palette_empty", ()
            decisions.append(ZoneDecision(
                zone_id=zid, zone_name=self._zone_name(zid),
                palette_state=palette_state, content=content,
                counts=dict(counts.get(zid, {})),
            ))
        return decisions

    # ---- internals ----

    def _project(self, det: Detection) -> tuple[float, float] | None:
        try:
            return self._projector.project(det)
        except Exception:
            return None

    def _zone_name(self, zone_id: str) -> str:
        zone = self._zones.by_id(zone_id)
        return zone.name if zone is not None else zone_id
