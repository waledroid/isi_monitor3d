"""``PalletStateManager`` — the single decision path for zone communication.

Consolidates every "what do we tell the AGV about this zone" call into one
class, decided from per-camera DETECTION EVIDENCE — tracks are never
consulted. Two independent signals feed each zone's decision:

* **Presence** (decision 2 of the design doc): a zone holds a class if ANY
  camera's detection of that class projects inside the zone polygon, within
  ``tol_m`` — the same tolerant 5-point-cross containment
  ``backbone.detection.zone_scope.build_zone_membership_filter`` uses,
  reused here rather than reinvented (``_in_zone``). Cross-camera OR: one
  camera's positive detection is proof, no other camera need agree.
* **Occupancy** (decision 3): reuses ``PalletOccupancy.frame_states`` per
  camera (the existing A+B image-overlap / metric-margin fusion, computed
  independently within each camera's own detections, always at Z=0 — see
  "Plane-aware bucketing" below), bucketed into zones by each pallet's
  position. A "full" verdict from ANY camera OUTRANKS another camera's
  "empty" for the same zone (an "empty" only claims absence from that angle
  — same rule as ``PalletOccupancy.enrich``'s cross-camera fallback).

**Plane-aware bucketing (zone-base-height, decision 5).** A zone's polygon
lives on its own plane (``Zone.z_base_m`` — 0.0 for the floor, e.g. 0.304 m
for a platform). Both presence and occupancy bucketing test containment on
EACH ZONE'S OWN PLANE, not a single shared Z=0 projection: constructed with
``rig`` set, this class builds one ``ZoneAwareProjector`` and re-projects
each detection's raw foot pixel (``det.camera_id``, ``det.foot_uv``) onto
every zone's plane it's tested against (``_plane_xy``, memoized per distinct
``z_base_m`` so zones sharing a height reuse one projection — the same
pattern ``build_zone_membership_filter`` uses). Occupancy's A/B
classification (``state``/``content``) is unaffected — ``PalletOccupancy``
still estimates full/empty from its own Z=0-projected ``pallets_m``
internally; only which ZONE a pallet's verdict lands in is re-derived on
that zone's plane. **Without ``rig`` (the default, and every pre-existing
caller)** this class falls back to the single Z=0 projection via the plain
``projector`` argument, computed once per detection exactly as before —
floor-only zones then produce BIT-IDENTICAL decisions to pre-zone-base-height
behavior.

**Fail-closed skip on projection failure (unchanged policy, now per-zone).**
Before zone-base-height, a detection whose single Z=0 projection failed
(unknown camera / degenerate ray / raised exception) contributed no evidence
to ANY zone that frame — the whole detection was skipped. With per-zone
planes, failure is now scoped to the (detection, zone) PAIR: if a
detection's camera is unknown to the ``ZoneAwareProjector`` or its ray is
degenerate on ONE zone's plane, only that zone gets no evidence from it —
other zones (especially ones sharing a different, valid plane) are
unaffected. This keeps the same "a bad projection proves nothing, never
counts as absence" fail-closed spirit CLAUDE.md principle 6 requires, just
correctly scoped now that different zones can disagree about whether a
given camera/pixel projects cleanly.

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

**Camera-loss gate (partial-frame protection).** A degraded Mode-2 pair (one
camera down) only carries the survivor's detections. Without a gate, a zone
covered only by the MISSING camera would accumulate consecutive "absent"
evidence every step and eventually exit presence — camera loss reading as
proof of absence, which is exactly the failure ``fail honestly`` (CLAUDE.md
principle 6) forbids. ``step()`` accepts ``reporting_cameras`` — the set of
cameras that actually reported *this* frame (defaults to the keys of
``detections_by_camera``, so callers who never pass it see today's
behaviour unchanged). The manager is constructed with the FULL configured
camera set (``camera_ids``); whenever ``reporting_cameras`` is a strict
subset of it, the frame is "partial" and each class's raw evidence is
unioned with that class's currently-held presence before it reaches the
hysteresis (see ``ZoneMembershipHysteresis.update``: a zid present in
``raw`` never advances its exit streak). Fresh evidence can still ENTER a
new zone on a partial frame (a surviving camera seeing a NEW palette is
real evidence); only EXIT is blocked. Occupancy needs no equivalent gate:
a zone with no detections this frame (partial OR a true single-frame
occlusion) already falls through to ``OccupancyStabilizer.last()`` — see
the ``zone_occ`` loop below.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from backbone.core.types import Detection
from backbone.homography.pallet_occupancy import (
    PALLET_CLASSES,
    OccupancyStabilizer,
    PalletOccupancy,
)
from backbone.shared.zones import (
    Zone,
    ZoneAwareProjector,
    ZoneMembershipHysteresis,
    ZoneRegistry,
)

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
    a detection straddling the boundary by projection error is still kept.

    Spec-approved trade-off: for two zones sharing a boundary, a detection
    within ``tol`` of that shared edge can satisfy BOTH zones' cross at once
    (double-counted into the adjacent zone), because the cross is centered
    on the detection, not clipped to a single zone's polygon. Losing a
    boundary detection entirely (no tolerance) was judged worse than the
    occasional double count in two neighbours.
    """
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
        rig=None,
        camera_ids: Collection[str] = (),
        tol_m: float = 0.15,
        enter_after: int = 2,
        exit_after: int = 15,
    ) -> None:
        self._zones = zones
        self._projector = projector
        self._occupancy = occupancy
        # Optional: enables per-zone-plane bucketing (zone-base-height,
        # decision 5). None (the default — every pre-existing caller) keeps
        # the single Z=0 `projector` path, bit-identical to before.
        self._zone_aware = ZoneAwareProjector(rig) if rig is not None else None
        # The FULL configured camera set — compared against each step's
        # `reporting_cameras` to detect a partial (degraded) frame. Empty
        # (the default) disables the camera-loss gate entirely, so callers
        # that never pass it keep pre-Finding-2 behaviour.
        self._camera_ids = frozenset(camera_ids)
        self._tol = float(tol_m)
        self._enter_after = int(enter_after)
        self._exit_after = int(exit_after)
        self._hyst: dict[str, ZoneMembershipHysteresis] = {}
        self._occ_stabilizer = OccupancyStabilizer()
        # Last step's post-hysteresis presence per class — the "currently
        # held" set fed into the camera-loss union, and the before/after
        # comparison that detects a palette zone's presence EXIT (to forget
        # its stale occupancy history; see OccupancyStabilizer.forget).
        self._present_prev: dict[str, set[str]] = {}

    def initial(self) -> list[ZoneDecision]:
        """Decisions before ``step`` has ever run: every zone reads
        ``no_data`` (no evidence observed yet — distinct from ``no_palette``,
        which means evidence WAS observed and simply found no palette)."""
        return [
            ZoneDecision(zone_id=zid, zone_name=self._zone_name(zid),
                        palette_state="no_data")
            for zid in self._zones.ids
        ]

    def step(
        self,
        detections_by_camera: Mapping[str, list[Detection]],
        *,
        reporting_cameras: Collection[str] | None = None,
    ) -> list[ZoneDecision]:
        """Decide every zone's state from this frame's per-camera detections.

        Args:
            detections_by_camera: This frame's detections, per camera.
            reporting_cameras: Cameras that actually reported this frame —
                defaults to ``detections_by_camera``'s keys. Pass the
                synchronizer's ``FramePair.frames`` keys explicitly (the
                orchestrator does) so a camera that reported zero detections
                still counts as "reporting" and one that's simply absent
                from the pair (degraded Mode 2) does not. See the module
                docstring's "Camera-loss gate" section.
        """
        zone_ids = self._zones.ids
        if reporting_cameras is None:
            reporting_cameras = detections_by_camera.keys()
        # A "partial" frame is missing at least one CONFIGURED camera. With
        # camera_ids unset (single-cam callers, most unit tests) the gate is
        # inert — every frame is treated as full, matching pre-Finding-2
        # behaviour exactly.
        partial = bool(self._camera_ids) and not self._camera_ids.issubset(
            set(reporting_cameras))

        # ---- per-zone per-class counts (raw, max-across-cameras) + this
        # frame's evidence set per class (union across cameras — OR).
        counts: dict[str, dict[str, int]] = {zid: {} for zid in zone_ids}
        evidence: dict[str, set[str]] = {}

        for cam_dets in detections_by_camera.values():
            cam_counts: dict[str, dict[str, int]] = {zid: {} for zid in zone_ids}
            for det in cam_dets:
                cls = _norm_cls(det.cls)
                if self._zone_aware is None:
                    # No zone-aware projector: one Z=0 projection shared by
                    # every zone, exactly as before zone-base-height
                    # (bit-identical). A failed projection skips the WHOLE
                    # detection, as it always did.
                    shared_xy = self._project(det)
                    if shared_xy is None:
                        continue
                    for zid in zone_ids:
                        zone = self._zones.by_id(zid)
                        if zone is not None and _in_zone(zone, shared_xy, self._tol):
                            cam_counts[zid][cls] = cam_counts[zid].get(cls, 0) + 1
                    continue
                plane_cache: dict[float, tuple[float, float] | None] = {}
                for zid in zone_ids:
                    zone = self._zones.by_id(zid)
                    if zone is None:
                        continue
                    xy = self._plane_xy(det, zone, plane_cache)
                    if xy is not None and _in_zone(zone, xy, self._tol):
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
            raw_set = set(evidence.get(cls, ()))
            if partial:
                # Camera loss must not read as evidence of absence: union in
                # the zones this class already holds so ZoneMembershipHysteresis
                # sees them as still-raw-present and cannot advance their exit
                # streak. Fresh evidence (a zone NOT already held) still enters
                # normally — only exiting is blocked on a partial view.
                raw_set |= self._present_prev.get(cls, set())
            present[cls] = set(hyst.update(_PSEUDO_TRACK, tuple(sorted(raw_set))))

        # A palette zone's presence just EXITED (full-frame absence, never
        # blocked above) ⇒ its occupancy vote history is now about a pallet
        # that's gone; forget it so a later, unrelated pallet doesn't inherit
        # a stale loaded/empty verdict (Finding 5 — see OccupancyStabilizer.forget).
        exited_palette = self._present_prev.get("palette", set()) - present.get("palette", set())
        for zid in exited_palette:
            self._occ_stabilizer.forget(zid)
        self._present_prev = present

        # ---- occupancy: per-camera A+B classification, bucketed into zones
        # by pallet position (each zone's OWN plane, same as presence above —
        # a platform pallet's occupancy state must land in the platform
        # zone), full-wins-across-cameras this frame. `frame_states`'
        # results are aligned in order with the pallet-class detections in
        # `cam_dets` (its own docstring guarantee), so zipping recovers each
        # verdict's raw detection (camera_id/foot_uv) for re-projection —
        # `PalletOccupancy`'s own A/B estimators still classify state/content
        # at Z=0 internally, unchanged (this only re-derives which ZONE a
        # verdict is bucketed into).
        frame_occ: dict[str, tuple[str, str | None]] = {}
        for cam_dets in detections_by_camera.values():
            pallets = [d for d in cam_dets if str(d.cls).lower() in PALLET_CLASSES]
            results = self._occupancy.frame_states(cam_dets)
            for det, (pallet_xy, state, content, _conf) in zip(pallets, results, strict=True):
                plane_cache: dict[float, tuple[float, float] | None] = {}
                for zid in zone_ids:
                    zone = self._zones.by_id(zid)
                    if zone is None:
                        continue
                    xy = pallet_xy if self._zone_aware is None else self._plane_xy(
                        det, zone, plane_cache)
                    if xy is None or not _in_zone(zone, xy, self._tol):
                        continue
                    cur = frame_occ.get(zid)
                    if cur is None or (state == "full" and cur[0] != "full"):
                        frame_occ[zid] = (state, content)

        # ---- temporal vote of the fused occupancy, keyed by ZONE id.
        # A zone absent from `frame_occ` this step — whether from a true
        # single-frame occlusion OR a partial/degraded frame that simply
        # carried no detections for it — already carries forward via
        # `last()` rather than voting fresh "empty" evidence. This is the
        # same mechanism that protects presence above; occupancy needs no
        # separate camera-loss gate (Finding 2, verified).
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

    def _plane_xy(
        self,
        det: Detection,
        zone: Zone,
        cache: dict[float, tuple[float, float] | None],
    ) -> tuple[float, float] | None:
        """``det``'s foot pixel projected onto ``zone``'s own base plane,
        memoized per distinct ``z_base_m`` (zones sharing a height reuse one
        projection — the same optimization
        ``zone_scope.build_zone_membership_filter`` applies). Only called
        when ``self._zone_aware`` is set (see module docstring).

        Returns ``None`` for an unknown camera or a degenerate/behind-camera
        ray — this (detection, zone) pair contributes no evidence for THIS
        zone only; other zones (particularly ones on a different, valid
        plane) are unaffected. See the module docstring's "Fail-closed skip"
        section.
        """
        z = float(zone.z_base_m)
        if z in cache:
            return cache[z]
        xy = None
        if det.camera_id in self._zone_aware:
            xy = self._zone_aware.position_in_zone(det.camera_id, det.foot_uv, zone)
        cache[z] = xy
        return xy

    def _zone_name(self, zone_id: str) -> str:
        zone = self._zones.by_id(zone_id)
        return zone.name if zone is not None else zone_id
