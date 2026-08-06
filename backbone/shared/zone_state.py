"""Per-zone occupancy state for Track2D positions — the WMS/FMS signal.

Concrete, single-implementation utility (not an ABC — same precedent as
``ZoneTransitionDetector``).

Where ``ZoneTransitionDetector`` emits boundary-crossing *events*, this
tracker maintains the *absolute state* of every configured zone: which
tracked objects are currently inside, with class and confidence. Downstream
(MQTT) the state is published retained on ``{prefix}/zone/{zone}`` so a
late-joining consumer reads every zone's contents immediately.

Publish policy (decided here, executed by the orchestrator):

* **on change** — the occupant set changed (track ids, classes, or a pallet's
  ``occupancy_state``). Emptying a zone yields an explicit empty state, never
  silence, so consumers can distinguish "empty" from "unknown".
* **periodic refresh** — an *occupied* zone is re-emitted every
  ``refresh_interval_s`` even without change, so ``ts`` freshness is
  observable without publishing at frame rate. Empty zones don't refresh;
  their retained message is already the truth.

Identity-loss policy: a vanished track simply drops out of the occupant list
on the next update — state is absolute, so no special-casing is needed
(unlike the transition detector's no-synthesised-"leave" rule).
"""

from __future__ import annotations

from dataclasses import dataclass

from backbone.core.types import Track2D
from backbone.shared.zones import ZoneRegistry


@dataclass(frozen=True)
class ZoneOccupant:
    """One tracked object currently inside a zone."""

    track_id: int
    cls: str
    confidence: float
    xy_m: tuple[float, float]
    occupancy_state: str | None
    occupancy_content: str | None
    occupancy_confidence: float


@dataclass(frozen=True)
class ZoneState:
    """The current contents of one zone at time ``ts``.

    ``zone_id`` is the STABLE identity (consumers key on it); ``zone`` is the
    current operator label, carried for display.
    """

    zone: str
    ts: float
    occupants: tuple[ZoneOccupant, ...]
    zone_id: str = ""


def _occupant(track: Track2D) -> ZoneOccupant:
    return ZoneOccupant(
        track_id=track.track_id,
        cls=track.cls,
        confidence=track.confidence,
        xy_m=track.xy_m,
        occupancy_state=track.occupancy_state,
        occupancy_content=track.occupancy_content,
        occupancy_confidence=track.occupancy_confidence,
    )


# Mirrors ``backbone.homography.pallet_occupancy.PALLET_CLASSES`` (kept local:
# shared/ must not import the homography layer).
_PALLET_CLASSES = frozenset({"palette", "pallet", "palette_vide"})


def _resolve_pallet_conflicts(occ: list[ZoneOccupant]) -> list[ZoneOccupant]:
    """Drop 'empty' pallet readings when the zone also holds a LOADED pallet.

    A single physical pallet is sometimes double-tracked with conflicting
    occupancy (one reading 'empty', one 'full'). Consumers (WMS/FMS) must not
    see both: zones are pallet-scale areas, so the agreed rule is that the
    loaded reading carries the information and concurrent 'empty' pallet
    readings in the same zone are noise. Pallets with UNKNOWN occupancy
    (``None``) are kept — unknown is not empty.
    """
    has_loaded = any(
        o.cls.lower() in _PALLET_CLASSES and o.occupancy_state == "full" for o in occ
    )
    if not has_loaded:
        return occ
    return [
        o for o in occ
        if not (o.cls.lower() in _PALLET_CLASSES and o.occupancy_state == "empty")
    ]


class ZoneStateTracker:
    """Maintain per-zone occupant lists; decide which zones to (re-)publish.

    Args:
        zones: The ``ZoneRegistry`` defining all known floor zones.
        refresh_interval_s: Re-emit an unchanged *occupied* zone after this
            many seconds (measured on the frame ``ts`` clock, i.e. capture_ts).
    """

    def __init__(self, zones: ZoneRegistry, refresh_interval_s: float = 1.0) -> None:
        self._zones = zones
        self._refresh_interval_s = float(refresh_interval_s)
        # Per zone ID (STABLE — renaming a zone must not reset its state): the
        # signature of the last published state + its publish ts.
        self._prev_sig: dict[str, tuple] = {}
        self._last_pub_ts: dict[str, float] = {}

    def initial_states(self, ts: float) -> list[ZoneState]:
        """Explicit empty state for every configured zone (startup retained pass).

        Seeds the change detector, so the following ``update`` calls only
        publish zones that actually gain occupants.
        """
        states = []
        for zid in self._zones.ids:
            name = self._zones.name_of(zid) or zid
            states.append(ZoneState(zone=name, ts=ts, occupants=(), zone_id=zid))
            self._prev_sig[zid] = ()
            self._last_pub_ts[zid] = ts
        return states

    def update(
        self,
        tracks: list[tuple[Track2D, tuple[str, ...]]],
        ts: float,
        decisions: dict[str, tuple] | None = None,
    ) -> list[ZoneState]:
        """Recompute every zone's contents; return the states to publish now.

        Args:
            tracks: This frame's stabilized tracks, each with its precomputed
                zone membership (``ZoneRegistry.which_ids(track.xy_m)`` — zone
                IDS) so point-in-polygon runs once per track for both this
                tracker and the transition detector.
            ts:     Frame capture timestamp.
            decisions: Optional per-zone-id decision signature (an opaque
                comparable tuple — the orchestrator passes the
                ``PalletStateManager`` verdict as
                ``(palette_state, content, sorted counts)``). When present for
                a zone it joins the change signature, so a decision flip
                republishes even with identical occupants. ``None`` (the
                default) keeps the pre-decision occupants-only semantics.
        """
        occupants: dict[str, list[ZoneOccupant]] = {zid: [] for zid in self._zones.ids}
        for track, member_zones in tracks:
            for zid in member_zones:
                if zid in occupants:
                    occupants[zid].append(_occupant(track))

        states: list[ZoneState] = []
        for zid, occ in occupants.items():
            occ = _resolve_pallet_conflicts(occ)
            occ.sort(key=lambda o: o.track_id)   # deterministic payload ordering
            sig: tuple = tuple((o.track_id, o.cls, o.occupancy_state) for o in occ)
            dec_sig = decisions.get(zid) if decisions else None
            if dec_sig is not None:
                sig = (sig, dec_sig)
            changed = sig != self._prev_sig.get(zid)
            refresh_due = bool(occ) and (
                ts - self._last_pub_ts.get(zid, float("-inf")) >= self._refresh_interval_s
            )
            if changed or refresh_due:
                name = self._zones.name_of(zid) or zid
                states.append(ZoneState(zone=name, ts=ts, occupants=tuple(occ), zone_id=zid))
                self._prev_sig[zid] = sig
                self._last_pub_ts[zid] = ts
        return states
