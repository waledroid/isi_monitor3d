"""``ZoneStateTracker`` — per-zone object-list state with change detection + refresh."""

from __future__ import annotations

import numpy as np

from backbone.core.types import Track2D
from backbone.shared.zone_state import ZoneStateTracker
from backbone.shared.zones import Zone, ZoneRegistry


def _zones() -> ZoneRegistry:
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    far = square + 10.0
    return ZoneRegistry([
        Zone(name="dock", type="palette", polygon=square),
        Zone(name="cold", type="palette", polygon=far),
    ])


def _track(track_id: int = 1, xy: tuple[float, float] = (1.0, 1.0), *,
           cls: str = "palette", occupancy_state: str | None = None) -> Track2D:
    t = Track2D(
        track_id=track_id, cls=cls, capture_ts=0.0,
        xy_m=xy, vxy_m=(0.0, 0.0),
        confidence=0.9, cameras_seeing=("cam_a",),
    )
    if occupancy_state is not None:
        t.occupancy_state = occupancy_state
    return t


def _feed(tracker: ZoneStateTracker, tracks: list[Track2D], ts: float):
    zones = _zones()
    # Membership is now keyed by STABLE zone id (which_ids), mirroring the
    # orchestrator — the tracker keys its internal state by id.
    return tracker.update([(t, zones.which_ids(t.xy_m)) for t in tracks], ts)


def test_initial_states_one_empty_per_zone() -> None:
    tracker = ZoneStateTracker(_zones())
    states = tracker.initial_states(ts=100.0)
    assert sorted(s.zone for s in states) == ["cold", "dock"]
    assert all(s.occupants == () for s in states)
    assert all(s.ts == 100.0 for s in states)


def test_enter_publishes_state_with_object() -> None:
    tracker = ZoneStateTracker(_zones())
    tracker.initial_states(ts=0.0)
    states = _feed(tracker, [_track(7, (1.0, 1.0))], ts=1.0)
    assert len(states) == 1
    s = states[0]
    assert s.zone == "dock"
    assert len(s.occupants) == 1
    assert s.occupants[0].track_id == 7
    assert s.occupants[0].cls == "palette"
    assert s.occupants[0].confidence == 0.9


def test_unchanged_state_not_republished_within_refresh() -> None:
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=10.0)
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7)], ts=1.0)
    states = _feed(tracker, [_track(7)], ts=1.5)   # same contents, refresh not due
    assert states == []


def test_refresh_republishes_occupied_zone() -> None:
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=1.0)
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7)], ts=1.0)
    states = _feed(tracker, [_track(7)], ts=2.5)   # refresh due
    assert [s.zone for s in states] == ["dock"]
    assert states[0].ts == 2.5


def test_empty_zone_does_not_refresh() -> None:
    """The retained empty message is enough — empty zones stay silent."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=1.0)
    tracker.initial_states(ts=0.0)
    assert _feed(tracker, [], ts=5.0) == []
    assert _feed(tracker, [], ts=50.0) == []


def test_leave_publishes_explicit_empty_state() -> None:
    tracker = ZoneStateTracker(_zones())
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7, (1.0, 1.0))], ts=1.0)
    states = _feed(tracker, [_track(7, (100.0, 100.0))], ts=2.0)  # left every zone
    assert [s.zone for s in states] == ["dock"]
    assert states[0].occupants == ()


def test_vanished_track_also_empties_the_zone() -> None:
    tracker = ZoneStateTracker(_zones())
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7)], ts=1.0)
    states = _feed(tracker, [], ts=2.0)   # track lost entirely
    assert [s.zone for s in states] == ["dock"]
    assert states[0].occupants == ()


def test_occupancy_change_triggers_publish() -> None:
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7, occupancy_state="empty")], ts=1.0)
    states = _feed(tracker, [_track(7, occupancy_state="full")], ts=1.5)
    assert [s.zone for s in states] == ["dock"]
    assert states[0].occupants[0].occupancy_state == "full"


def test_membership_change_between_zones() -> None:
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7, (1.0, 1.0))], ts=1.0)
    states = _feed(tracker, [_track(7, (11.0, 11.0))], ts=2.0)   # dock → cold
    by_zone = {s.zone: s for s in states}
    assert set(by_zone) == {"dock", "cold"}
    assert by_zone["dock"].occupants == ()
    assert by_zone["cold"].occupants[0].track_id == 7


def test_first_update_without_initial_states_publishes_everything() -> None:
    """Without the startup pass, the first update announces every zone once."""
    tracker = ZoneStateTracker(_zones())
    states = _feed(tracker, [_track(7)], ts=1.0)
    by_zone = {s.zone: s for s in states}
    assert set(by_zone) == {"dock", "cold"}
    assert len(by_zone["dock"].occupants) == 1
    assert by_zone["cold"].occupants == ()


def test_zone_state_message_from_state() -> None:
    """comms conversion: ZoneStateMessage.from_state maps occupants → objects."""
    from backbone.comms.schemas import ZoneStateMessage

    tracker = ZoneStateTracker(_zones())
    tracker.initial_states(ts=0.0)
    state = _feed(tracker, [_track(7, occupancy_state="full")], ts=1.0)[0]
    msg = ZoneStateMessage.from_state(state)
    assert msg.zone == "dock"
    assert msg.zone_id == "dock"          # stable id carried onto the wire
    assert msg.count == 1
    assert msg.objects[0].track_id == 7
    assert msg.objects[0].occupancy_state == "full"
    assert msg.ts == 1.0


def _named_zones() -> ZoneRegistry:
    """One zone whose display NAME differs from its stable id."""
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    return ZoneRegistry([Zone(name="Zone 1", type="palette", polygon=square, id="z1")])


def test_state_carries_stable_id_and_display_name() -> None:
    zones = _named_zones()
    tracker = ZoneStateTracker(zones)
    tracker.initial_states(ts=0.0)
    states = tracker.update([(_track(7, (1.0, 1.0)), zones.which_ids((1.0, 1.0)))], ts=1.0)
    assert len(states) == 1
    assert states[0].zone == "Zone 1"     # display label
    assert states[0].zone_id == "z1"      # stable identity


def test_rename_does_not_republish_or_reset_state() -> None:
    """Renaming a zone (same id) keeps its occupancy signature — no spurious publish.

    The tracker keys its change detector by id, so swapping the registry to
    one where id "z1" has a new NAME leaves the signature unchanged: an
    unchanged occupied zone within the refresh window republishes nothing.
    """
    tracker = ZoneStateTracker(_named_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    zones = _named_zones()
    tracker.update([(_track(7, (1.0, 1.0)), zones.which_ids((1.0, 1.0)))], ts=1.0)

    # Operator renames "Zone 1" → "Loading Bay"; id "z1" preserved.
    renamed = ZoneRegistry([
        Zone(name="Loading Bay", type="palette",
             polygon=np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]), id="z1")
    ])
    tracker._zones = renamed
    states = tracker.update(
        [(_track(7, (1.1, 1.1)), renamed.which_ids((1.1, 1.1)))], ts=1.5
    )
    assert states == []   # identity stable ⇒ no reset, no spurious republish


def test_six_zones_per_node_all_monitored() -> None:
    """The site contract: up to 6 zones per node (zone1-zone6), each monitored.

    Matches the dashboard's MAX_ZONES=6 config limit — the tracker must give
    every one of the 6 zones its own initial state and track each zone's
    contents independently.
    """
    size = 4.0
    zones = ZoneRegistry([
        Zone(
            name=f"zone{i}", type="palette",
            polygon=np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])
            + i * 10.0,   # disjoint squares at (10i, 10i)
        )
        for i in range(1, 7)
    ])
    tracker = ZoneStateTracker(zones, refresh_interval_s=100.0)

    # Startup: one retained empty state per zone — the whole folder exists.
    initial = tracker.initial_states(ts=0.0)
    assert sorted(s.zone for s in initial) == [f"zone{i}" for i in range(1, 7)]

    # One object in each of the 6 zones → 6 independent states, right occupants.
    tracks = [
        (_track(track_id=i, xy=(i * 10.0 + 1.0, i * 10.0 + 1.0)),
         zones.which_ids((i * 10.0 + 1.0, i * 10.0 + 1.0)))
        for i in range(1, 7)
    ]
    states = tracker.update(tracks, ts=1.0)
    assert sorted(s.zone for s in states) == [f"zone{i}" for i in range(1, 7)]
    for s in states:
        assert len(s.occupants) == 1
        assert f"zone{s.occupants[0].track_id}" == s.zone

    # One zone changes (zone3 empties) → ONLY zone3 republishes.
    tracks_minus_3 = [t for t in tracks if t[0].track_id != 3]
    states = tracker.update(tracks_minus_3, ts=2.0)
    assert [s.zone for s in states] == ["zone3"]
    assert states[0].occupants == ()


def test_decision_only_change_triggers_publish() -> None:
    """The PalletStateManager decision is part of the change signature: a zone
    whose occupants did NOT change still republishes when its decision flips
    (e.g. palette_empty → palette_loaded), so the wire always carries the
    latest verdict. Same decision + same occupants stays silent.

    The decision signature is ``(palette_state, content)`` ONLY — see
    ``test_counts_only_change_does_not_republish`` below for why counts are
    deliberately excluded (Finding 1)."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    zones = _zones()
    tracks = [(_track(7), zones.which_ids((1.0, 1.0)))]
    dec_empty = {"dock": ("palette_empty", ())}
    tracker.update(tracks, ts=1.0, decisions=dec_empty)
    # Unchanged occupants + unchanged decision → nothing republished.
    assert tracker.update(tracks, ts=1.5, decisions=dec_empty) == []
    # Occupants identical, decision flips → the zone republishes.
    dec_loaded = {"dock": ("palette_loaded", ("carton",))}
    states = tracker.update(tracks, ts=2.0, decisions=dec_loaded)
    assert [s.zone for s in states] == ["dock"]


def test_counts_only_change_does_not_republish() -> None:
    """Finding 1 — counts are NOT part of the decision change-signature: a
    raw per-frame count flap (1↔2 duplicate boxes, a dropout frame) with the
    SAME ``palette_state``/``content`` must not trigger a republish, even
    though the orchestrator's ``ZoneDecision.counts`` differs frame to frame.
    The tracker only ever sees the 2-tuple the orchestrator hands it, so a
    counts change can't reach the signature at all — this pins that contract
    at the call boundary the orchestrator uses."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    zones = _zones()
    tracks = [(_track(7), zones.which_ids((1.0, 1.0)))]
    # Same (palette_state, content) every step — the orchestrator would build
    # exactly this whether the frame carried 1 or 2 duplicate palette boxes.
    dec = {"dock": ("palette_empty", ())}
    tracker.update(tracks, ts=1.0, decisions=dec)
    assert tracker.update(tracks, ts=1.5, decisions=dec) == []
    assert tracker.update(tracks, ts=2.0, decisions=dec) == []


def test_update_without_decisions_keeps_old_behavior() -> None:
    """Backward compat: callers that never pass ``decisions`` see the exact
    pre-decision signature semantics (occupants-only change detection)."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7)], ts=1.0)
    assert _feed(tracker, [_track(7)], ts=1.5) == []


def test_conflicting_pallet_readings_resolve_to_loaded() -> None:
    """A double-tracked pallet reading both 'empty' and 'full' in one zone must
    publish only the LOADED reading (the agreed WMS/FMS rule — zones are
    pallet-scale, so a concurrent 'empty' pallet is the same pallet misread).
    Non-pallet objects and unknown-occupancy pallets are untouched."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    states = _feed(tracker, [
        _track(7, occupancy_state="full"),
        _track(8, occupancy_state="empty"),      # dual read of the same pallet
        _track(9, cls="carton"),                 # loose object stays listed
        _track(10, occupancy_state=None),        # unknown is NOT empty — kept
    ], ts=1.0)
    dock = next(s for s in states if s.zone == "dock")
    ids = [o.track_id for o in dock.occupants]
    assert ids == [7, 9, 10]
    assert all(o.occupancy_state != "empty" for o in dock.occupants)


def test_empty_pallet_alone_still_published() -> None:
    """No loaded pallet in the zone → the 'empty' reading is the truth."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    states = _feed(tracker, [_track(7, occupancy_state="empty")], ts=1.0)
    dock = next(s for s in states if s.zone == "dock")
    assert [o.track_id for o in dock.occupants] == [7]
    assert dock.occupants[0].occupancy_state == "empty"


def test_conflict_resolution_stabilises_change_detection() -> None:
    """The dual-read pallet flapping between (full+empty) and (full) must NOT
    spam on-change publishes — after resolution both frames have the same
    signature."""
    tracker = ZoneStateTracker(_zones(), refresh_interval_s=100.0)
    tracker.initial_states(ts=0.0)
    _feed(tracker, [_track(7, occupancy_state="full"),
                    _track(8, occupancy_state="empty")], ts=1.0)
    states = _feed(tracker, [_track(7, occupancy_state="full")], ts=1.5)
    assert states == []          # nothing changed after resolution
