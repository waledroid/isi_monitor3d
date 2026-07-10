"""``ZoneTransitionDetector`` / ``PassingEvent`` — pure unit tests.

No real cameras, no real network. Uses an in-memory ``ZoneRegistry`` built
with the same ``Zone.from_dict`` / ``ZoneRegistry`` pattern as
``tests/test_zones.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.shared.zone_transitions import PassingEvent, ZoneTransitionDetector
from backbone.shared.zones import Zone, ZoneRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A 2 m x 2 m square with one corner at the origin.
_SQUARE_B3D = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

# A second, slightly larger square (partially overlapping) offset to the right.
_SQUARE_OVERLAP = np.array([[1.0, 0.0], [4.0, 0.0], [4.0, 2.0], [1.0, 2.0]])


def _single_zone_registry() -> ZoneRegistry:
    """Registry containing one zone named 'B3D'."""
    return ZoneRegistry([Zone("B3D", "storage", _SQUARE_B3D)])


def _two_zone_registry() -> ZoneRegistry:
    """Registry containing two overlapping zones."""
    return ZoneRegistry([
        Zone("B3D", "storage", _SQUARE_B3D),
        Zone("RIGHT", "danger", _SQUARE_OVERLAP),
    ])


def _empty_registry() -> ZoneRegistry:
    return ZoneRegistry.empty()


# ---------------------------------------------------------------------------
# PassingEvent dataclass
# ---------------------------------------------------------------------------


def test_passing_event_is_frozen() -> None:
    ev = PassingEvent(track_id=1, cls="person", zone="B3D", direction="enter", ts=0.0)
    with pytest.raises((AttributeError, TypeError)):
        ev.zone = "other"  # type: ignore[misc]


def test_passing_event_fields() -> None:
    ev = PassingEvent(track_id=7, cls="palette", zone="B3D", direction="leave", ts=42.5)
    assert ev.track_id == 7
    assert ev.cls == "palette"
    assert ev.zone == "B3D"
    assert ev.direction == "leave"
    assert ev.ts == pytest.approx(42.5)


def test_event_carries_stable_zone_id() -> None:
    """Every emitted event tags the STABLE id alongside the display name."""
    det = ZoneTransitionDetector(
        ZoneRegistry([Zone("Zone 1", "storage", _SQUARE_B3D, id="z-stable-1")])
    )
    events = det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    assert len(events) == 1
    assert events[0].zone == "Zone 1"
    assert events[0].zone_id == "z-stable-1"


def test_rename_zone_emits_no_spurious_events() -> None:
    """Renaming a zone (id unchanged) must NOT fire a leave-old/enter-new pair.

    The detector keys internal membership by id, so swapping the registry to
    one where the same id carries a new NAME leaves membership unchanged.
    """
    det = ZoneTransitionDetector(
        ZoneRegistry([Zone("Zone 1", "storage", _SQUARE_B3D, id="z1")])
    )
    e1 = det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    assert [ev.direction for ev in e1] == ["enter"]

    # Operator renames "Zone 1" → "Loading Bay"; id "z1" is preserved.
    det._zones = ZoneRegistry([Zone("Loading Bay", "storage", _SQUARE_B3D, id="z1")])
    e2 = det.update(track_id=1, cls="person", xy_m=(1.1, 1.1), ts=2.0)
    assert e2 == []   # no spurious leave/enter — identity is stable


# ---------------------------------------------------------------------------
# ZoneTransitionDetector.update — core behaviour
# ---------------------------------------------------------------------------


def test_enter_event_on_first_update_inside_zone() -> None:
    """A track first seen inside a zone fires one 'enter' event."""
    det = ZoneTransitionDetector(_single_zone_registry())
    events = det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 1
    assert ev.cls == "person"
    assert ev.zone == "B3D"
    assert ev.direction == "enter"
    assert ev.ts == pytest.approx(1.0)


def test_no_event_when_staying_inside_zone() -> None:
    """No events while the track stays inside the same zone."""
    det = ZoneTransitionDetector(_single_zone_registry())
    det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)   # enter
    events = det.update(track_id=1, cls="person", xy_m=(1.5, 1.5), ts=2.0)  # still inside
    assert events == []


def test_leave_event_when_exiting_zone() -> None:
    """Moving outside the zone fires one 'leave' event."""
    det = ZoneTransitionDetector(_single_zone_registry())
    det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)   # enter
    events = det.update(track_id=1, cls="person", xy_m=(5.0, 5.0), ts=2.0)  # leave
    assert len(events) == 1
    ev = events[0]
    assert ev.direction == "leave"
    assert ev.zone == "B3D"


def test_no_event_when_staying_outside_zone() -> None:
    """No events while the track remains outside every zone."""
    det = ZoneTransitionDetector(_single_zone_registry())
    # First update: outside zone — no zones to enter.
    events = det.update(track_id=1, cls="person", xy_m=(10.0, 10.0), ts=1.0)
    assert events == []
    # Second update: still outside.
    events = det.update(track_id=1, cls="person", xy_m=(11.0, 11.0), ts=2.0)
    assert events == []


def test_enter_then_leave_sequence() -> None:
    """Full enter → stay → leave sequence produces exactly 2 events (not during stay)."""
    det = ZoneTransitionDetector(_single_zone_registry())
    e1 = det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    e2 = det.update(track_id=1, cls="person", xy_m=(1.2, 1.2), ts=2.0)
    e3 = det.update(track_id=1, cls="person", xy_m=(5.0, 5.0), ts=3.0)

    assert len(e1) == 1 and e1[0].direction == "enter"
    assert e2 == []
    assert len(e3) == 1 and e3[0].direction == "leave"


def test_two_overlapping_zones_fire_two_enters() -> None:
    """Moving into a region covered by two zones fires two 'enter' events."""
    det = ZoneTransitionDetector(_two_zone_registry())
    # (1.5, 1.0) is inside both B3D and RIGHT.
    events = det.update(track_id=1, cls="person", xy_m=(1.5, 1.0), ts=1.0)
    assert len(events) == 2
    zones_entered = {ev.zone for ev in events}
    assert zones_entered == {"B3D", "RIGHT"}
    assert all(ev.direction == "enter" for ev in events)


def test_partial_exit_from_overlapping_zones() -> None:
    """Moving from overlap region into only one zone fires one 'leave' for the other."""
    det = ZoneTransitionDetector(_two_zone_registry())
    det.update(track_id=1, cls="person", xy_m=(1.5, 1.0), ts=1.0)  # inside B3D + RIGHT

    # (0.5, 1.0) is inside B3D but outside RIGHT.
    events = det.update(track_id=1, cls="person", xy_m=(0.5, 1.0), ts=2.0)
    assert len(events) == 1
    assert events[0].direction == "leave"
    assert events[0].zone == "RIGHT"


def test_empty_registry_never_fires_events() -> None:
    """With no zones configured, no events are ever emitted."""
    det = ZoneTransitionDetector(_empty_registry())
    for xy in [(0.0, 0.0), (1.0, 1.0), (-5.0, 10.0)]:
        events = det.update(track_id=1, cls="person", xy_m=xy, ts=1.0)
        assert events == []


def test_independent_tracks_do_not_interfere() -> None:
    """Two distinct track_ids maintain independent zone state."""
    det = ZoneTransitionDetector(_single_zone_registry())
    # Track 1 enters.
    det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    # Track 2 is outside — should not inherit track 1's state.
    events2 = det.update(track_id=2, cls="person", xy_m=(5.0, 5.0), ts=1.0)
    assert events2 == []
    # Track 1 stays inside — no new event.
    events1 = det.update(track_id=1, cls="person", xy_m=(1.5, 1.5), ts=2.0)
    assert events1 == []


def test_cls_is_propagated_in_event() -> None:
    """The ``cls`` field of the emitted event matches what was passed to ``update``."""
    det = ZoneTransitionDetector(_single_zone_registry())
    events = det.update(track_id=3, cls="palette", xy_m=(1.0, 1.0), ts=1.0)
    assert events[0].cls == "palette"


# ---------------------------------------------------------------------------
# ZoneTransitionDetector.forget
# ---------------------------------------------------------------------------


def test_forget_drops_state_so_reentry_re_fires_enter() -> None:
    """After ``forget``, a re-appearing track fires 'enter' again."""
    det = ZoneTransitionDetector(_single_zone_registry())
    det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)   # enter B3D
    det.forget({})  # track 1 is gone from live set

    # Track re-appears (or a new track gets the same ID) — must fire 'enter' again.
    events = det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=3.0)
    assert len(events) == 1
    assert events[0].direction == "enter"


def test_forget_does_not_synthesise_leave_events() -> None:
    """``forget`` is silent — it drops state without returning or publishing events."""
    det = ZoneTransitionDetector(_single_zone_registry())
    det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)   # inside B3D
    det.forget(set())   # forget returns None, not a list of events


def test_forget_keeps_live_tracks() -> None:
    """``forget({live_id})`` retains state for live tracks."""
    det = ZoneTransitionDetector(_single_zone_registry())
    det.update(track_id=1, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    det.update(track_id=2, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    # Forget track 2 only; track 1 stays.
    det.forget({1})

    # Track 1 still has state → no re-enter.
    e1 = det.update(track_id=1, cls="person", xy_m=(1.5, 1.5), ts=2.0)
    assert e1 == []

    # Track 2 was forgotten → re-enter fires.
    e2 = det.update(track_id=2, cls="person", xy_m=(1.0, 1.0), ts=2.0)
    assert len(e2) == 1 and e2[0].direction == "enter"


def test_forget_with_all_live_keeps_all_state() -> None:
    """Calling ``forget`` with the full live set is a no-op."""
    det = ZoneTransitionDetector(_single_zone_registry())
    det.update(track_id=5, cls="person", xy_m=(1.0, 1.0), ts=1.0)
    det.forget({5})  # keep track 5

    events = det.update(track_id=5, cls="person", xy_m=(1.5, 1.5), ts=2.0)
    assert events == []   # still inside → no event
