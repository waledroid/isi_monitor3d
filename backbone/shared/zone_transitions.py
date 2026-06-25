"""Zone entry/leave event detection for Track2D positions.

Concrete, single-implementation utility (not an ABC — same precedent as
``FootProjector``, ``CrossCamFusion``, etc.).

A ``PassingEvent`` is emitted whenever a tracked object crosses a zone
boundary: ``"enter"`` when it moves into a zone it wasn't in before,
``"leave"`` when it moves out of one it was in.

**Identity-loss policy:** when a track disappears (``forget`` call), its
state is silently dropped. We do **not** synthesise "leave" events because:

1. The track may have been lost due to occlusion, not a real exit.
2. Downstream consumers subscribe to enter/leave to trigger alerts; a
   spurious "leave" on ID loss would misfire.
3. If the track re-appears at the same position, it will re-enter on the
   next ``update`` call (correct behaviour for a new or recovered track).
"""

from __future__ import annotations

from dataclasses import dataclass

from backbone.shared.zones import ZoneRegistry


@dataclass(frozen=True)
class PassingEvent:
    """One zone-boundary crossing by a tracked object."""

    track_id: int
    cls: str
    zone: str
    direction: str   # "enter" | "leave"
    ts: float


class ZoneTransitionDetector:
    """Detect enter/leave events from a stream of (track_id, position) updates.

    Args:
        zones: The ``ZoneRegistry`` defining all known floor zones.
    """

    def __init__(self, zones: ZoneRegistry) -> None:
        self._zones = zones
        self._prev: dict[int, frozenset[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        cls: str,
        xy_m: tuple[float, float],
        ts: float,
    ) -> list[PassingEvent]:
        """Compute zone membership for ``xy_m`` and diff against the previous
        set for this track.

        Returns a list of ``PassingEvent``s — "enter" for each new zone, "leave"
        for each exited zone. Returns an empty list if membership is unchanged.

        Args:
            track_id: Unique track identifier (from the homography tracker).
            cls:      Object class string (e.g. "person", "palette").
            xy_m:     Floor-plane position in meters ``(X, Y)``.
            ts:       Capture timestamp for the event.
        """
        now: frozenset[str] = frozenset(self._zones.which(xy_m))
        before: frozenset[str] = self._prev.get(track_id, frozenset())
        self._prev[track_id] = now

        events: list[PassingEvent] = []
        for zone in sorted(now - before):   # sorted for deterministic test ordering
            events.append(PassingEvent(track_id=track_id, cls=cls, zone=zone,
                                       direction="enter", ts=ts))
        for zone in sorted(before - now):
            events.append(PassingEvent(track_id=track_id, cls=cls, zone=zone,
                                       direction="leave", ts=ts))
        return events

    def forget(self, live_ids: set[int]) -> None:
        """Drop state for tracks that are no longer live.

        Vanished tracks are identity-loss events, **not** physical zone exits —
        no "leave" events are synthesised. The dropped state means that if a
        track re-appears (or a new track happens to reuse the ID) it will fire
        "enter" events fresh, which is the correct behaviour.

        Args:
            live_ids: Set of ``track_id``s that are still active this frame.
        """
        dead = [tid for tid in self._prev if tid not in live_ids]
        for tid in dead:
            del self._prev[tid]
