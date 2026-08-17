"""Temporal stabilisation of étagère cell states.

Mirrors ``OccupancyStabilizer`` (backbone/homography/pallet_occupancy.py) but
keyed per (zone_id, r, c) over the two-state alphabet filled/empty: the first
observation sets a cell immediately; a held state flips only when the
challenger wins ≥ ``flip_ratio`` of the vote window. ``unknown`` never votes
(a hand over the bin must not flip it) but a cell with no vote for
``unknown_after_s`` decays to ``unknown`` (fail honestly). ``update`` returns
a message only on change or heartbeat — retained MQTT + heartbeat is the
same late-joiner hygiene zone_state uses.

Establishing a held state (the very first observation, or re-establishing
after an ``unknown`` decay) seeds the vote history with ``window`` copies of
that state rather than a single entry. Without this, ``len(hist)`` ramps up
from 1 to ``window`` across the first few votes and ``ceil(flip_ratio *
len(hist))`` ramps up with it — a cell would flip after only 3-4 challenger
votes instead of the intended supermajority of the full window. Seeding
keeps ``len(hist) == window`` from the first vote onward, so the flip
threshold is always ``ceil(flip_ratio * window)``.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage

_VOTING = ("filled", "empty")

_Key = tuple[str, int, int]


class EtagereStateTracker:
    def __init__(self, *, window: int = 15, flip_ratio: float = 0.7,
                 unknown_after_s: float = 5.0, heartbeat_s: float = 5.0) -> None:
        self._window = int(window)
        self._flip = float(flip_ratio)
        self._unknown_after = float(unknown_after_s)
        self._heartbeat = float(heartbeat_s)
        self._hist: dict[_Key, deque] = {}
        self._held: dict[_Key, str] = {}
        self._conf: dict[_Key, float] = {}
        self._last_vote_t: dict[_Key, float] = {}
        self._last_emit_t: dict[str, float] = {}
        self._last_emit_grid: dict[str, tuple[str, ...]] = {}

    def forget_zone(self, zone_id: str) -> None:
        """Drop all per-cell history/held state for a zone (and its emit gate).

        Without this, a stale ``et_1`` history (e.g. a zone deleted then
        redefined at the same id) would keep flip hysteresis from an
        unrelated prior occupancy pattern."""
        for d in (self._hist, self._held, self._conf, self._last_vote_t):
            for key in [k for k in d if k[0] == zone_id]:
                d.pop(key, None)
        self._last_emit_t.pop(zone_id, None)
        self._last_emit_grid.pop(zone_id, None)

    def _vote(self, key: _Key, state: str, conf: float, now: float) -> str:
        held = self._held.get(key)
        if state in _VOTING:
            self._last_vote_t[key] = now
            if held not in _VOTING:
                # First vote (ever, or after an unknown decay): establish
                # immediately, and seed the window so the NEXT flip needs a
                # real supermajority rather than one still ramping up.
                self._hist[key] = deque([state] * self._window, maxlen=self._window)
                new = state
            else:
                hist = self._hist.setdefault(key, deque(maxlen=self._window))
                hist.append(state)
                counts = Counter(hist)
                challenger = "empty" if held == "filled" else "filled"
                need = math.ceil(self._flip * len(hist))
                new = challenger if counts.get(challenger, 0) >= need else held
            self._held[key] = new
            if new == state:
                self._conf[key] = conf
            return new
        # unknown observation never votes: hold the current state unless
        # it has gone stale (no real vote for unknown_after_s).
        last = self._last_vote_t.get(key)
        if held in _VOTING and last is not None and now - last <= self._unknown_after:
            return held
        self._held[key] = "unknown"
        self._conf[key] = 0.0
        self._hist.pop(key, None)
        return "unknown"

    def update(self, msg: EtagereStateMessage, now: float | None = None) -> EtagereStateMessage | None:
        t = time.time() if now is None else float(now)
        out_cells = []
        for cell in msg.cells:
            key = (msg.zone_id, cell.r, cell.c)
            state = self._vote(key, cell.state, cell.confidence, t)
            out_cells.append(EtagereCellState(r=cell.r, c=cell.c, state=state,
                                              confidence=self._conf.get(key, 0.0)))
        grid = tuple(c.state for c in out_cells)
        last_t = self._last_emit_t.get(msg.zone_id)
        changed = grid != self._last_emit_grid.get(msg.zone_id)
        due = last_t is None or (t - last_t) >= self._heartbeat
        if not (changed or due):
            return None
        self._last_emit_t[msg.zone_id] = t
        self._last_emit_grid[msg.zone_id] = grid
        return EtagereStateMessage(
            ts=msg.ts, camera_id=msg.camera_id, zone_id=msg.zone_id, name=msg.name,
            rows=msg.rows, cols=msg.cols, cells=tuple(out_cells), seq=msg.seq,
            producer_id=msg.producer_id, config_fingerprint=msg.config_fingerprint,
            stabilized=True,
        )
