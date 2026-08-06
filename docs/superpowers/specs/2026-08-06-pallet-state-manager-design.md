# PalletStateManager — one decision path for zone communication

**Date:** 2026-08-06
**Goal:** consolidate every "what do we tell the AGV about this zone" decision
into ONE backbone class, decided from per-camera DETECTION EVIDENCE (any
camera's positive detection is proof), with tracks demoted to enrichment.

## Problem

Communication decisions are scattered: per-camera occupancy in
`pallet_occupancy.py`, zone occupant lists (track-based) in `zone_state.py`,
membership debounce in the orchestrator, and the final French sentence in
`comms_nodes.js`. The track-dependent chain kept failing (fusion state,
membership flap, parallax wander) while the per-camera zone-filtered
detections were correct the whole time — live 2026-08-06: both cameras
detect palettes in both zones (0.95-0.98, zone-filter PASS) yet the panel
said "Aucune palette disponible".

## Decisions (approved 2026-08-06)

1. New class `PalletStateManager` (`backbone/homography/pallet_state_manager.py`)
   owns ALL zone communication decisions. Everything else renders its output.
2. **Presence = detection evidence, OR across cameras.** A zone contains a
   palette if ANY camera has a palette detection whose floor projection lies
   in the zone polygon (tol ±0.15 m). Same per class for carton/polybag.
   No dependency on tracks/fusion/membership.
3. **Occupancy** reuses the existing estimators (`PalletOccupancy`
   `frame_states` per camera) bucketed into zones by pallet floor position;
   cross-camera **full-wins**; `OccupancyStabilizer` re-keyed by **zone id**
   (string key) instead of pallet track_id.
4. **Count per class** = `max` across cameras of that camera's in-zone count
   (never sum — the same object seen twice must not double-count).
5. **decide(zone) → state enum** on the wire:
   `no_data | no_palette | palette_empty | palette_loaded` (+ `content`
   list + `counts` per class). Presence hysteresis (appear after 2
   consecutive evidence frames, disappear after 15 absent frames — the
   `ZoneMembershipHysteresis` pattern, generalized or reused with string
   "track" keys) so the enum cannot flap.
6. **Wire:** `ZoneStateMessage` gains an OPTIONAL `decision` object
   (pydantic model: `palette_state`, `content: list[str]`,
   `counts: dict[str, int]`). Occupants stay as today (enrichment).
7. **Renderers become dumb:** `comms_nodes.js` `_paletteLine` maps the enum
   to i18n text when `decision` is present (falls back to the old
   object-list heuristic when absent — mixed-version tolerance); isicomms
   `/ui` zones table shows `decision.palette_state` when present.

## Rollout ordering (CRITICAL)

The dockerized isicomms gateway parses with `extra="forbid"` — it must be
REBUILT (with the updated schema) **before** the backbone restarts and
starts publishing the new field, or it will reject every zone state.
Order: commit → rebuild gateway (`docker compose -p on-prem build gateway &&
up -d gateway`) → restart backbone (STOP/START).

## Testing

- Unit (new `tests/test_pallet_state_manager.py`): cam_b-only palette ⇒
  present (the live regression); full-wins occupancy across cameras;
  count=max; presence hysteresis (2-in/15-out); enum for all four states;
  zone bucketing tol.
- Schema round-trip with and without `decision` (old parsers must still
  accept messages WITHOUT it; new parser accepts both).
- isicomms: subscriber stores decision; /ui renders it (string smoke).
- monitor_web: `_paletteLine`-equivalent JS is render-only (node --check;
  existing suite green).
- e2e (`test_orchestrator.py`): a stepped pair with a palette+carton yields
  a zone_state carrying `decision.palette_state == "palette_loaded"`.

## Out of scope

- Removing the old track-based occupants list (stays as enrichment).
- passings/transitions (already debounced separately).
- Any RTSP/tracking investigation (presence no longer depends on it).
