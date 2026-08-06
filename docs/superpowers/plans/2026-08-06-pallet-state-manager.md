# PalletStateManager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One backbone class deciding every zone-communication state from per-camera detection evidence (any camera's positive detection is proof), published as an optional `decision` object on `ZoneStateMessage`, rendered dumbly everywhere.

**Architecture:** `PalletStateManager` composes existing pieces — `FootProjector` (per-camera floor projection), `PalletOccupancy.frame_states` (per-camera occupancy estimators), `ZoneRegistry` polygons (±0.15 m tol), presence hysteresis (2-in/15-out, `ZoneMembershipHysteresis` reused with string keys or a sibling) — into per-zone `ZoneDecision`s. Orchestrator feeds it per step and attaches decisions to zone-state publishes. isicomms stores/renders it; `comms_nodes.js` maps enum→i18n with the old heuristic as fallback.

**Tech Stack:** Python 3.10, numpy, pydantic v2 (schemas), pytest hermetic; JS render-only change.

**Spec:** `docs/superpowers/specs/2026-08-06-pallet-state-manager-design.md` (binding: decisions 1-7 + rollout ordering).

## Global Constraints

- Python 3.10 only; run everything in monitor3d env; full repo suite + isicomms + monitor_web suites green at每 task end; ruff clean on changed files.
- Presence = detection evidence OR-ed across cameras (zone polygon ±0.15 m on the detection's projected floor point). Tracks are NEVER consulted for presence/occupancy/count.
- Count per class = MAX across cameras, never sum.
- Enum values exactly: `no_data`, `no_palette`, `palette_empty`, `palette_loaded`.
- `ZoneStateMessage.decision` is OPTIONAL (`None` default) — messages without it must parse everywhere (mixed-version tolerance).
- Presence hysteresis: appear after 2 consecutive evidence frames, disappear after 15 absent frames; per (zone, cls).
- Renderers must not decide: JS/isicomms map the enum; fallback to the old heuristic only when `decision` absent.
- Commit per task, only that task's files, `git commit --no-verify`, trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01CF51vq87B9hoGAkRkzWjf7`

---

### Task 1: `PalletStateManager` core (backbone) — TDD

**Files:** Create `backbone/homography/pallet_state_manager.py`, `tests/test_pallet_state_manager.py`; export from `backbone/homography/__init__.py`.

**Interfaces (produced):**
```python
@dataclass(frozen=True)
class ZoneDecision:
    zone_id: str
    zone_name: str
    palette_state: str          # no_data|no_palette|palette_empty|palette_loaded
    content: tuple[str, ...]    # e.g. ("carton",) when loaded
    counts: dict[str, int]      # per detected class, max-across-cameras

class PalletStateManager:
    def __init__(self, zones: ZoneRegistry, projector: FootProjector,
                 occupancy: PalletOccupancy, *, tol_m: float = 0.15,
                 enter_after: int = 2, exit_after: int = 15) -> None: ...
    def step(self, detections_by_camera: Mapping[str, list[Detection]]) -> list[ZoneDecision]: ...
```
Implementation notes: project each detection per camera (`projector.project`, guard exceptions → skip det); zone bucket via polygon containment with tol (use/extend the same tolerant containment `build_zone_membership_filter` relies on — read `backbone/shared/zones.py` `Zone.contains` and the tol mechanics in `zone_scope.build_zone_membership_filter`, reuse rather than reinvent); per camera per zone per cls counts; presence evidence = count>0; hysteresis per (zone_id, cls) using `ZoneMembershipHysteresis` with `(hash-key)` string zone ids — it already takes arbitrary track ids; keys can be `f"{zone_id}:{cls}"` with a single pseudo-zone, or instantiate one hysteresis per cls — implementer's choice, tested behavior is what counts. Occupancy: call `occupancy.frame_states(dets)` per camera, bucket each pallet state into its zone by the returned pallet floor pos; full-wins across cameras per zone; zone-keyed `OccupancyStabilizer` (string keys — verify it accepts them; its dict keys are opaque). `no_data` only when the manager received no detections at all for ≥1 full absence window AND no presence held (keep simple: zones with no held presence and no evidence this step AND palette hysteresis empty → `no_palette`; `no_data` reserved for "manager never stepped" — return it only from a `initial()` helper used before the first step).

**Tests (all hermetic, synthetic Detections + `_FakeProjector` pattern from `tests/test_pallet_occupancy.py`):**
1. `test_single_camera_evidence_is_proof` — palette det in zone from cam_b only ⇒ `palette_empty` (the live regression).
2. `test_presence_survives_one_camera_losing_it` — cam_a evidence disappears, cam_b persists ⇒ state unchanged every frame.
3. `test_full_wins_across_cameras_for_occupancy` — cam_a says empty, cam_b's estimators say full+carton ⇒ `palette_loaded`, content ("carton",).
4. `test_count_is_max_not_sum` — same zone: cam_a 2 palettes, cam_b 1 ⇒ counts["palette"] == 2.
5. `test_presence_hysteresis_two_in_fifteen_out` — 1 evidence frame ⇒ not yet present; 2 ⇒ present; then absent 14 frames ⇒ still present; 15th ⇒ `no_palette`.
6. `test_carton_alone_no_palette` — carton evidence, no palette ⇒ `no_palette` with counts["carton"]==1.
7. `test_tolerance_catches_edge_detection` — det 0.1 m outside polygon ⇒ still bucketed (tol 0.15).
8. `test_decision_is_stable_over_repeated_identical_frames` — 30 identical frames ⇒ one distinct decision tuple.

TDD: write tests → RED → implement → GREEN → `pytest -q` root suite green → ruff → commit.

### Task 2: Wire schema + isicomms — TDD

**Files:** Modify `backbone/comms/schemas.py` (add `ZoneDecisionModel` pydantic + optional `decision` on `ZoneStateMessage`); `isicomms/isicomms/mqtt_subscriber.py` (store last decision per zone on NodeState); `isicomms/isicomms/api/routes_zones.py` + `api/ui_page.py` (expose/render `palette_state` when present); tests in `tests/test_metadata_schemas.py`, `isicomms/tests/`.

**Interfaces:** `ZoneDecisionModel(palette_state: str, content: tuple[str, ...] = (), counts: dict[str, int] = {})`; `ZoneStateMessage.decision: ZoneDecisionModel | None = None`.

**Tests:** schema round-trip with/without decision (old-style payload parses; new payload parses; `extra="forbid"` semantics preserved); isicomms subscriber stores decision (extend existing zone-state test fixture); `/zones` REST includes `palette_state` when present; `/ui` zones table renders it (string smoke). All three suites green.

### Task 3: Orchestrator wiring + renderers — TDD

**Files:** `backbone/runtime/orchestrator.py` (build manager in `_build` next to `self._occupancy`; call `manager.step(detections_by_camera)` in `step()` where zone states are assembled; attach the matching `ZoneDecision` to each published `ZoneStateMessage`; publish a zone state when its decision CHANGES even if occupants didn't — extend the change signature in `backbone/shared/zone_state.py` with the decision tuple); `monitor_web/monitor_web/static/js/comms_nodes.js` (`_paletteLine` uses `st.decision.palette_state` when present → i18n keys `zone_no_palette`/`zone_palette_empty`/`zone_palette_with` + content join; fallback to old heuristic when absent); i18n en/fr keys if any new needed (reuse existing three).

**Tests:** e2e in `tests/test_orchestrator.py` — stepped pallet+carton pair produces a zone_state whose `decision.palette_state == "palette_loaded"` (scan like the existing occupied-zone-state test); zone_state change-detection test for decision-only change; `node --check` comms_nodes.js; monitor_web suite green.

### Task 4 (controller, not a subagent): rollout + live verify

1. Rebuild gateway FIRST: `docker compose -p on-prem build gateway && docker compose -p on-prem up -d gateway` (schema with optional field must be deployed before the backbone emits it).
2. Restart backbone (dashboard control API stop → start ×2).
3. Live watch (existing `shake_watch.py` style): Sortie_1 ⇒ `palette_loaded`/carton stable; Sortie_2 ⇒ `palette_empty`-or-loaded per reality; dashboard card French text matches; `/ui` zones table shows the state.

## Verification commands
- `conda run -n monitor3d pytest -q` (root), `cd isicomms && pytest -q`, `cd monitor_web && pytest -q`, ruff on changed files, `node --check` for JS.
