# Zone Base Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** zones carry `z_base_m`; zone projection into cameras, the in-zone membership guarantee, and PalletStateManager bucketing all use the zone's own plane instead of z=0.

**Architecture:** one geometry primitive (`pixel_to_plane`), schema field on `Zone` (+ `ZoneSpec` advert), plane-aware projection in `zone_scope`, a `ZoneAwareProjector` consumed by the membership filter and PSM, and a Settings-editor numeric field persisted through the existing zones payload.

**Tech Stack:** Python 3.10, numpy/cv2, pytest hermetic; JS zone editor field; i18n en+fr.

**Spec:** `docs/superpowers/specs/2026-08-06-zone-base-height-design.md` (decisions 1-6 binding; out-of-scope list binding too).

## Global Constraints

- Backward compat everywhere: zones.yaml without `z_base_m` ⇒ 0.0; behavior for z_base_m==0 must be BIT-IDENTICAL to today (pinned by parity tests).
- `pixel_to_plane(uv, K, D, R, t, z_m)`: z=0 parity with the existing floor path to <1e-6 m; degenerate rays → None.
- Mode-1 H-only rigs: raised projection paths fall back to current behavior (no crash).
- Tracks/triangulation/passings untouched (spec §6).
- All suites green at each task end (root / isicomms untouched / monitor_web), ruff clean, `node --check` for JS.
- Commit per task, only that task's files, `git commit --no-verify`, trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01CF51vq87B9hoGAkRkzWjf7`

---

### Task 1: geometry primitive + schema

**Files:** `backbone/shared/geometry.py` (+`pixel_to_plane`), `backbone/shared/zones.py` (`Zone.z_base_m: float = 0.0`, load/save round-trip), `backbone/comms/schemas.py` (`ZoneSpec.z_base_m: float = 0.0` — additive), tests in `tests/test_geometry.py`, `tests/test_zones.py`, `tests/test_metadata_schemas.py`.

**Produced:** `pixel_to_plane(uv: np.ndarray, K, D, R, t, z_m: float) -> np.ndarray | None` ((N,2) in → (N,2) world XY on plane z=z_m, or None per-point/whole — read the file's conventions and match `pixel_to_floor`'s shape contract; document the choice). Zone dataclass field + YAML round-trip; ZoneSpec advert field.

**Tests:** z=0 parity vs `undistort+pixel_to_floor` (<1e-6, several pixels incl. edges); analytic raised-plane case (synthetic K=eye-ish, known R,t: a pixel whose ray hits (1,2,0.304) exactly); degenerate ray → None; zones.yaml round-trip with and without the field; ZoneSpec parses without the field (old adverts).

### Task 2: plane-aware zone projection + membership filter

**Files:** `backbone/detection/zone_scope.py` (`zone_crop_boxes`, `zone_fill_polygons` project at `zone.z_base_m` / `+crop_height_m`; `build_zone_membership_filter` uses per-zone plane via the new helper), `backbone/shared/zones.py` if a `ZoneAwareProjector` helper naturally lives there (implementer's judgment — keep ONE home, no duplication), tests in `tests/test_zone_scope.py`.

**Tests:** raised zone's crop box shifts vs z=0 projection on a synthetic metric rig (assert direction/magnitude sanity, not magic pixels); z_base_m=0 zones produce byte-identical boxes to before (parity vs current function output pinned); membership filter accepts a detection whose FLOOR projection lands outside the raised zone but whose PLANE projection lands inside (the live platform case, synthesized); Mode-1 H-only rig → unchanged behavior.

### Task 3: PalletStateManager on the zone plane

**Files:** `backbone/homography/pallet_state_manager.py` (bucketing via per-zone plane projection — accept the rig or a `ZoneAwareProjector` at construction; keep the plain projector for occupancy's existing estimators), `backbone/runtime/orchestrator.py` (pass the new dependency), tests in `tests/test_pallet_state_manager.py`.

**Tests:** platform-zone detection bucketed correctly where floor projection would miss (synthetic); floor zones bit-identical decisions (parity with existing tests — they must all still pass unchanged); camera-loss gate + hysteresis unaffected.

### Task 4: Settings UI + persistence

**Files:** `monitor_web/monitor_web/static/js/zone_manager.js` (+ numeric "Base height (m)" per floor-zone row; collect into the zones payload), `monitor_web/monitor_web/templates/dashboard.html` if markup needed, `monitor_web/monitor_web/api/routes_config.py` (accept/persist `z_base_m` per zone into zones.yaml; validate 0 ≤ z ≤ 5, 422 outside), i18n `en.json`+`fr.json` (`zone_base_height` — "Base height (m)" / "Hauteur de base (m)"), monitor_web tests (payload round-trip persists z_base_m; absent field defaults 0; validation 422).

### Task 5 (controller): rollout + live verify

User sets 0.304 on sortie_machine_1 → Save → STOP/START. Verify: zone outline/crop lands on the physical platform in both cameras (zonefit-style render), PSM decisions stable, cross-cam separation for the platform palette collapses (sep_probe re-run), no regressions on sortie_machine_2.

## Verification commands
Root/isicomms/monitor_web pytest; ruff; node --check.
