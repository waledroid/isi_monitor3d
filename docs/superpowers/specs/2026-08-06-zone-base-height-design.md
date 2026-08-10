# Zone base height (z_base_m) — platforms stop lying about geometry

**Date:** 2026-08-06
**Goal:** zones declare the height of the plane they live on; every consumer
that today assumes z=0 uses the zone's plane instead.

## Problem

The whole pipeline assumes one plane (z=0). `sortie_machine_1` is drawn on a
30.4 cm platform: its polygon projects DISPLACED into every camera
(`zone_crop_boxes` projects at z=0), platform objects' floor-projected feet
land 0.5-0.6 m apart between cameras (measured), membership/decisions wobble,
and fusion needed widened thresholds to cope. Floor zones are exact.

## Decisions (approved 2026-08-06)

1. **Schema:** `Zone` gains `z_base_m: float = 0.0` (zones.yaml round-trips
   it; absent ⇒ 0.0 — full backward compat). `kind` stays as-is (colors/
   severity); the height is orthogonal. ZoneSpec config adverts carry it.
2. **UI:** the Settings zone editor gains a numeric "Base height (m)" field
   per zone (default 0; the operator types 0.304 for the platform;
   étagère shelves likewise). i18n en+fr. Persisted via the existing
   zones payload → zones.yaml atomic write.
3. **Geometry helper:** `pixel_to_plane(uv, K, D, R, t, z_m)` in
   `backbone/shared/geometry.py` — undistorted pixel ray ∩ plane z=z_m.
   For z_m=0 it must agree with the existing floor path to <1e-6 m
   (pinned by test). Degenerate rays (parallel to plane / behind camera)
   return None.
4. **Zone projection into cameras:** `zone_crop_boxes` and
   `zone_fill_polygons` project each zone's polygon at ITS `z_base_m`
   (and `z_base_m + crop_height_m` for the headroom variant) instead of 0.
   Mode-1 H-only rigs (no metric extrinsics) keep today's behavior.
5. **Membership & decisions on the zone's plane:** a `ZoneAwareProjector`
   (rig + zones) exposes `position_in_zone(cam_id, foot_uv, zone) ->
   (x, y) | None` projecting the pixel onto THAT zone's plane. Consumers:
   `build_zone_membership_filter` (isistream in-zone guarantee) and
   `PalletStateManager` bucketing test containment per zone using the
   zone's own plane (floor zones: identical to today). Platform objects'
   cross-camera positions then agree to detector noise.
6. **Out of scope (documented limitations):** global track xy_m and
   triangulation stay floor/raw — the Track3D Z badge remains the true
   ray-crossing (platform-object Z accuracy needs better correspondence,
   a future feature); the orchestrator's zone-membership for TRACKS
   (passings) stays floor-based v1.

## Rollout

Code → user sets 0.304 on sortie_machine_1 in Settings → Save →
STOP/START (zones load at boot in backbone + isistream) → verify: crops/
outlines land on the physical platform in BOTH cameras; PSM decisions
stable; cross-cam separation for the platform palette collapses.

## Testing

Hermetic per task: geometry z=0 parity + known-height analytic case +
degenerate rays; schema round-trip incl. absent field; crop-box projection
moves as expected for a raised zone (synthetic rig); membership filter
accepts a platform detection that floor-projection would misplace; PSM
buckets via zone plane; UI payload round-trip persists z_base_m; all
suites + ruff green.
