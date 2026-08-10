// Metric floor zones (zones.yaml) — authored by clicking points on a CAMERA.
//
// These are the shared, world-space zones the Backbone consumes (passings,
// zone-state MQTT topics, `in_zone` subscriptions) — distinct from the
// per-camera pixel zone-patches (zone_patch.js). Because they are stored in
// floor metres, every zone automatically renders in BOTH camera views
// (server-drawn on the cam views) and on the 3D map, regardless of which
// camera it was drawn from.
//
// Flow: Settings ▸ Zones ▸ "+ Draw floor zone on camera" → pick cam_a/cam_b
// (calibration-gated picker) → click ≥3 floor points on the live view (each
// click is projected pixel→world via /api/project/pixel-to-floor) → the zone
// persists immediately through POST /api/config {cameras, zones} and the
// Settings modal reopens on the Zones tab. Edits (name/kind/severity) and
// deletes persist the same way. The Backbone applies zones on next START.
//
// Raised zones (platforms/shelves): clicks are decoded ON THE ZONE'S PLANE
// (z_base_m — the ray/plane intersection, `z_m` on the endpoint), not the
// floor. A click on a platform edge decoded at z=0 stores the platform's
// displaced floor SHADOW, which is why each row has a redraw button: set the
// base height first, then redraw by clicking the platform's edges — the same
// clicks then store the true footprint. New zones decode at z=0 (their
// height isn't known yet); for a raised zone, set the height and redraw.

import { promptZoneName, startDraw } from "/static/js/draw_mode.js";
import { openPicker } from "/static/js/draw_target_picker.js";

function t(key, fallback) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fallback;
}

function el(id) { return document.getElementById(id); }

const KINDS = ["palette", "etagere", "danger"];
const SEVERITIES = ["info", "warning", "critical"];

let zones = [];        // [{id, name, kind, type, severity, polygon: [[X,Y] m], z_base_m}]
let maxZones = 6;
let loaded = false;    // guard: never persist a list we never loaded

// Immutable per-zone id, generated ONCE at creation. External systems key on
// it, so it must survive renames + never be reused after a delete.
function newZoneId() {
  return `fz_${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}`;
}

// Next UNUSED default label ("Zone N"): max existing index + 1, never
// `length + 1` (which reuses a freed number and retargets name-keyed consumers).
function nextDefaultName() {
  let max = 0;
  for (const z of zones) {
    const m = /^Zone (\d+)$/.exec(z.name || "");
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `Zone ${max + 1}`;
}

// Called by zone_manager.open() with the fresh GET /api/config body.
export function loadFloorZones(configData) {
  zones = (configData && Array.isArray(configData.zones)) ? configData.zones.map((z) => ({
    // Preserve the immutable id; mint one for a legacy zone that lacks it so a
    // later rename can't shift its identity (persisted on the next save).
    id: z.id || newZoneId(),
    name: z.name || "",
    kind: KINDS.includes(z.kind) ? z.kind : "palette",
    type: z.type || z.kind || "palette",
    severity: SEVERITIES.includes(z.severity) ? z.severity : "info",
    polygon: z.polygon || [],
    // Height of the zone's plane above the floor (m); absent/invalid → 0.
    z_base_m: Number.isFinite(+z.z_base_m) ? Math.max(0, Math.min(5, +z.z_base_m)) : 0,
  })) : [];
  maxZones = (configData && configData.max_zones) || 6;
  loaded = !!configData;
  renderList();
}

// Persist the current zone list. POST /api/config requires `cameras`, so the
// current camera config is round-tripped from a fresh GET (same mapping the
// Settings Save uses: v4l2 → {device}, else {url}).
async function persist() {
  if (!loaded) return false;
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`GET /api/config ${res.status}`);
    const cfg = await res.json();
    const cameras = {};
    for (const [id, c] of Object.entries(cfg.cameras || {})) {
      if (c.name === "v4l2" && c.device) cameras[id] = { device: c.device };
      else if (c.url) cameras[id] = { url: c.url };
    }
    const post = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cameras, zones }),
    });
    if (!post.ok) throw new Error(await post.text());
    // Refresh every consumer: per-cam projections (server-side listeners for
    // config:saved), the 3D map's zone prisms, and the WS video subscriptions.
    document.dispatchEvent(new CustomEvent("config:saved"));
    window.__floor_map?.reloadZones?.();
    return true;
  } catch (err) {
    console.warn("floor_zones: save failed", err);
    window.alert(`${t("save_failed", "Save failed")}: ${err}`);
    return false;
  }
}

function reopenSettingsOnZonesTab() {
  el("btn-add-zone")?.click();   // zone_manager.open()
  // open() resets to the first tab — jump back to Zones once the modal is up.
  setTimeout(() => {
    document.querySelector('.settings-tab-btn[data-tab="zones"]')?.click();
  }, 120);
}

// Shared picker → draw plumbing. `zM` is the decode plane (0 = floor);
// `onPoints` receives the ≥3 world points and persists them.
function pickAndDraw({ zM, label, onPoints }) {
  // The cam view must be visible + clickable: hide the Settings overlay first.
  el("zone-manager")?.classList.add("hidden");
  openPicker({
    onPick: (cam) => {
      const store = window.Alpine?.store?.("bigPanel");
      if (store && store.view !== cam) store.select(cam);
      setTimeout(() => startDraw({
        target: cam,
        mode: "project",     // clicks → world metres on the plane z = zM
        zM,
        label,
        minPoints: 3,
        onDone: async (worldPoints) => {
          if (Array.isArray(worldPoints) && worldPoints.length >= 3) {
            await onPoints(worldPoints);
          }
          reopenSettingsOnZonesTab();
        },
        onCancel: reopenSettingsOnZonesTab,
      }), 200);   // let the cam view mount before attaching the draw layer
    },
    onCancel: reopenSettingsOnZonesTab,
  });
}

function drawNewZone() {
  if (zones.length >= maxZones) {
    window.alert(t("floor_zones_max", `Maximum of ${maxZones} floor zones reached. Delete one first.`));
    return;
  }
  pickAndDraw({
    zM: 0,   // a new zone's height isn't known yet — raised zones: set the
             // base height on the row, then redraw (plane-aware decode)
    label: t("floor_zone_draw_hint", "Floor zone — click ≥3 points on the floor, then Done"),
    onPoints: async (worldPoints) => {
      const name = promptZoneName(zones.map((z) => z.name), nextDefaultName());
      if (!name) return;
      zones.push({
        id: newZoneId(),
        name,
        kind: "palette",
        type: "palette",
        severity: "info",
        polygon: worldPoints,
        z_base_m: 0,
      });
      await persist();
    },
  });
}

// Re-click an existing zone's outline, decoding at ITS base height — the fix
// for a platform zone originally drawn with the floor decode (its stored
// polygon is the platform's displaced floor shadow, not its footprint).
function redrawZone(zone) {
  const h = zone.z_base_m || 0;
  const hint = h > 0
    ? t("floor_zone_redraw_hint_plane", "Click ≥3 edge points ON the raised surface")
      + ` (z = ${h.toFixed(2)} m)`
    : t("floor_zone_draw_hint", "Floor zone — click ≥3 points on the floor, then Done");
  pickAndDraw({
    zM: h,
    label: `${zone.name} — ${hint}`,
    onPoints: async (worldPoints) => {
      zone.polygon = worldPoints;
      await persist();
    },
  });
}

function buildRow(zone, idx) {
  const row = document.createElement("div");
  // zm-zone-row: floor-zone rows carry one extra control (base height) vs the
  // shared 5-column .config-link-row grid — scoped 6-column override in CSS.
  row.className = "config-link-row zm-zone-row";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "zm-input";
  nameInput.value = zone.name;
  nameInput.addEventListener("change", () => {
    const val = nameInput.value.trim();
    const taken = zones.filter((z) => z.id !== zone.id)
      .map((z) => String(z.name || "").trim().toLowerCase());
    if (!val || taken.includes(val.toLowerCase())) {
      window.alert(!val
        ? "Zone name cannot be empty."
        : `"${val}" is already used by another zone — names must be unique.`);
      nameInput.value = zone.name;
      return;
    }
    zone.name = val;
    persist();
  });

  const kindSel = document.createElement("select");
  for (const k of KINDS) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k === "etagere" ? "étagère" : k;
    kindSel.appendChild(o);
  }
  kindSel.value = zone.kind;
  kindSel.addEventListener("change", () => {
    zone.kind = kindSel.value;
    zone.type = kindSel.value;
    sevSel.style.visibility = zone.kind === "danger" ? "visible" : "hidden";
    persist();
  });

  const sevSel = document.createElement("select");
  for (const s of SEVERITIES) {
    const o = document.createElement("option");
    o.value = s;
    o.textContent = s;
    sevSel.appendChild(o);
  }
  sevSel.value = zone.severity;
  sevSel.style.visibility = zone.kind === "danger" ? "visible" : "hidden";
  sevSel.addEventListener("change", () => { zone.severity = sevSel.value; persist(); });

  // Base height (m) of the zone's plane — 0 = floor; platforms/shelves > 0.
  // Persisted as z_base_m; the Backbone projects the zone at this plane.
  const baseInput = document.createElement("input");
  baseInput.type = "number";
  baseInput.className = "zm-input zm-zone-base";
  baseInput.step = "0.01";
  baseInput.min = "0";
  baseInput.max = "5";
  baseInput.value = zone.z_base_m || 0;
  baseInput.title = t("zone_base_height", "Base height (m)");
  baseInput.setAttribute("aria-label", t("zone_base_height", "Base height (m)"));
  baseInput.addEventListener("change", () => {
    const v = parseFloat(baseInput.value);
    if (!Number.isFinite(v)) {
      // Unparseable/empty input restores the PREVIOUS value rather than
      // silently persisting 0 — an operator's fat-fingered edit must not
      // quietly reset a platform zone back to the floor.
      baseInput.value = zone.z_base_m || 0;
      return;
    }
    const clamped = Math.max(0, Math.min(5, v));
    baseInput.value = clamped;
    zone.z_base_m = clamped;
    persist();
  });
  // Visible small label + tooltip so the bare number isn't cryptic in the row.
  const baseField = document.createElement("label");
  baseField.className = "zm-zone-base-field";
  baseField.title = t("zone_base_height", "Base height (m)");
  const baseLabel = document.createElement("span");
  baseLabel.textContent = t("zone_base_height", "Base height (m)");
  baseField.appendChild(baseLabel);
  baseField.appendChild(baseInput);

  const meta = document.createElement("span");
  meta.className = "layout-hint";
  meta.textContent = `${zone.polygon.length} pts`;

  // Redraw: re-click the outline decoded at the zone's CURRENT base height
  // (plane-aware) — required after raising a zone that was drawn on the floor.
  const redraw = document.createElement("button");
  redraw.type = "button";
  redraw.className = "glass-btn zm-iconbtn";
  redraw.title = t("floor_zone_redraw", "Redraw on camera (clicks decode at the base height)");
  redraw.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>';
  redraw.addEventListener("click", () => redrawZone(zone));

  const del = document.createElement("button");
  del.type = "button";
  del.className = "glass-btn zm-iconbtn";
  del.title = t("clear", "Delete");
  del.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
  del.addEventListener("click", async () => {
    zones.splice(idx, 1);
    if (await persist()) renderList();
  });

  row.appendChild(nameInput);
  row.appendChild(kindSel);
  row.appendChild(sevSel);
  row.appendChild(baseField);
  row.appendChild(meta);
  row.appendChild(redraw);
  row.appendChild(del);
  return row;
}

function renderList() {
  const host = el("zm-floor-zones");
  if (!host) return;
  host.innerHTML = "";
  if (!zones.length) {
    const p = document.createElement("p");
    p.className = "layout-hint";
    p.textContent = t("floor_zones_empty",
      "No floor zones yet — draw one on a camera; it appears in both cam views and on the map.");
    host.appendChild(p);
    return;
  }
  zones.forEach((z, i) => host.appendChild(buildRow(z, i)));
}

function wire() {
  el("zm-floor-zone-add")?.addEventListener("click", drawNewZone);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
