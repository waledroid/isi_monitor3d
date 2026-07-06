// Metric floor zones (zones.yaml) — authored by clicking points on a CAMERA.
//
// These are the shared, world-space zones the Backbone consumes (passings,
// zone-state MQTT topics, `in_zone` subscriptions) — distinct from the
// per-camera pixel zone-patches (zone_patch.js). Because they are stored in
// floor metres, every zone automatically renders in BOTH camera views
// (camera_zones.js + live_overlay.js) and on the 3D map, regardless of which
// camera it was drawn from.
//
// Flow: Settings ▸ Zones ▸ "+ Draw floor zone on camera" → pick cam_a/cam_b
// (calibration-gated picker) → click ≥3 floor points on the live view (each
// click is projected pixel→world via /api/project/pixel-to-floor) → the zone
// persists immediately through POST /api/config {cameras, zones} and the
// Settings modal reopens on the Zones tab. Edits (name/kind/severity) and
// deletes persist the same way. The Backbone applies zones on next START.

import { startDraw } from "/static/js/draw_mode.js";
import { openPicker } from "/static/js/draw_target_picker.js";

function t(key, fallback) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fallback;
}

function el(id) { return document.getElementById(id); }

const KINDS = ["palette", "etagere", "danger"];
const SEVERITIES = ["info", "warning", "critical"];

let zones = [];        // [{name, kind, type, severity, polygon: [[X,Y] m]}]
let maxZones = 6;
let loaded = false;    // guard: never persist a list we never loaded

// Called by zone_manager.open() with the fresh GET /api/config body.
export function loadFloorZones(configData) {
  zones = (configData && Array.isArray(configData.zones)) ? configData.zones.map((z) => ({
    name: z.name || "",
    kind: KINDS.includes(z.kind) ? z.kind : "palette",
    type: z.type || z.kind || "palette",
    severity: SEVERITIES.includes(z.severity) ? z.severity : "info",
    polygon: z.polygon || [],
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
    // Refresh every consumer: per-cam projections (camera_zones listens for
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

function drawNewZone() {
  if (zones.length >= maxZones) {
    window.alert(t("floor_zones_max", `Maximum of ${maxZones} floor zones reached. Delete one first.`));
    return;
  }
  // The cam view must be visible + clickable: hide the Settings overlay first.
  el("zone-manager")?.classList.add("hidden");
  openPicker({
    onPick: (cam) => {
      const store = window.Alpine?.store?.("bigPanel");
      if (store && store.view !== cam) store.select(cam);
      setTimeout(() => startDraw({
        target: cam,
        mode: "project",     // clicks → world metres via the camera's H
        label: t("floor_zone_draw_hint", "Floor zone — click ≥3 points on the floor, then Done"),
        minPoints: 3,
        onDone: async (worldPoints) => {
          if (Array.isArray(worldPoints) && worldPoints.length >= 3) {
            zones.push({
              name: `Zone ${zones.length + 1}`,
              kind: "palette",
              type: "palette",
              severity: "info",
              polygon: worldPoints,
            });
            await persist();
          }
          reopenSettingsOnZonesTab();
        },
        onCancel: reopenSettingsOnZonesTab,
      }), 200);   // let the cam view mount before attaching the draw layer
    },
    onCancel: reopenSettingsOnZonesTab,
  });
}

function buildRow(zone, idx) {
  const row = document.createElement("div");
  row.className = "config-link-row";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "zm-input";
  nameInput.value = zone.name;
  nameInput.addEventListener("change", () => {
    zone.name = nameInput.value.trim() || zone.name;
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

  const meta = document.createElement("span");
  meta.className = "layout-hint";
  meta.textContent = `${zone.polygon.length} pts`;

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
  row.appendChild(meta);
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
