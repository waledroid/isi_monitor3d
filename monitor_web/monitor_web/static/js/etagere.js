// Étagère (bin-rack) zones — Settings editor (4 corners → auto-split →
// drag-adjust), the data source `live_overlay.js` reads to draw cell
// outlines + live fill state on the cam views, and the poller that feeds
// the COMMUNICATION matrix widget (comms_nodes.js). Authoring only: NO
// inference runs here — occupancy comes from GET /api/etagere/state.
//
// Cam picking reuses the SAME plumbing as floor_zones.js's `pickAndDraw`
// (draw_target_picker.js's `openPicker`, hiding #zone-manager so the cam
// view underneath is visible/clickable, selecting it via the bigPanel
// Alpine store) rather than inventing a new cam-choice mechanism. Corner
// drag-adjust is not a click-collect flow like startDraw's other modes, so
// it drives the shared #draw-toolbar's Done/Cancel buttons directly.
import { startDraw } from "/static/js/draw_mode.js";
import { openPicker } from "/static/js/draw_target_picker.js";
import { applyDrag, hitTest } from "/static/js/etagere_geom.js";

export { applyDrag, hitTest };

const DEFAULT_ROWS = 3;
const DEFAULT_COLS = 3;
const STATE_POLL_MS = 1000;

let cfg = { model: null, zones: [] };
let states = {};        // zone_id -> {name, camera_id, rows, cols, matrix, cells, ts}
let editing = null;     // {zoneId} while the drag-adjust overlay is active
let stopEditing = null; // teardown for the CURRENT drag-adjust session, if any

export function getZones(camId) {
  return (cfg.zones || []).filter((z) => !camId || z.camera === camId);
}
export function getStates() { return states; }
export function isEditing(zoneId) { return !!editing && editing.zoneId === zoneId; }

function el(id) { return document.getElementById(id); }

async function load() {
  try {
    const r = await fetch("/api/etagere");
    if (r.ok) cfg = await r.json();
  } catch (e) {
    console.warn("etagere: load failed", e);
  }
  cfg.zones = cfg.zones || [];
  renderSettingsList();
}

async function save() {
  try {
    const r = await fetch("/api/etagere", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    if (!r.ok) window.alert("Étagère save failed: " + (await r.text()));
  } catch (e) {
    window.alert("Étagère save failed: " + e);
  }
  renderSettingsList();
}

async function pollStates() {
  try {
    const r = await fetch("/api/etagere/state");
    if (r.ok) states = (await r.json()).states || {};
  } catch (_) { /* keep the last snapshot on a network blip */ }
}

function nextDefaultName() {
  let max = 0;
  for (const z of cfg.zones || []) {
    const m = /^Étagère (\d+)$/.exec(z.name || "");
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `Étagère ${max + 1}`;
}

// Same convention as floor_zones.js: hide the Settings overlay so the cam
// view underneath is visible/clickable, then reopen it (back on the Zones
// tab) once the cam-side interaction is done.
function hideSettings() { el("zone-manager")?.classList.add("hidden"); }
function reopenSettingsOnZonesTab() {
  el("btn-add-zone")?.click();   // zone_manager.open() — resets to tab 1
  setTimeout(() => {
    document.querySelector('.settings-tab-btn[data-tab="zones"]')?.click();
  }, 120);
}
function selectCam(cam) {
  const store = window.Alpine?.store?.("bigPanel");
  if (store && store.view !== cam) store.select(cam);
}

// ---- draw a new étagère: pick a cam → click 4 corners → autosplit → adjust ----
export function startEtagereDraw(camId) {
  hideSettings();
  const beginOn = (cam) => {
    selectCam(cam);
    setTimeout(() => startDraw({
      target: cam,
      mode: "raw",                 // source-frame pixels, no calibration needed
      minPoints: 4,
      maxPoints: 4,
      label: "Étagère — click the rack's 4 outer corners (TL, TR, BR, BL)",
      onDone: async (points) => {
        if (!points || points.length !== 4) { reopenSettingsOnZonesTab(); return; }
        const name = window.prompt("Étagère name", nextDefaultName());
        if (!name) { reopenSettingsOnZonesTab(); return; }
        const img = el(`${cam}-img`);
        let cells = [];
        try {
          const r = await fetch("/api/etagere/autosplit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ corners: points, rows: DEFAULT_ROWS, cols: DEFAULT_COLS }),
          });
          if (r.ok) cells = (await r.json()).cells || [];
          else window.alert("Auto-split failed: " + (await r.text()));
        } catch (e) {
          window.alert("Auto-split failed: " + e);
        }
        const zone = {
          id: "et_" + Date.now().toString(36),
          name,
          camera: cam,
          frame_wh: [img?.naturalWidth || 0, img?.naturalHeight || 0],
          corners: points,
          rows: DEFAULT_ROWS,
          cols: DEFAULT_COLS,
          cells,
        };
        cfg.zones.push(zone);
        await save();
        startAdjust(zone.id);   // straight into drag-adjust to fine-tune the split
      },
      onCancel: reopenSettingsOnZonesTab,
    }), 200);   // let the cam view mount before attaching the draw layer
  };
  if (camId === "cam_a" || camId === "cam_b") { beginOn(camId); return; }
  openPicker({ onPick: beginOn, onCancel: reopenSettingsOnZonesTab });
}

// ---- drag-adjust an existing étagère's cell corners on the cam overlay ----
// Not a click-collect flow (startDraw's contract), so it wires the SAME
// shared #draw-toolbar DOM directly instead of going through startDraw.
export function startAdjust(zoneId) {
  const zone = (cfg.zones || []).find((z) => z.id === zoneId);
  if (!zone) return;
  if (editing) stopEditing?.();   // only one edit session at a time — tear down the old one first

  hideSettings();
  selectCam(zone.camera);
  editing = { zoneId };

  setTimeout(() => {
    const canvas = el(`${zone.camera}-overlay`);
    if (!canvas) { editing = null; reopenSettingsOnZonesTab(); return; }
    canvas.style.pointerEvents = "auto";

    let drag = null;
    const toSrc = (ev) => window.__displayToSource(canvas, zone.camera, ev.offsetX, ev.offsetY, zone.frame_wh);
    const down = (ev) => {
      const p = toSrc(ev);
      const h = hitTest(zone, p[0], p[1]);
      if (h.cellIdx >= 0) drag = { ...h, last: p };
    };
    const move = (ev) => {
      if (!drag) return;
      const p = toSrc(ev);
      zone.cells[drag.cellIdx].rect = applyDrag(
        zone.cells[drag.cellIdx].rect, drag.handle, p[0] - drag.last[0], p[1] - drag.last[1]);
      drag.last = p;
    };
    const up = () => { drag = null; };
    canvas.addEventListener("mousedown", down);
    canvas.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);

    const bar = el("draw-toolbar");
    const label = el("draw-zone-label");
    const doneBtn = el("draw-done");
    const cancelBtn = el("draw-cancel");
    const countWrap = el("draw-count")?.parentElement;
    const undoBtn = el("draw-undo");
    if (label) label.textContent = `${zone.name} — drag a corner or cell, then Done`;
    if (countWrap) countWrap.style.display = "none";
    if (undoBtn) undoBtn.style.display = "none";
    bar?.classList.remove("hidden");

    const teardown = () => {
      canvas.removeEventListener("mousedown", down);
      canvas.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      canvas.style.pointerEvents = "none";
      if (doneBtn) doneBtn.onclick = null;
      if (cancelBtn) cancelBtn.onclick = null;
      bar?.classList.add("hidden");
      if (countWrap) countWrap.style.display = "";
      if (undoBtn) undoBtn.style.display = "";
      editing = null;
      stopEditing = null;
    };
    stopEditing = teardown;
    if (doneBtn) doneBtn.onclick = async () => { teardown(); await save(); reopenSettingsOnZonesTab(); };
    if (cancelBtn) cancelBtn.onclick = async () => { teardown(); await load(); reopenSettingsOnZonesTab(); };
  }, 200);
}

export function deleteZone(id) {
  cfg.zones = (cfg.zones || []).filter((z) => z.id !== id);
  save();
}

function _esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderSettingsList() {
  const host = el("zm-etagere-list");
  if (!host) return;
  const zones = cfg.zones || [];
  if (!zones.length) {
    host.innerHTML = '<p class="layout-hint" data-i18n="etagere_empty">' +
      "No étagère yet — draw one on a camera.</p>";
    return;
  }
  host.innerHTML = "";
  zones.forEach((z, i) => {
    const camLbl = z.camera === "cam_b" ? "CAM 2" : "CAM 1";
    const row = document.createElement("div");
    row.className = "config-zone-row";
    row.dataset.zoneId = z.id;
    row.innerHTML =
      `<div class="czr-header">` +
        `<span class="config-zone-num" title="Étagère ${i + 1}">${i + 1}</span>` +
        `<span class="zm-name etag-name">${_esc(z.name || "")}</span>` +
        `<span class="czr-cam-badge">${camLbl}</span>` +
        `<button type="button" class="glass-btn zm-iconbtn zm-delete" ` +
          `title="Delete ${_esc(z.name || "")}" aria-label="Delete étagère">` +
          `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" ` +
              `stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">` +
            `<path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>` +
            `<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>` +
        `</button>` +
      `</div>` +
      `<div class="czr-controls">` +
        `<span class="etag-dims">${z.rows}×${z.cols}</span>` +
        `<button type="button" class="glass-btn zm-small-btn" data-adjust>Adjust cells</button>` +
      `</div>`;
    row.querySelector("[data-adjust]").addEventListener("click", () => startAdjust(z.id));
    row.querySelector(".zm-delete").addEventListener("click", () => deleteZone(z.id));
    host.appendChild(row);
  });
}

if (typeof document !== "undefined") {
  const boot = () => {
    load();
    el("zm-etagere-add")?.addEventListener("click", () => startEtagereDraw());
    pollStates();
    setInterval(pollStates, STATE_POLL_MS);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.__etagere = { startEtagereDraw, deleteZone, getZones, getStates, startAdjust };
}
