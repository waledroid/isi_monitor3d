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
import { promptZoneName, startDraw } from "/static/js/draw_mode.js";
import { openPicker } from "/static/js/draw_target_picker.js";
import { applyDrag, frameWhOrNull, hitTest, rotateCorners } from "/static/js/etagere_geom.js";

export { applyDrag, hitTest };

const DEFAULT_ROWS = 3;
const ROTATE_STEP_DEG = 2;    // per ‹ › click
const DEFAULT_COLS = 3;
const STATE_POLL_MS = 1000;

let cfg = { model: null, zones: [] };
let availableModels = [];   // [{path, label, mtime}] from GET /api/etagere
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
  // Read-only picker candidates ride on GET; keep them out of the saved cfg.
  availableModels = Array.isArray(cfg.available_models) ? cfg.available_models : [];
  delete cfg.available_models;
  cfg.zones = cfg.zones || [];
  renderSettingsList();
  renderModelPicker();
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

// Étagères share the "Zone N" numbering + uniqueness with camera zones
// (zone_patch.js owns the helpers; fall back to our own list if it's absent).
function allZoneNames() {
  const zp = window.__zonePatch;
  if (zp?.allZoneNames) return zp.allZoneNames();
  return (cfg.zones || []).map((z) => z.name).filter(Boolean);
}
function nextDefaultName() {
  const zp = window.__zonePatch;
  const names = allZoneNames();
  if (zp?.nextZoneNumberName) return zp.nextZoneNumberName(names);
  let max = 0;
  for (const n of names) { const m = /^Zone (\d+)$/.exec(n || ""); if (m) max = Math.max(max, +m[1]); }
  return `Zone ${max + 1}`;
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
        const name = promptZoneName(allZoneNames(), nextDefaultName());
        if (!name) { reopenSettingsOnZonesTab(); return; }
        const img = el(`${cam}-img`);
        // Read the frame size the SAME way live_overlay.js's overlays do
        // (naturalWidth/Height, falling back to the passthrough player's
        // decoded size when the <img> has no src) — etagere.js must not
        // import live_overlay.js (browser-absolute-path import cycle risk /
        // Node test unimportability), so read its helper off `window` at
        // call time instead. A still-0x0 result means the live view hasn't
        // produced a frame yet; saving that would ZeroDivisionError every
        // crop at inference time (I2), so abort the draw instead.
        const natural = window.__naturalSize
          ? window.__naturalSize(img, cam)
          : [img?.naturalWidth || 0, img?.naturalHeight || 0];
        const frameWh = frameWhOrNull(natural[0], natural[1]);
        if (!frameWh) {
          window.alert("camera frame size unknown — wait for the live view, then draw again");
          reopenSettingsOnZonesTab();
          return;
        }
        // Autosplit is load-bearing: a failed/short response must NOT reach
        // cfg.zones. Pushing a zone with too few/no cells would (a) make the
        // very next save() 422 server-side and (b) leave cfg permanently
        // holding that invalid zone client-side (every later save() 422s
        // too, until a full page reload) — so bail out before touching cfg.
        const expectedCells = DEFAULT_ROWS * DEFAULT_COLS;
        let cells;
        try {
          const r = await fetch("/api/etagere/autosplit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ corners: points, rows: DEFAULT_ROWS, cols: DEFAULT_COLS }),
          });
          if (!r.ok) {
            window.alert("Auto-split failed: " + (await r.text()));
            reopenSettingsOnZonesTab();
            return;
          }
          cells = (await r.json()).cells;
        } catch (e) {
          window.alert("Auto-split failed: " + e);
          reopenSettingsOnZonesTab();
          return;
        }
        if (!Array.isArray(cells) || cells.length !== expectedCells) {
          window.alert(`Auto-split failed: expected ${expectedCells} cells, got ` +
            `${Array.isArray(cells) ? cells.length : 0}.`);
          reopenSettingsOnZonesTab();
          return;
        }
        const zone = {
          id: "et_" + Date.now().toString(36),
          name,
          camera: cam,
          frame_wh: frameWh,
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
  // Étagère drawing is pure image-space (mode:"raw") — it must work on an
  // UNCALIBRATED rig, so the calibration gate other pickers apply is skipped.
  openPicker({ onPick: beginOn, onCancel: reopenSettingsOnZonesTab, ignoreCalibration: true });
}

// ---- drag-adjust an existing étagère's cell corners on the cam overlay ----
// Not a click-collect flow (startDraw's contract), so it wires the SAME
// shared #draw-toolbar DOM directly instead of going through startDraw.
let adjustTimer = null;   // pending "attach after the cam view mounts" timer, if any

export function startAdjust(zoneId) {
  const zone = (cfg.zones || []).find((z) => z.id === zoneId);
  if (!zone) return;
  // Re-entry guard, BOTH cases: (a) a session already attached — tear it
  // down; (b) a PREVIOUS call's 200 ms mount-wait hasn't fired yet — cancel
  // it synchronously so it can never attach after this one (that race would
  // otherwise leak the first session's listeners + stomp its toolbar wiring).
  if (adjustTimer) { clearTimeout(adjustTimer); adjustTimer = null; }
  if (editing) stopEditing?.();

  hideSettings();
  selectCam(zone.camera);
  editing = { zoneId };

  adjustTimer = setTimeout(() => {
    adjustTimer = null;
    const canvas = el(`${zone.camera}-overlay`);
    if (!canvas) { editing = null; reopenSettingsOnZonesTab(); return; }
    canvas.style.pointerEvents = "auto";

    let drag = null;
    // A missing live_overlay.js (load-order hiccup, or this module used
    // outside the dashboard) must never throw inside a mouse handler.
    const toSrc = (ev) => window.__displayToSource
      ? window.__displayToSource(canvas, zone.camera, ev.offsetX, ev.offsetY, zone.frame_wh)
      : null;
    const down = (ev) => {
      const p = toSrc(ev);
      if (!p) return;
      const h = hitTest(zone, p[0], p[1]);
      if (h.cellIdx >= 0) drag = { ...h, last: p };
    };
    const move = (ev) => {
      if (!drag) return;
      const p = toSrc(ev);
      if (!p) return;
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
    if (label) label.textContent = `${zone.name} — rotate ‹ ›, drag a corner or cell, then Done`;
    if (countWrap) countWrap.style.display = "none";
    if (undoBtn) undoBtn.style.display = "none";
    const rotCcw = el("draw-rotate-ccw");
    const rotCw = el("draw-rotate-cw");
    // Rotate the WHOLE grid: spin the outer corners around their centre and
    // re-derive the 9 cells from them (server auto-split). Coarse first, then
    // per-cell drags fine-tune — a rotate after drags re-derives the cells.
    let rotating = false;
    const rotateBy = async (deg) => {
      if (rotating || !Array.isArray(zone.corners) || zone.corners.length !== 4) return;
      rotating = true;
      try {
        const corners = rotateCorners(zone.corners, deg);
        const r = await fetch("/api/etagere/autosplit", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ corners, rows: zone.rows || DEFAULT_ROWS, cols: zone.cols || DEFAULT_COLS }),
        });
        if (!r.ok) { window.alert("Rotate failed: " + (await r.text())); return; }
        const cells = (await r.json()).cells;
        if (!Array.isArray(cells) || !cells.length) return;
        zone.corners = corners;
        zone.cells = cells;
      } catch (e) {
        window.alert("Rotate failed: " + e);
      } finally {
        rotating = false;
      }
    };
    if (rotCcw) { rotCcw.classList.remove("hidden"); rotCcw.onclick = () => rotateBy(-ROTATE_STEP_DEG); }
    if (rotCw)  { rotCw.classList.remove("hidden");  rotCw.onclick  = () => rotateBy(ROTATE_STEP_DEG); }
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
      if (rotCcw) { rotCcw.classList.add("hidden"); rotCcw.onclick = null; }
      if (rotCw)  { rotCw.classList.add("hidden");  rotCw.onclick = null; }
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

// --- Étagère model picker (Settings) --------------------------------------
// The étagère detector is its OWN 2-class model, separate from the object
// model; it lives in etagere.yaml's `model:` block. The dropdown lists the
// same trainer exports the object-model picker offers; the current path is
// kept selectable even when it is not among the scanned candidates.
const DEFAULT_MODEL = { class_names: ["empty_box", "filled_box"], imgsz: 320,
  confidence_threshold: 0.3, crop_margin: 0.08, max_fps: 2.0 };

function renderModelPicker() {
  const sel = el("zm-etagere-model-onnx");
  if (!sel) return;
  const current = cfg.model?.onnx_path || "";
  const seen = new Set();
  const opts = ['<option value="">— none —</option>'];
  for (const m of availableModels) {
    if (!m?.path || seen.has(m.path)) continue;
    seen.add(m.path);
    opts.push(`<option value="${_esc(m.path)}">${_esc(m.label || m.path)}</option>`);
  }
  if (current && !seen.has(current)) {
    opts.push(`<option value="${_esc(current)}">${_esc(current)} (current)</option>`);
  }
  sel.innerHTML = opts.join("");
  sel.value = current;
  const conf = el("zm-etagere-model-conf");
  if (conf) conf.value = cfg.model?.confidence_threshold ?? "";
  const hint = el("zm-etagere-model-imgsz");
  if (hint) hint.textContent = String(cfg.model?.imgsz ?? DEFAULT_MODEL.imgsz);
}

async function applyModel() {
  const path = (el("zm-etagere-model-onnx")?.value || "").trim();
  const confRaw = el("zm-etagere-model-conf")?.value;
  const conf = confRaw === "" || confRaw == null ? null : Number(confRaw);
  if (!path) {
    cfg.model = null;                       // no model ⇒ feature off (isistream skips it)
  } else {
    const base = cfg.model && cfg.model.onnx_path === path ? cfg.model : { ...DEFAULT_MODEL };
    cfg.model = { ...DEFAULT_MODEL, ...base, onnx_path: path };
    if (conf != null && !Number.isNaN(conf)) cfg.model.confidence_threshold = conf;
  }
  await save();                             // server validates + hot-restarts isistream
  renderModelPicker();
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
    el("zm-etagere-model-save")?.addEventListener("click", () => applyModel());
    pollStates();
    setInterval(pollStates, STATE_POLL_MS);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.__etagere = { startEtagereDraw, deleteZone, getZones, getStates, startAdjust };
}
