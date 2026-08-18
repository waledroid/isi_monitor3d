// Zone-patch ROIs — pixel-space "watch boxes" drawn directly on a camera frame,
// cropped into a detected patch (targeted SAHI) shown in the ZONE panels.
//
// No calibration needed: the box is captured in SOURCE-frame pixels via draw_mode's
// `raw` cam mode, stored in zone_patches.yaml, and the /stream/zone/{id} endpoint
// crops + detects that region. live_overlay.js reads getPatches() to draw the red ROI.

import { promptZoneName, startDraw } from "/static/js/draw_mode.js";

const MAX_PATCHES = 6;   // max zones the operator can create (drawn on CAM)

let patches = [];   // [{id, name, camera, polygon:[[u,v]..], rect:[x0,y0,x1,y1], frame_wh:[W,H], color}]
let panelStreamIds = [];   // /ws/video streams currently attached to the ZONE panels

export function getPatches(camId) {
  return camId ? patches.filter((p) => (p.camera || "cam_a") === camId) : patches.slice();
}

// Operator-authored patches only — twins (server-derived cross-camera copies,
// `twin_of` set) detect and draw like patches but are never edited/POSTed.
function userPatches() {
  return patches.filter((p) => !p.twin_of);
}

// Ghost outlines TARGETED at this camera: patches drawn on the OTHER camera,
// projected through the floor by the server (GET /api/zone-patches `ghost`).
// Empty without Mode-2 calibration or when a patch has no overlap here.
export function getGhosts(camId) {
  const out = [];
  for (const p of patches) {
    const g = p.ghost;
    if (g && g.camera === camId && Array.isArray(g.polygon) && g.polygon.length >= 3) {
      out.push({ ...g, name: p.name || "", color: p.color, from: p.camera || "cam_a" });
    }
  }
  return out;
}

async function load() {
  try {
    const r = await fetch("/api/zone-patches");
    if (r.ok) patches = (await r.json()).patches || [];
  } catch { /* keep what we have on a network blip */ }
  renderPanels();
  renderSettingsList();
  updateDrawButton();
}

async function save() {
  try {
    await fetch("/api/zone-patches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Twins are derived server-side on every save — never sent back.
      body: JSON.stringify({ patches: userPatches() }),
    });
  } catch (e) {
    console.warn("zone_patch: save failed", e);
  }
  // Re-fetch rather than render the local array: the GET enriches each patch
  // with its cross-camera `ghost` polygon, which only the server can compute.
  await load();
  window.dispatchEvent(new CustomEvent("zone-patches:saved"));
}

function activeCam() {
  const v = window.Alpine?.store?.("bigPanel")?.view || sessionStorage.getItem("active_view");
  return v === "cam_a" || v === "cam_b" ? v : null;
}

export function startPatchDraw(camId) {
  const cam = camId || activeCam();
  if (!cam) return;
  if (userPatches().length >= MAX_PATCHES) {   // cap counts OPERATOR zones, not twins
    window.alert(`Maximum of ${MAX_PATCHES} zones reached. Delete one before adding another.`);
    return;
  }
  const img = document.getElementById(`${cam}-img`);
  startDraw({
    target: cam,
    mode: "raw",                 // source-frame pixels, no pixel→world projection
    label: "",                   // no label — the toolbar hint says what to do
    minPoints: 3,                // a polygon (finish via the Done button)
    onDone: (points) => {
      if (!points || points.length < 3) return;
      // Name is COMPULSORY and UNIQUE — ask before the zone exists, so no
      // autonamed zone ever reaches zones.yaml / the MQTT zone_state payload.
      const name = promptZoneName(allZoneNames(), nextDefaultName());
      if (!name) return;                       // cancelled → no zone
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      patches.push({
        id: "zp_" + Date.now().toString(36),
        name,
        camera: cam,
        polygon: points,                                   // red overlay + crop region
        rect: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
        frame_wh: [img?.naturalWidth || 0, img?.naturalHeight || 0],
      });
      save();
    },
  });
}

// Next UNUSED default label ("Zone N"): max existing index + 1, never
// `length + 1`. A zone's identity is its immutable `id`; the label is cosmetic.
// Reusing a freed number would silently retarget anything keyed on the name.
function nextDefaultName() {
  return nextZoneNumberName(allZoneNames());
}

// "Zone N" numbering and name uniqueness are SHARED between camera zones
// (this module) and étagères (etagere.js): an operator adding "an étagère
// as zone 3" gets the next free number across both lists, and no two zones
// of either kind can carry the same name.
export function allZoneNames() {
  const mine = userPatches().map((p) => p.name);
  const theirs = (window.__etagere?.getZones?.() || []).map((z) => z.name);
  return [...mine, ...theirs].filter(Boolean);
}

export function nextZoneNumberName(names) {
  let max = 0;
  for (const n of names) {
    const m = /^Zone (\d+)$/.exec(n || "");
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `Zone ${max + 1}`;
}

// First step of "add zone": ask which KIND, then hand off to the right drawer.
// Floor/camera zone → the polygon flow below; étagère → etagere.js (4 corners
// → auto-split grid). Cancel/ESC/backdrop-click closes without drawing.
export function chooseKindThenDraw(camId) {
  const root = document.getElementById("zm-kind-picker");
  if (!root) { startPatchDraw(camId); return; }
  const close = () => {
    root.classList.add("hidden");
    document.removeEventListener("keydown", onKey);
    root.removeEventListener("click", onBackdrop);
  };
  const onKey = (ev) => { if (ev.key === "Escape") close(); };
  const onBackdrop = (ev) => { if (ev.target === root) close(); };
  for (const btn of root.querySelectorAll(".zm-kind-btn")) {
    btn.onclick = () => {
      close();
      if (btn.dataset.kind === "etagere") window.__etagere?.startEtagereDraw?.(camId);
      else startPatchDraw(camId);
    };
  }
  const cancel = document.getElementById("zm-kind-picker-cancel");
  if (cancel) cancel.onclick = close;
  document.addEventListener("keydown", onKey);
  root.addEventListener("click", onBackdrop);
  root.classList.remove("hidden");
}

// Delete ONE zone by id (Settings per-row delete icon) and persist. Deleting a
// zone must NOT rename or reorder any other — every surviving zone keeps its id
// AND its label. (The list ordinal in the Settings rows is a display row-number,
// not identity.) Deleting a zone also drops its twin (save() sends operator
// patches; twins are regenerated).
export function deletePatch(id) {
  const before = patches.length;
  patches = patches.filter((p) => p.id !== id && p.twin_of !== id);
  if (patches.length !== before) save();
}

// Start drawing a new zone from a panel's "+ Add zone" placeholder. If a camera
// view isn't active (e.g. MAP), switch to CAM 1 first so the draw has a target.
function addZoneFromPanel() {
  const store = window.Alpine?.store?.("bigPanel");
  if (store && store.view !== "cam_a" && store.view !== "cam_b") {
    store.select("cam_a");
    setTimeout(() => chooseKindThenDraw(), 150);   // let the cam view mount first
    return;
  }
  chooseKindThenDraw();
}

// --- Edit an existing camera zone's polygon (drag its vertices) ---------------
// Same interaction model as the étagère adjuster: Settings hides, the cam
// shows the zone with vertex handles (live_overlay draws them while
// isPatchEditing(id)), the shared #draw-toolbar's Done saves / Cancel reverts.
let editingPatch = null;      // { id, cam, orig } while a vertex-drag session is live
let stopPatchEdit = null;
const VERTEX_TOL_PX = 8;

export function isPatchEditing(id) { return !!editingPatch && editingPatch.id === id; }

function selectCamView(cam) {
  const store = window.Alpine?.store?.("bigPanel");
  if (store && store.view !== cam) store.select(cam);
}

export function startPatchEdit(id) {
  const p = patches.find((q) => q.id === id && !q.twin_of);
  if (!p || !Array.isArray(p.polygon) || p.polygon.length < 3) return;
  if (editingPatch) stopPatchEdit?.();
  document.getElementById("zone-manager")?.classList.add("hidden");   // reveal the cam
  const cam = p.camera || "cam_a";
  selectCamView(cam);
  editingPatch = { id, cam, orig: p.polygon.map((v) => [v[0], v[1]]) };

  setTimeout(() => {
    const canvas = document.getElementById(`${cam}-overlay`);
    const bar = document.getElementById("draw-toolbar");
    const label = document.getElementById("draw-zone-label");
    const doneBtn = document.getElementById("draw-done");
    const cancelBtn = document.getElementById("draw-cancel");
    const countWrap = document.getElementById("draw-count")?.parentElement;
    const undoBtn = document.getElementById("draw-undo");
    if (!canvas) { editingPatch = null; return; }
    canvas.style.pointerEvents = "auto";
    let drag = null;   // vertex index being dragged
    const toSrc = (ev) => window.__displayToSource
      ? window.__displayToSource(canvas, cam, ev.offsetX, ev.offsetY, p.frame_wh)
      : null;
    const down = (ev) => {
      const s = toSrc(ev); if (!s) return;
      // pick the nearest vertex within tolerance (source px scaled ≈ display px
      // is close enough for a handle grab)
      let best = -1, bestD = Infinity;
      p.polygon.forEach(([vx, vy], i) => {
        const d = Math.hypot(vx - s[0], vy - s[1]);
        if (d < bestD) { bestD = d; best = i; }
      });
      const scale = window.__displayScale ? window.__displayScale(canvas, cam, p.frame_wh) : 1;
      if (best >= 0 && bestD * scale <= VERTEX_TOL_PX * 1.5) drag = best;
    };
    const move = (ev) => {
      if (drag == null) return;
      const s = toSrc(ev); if (!s) return;
      p.polygon[drag] = [s[0], s[1]];
      const xs = p.polygon.map((v) => v[0]), ys = p.polygon.map((v) => v[1]);
      p.rect = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    };
    const up = () => { drag = null; };
    canvas.addEventListener("mousedown", down);
    canvas.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    if (label) label.textContent = `${p.name} — drag the polygon points, then Done`;
    if (countWrap) countWrap.style.display = "none";
    if (undoBtn) undoBtn.style.display = "none";
    bar?.classList.remove("hidden");

    const teardown = () => {
      canvas.removeEventListener("mousedown", down);
      canvas.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      canvas.style.pointerEvents = "";
      if (doneBtn) doneBtn.onclick = null;
      if (cancelBtn) cancelBtn.onclick = null;
      bar?.classList.add("hidden");
      if (countWrap) countWrap.style.display = "";
      if (undoBtn) undoBtn.style.display = "";
      editingPatch = null;
      stopPatchEdit = null;
    };
    stopPatchEdit = teardown;
    const reopen = () => {
      document.getElementById("zone-manager")?.classList.remove("hidden");
      document.querySelector('.settings-tab-btn[data-tab="zones"]')?.click();
    };
    if (doneBtn) doneBtn.onclick = () => { teardown(); save(); reopen(); };
    if (cancelBtn) cancelBtn.onclick = () => {
      p.polygon = editingPatch?.orig || p.polygon;   // captured before teardown
      const xs = p.polygon.map((v) => v[0]), ys = p.polygon.map((v) => v[1]);
      p.rect = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      teardown(); reopen();
    };
  }, 200);
}

// Fill the three ZONE panels with the first three ROIs' cropped streams (over
// the shared /ws/video socket — no per-panel HTTP connection). An undeclared
// slot shows a clickable "+ Add zone" placeholder. Vanilla DOM — the panels'
// tall expand stays Alpine-driven.
function renderPanels() {
  const VWS = window.__videoWS;
  // Drop the previous render's subscriptions; re-attached below for the
  // current zone set (a deleted/re-ordered zone must not keep streaming).
  if (VWS) panelStreamIds.forEach((id) => VWS.detach(id));
  panelStreamIds = [];
  // Six slots (MAX_PATCHES). Slots 1-3 are always visible (empty ⇒ the
  // "+ Add zone" placeholder); slots 4-6 appear only while a zone occupies
  // them, and the grid flips to 3x2 beyond three zones.
  const slots = Array.from({ length: MAX_PATCHES }, (_, i) => ({
    body: `zone-${i + 1}-body`, badge: `zone-${i + 1}-badge`,
    fallback: `ZONE ${i + 1}`,
  }));
  // ZONE panels are shared by camera zones (this module) and étagères
  // (etagere.js): one list, ordered by the "Zone N" number in the name so an
  // étagère named "Zone 3" lands in panel 3 next to its siblings; unnumbered
  // names keep creation order after the numbered ones. Twins never occupy
  // panels; the 6-slot cap applies to the combined list.
  const zoneNo = (n) => { const m = /^Zone (\d+)$/.exec(n || ""); return m ? +m[1] : Infinity; };
  const own = [
    ...userPatches().map((p) => ({ kind: "patch", id: p.id, name: p.name })),
    ...(window.__etagere?.getZones?.() || []).map((z) => ({ kind: "etagere", id: z.id, name: z.name })),
  ].map((e, i) => ({ ...e, order: i }))
   .sort((a, b) => (zoneNo(a.name) - zoneNo(b.name)) || (a.order - b.order))
   .slice(0, MAX_PATCHES);
  const container = document.getElementById("zones-container");
  if (container) {
    container.classList.toggle("zones-grid-6", own.length > 3);
    container.classList.toggle("zones-grid-3", own.length <= 3);
  }
  slots.forEach((slot, i) => {
    const body = document.getElementById(slot.body);
    const badge = document.getElementById(slot.badge);
    if (!body) return;
    const panel = body.closest(".panel");          // for the synced (light-green) state
    const p = own[i];
    if (panel && i >= 3) panel.style.display = p ? "" : "none";
    if (!p) {
      body.innerHTML =
        '<button type="button" class="zone-add" title="Draw a new zone on the camera">' +
        '<span class="zone-add-plus">+</span><span>Add zone</span></button>';
      body.querySelector(".zone-add").addEventListener("click", addZoneFromPanel);
      if (badge) badge.textContent = slot.fallback;
      if (panel) panel.classList.remove("zone-synced");
      return;
    }
    if (panel) panel.classList.add("zone-synced");   // a zone is declared + saved here
    // Just the centred slice — the model + detect-size live in Settings ▸ Camera
    // zones, not overlaid on the panel.
    body.innerHTML = `<img class="zone-patch-img" alt="${p.name}">`;
    if (VWS) {
      // étagère panels stream the warped-cell mosaic (what the model sees)
      const sid = p.kind === "etagere" ? `etagere:${p.id}` : `zone:${p.id}`;
      VWS.attach(body.querySelector("img"), sid);
      panelStreamIds.push(sid);
    }
    if (badge) badge.textContent = p.name || slot.fallback;
  });
}

// Settings sync: render the camera zones as editable rows in the Settings modal
function renderSettingsList() {
  const host = document.getElementById("zm-cam-zones");
  if (!host) return;
  const own = userPatches();   // twins are derived — not editable rows
  if (!own.length) {
    host.innerHTML = '<p class="layout-hint">Draw a zone on a camera view — it appears here.</p>';
    return;
  }
  host.innerHTML = "";
  own.forEach((p, i) => {
    const r = p.rect || [0, 0, 0, 0];
    const x0 = Math.round(Math.min(r[0], r[2])), y0 = Math.round(Math.min(r[1], r[3]));
    const w  = Math.round(Math.abs(r[2] - r[0])), h = Math.round(Math.abs(r[3] - r[1]));
    const verts   = Array.isArray(p.polygon) ? p.polygon : [];
    const npts    = verts.length;
    const polyStr = verts.map(([u, v]) => `(${Math.round(u)},${Math.round(v)})`).join(" ");
    const color   = p.color || "#ff3b3b";
    const camLbl  = p.camera === "cam_b" ? "CAM 2" : "CAM 1";

    const row = document.createElement("div");
    row.className = "config-zone-row";
    row.dataset.zoneId = p.id;

    row.innerHTML =
      /* ── top line: number · name · camera badge · spacer · delete ──── */
      `<div class="czr-header">` +
        `<span class="config-zone-num" title="Zone ${i + 1}">${i + 1}</span>` +
        `<input class="zm-name" value="${p.name || ""}" placeholder="zone name" title="Zone name" />` +
        `<span class="czr-cam-badge">${camLbl}</span>` +
        `<button type="button" class="glass-btn zm-small-btn zm-edit-points" title="Drag this zone's points on the camera">Edit points</button>` +
        `<button type="button" class="glass-btn zm-iconbtn zm-delete"` +
          ` title="Delete zone ${i + 1}" aria-label="Delete zone ${i + 1}">` +
          `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"` +
              ` stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">` +
            `<path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>` +
            `<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>` +
        `</button>` +
      `</div>` +
      /* ── controls line: colour ───────────────────────────────── */
      `<div class="czr-controls">` +
        `<span class="czr-field czr-field--color">` +
          `<label class="czr-label" title="Zone outline colour on the camera view">Color</label>` +
          `<input class="zm-color" type="color" value="${color}" />` +
        `</span>` +
      `</div>` +
      /* ── info bar: slice chip · polygon chip ─────────────────────────── */
      `<div class="czr-info-bar">` +
        `<span class="czr-chip czr-chip--slice" title="Bounding-box crop in source-frame pixels">` +
          `<span class="czr-chip-key">slice</span>` +
          `@(${x0},\u200B${y0}) ${w}×${h}\u202Fpx · ${npts}\u202Fpts` +
        `</span>` +
        `<span class="czr-chip czr-chip--poly" title="Polygon vertices in source-frame pixels">` +
          `<span class="czr-chip-key">polygon</span>` +
          `<span class="czr-chip-verts">${polyStr || "—"}</span>` +
        `</span>` +
      `</div>`;

    row.querySelector(".zm-name").addEventListener("change", (e) => {
      const val = e.target.value.trim();
      const taken = userPatches().filter((q) => q.id !== p.id)
        .map((q) => String(q.name || "").trim().toLowerCase());
      if (!val || taken.includes(val.toLowerCase())) {
        window.alert(!val
          ? "Zone name cannot be empty."
          : `"${val}" is already used by another zone — names must be unique.`);
        e.target.value = p.name;               // revert the field
        return;
      }
      p.name = val;
      save();
    });
    // No per-zone confidence: ONE global model with ONE threshold
    // (Settings ▸ Isistream) serves every zone — the dashboard only renders.
    row.querySelector(".zm-color").addEventListener("change", (e) => { p.color = e.target.value; save(); });
    row.querySelector(".zm-edit-points").addEventListener("click", () => startPatchEdit(p.id));
    row.querySelector(".zm-delete").addEventListener("click", () => deletePatch(p.id));
    host.appendChild(row);
  });
}

// Reflect the 6-zone cap on the draw button: disabled + a hint once full.
function updateDrawButton() {
  const draw = document.getElementById("btn-draw-patch");
  if (!draw) return;
  const full = userPatches().length >= MAX_PATCHES;
  draw.disabled = full;
  draw.classList.toggle("is-disabled", full);
  draw.title = full ? `Maximum of ${MAX_PATCHES} zones reached` : "Draw a zone on this camera";
}

function wire() {
  // étagères share the ZONE panels — re-render when their list changes
  window.addEventListener("etagere:changed", () => renderPanels());
  const draw = document.getElementById("btn-draw-patch");
  if (draw) draw.onclick = () => chooseKindThenDraw();
  load();
}

window.__zonePatch = { startPatchDraw, deletePatch, chooseKindThenDraw, allZoneNames, nextZoneNumberName, startPatchEdit, isPatchEditing };
// NOTE: do NOT re-render here on "zone-patches:saved" — save() already re-rendered
// the panels + settings list. A second render rebuilds the zone <img> (re-opening
// its /stream/zone connection) for nothing. The event stays for OTHER listeners
// (e.g. floor_map_3d.reloadZones).

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
