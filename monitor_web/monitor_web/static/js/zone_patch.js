// Zone-patch ROIs — pixel-space "watch boxes" drawn directly on a camera frame,
// cropped into a detected patch (targeted SAHI) shown in the ZONE panels.
//
// No calibration needed: the box is captured in SOURCE-frame pixels via draw_mode's
// `raw` cam mode, stored in zone_patches.yaml, and the /stream/zone/{id} endpoint
// crops + detects that region. live_overlay.js reads getPatches() to draw the red ROI.

import { startDraw } from "/static/js/draw_mode.js";

const MAX_PATCHES = 6;   // max zones the operator can create (drawn on CAM)

let patches = [];   // [{id, name, camera, polygon:[[u,v]..], rect:[x0,y0,x1,y1], frame_wh:[W,H], model, infer_size, confidence, color}]
let models = [];    // [{path, label}] trained detection models for the per-zone picker
let panelStreamIds = [];   // /ws/video streams currently attached to the ZONE panels

async function loadModels() {
  try {
    const r = await fetch("/api/detection/onnx-files");
    if (r.ok) models = (await r.json()).files || [];
  } catch { /* keep what we have */ }
}

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
    label: "Zone — click polygon points, then Done",
    minPoints: 3,                // a polygon (finish via the Done button)
    onDone: (points) => {
      if (!points || points.length < 3) return;
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      patches.push({
        id: "zp_" + Date.now().toString(36),
        name: `Zone ${userPatches().length + 1}`,
        camera: cam,
        polygon: points,                                   // red overlay + crop region
        rect: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
        frame_wh: [img?.naturalWidth || 0, img?.naturalHeight || 0],
        infer_size: 320,                                   // detection input size (INTER_AREA)
        sahi: false,                                       // SAHI slicing off by default
        enhance: false,                                    // slice enhancement off by default
      });
      save();
    },
  });
}

// Default names ("Zone N") follow the slot they occupy. After a delete the list
// compacts (the next zone takes the freed slot), so renumber the DEFAULT-named
// zones to match their new position — custom names are never touched.
function renumberDefaults() {
  userPatches().forEach((p, i) => {
    if (/^Zone \d+$/.test(p.name || "")) p.name = `Zone ${i + 1}`;
  });
}

// Delete ONE zone by id (Settings per-row delete icon) and persist. The list
// compacts — the panel/numbering shift up to fill the gap. Deleting a zone
// also drops its twin (save() sends operator patches; twins are regenerated).
export function deletePatch(id) {
  const before = patches.length;
  patches = patches.filter((p) => p.id !== id && p.twin_of !== id);
  if (patches.length !== before) {
    renumberDefaults();
    save();
  }
}

// Start drawing a new zone from a panel's "+ Add zone" placeholder. If a camera
// view isn't active (e.g. MAP), switch to CAM 1 first so the draw has a target.
function addZoneFromPanel() {
  const store = window.Alpine?.store?.("bigPanel");
  if (store && store.view !== "cam_a" && store.view !== "cam_b") {
    store.select("cam_a");
    setTimeout(() => startPatchDraw(), 150);   // let the cam view mount first
    return;
  }
  startPatchDraw();
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
  const slots = [
    { body: "zone-1-body", badge: "zone-1-badge", fallback: "ZONE 1" },
    { body: "zone-2-body", badge: "zone-2-badge", fallback: "ZONE 2" },
    { body: "zone-3-body", badge: "zone-3-badge", fallback: "ZONE 3" },
  ];
  const own = userPatches();   // twins never occupy ZONE panels
  slots.forEach((slot, i) => {
    const body = document.getElementById(slot.body);
    const badge = document.getElementById(slot.badge);
    if (!body) return;
    const panel = body.closest(".panel");          // for the synced (light-green) state
    const p = own[i];
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
      const sid = `zone:${p.id}`;
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
        `<button type="button" class="glass-btn zm-iconbtn zm-delete"` +
          ` title="Delete zone ${i + 1}" aria-label="Delete zone ${i + 1}">` +
          `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"` +
              ` stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">` +
            `<path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>` +
            `<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>` +
        `</button>` +
      `</div>` +
      /* ── controls line: model · size · conf · colour · SAHI ─────────── */
      `<div class="czr-controls">` +
        `<span class="czr-field">` +
          `<label class="czr-label" title="Detection model for this zone">Model</label>` +
          `<select class="zm-model">${modelOptions(p.model || "")}</select>` +
        `</span>` +
        `<span class="czr-field czr-field--narrow">` +
          `<label class="czr-label" title="Input resolution fed to the detector (px)">Size</label>` +
          `<input class="zm-size" type="number" min="64" max="1280" step="32"` +
                ` value="${p.infer_size || 320}" placeholder="px" />` +
        `</span>` +
        `<span class="czr-field czr-field--narrow">` +
          `<label class="czr-label" title="Confidence threshold 0–1 (blank = global)">Conf</label>` +
          `<input class="zm-conf" type="number" min="0" max="1" step="0.05"` +
                ` value="${p.confidence ?? ""}" placeholder="—" />` +
        `</span>` +
        `<span class="czr-field czr-field--color">` +
          `<label class="czr-label" title="Zone outline colour on the camera view">Color</label>` +
          `<input class="zm-color" type="color" value="${color}" />` +
        `</span>` +
        `<label class="czr-sahi-toggle" title="SAHI slicing — fixed tile grid for small/far objects">` +
          `<input class="zm-sahi-cb" type="checkbox" ${p.sahi ? "checked" : ""} />` +
          `<span class="czr-sahi-label">SAHI</span>` +
        `</label>` +
        `<label class="czr-sahi-toggle" title="Enhance slice for far/dim objects: temporal denoise + upscale + contrast + sharpen. Tip: raising this zone's Size is the biggest single win.">` +
          `<input class="zm-enh-cb" type="checkbox" ${p.enhance ? "checked" : ""} />` +
          `<span class="czr-sahi-label">ENH</span>` +
        `</label>` +
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

    row.querySelector(".zm-name").addEventListener("change", (e) => { p.name = e.target.value.trim(); save(); });
    row.querySelector(".zm-model").addEventListener("change", (e) => { p.model = e.target.value || null; save(); });
    row.querySelector(".zm-size").addEventListener("change", (e) => {
      const v = parseInt(e.target.value, 10);
      p.infer_size = Number.isFinite(v) ? Math.max(64, Math.min(1280, v)) : 320;
      save();
    });
    row.querySelector(".zm-conf").addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      p.confidence = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : null;
      save();
    });
    row.querySelector(".zm-color").addEventListener("change", (e) => { p.color = e.target.value; save(); });
    row.querySelector(".zm-sahi-cb").addEventListener("change", (e) => { p.sahi = e.target.checked; save(); });
    row.querySelector(".zm-enh-cb").addEventListener("change", (e) => { p.enhance = e.target.checked; save(); });
    row.querySelector(".zm-delete").addEventListener("click", () => deletePatch(p.id));
    host.appendChild(row);
  });
}

// Build <option>s for the per-zone model picker: "global" + each trained model.
function modelOptions(selected) {
  const opts = ['<option value="">— global model —</option>'];
  for (const m of models) {
    const path = m.path || m;
    const label = m.label || path;
    opts.push(`<option value="${path}"${path === selected ? " selected" : ""}>${label}</option>`);
  }
  return opts.join("");
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
  const draw = document.getElementById("btn-draw-patch");
  if (draw) draw.onclick = () => startPatchDraw();
  loadModels().then(load);   // models first so the per-zone picker has options
}

window.__zonePatch = { startPatchDraw, deletePatch };
// NOTE: do NOT re-render here on "zone-patches:saved" — save() already re-rendered
// the panels + settings list. A second render rebuilds the zone <img> (re-opening
// its /stream/zone connection) for nothing. The event stays for OTHER listeners
// (e.g. floor_map_3d.reloadZones).

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
