// Warehouse-layout authoring (3D): trace rack/wall footprints on the MAP by clicking
// two opposite floor corners (grid-snapped raycast onto the Three.js floor), set
// per-element height / shelf levels / rotation / label, accumulate, save. The racks
// render as 3-level white shelving in floor_map_3d.js. Map-view only. Reuses
// draw_mode (now Three.js, no Pixi).
import { startDraw } from "/static/js/draw_mode.js";

const el = (id) => document.getElementById(id);
let elements = [];                 // working set, committed on Save
let outline = null;                // floor boundary {footprint: [[x,y],...]} | null

function updateCount() {
  const counter = el("layout-count");
  if (!counter) return;
  const n = elements.length;
  counter.textContent = `${n} element${n !== 1 ? "s" : ""}`;
  counter.style.display = n ? "inline-block" : "none";
  const btn = el("layout-delete-last");
  if (btn) btn.disabled = n === 0;
}

// ---------- add element (rect-snap on the 3D floor) ----------
function addElement() {
  const type = el("layout-type").value;                                  // rack | wall
  const height = parseFloat(el("layout-height").value) || 2.4;
  const levels = Math.max(1, Math.min(8, parseInt(el("layout-levels")?.value, 10) || 3));
  const rotation_deg = parseFloat(el("layout-rotation")?.value) || 0;
  const label = (el("layout-label")?.value || "").trim();
  startDraw({
    target: "map",
    mode: "rectSnap",
    gridStep: 0.1,
    label: `${type} · click two opposite floor corners (0.1 m grid)`,
    onDone: (footprint) => {                       // world (X,Y) rectangle
      elements.push({ id: `${type}_${Date.now()}`, type, shape: "rectangle",
                      footprint, height_m: height, levels, rotation_deg, label });
      window.__floor_map.drawLayout(elements);     // live 3D preview
      updateCount();
    },
  });
}

function deleteLast() {
  if (!elements.length) return;
  elements.pop();
  window.__floor_map.drawLayout(elements);
  updateCount();
}

// ---------- floor outline (work-area boundary) ----------
function setFloorOutline() {
  startDraw({
    target: "map", mode: "rectSnap", gridStep: 0.1,
    label: "Work area · click 2 opposite corners",
    onDone: (footprint) => {
      outline = { footprint };
      window.__floor_map?.drawOutline(outline);
      save();   // persist immediately
    },
  });
}

// ---------- save / clear ----------
async function save() {
  const res = await fetch("/api/warehouse-map", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ elements, outline }),
  });
  if (res.ok) document.dispatchEvent(new CustomEvent("layout:changed"));
}

async function clearAll() {
  elements = [];
  outline = null;
  window.__floor_map.drawLayout([]);
  window.__floor_map?.drawOutline(null);
  updateCount();
  try {
    await fetch("/api/warehouse-map", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ elements: [], outline: null }),
    });
  } catch { /* server save failed silently; visual already cleared */ }
  document.dispatchEvent(new CustomEvent("layout:changed"));
}

// ---------- panel toggle ----------
async function openAndLoad() {
  try {
    const d = await fetch("/api/warehouse-map").then((r) => r.json());
    elements = d.elements || [];
    outline = d.outline || null;
    updateCount();
    window.__floor_map?.drawLayout(elements);
    window.__floor_map?.drawOutline(outline);
  } catch { /* keep in-memory state */ }
}

function togglePanel() {
  const panel = el("layout-panel");
  if (!panel) return;
  const opening = panel.classList.contains("hidden");
  panel.classList.toggle("hidden");
  if (opening) openAndLoad();
}

// Map-only editor: hide the panel whenever the big panel leaves MAP view. Alpine's
// reactive effect re-runs on `$store.bigPanel.view` change.
function gateToMapView() {
  const A = window.Alpine;
  if (!A || !A.store || !A.effect) return false;
  const store = A.store("bigPanel");
  if (!store) return false;
  A.effect(() => {
    if (store.view !== "map") {
      const p = el("layout-panel");
      if (p) p.classList.add("hidden");
    }
  });
  return true;
}

function wire() {
  if (!el("layout-panel")) return;
  // Seed working set from the saved layout.
  fetch("/api/warehouse-map").then(r => r.json()).then(d => {
    elements = d.elements || [];
    outline  = d.outline  || null;
    updateCount();
  }).catch(() => {});
  el("layout-add")?.addEventListener("click", addElement);
  el("layout-save")?.addEventListener("click", save);
  el("layout-clear")?.addEventListener("click", clearAll);
  el("layout-delete-last")?.addEventListener("click", deleteLast);
  el("layout-outline")?.addEventListener("click", setFloorOutline);
  const toggle = el("btn-layout");
  if (toggle) toggle.addEventListener("click", togglePanel);
  // Shelf levels only apply to racks — disable for walls.
  const typeSel = el("layout-type");
  const levelsInput = el("layout-levels");
  if (typeSel && levelsInput) {
    const sync = () => { levelsInput.disabled = typeSel.value !== "rack"; };
    typeSel.addEventListener("change", sync); sync();
  }
  updateCount();
  if (!gateToMapView()) document.addEventListener("alpine:init", () => setTimeout(gateToMapView, 0));
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
else wire();
