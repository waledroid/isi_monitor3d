import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

// Shared by BOTH the Control-maps (phase 2) and Ground-truth-masks (phase 3)
// pages — each page only contains the elements it needs, so every wiring below
// is feature-detected (guarded) and no-ops when its element is absent.
const el = (id) => document.getElementById(id);

const root = el("maps-root");
const project = root.dataset.project;
const classes = JSON.parse(root.dataset.classes);
let records = [];
let current = null;          // active record
let prompts = [];            // working prompt list for `current`
let dragStart = null;

const img = el("v-image");
const canvas = el("prompt-canvas");
const clsSel = el("prompt-class");
if (clsSel) {
  clsSel.innerHTML = classes.map((c) => `<option value="${c.name}">${c.name}</option>`).join("");
}

function colorOf(name) {
  const c = classes.find((x) => x.name === name);
  return c ? `rgb(${c.color.join(",")})` : "#fff";
}

async function load() {
  records = (await getJSON(`/api/p/${project}/records`)).records.filter((r) => !r.excluded);
  renderStrip();
  // Land on the first VISIBLE record (skips backgrounds on the masks page, which
  // hides them by default — they're never masked anyway).
  if (records.length && !current) select(records.find(passesFilters) || records[0]);
}

// Backgrounds are hidden by default on the masks page (toggle present); the
// control-maps page has no toggle, so it shows everything (it needs bg depth).
function passesFilters(r) {
  if (el("only-review")?.checked && !r.needs_review) return false;
  const bgToggle = el("show-backgrounds");
  if (bgToggle && !bgToggle.checked && r.background) return false;
  return true;
}

function renderStrip() {
  const strip = el("record-strip");
  strip.innerHTML = "";
  for (const r of records) {
    if (!passesFilters(r)) continue;
    const t = document.createElement("img");
    t.src = `/media/${project}/thumb/${r.id}`;
    t.className = (current && r.id === current.id ? "active " : "") + (r.needs_review ? "review" : "");
    t.onclick = () => select(r);
    strip.appendChild(t);
  }
}

function setSrc(id, url) {
  const e = el(id);
  if (e) e.src = url;
}

function select(r) {
  flushSave();                 // persist any pending edit before switching records
  current = r;
  prompts = (r.mask_prompts || []).map((p) => ({ ...p }));
  const n = Date.now();
  setSrc("v-image", `/media/${project}/image/${r.id}?n=${n}`);
  setSrc("v-depth", r.depth_map ? `/media/${project}/depth/${r.id}?n=${n}` : "");
  setSrc("v-canny", r.canny_map ? `/media/${project}/canny/${r.id}?n=${n}` : "");
  setSrc("v-mask", r.mask ? `/media/${project}/mask/${r.id}?n=${n}` : "");
  renderStrip();
  if (img) img.onload = drawPrompts;
}

function canvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  // displayed px → SOURCE px (image rendered full-width, aspect preserved)
  const sx = current.width / rect.width, sy = current.height / rect.height;
  return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
}

// `live` (optional [x1,y1,x2,y2] in SOURCE px) draws the in-progress drag box.
function drawPrompts(live) {
  if (!canvas || !img) return;
  const rect = img.getBoundingClientRect();
  canvas.width = rect.width; canvas.height = rect.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!current) return;
  const kx = rect.width / current.width, ky = rect.height / current.height;
  for (const p of prompts) {
    ctx.strokeStyle = ctx.fillStyle = colorOf(p.class_name);
    if (p.kind === "point" && p.xy) {
      ctx.beginPath();
      ctx.arc(p.xy[0] * kx, p.xy[1] * ky, 5, 0, Math.PI * 2);
      p.label ? ctx.fill() : ctx.stroke();      // filled = fg, hollow = bg
    } else if (p.kind === "box" && p.xyxy) {
      const [a, b, c, d] = p.xyxy;
      ctx.strokeRect(a * kx, b * ky, (c - a) * kx, (d - b) * ky);
    }
  }
  if (live) {                                    // dashed preview while dragging
    ctx.strokeStyle = colorOf(clsSel.value);
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(live[0] * kx, live[1] * ky,
                   (live[2] - live[0]) * kx, (live[3] - live[1]) * ky);
    ctx.setLineDash([]);
  }
}

if (canvas) {
  canvas.addEventListener("pointerdown", (e) => {
    dragStart = canvasPos(e);
    canvas.setPointerCapture(e.pointerId);       // keep move events if cursor leaves
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!current || !dragStart) return;
    const cur = canvasPos(e);
    drawPrompts([dragStart[0], dragStart[1], cur[0], cur[1]]);
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!current || !dragStart) return;
    const end = canvasPos(e);
    const dist = Math.hypot(end[0] - dragStart[0], end[1] - dragStart[1]);
    if (dist > 8) {
      prompts.push({ kind: "box", class_name: clsSel.value,
                     xyxy: [Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]),
                            Math.max(dragStart[0], end[0]), Math.max(dragStart[1], end[1])] });
    } else {
      prompts.push({ kind: "point", class_name: clsSel.value, xy: end,
                     label: e.shiftKey ? 0 : 1 });
    }
    dragStart = null;
    drawPrompts();
    scheduleSave();
  });
  window.addEventListener("resize", () => drawPrompts());
}

// ── Auto-save prompts (debounced) — replaces the old Save-prompts button ────
// Every canvas edit schedules a save; the snapshot pins the edited record so a
// late flush never writes to the wrong record after the user switches images.
let saveTimer = null;
let pendingSave = null;        // { rec, prompts } snapshot of the last edit

function cancelPendingSave() {
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
  pendingSave = null;
}

async function flushSave() {
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
  if (!pendingSave) return;
  const { rec, prompts: snap } = pendingSave;
  pendingSave = null;
  try {
    await sendJSON(`/api/p/${project}/records/${rec.id}/prompts`, "PUT", { prompts: snap });
    rec.mask_prompts = snap;
    flash(el("maps-msg"), "prompts saved — run masks to apply", true);
  } catch (e) { flash(el("maps-msg"), `prompt save FAILED: ${e.message}`, false); }
}

function scheduleSave() {
  if (!current) return;
  pendingSave = { rec: current, prompts: prompts.map((p) => ({ ...p })) };
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSave, 400);
}

const clearBtn = el("prompt-clear");
if (clearBtn) clearBtn.onclick = async () => {
  cancelPendingSave();         // a queued edit must not resurrect cleared prompts
  prompts = [];
  drawPrompts();                                   // wipe the canvas dots
  if (!current) return;
  // Persist the cleared state: empty prompts also drop rec.mask server-side,
  // and blank the shown mask so "clear" actually clears (not just the dots).
  try {
    await sendJSON(`/api/p/${project}/records/${current.id}/prompts`, "PUT", { prompts: [] });
    current.mask_prompts = [];
    current.mask = null;
    setSrc("v-mask", "");
    flash(el("maps-msg"), "prompts + mask cleared", true);
  } catch (e) { flash(el("maps-msg"), e.message, false); }
};

// ── Auto-prompt detector (masks page only) ──────────────────────────────────
const detSel = el("prompt-detector");
async function loadDetectors() {
  if (!detSel) return;
  try {
    const { models, current: cur } = await getJSON(`/api/p/${project}/detector-models`);
    for (const m of models) {
      const o = document.createElement("option");
      o.value = m.path; o.textContent = m.label;
      detSel.appendChild(o);
    }
    if (cur) detSel.value = cur;          // reflect the persisted selection
  } catch { /* dropdown stays "none" if isidet models can't be listed */ }
}

const detectBtn = el("prompt-detect");
if (detectBtn) detectBtn.onclick = async () => {
  if (!current) return;
  if (!detSel?.value) { flash(el("maps-msg"), "pick an auto-prompt detector first", false); return; }
  detectBtn.disabled = true;
  flash(el("maps-msg"), "detecting…", true);
  try {
    const { prompts: found } = await sendJSON(
      `/api/p/${project}/records/${current.id}/detect-prompts`, "POST", { onnx_path: detSel.value });
    // Replace existing box prompts with the detector's; keep hand-drawn points.
    prompts = prompts.filter((p) => p.kind !== "box").concat(found || []);
    drawPrompts();
    scheduleSave();
    flash(el("maps-msg"), `${(found || []).length} box(es) — saving…`, true);
  } catch (e) { flash(el("maps-msg"), e.message, false); }
  finally { detectBtn.disabled = false; }
};

function runner(phase) {
  return async () => {
    try {
      await flushSave();       // just-drawn prompts must reach the run
      const body = phase === "masks" && detSel ? { prompt_detector: detSel.value || "none" } : {};
      const { job } = await sendJSON(`/api/p/${project}/run/${phase}`, "POST", body);
      flash(el("maps-msg"), `${phase} running…`, true);
      watchJob(job.id, null, async (j) => {
        flash(el("maps-msg"), j.state === "done" ? `${phase} done` : j.error, j.state === "done");
        const keep = current?.id;
        await load();
        const again = records.find((r) => r.id === keep);
        if (again) select(again);
      });
    } catch (e) { flash(el("maps-msg"), e.message, false); }
  };
}
if (el("run-maps")) el("run-maps").onclick = runner("maps");
if (el("run-masks")) el("run-masks").onclick = runner("masks");
el("only-review")?.addEventListener("change", renderStrip);
el("show-backgrounds")?.addEventListener("change", renderStrip);

load();
loadDetectors();
