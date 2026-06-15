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
  if (records.length && !current) select(records[0]);
}

function renderStrip() {
  const onlyReview = el("only-review")?.checked;
  const strip = el("record-strip");
  strip.innerHTML = "";
  for (const r of records) {
    if (onlyReview && !r.needs_review) continue;
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

function drawPrompts() {
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
}

if (canvas) {
  canvas.addEventListener("pointerdown", (e) => { dragStart = canvasPos(e); });
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
  });
  window.addEventListener("resize", drawPrompts);
}

const clearBtn = el("prompt-clear");
if (clearBtn) clearBtn.onclick = async () => {
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

const saveBtn = el("prompt-save");
if (saveBtn) saveBtn.onclick = async () => {
  if (!current) return;
  await sendJSON(`/api/p/${project}/records/${current.id}/prompts`, "PUT", { prompts });
  current.mask_prompts = prompts;
  flash(el("maps-msg"), "prompts saved — run masks to apply", true);
};

function runner(phase) {
  return async () => {
    try {
      const { job } = await sendJSON(`/api/p/${project}/run/${phase}`, "POST", {});
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

load();
