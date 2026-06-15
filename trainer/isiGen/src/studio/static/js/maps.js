import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const root = document.getElementById("maps-root");
const project = root.dataset.project;
const classes = JSON.parse(root.dataset.classes);
let records = [];
let current = null;          // active record
let prompts = [];            // working prompt list for `current`
let dragStart = null;

const img = document.getElementById("v-image");
const canvas = document.getElementById("prompt-canvas");
const clsSel = document.getElementById("prompt-class");
clsSel.innerHTML = classes.map((c) => `<option value="${c.name}">${c.name}</option>`).join("");

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
  const onlyReview = document.getElementById("only-review").checked;
  const strip = document.getElementById("record-strip");
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

function select(r) {
  current = r;
  prompts = (r.mask_prompts || []).map((p) => ({ ...p }));
  const n = Date.now();
  img.src = `/media/${project}/image/${r.id}?n=${n}`;
  document.getElementById("v-depth").src = r.depth_map ? `/media/${project}/depth/${r.id}?n=${n}` : "";
  document.getElementById("v-canny").src = r.canny_map ? `/media/${project}/canny/${r.id}?n=${n}` : "";
  document.getElementById("v-mask").src = r.mask ? `/media/${project}/mask/${r.id}?n=${n}` : "";
  renderStrip();
  img.onload = drawPrompts;
}

function canvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  // displayed px → SOURCE px (image rendered full-width, aspect preserved)
  const sx = current.width / rect.width, sy = current.height / rect.height;
  return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
}

function drawPrompts() {
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

document.getElementById("prompt-clear").onclick = async () => {
  prompts = [];
  drawPrompts();                                   // wipe the canvas dots
  if (!current) return;
  // Persist the cleared state: empty prompts also drop rec.mask server-side,
  // and blank the shown mask so "clear" actually clears (not just the dots).
  try {
    await sendJSON(`/api/p/${project}/records/${current.id}/prompts`, "PUT", { prompts: [] });
    current.mask_prompts = [];
    current.mask = null;
    document.getElementById("v-mask").src = "";
    flash(document.getElementById("maps-msg"), "prompts + mask cleared", true);
  } catch (e) { flash(document.getElementById("maps-msg"), e.message, false); }
};
document.getElementById("prompt-save").onclick = async () => {
  if (!current) return;
  await sendJSON(`/api/p/${project}/records/${current.id}/prompts`, "PUT", { prompts });
  current.mask_prompts = prompts;
  flash(document.getElementById("maps-msg"), "prompts saved — run masks to apply", true);
};

function runner(phase) {
  return async () => {
    try {
      const { job } = await sendJSON(`/api/p/${project}/run/${phase}`, "POST", {});
      flash(document.getElementById("maps-msg"), `${phase} running…`, true);
      watchJob(job.id, null, async (j) => {
        flash(document.getElementById("maps-msg"),
              j.state === "done" ? `${phase} done` : j.error, j.state === "done");
        const keep = current?.id;
        await load();
        const again = records.find((r) => r.id === keep);
        if (again) select(again);
      });
    } catch (e) { flash(document.getElementById("maps-msg"), e.message, false); }
  };
}
document.getElementById("run-maps").onclick = runner("maps");
document.getElementById("run-masks").onclick = runner("masks");
document.getElementById("only-review").addEventListener("change", renderStrip);
window.addEventListener("resize", drawPrompts);

load();
