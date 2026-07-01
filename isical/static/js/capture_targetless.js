// Targetless extrinsics UI (experimental) — method selector + interactive
// floor scale-reference marking + key-stage images + validation report.
//
// Loaded only on the extrinsic capture page. The AprilGrid path is untouched;
// this module toggles a targetless panel when the method selector is "targetless"
// and drives the Stage-3 backend routes (extrinsic-method, scale-references,
// targetless-stages/-report). Click-to-mark reuses the floor/zone click pattern:
// click the same floor landmark on cam_a then cam_b for point 1, then point 2,
// enter the measured metres, and add the reference (≥3 needed).

import { getJSON, sendJSON, flash } from "./api.js";

export function initTargetless(root) {
  const project = root.dataset.project;
  const phase = root.dataset.phase;
  if (phase !== "extrinsic") return;
  const cameras = JSON.parse(root.dataset.cameras || "[]");

  const sel = document.getElementById("extrinsic-method");
  const methodMsg = document.getElementById("method-msg");
  const panel = document.getElementById("targetless-panel");
  if (!sel || !panel) return;

  const applyMethod = (method) => {
    panel.hidden = method !== "targetless";
    // Targetless hides the AprilGrid tag-measurement inputs (irrelevant).
    document.querySelectorAll(".board-measure-label").forEach((el) => {
      el.style.display = method === "targetless" ? "none" : "";
    });
  };

  getJSON(`/api/p/${project}/extrinsic-method`)
    .then((r) => { sel.value = r.method; applyMethod(r.method); })
    .catch(() => {});

  sel.addEventListener("change", async () => {
    try {
      await sendJSON(`/api/p/${project}/extrinsic-method`, "PUT", { method: sel.value });
      applyMethod(sel.value);
      flash(methodMsg, `method: ${sel.value}`);
    } catch (e) { flash(methodMsg, e.message, false); }
  });

  setupScaleMarking(project, cameras);
  refreshStages(project);
  refreshReport(project);
}

// --- interactive scale-reference marking ------------------------------------

function setupScaleMarking(project, cameras) {
  const views = document.getElementById("scale-mark-views");
  const list = document.getElementById("scale-ref-list");
  const distIn = document.getElementById("scale-dist");
  const addBtn = document.getElementById("scale-add");
  const clearBtn = document.getElementById("scale-clear");
  const count = document.getElementById("scale-count");
  const hint = document.getElementById("scale-click-hint");
  if (!views) return;

  // Live stream figures per camera (reuse the MJPEG /stream transport the cam
  // views use), overlaid with clicked marks.
  views.innerHTML = "";
  const imgs = {};
  cameras.forEach((c) => {
    const fig = document.createElement("figure");
    fig.className = "cap-figure";
    fig.innerHTML = `<figcaption>${c} — click floor landmarks</figcaption>
      <div class="canvas-wrap"><img class="scale-stream" data-cam="${c}"
        src="/stream/${project}/${c}" alt="${c} live"></div>`;
    views.appendChild(fig);
    imgs[c] = fig.querySelector("img");
  });

  // Click sequence: p1@cam_a, p1@cam_b, p2@cam_a, p2@cam_b.
  const order = ["p1_a", "p1_b", "p2_a", "p2_b"];
  const camForStep = [cameras[0], cameras[1], cameras[0], cameras[1]];
  let step = 0;
  let pending = {};

  const updateHint = () => {
    if (step >= order.length) { hint.textContent = "all 4 points marked — enter distance + Add"; return; }
    hint.textContent = `click ${order[step]} on ${camForStep[step]}`;
  };
  updateHint();

  cameras.forEach((c) => {
    imgs[c].addEventListener("click", (ev) => {
      if (step >= order.length || camForStep[step] !== c) return;
      const rect = imgs[c].getBoundingClientRect();
      const x = (ev.clientX - rect.left) / rect.width * imgs[c].naturalWidth;
      const y = (ev.clientY - rect.top) / rect.height * imgs[c].naturalHeight;
      pending[order[step]] = [x, y];
      step += 1;
      addBtn.disabled = step < order.length || !(distIn.value > 0);
      updateHint();
    });
  });

  distIn.addEventListener("input", () => {
    addBtn.disabled = step < order.length || !(distIn.value > 0);
  });

  let refs = [];
  const render = () => {
    list.innerHTML = refs.map((r, i) =>
      `<li>#${i}: ${r.distance_m} m (p1 [${r.p1_a.map(Math.round)}]…)</li>`).join("");
    count.textContent = `${refs.length} reference(s)${refs.length >= 3 ? " ✓" : " (need ≥3)"}`;
  };

  getJSON(`/api/p/${project}/scale-references`)
    .then((r) => { refs = r.references || []; render(); }).catch(() => {});

  addBtn.addEventListener("click", async () => {
    if (step < order.length) return;
    refs.push({ ...pending, distance_m: parseFloat(distIn.value) });
    try {
      await sendJSON(`/api/p/${project}/scale-references`, "PUT", { references: refs });
      pending = {}; step = 0; distIn.value = ""; addBtn.disabled = true;
      render(); updateHint();
    } catch (e) { flash(count, e.message, false); }
  });

  clearBtn.addEventListener("click", async () => {
    refs = []; pending = {}; step = 0; addBtn.disabled = true;
    await sendJSON(`/api/p/${project}/scale-references`, "PUT", { references: [] }).catch(() => {});
    render(); updateHint();
  });
}

// --- key-stage images + report (post-solve) ---------------------------------

async function refreshStages(project) {
  const wrap = document.getElementById("targetless-stages");
  const holder = document.getElementById("stage-imgs");
  if (!wrap) return;
  try {
    const r = await getJSON(`/api/p/${project}/targetless-stages`);
    if (!r.stages.length) { wrap.hidden = true; return; }
    holder.innerHTML = r.stages.map((s) =>
      `<figure class="stage-fig"><figcaption>${s}</figcaption>
       <img src="/targetless-stage/${project}/${s}?t=${Date.now()}" alt="${s}"></figure>`).join("");
    wrap.hidden = false;
  } catch { wrap.hidden = true; }
}

async function refreshReport(project) {
  const wrap = document.getElementById("targetless-report");
  const body = document.getElementById("report-body");
  if (!wrap) return;
  try {
    const r = await getJSON(`/api/p/${project}/targetless-report`);
    if (!r.report) { wrap.hidden = true; return; }
    const rep = r.report;
    body.textContent = (rep.summary_lines || []).join("\n") +
      `\n\naccepted=${rep.accepted} (${rep.acceptance_reason})`;
    wrap.hidden = false;
  } catch { wrap.hidden = true; }
}
