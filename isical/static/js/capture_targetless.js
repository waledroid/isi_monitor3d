// Targetless extrinsics UI (experimental) — notebook-style, stage-by-stage.
//
// Loaded only on the extrinsic capture page. The Boards (AprilGrid) path is
// untouched; this module toggles a targetless notebook panel when the method
// selector is "targetless". Layout (top → bottom, like Jupyter cells):
//
//   top   : BOTH live cam views (cam_a | cam_b) + Capture pairs / Stop / Solve.
//   cell ①: captured stereo pairs   (stage image "pair")
//   cell ②: feature matches         (stage image "matches")
//   cell ③: scale references        (interactive ≥3-point marking + "scale_refs")
//   cell ④: triangulation + floor   (stage image "triangulation")
//   cell ⑤: result                  (R/t matrices text + validation summary)
//
// Session lifecycle (capturing sync pairs, floor prompt, status polling) stays
// owned by capture.js — the targetless top controls delegate to its hidden
// #cap-start / #cap-stop buttons so nothing is duplicated. Solve POSTs the
// extrinsic job then polls /api/jobs; on completion the cells refresh.

import { getJSON, sendJSON, flash } from "./api.js";

export function initTargetless(root) {
  const project = root.dataset.project;
  const phase = root.dataset.phase;
  if (phase !== "extrinsic") return;
  const cameras = JSON.parse(root.dataset.cameras || "[]");

  const sel = document.getElementById("extrinsic-method");
  const toggle = document.getElementById("method-toggle");
  const methodMsg = document.getElementById("method-msg");
  const panel = document.getElementById("targetless-panel");
  if (!sel || !panel) return;

  const applyMethod = (method) => {
    panel.hidden = method !== "targetless";
    // Reflect the two-way toggle's active state.
    if (toggle) {
      toggle.querySelectorAll(".method-opt").forEach((b) =>
        b.classList.toggle("active", b.dataset.method === method));
    }
    // Targetless hides the AprilGrid tag-measurement inputs (irrelevant).
    document.querySelectorAll(".board-measure-label").forEach((el) => {
      el.style.display = method === "targetless" ? "none" : "";
    });
    // The default capture fourup IS redundant in targetless mode (its live views
    // are re-hosted at the top of the notebook), so hide it there.
    const capViews = document.getElementById("cap-views");
    if (capViews) capViews.style.display = method === "targetless" ? "none" : "";
    if (method === "targetless") mirrorLiveViews(project, cameras);
  };

  getJSON(`/api/p/${project}/extrinsic-method`)
    .then((r) => { sel.value = r.method; applyMethod(r.method); })
    .catch(() => {});

  const setMethod = async (method) => {
    try {
      await sendJSON(`/api/p/${project}/extrinsic-method`, "PUT", { method });
      sel.value = method;
      applyMethod(method);
      flash(methodMsg, `method: ${method}`);
    } catch (e) { flash(methodMsg, e.message, false); }
  };
  if (toggle) {
    toggle.querySelectorAll(".method-opt").forEach((b) =>
      b.addEventListener("click", () => setMethod(b.dataset.method)));
  }
  sel.addEventListener("change", () => setMethod(sel.value));

  setupLiveControls(project);
  setupScaleMarking(project, cameras);
  refreshStages(project);
  refreshReport(project);
  refreshResult(project);
}

// --- top: both live views + capture/solve controls --------------------------

// Re-host cam_a | cam_b live streams at the top of the notebook so the operator
// aims for overlap/texture. Uses the same MJPEG /stream transport the cam views
// use; the <img>s only carry a src while a capture session is running.
function mirrorLiveViews(project, cameras) {
  const wrap = document.getElementById("tl-live-views");
  if (!wrap || wrap.dataset.built) return;
  wrap.dataset.built = "1";
  wrap.innerHTML = cameras.map((c) => `
    <figure class="cap-figure" data-cam="${c}">
      <figcaption>${c}</figcaption>
      <div class="canvas-wrap"><img class="tl-stream" data-cam="${c}" alt="${c} live"></div>
    </figure>`).join("");
}

function tlStreams(project, on) {
  document.querySelectorAll(".tl-stream").forEach((img) => {
    img.src = on ? `/stream/${project}/${img.dataset.cam}?t=${Date.now()}` : "";
  });
}

function setupLiveControls(project) {
  const capBtn = document.getElementById("tl-capture");
  const stopBtn = document.getElementById("tl-stop");
  const solveBtn = document.getElementById("tl-solve");
  const msg = document.getElementById("tl-live-msg");
  // Delegate session lifecycle to capture.js's hidden main buttons.
  const mainStart = document.getElementById("cap-start");
  const mainStop = document.getElementById("cap-stop");
  if (!capBtn) return;

  capBtn.addEventListener("click", () => {
    mainStart?.click();
    tlStreams(project, true);
    capBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    flash(msg, "capturing sync pairs — aim for floor overlap + texture", true);
  });
  stopBtn?.addEventListener("click", () => {
    mainStop?.click();
    tlStreams(project, false);
    capBtn.disabled = false;
    stopBtn.disabled = true;
    flash(msg, "stopped", true);
  });

  solveBtn?.addEventListener("click", async () => {
    // Capture is exclusive with solving — stop any live session first.
    mainStop?.click();
    tlStreams(project, false);
    if (stopBtn) stopBtn.disabled = true;
    capBtn.disabled = false;
    solveBtn.disabled = true;
    flash(msg, "solving… (SuperPoint+LightGlue → BA → floor fit)", true);
    try {
      await sendJSON(`/api/p/${project}/run/extrinsic`, "POST", {});
      await waitForJob(project, msg);
    } catch (e) {
      flash(msg, e.message, false);
    } finally {
      solveBtn.disabled = false;
      // Refresh the cells regardless — a failed solve still updates the report.
      refreshStages(project);
      refreshReport(project);
      refreshResult(project);
    }
  });
}

async function waitForJob(project, msg) {
  for (let i = 0; i < 600; i += 1) {
    let jobs = [];
    try { jobs = (await getJSON(`/api/jobs`)).jobs || []; } catch { /* */ }
    const j = jobs.find((x) => x.phase === "extrinsic" && x.project === project);
    if (j && (j.state === "done" || j.state === "failed")) {
      flash(msg, j.state === "done" ? "solve complete ✓" : `solve failed: ${j.error || ""}`,
            j.state === "done");
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  flash(msg, "solve still running — check the jobs panel", true);
}

// --- cell ③: interactive scale-reference marking ----------------------------

function setupScaleMarking(project, cameras) {
  const views = document.getElementById("scale-mark-views");
  const list = document.getElementById("scale-ref-list");
  const distIn = document.getElementById("scale-dist");
  const addBtn = document.getElementById("scale-add");
  const clearBtn = document.getElementById("scale-clear");
  const count = document.getElementById("scale-count");
  const hint = document.getElementById("scale-click-hint");
  if (!views) return;

  // Live stream figures per camera (reuse the MJPEG /stream transport), overlaid
  // with clicked marks.
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

// --- cells: stage images fill in as the solve runs --------------------------

// Map each stage image to its notebook cell + status label.
const _STAGE_CELL = {
  pair: "cell-pair",
  matches: "cell-matches",
  scale_refs: "cell-scale",
  triangulation: "cell-triangulation",
  result: "cell-result",
};

function revealCell(cellId) {
  const cell = document.getElementById(cellId);
  if (!cell) return;
  const motionOk = !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (motionOk) cell.classList.add("nb-reveal");
  cell.classList.add("nb-has-output");
}

async function refreshStages(project) {
  let stages = [];
  try { stages = (await getJSON(`/api/p/${project}/targetless-stages`)).stages || []; }
  catch { return; }
  const have = new Set(stages);
  for (const [stage, cellId] of Object.entries(_STAGE_CELL)) {
    const cell = document.getElementById(cellId);
    if (!cell) continue;
    const status = cell.querySelector(".nb-status");
    if (!have.has(stage)) continue;
    const src = `/targetless-stage/${project}/${stage}?t=${Date.now()}`;
    if (stage === "scale_refs") {
      // Cell ③ keeps its interactive marking UI; append the solve output image.
      const fig = document.getElementById("scale-stage-fig");
      const img = document.getElementById("scale-stage-img");
      if (fig && img) { img.src = src; fig.hidden = false; }
    } else {
      const out = cell.querySelector(".nb-out");
      if (out) {
        out.innerHTML = `<figure class="stage-fig"><img src="${src}" alt="${stage}"></figure>`;
      }
    }
    if (status && stage !== "scale_refs") { status.textContent = "done ✓"; status.classList.add("ok"); }
    revealCell(cellId);
  }
}

async function refreshResult(project) {
  const pre = document.getElementById("result-matrices");
  const status = document.querySelector("#cell-result .nb-status");
  if (!pre) return;
  let m = null;
  try { m = (await getJSON(`/api/p/${project}/calibration-matrices`)).matrices; }
  catch { return; }
  if (!m || !m.cameras || !Object.keys(m.cameras).length) return;
  pre.textContent = formatMatrices(m);
  if (status) { status.textContent = "solved ✓"; status.classList.add("ok"); }
  revealCell("cell-result");
}

// Notebook print-out of each camera's R (3x3) / t (3x1) + RMS, monospace-aligned.
function formatMatrices(m) {
  const fmt = (x) => {
    const s = (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(4);
    return s.padStart(9);
  };
  const lines = [];
  if (m.floor_anchor_method) lines.push(`floor anchor : ${m.floor_anchor_method}`);
  if (m.calibration_mode) lines.push(`calib mode   : ${m.calibration_mode}`);
  lines.push("");
  for (const [cid, cam] of Object.entries(m.cameras)) {
    lines.push(`── ${cid} ─────────────────────────────`);
    const R = cam.R || [[null, null, null], [null, null, null], [null, null, null]];
    const t = cam.t || [null, null, null];
    lines.push("R =");
    for (let i = 0; i < 3; i += 1) {
      const row = (R[i] || []).slice(0, 3);
      while (row.length < 3) row.push(null);
      lines.push(`   [ ${row.map(fmt).join("  ")} ]`);
    }
    const tv = Array.isArray(t) ? t.slice(0, 3) : [null, null, null];
    while (tv.length < 3) tv.push(null);
    lines.push(`t = [ ${tv.map(fmt).join("  ")} ]  (m)`);
    const rms = cam.reprojection_rms_px;
    lines.push(`reprojection RMS = ${rms == null ? "—" : Number(rms).toFixed(4)} px`);
    lines.push("");
  }
  return lines.join("\n");
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
