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

  const boardsPanel = document.getElementById("boards-panel");

  const applyMethod = (method) => {
    panel.hidden = method !== "targetless";
    // Boards (AprilGrid): show the read-only stage notebook (galleries of the
    // already-captured pairs + floor + solved result).
    if (boardsPanel) {
      boardsPanel.hidden = method === "targetless";
      if (method === "aprilgrid") window.__boardsNotebook?.refresh();
    }
    // Reflect the two-way toggle's active state.
    if (toggle) {
      toggle.querySelectorAll(".method-opt").forEach((b) =>
        b.classList.toggle("active", b.dataset.method === method));
    }
    // Targetless hides the AprilGrid tag-measurement inputs and derived text (irrelevant).
    document.querySelectorAll(".board-measure-label, #board-derived").forEach((el) => {
      el.style.display = method === "targetless" ? "none" : "";
    });
    // Targetless also hides the top main capture buttons/messages to prevent redundancy.
    document.querySelectorAll("#cap-start, #cap-stop, #cap-restart, #cap-msg").forEach((el) => {
      el.style.display = method === "targetless" ? "none" : "";
    });
    // Hide Boards-specific floor anchor tools and previews in Targetless mode
    const floorTools = document.getElementById("floor-tools");
    const floorLive = document.getElementById("floor-live");
    const floorPrompt = document.getElementById("floor-prompt");
    if (floorTools) floorTools.style.display = method === "targetless" ? "none" : "";
    if (floorLive && method === "targetless") floorLive.hidden = true;
    if (floorPrompt && method === "targetless") floorPrompt.hidden = true;

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
      <figcaption>${c} <span class="counts" data-cam="${c}">—</span></figcaption>
      <div class="canvas-wrap">
        <div class="stream-placeholder">
          <svg class="placeholder-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4zM14 13h-3v3H9v-3H6v-2h3V8h2v3h3v2z"/></svg>
          <span class="placeholder-text">Stream Inactive</span>
        </div>
        <img class="tl-stream" data-cam="${c}" alt="${c} live">
      </div>
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

// Snap radius (source-frame pixels): a cam-A click within this distance of a
// matched keypoint snaps to it and auto-fills the cam-B point from its match.
const SNAP_RADIUS_PX = 24;

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

  // --- snap assist: fetch verified cam_a↔cam_b correspondences ----------------
  // Snap is a COMPLEMENT to manual marking: a cam-A click near a matched keypoint
  // snaps sub-pixel-exact and auto-fills its cam-B match (one click per landmark).
  // If matches are unavailable (no weights / no pair / matcher off) OR no match is
  // near the click, we fall back to the current manual 4-click flow untouched.
  let matches = [];            // [{a:[x,y], b:[x,y], score}]
  let snapEnabled = true;
  const snapToggle = document.getElementById("scale-snap-toggle");
  const snapStatus = document.getElementById("scale-snap-status");
  const setSnapStatus = (txt) => { if (snapStatus) snapStatus.textContent = txt; };
  if (snapToggle) {
    snapToggle.addEventListener("change", () => { snapEnabled = snapToggle.checked; updateHint(); });
  }
  getJSON(`/api/p/${project}/feature-matches`)
    .then((r) => {
      matches = r.matches || [];
      if (r.count > 0) {
        setSnapStatus(`snap ready — ${r.count} matched features`);
        if (snapToggle) snapToggle.disabled = false;
      } else {
        snapEnabled = false;
        if (snapToggle) { snapToggle.checked = false; snapToggle.disabled = true; }
        setSnapStatus(r.reason || "snap unavailable — mark manually");
      }
      updateHint();
    })
    .catch(() => {
      snapEnabled = false;
      if (snapToggle) { snapToggle.checked = false; snapToggle.disabled = true; }
      setSnapStatus("snap unavailable — mark manually");
    });

  // Nearest matched keypoint (in cam A) to a click, within the snap radius.
  const nearestMatch = (x, y) => {
    let best = null;
    let bestD = SNAP_RADIUS_PX;
    for (const m of matches) {
      const d = Math.hypot(m.a[0] - x, m.a[1] - y);
      if (d <= bestD) { bestD = d; best = m; }
    }
    return best;
  };

  // Click sequence for the MANUAL path: p1@cam_a, p1@cam_b, p2@cam_a, p2@cam_b.
  // A snapped cam-A click auto-fills the paired cam-B point, so the sequence
  // advances by two (skipping the cam-B click) for that landmark.
  const order = ["p1_a", "p1_b", "p2_a", "p2_b"];
  const camForStep = [cameras[0], cameras[1], cameras[0], cameras[1]];
  let step = 0;
  let pending = {};
  const snapped = {};          // per-point: true=snapped ●, false=manual ○

  const canAdd = () => step >= order.length && distIn.value > 0;

  const updateHint = () => {
    if (step >= order.length) { hint.textContent = "both points marked — enter distance + Add"; return; }
    const key = order[step];
    const cam = camForStep[step];
    if (key.endsWith("_a") && snapEnabled && matches.length) {
      hint.textContent = `click ${key} on ${cam} (near a landmark → snaps + auto-fills ${cam === cameras[0] ? cameras[1] : ""} cam-B)`;
    } else {
      hint.textContent = `click ${key} on ${cam}`;
    }
  };
  updateHint();

  cameras.forEach((c) => {
    imgs[c].addEventListener("click", (ev) => {
      if (step >= order.length || camForStep[step] !== c) return;
      const rect = imgs[c].getBoundingClientRect();
      const x = (ev.clientX - rect.left) / rect.width * imgs[c].naturalWidth;
      const y = (ev.clientY - rect.top) / rect.height * imgs[c].naturalHeight;
      const key = order[step];

      // Snap only engages on a cam-A click when enabled and a match is near it.
      const m = (key.endsWith("_a") && snapEnabled) ? nearestMatch(x, y) : null;
      if (m) {
        pending[key] = [m.a[0], m.a[1]];        // sub-pixel exact cam-A keypoint
        const bKey = key.replace("_a", "_b");
        pending[bKey] = [m.b[0], m.b[1]];        // auto-filled cam-B match
        snapped[key] = true; snapped[bKey] = true;
        step += 2;                               // cam-B point already placed
      } else {
        pending[key] = [x, y];
        snapped[key] = false;
        step += 1;
      }
      addBtn.disabled = !canAdd();
      updatePendingRow(); updateHint();
    });
  });

  distIn.addEventListener("input", () => { addBtn.disabled = !canAdd(); });

  let refs = [];
  const marker = (key) => snapped[key] ? "● snapped" : "○ manual";
  const updatePendingRow = () => {
    const row = document.getElementById("scale-pending");
    if (!row) return;
    const parts = order
      .filter((k) => pending[k])
      .map((k) => `${k}: [${pending[k].map(Math.round)}] ${marker(k)}`);
    row.textContent = parts.length ? `pending — ${parts.join("  ·  ")}` : "";
  };

  const render = () => {
    list.innerHTML = refs.map((r, i) => {
      const tag = r.snapped ? "●" : "○";
      return `<li>#${i}: ${r.distance_m} m ${tag} (p1 [${r.p1_a.map(Math.round)}]…)</li>`;
    }).join("");
    count.textContent = `${refs.length} reference(s)${refs.length >= 3 ? " ✓" : " (need ≥3)"}`;
  };

  const resetPending = () => {
    pending = {}; step = 0;
    for (const k of order) delete snapped[k];
    updatePendingRow();
  };

  getJSON(`/api/p/${project}/scale-references`)
    .then((r) => { refs = r.references || []; render(); }).catch(() => {});

  addBtn.addEventListener("click", async () => {
    if (step < order.length) return;
    // snapped flag is display-only; the stored ScaleReference shape is unchanged.
    const wasSnapped = !!(snapped.p1_a && snapped.p2_a);
    refs.push({
      p1_a: pending.p1_a, p1_b: pending.p1_b,
      p2_a: pending.p2_a, p2_b: pending.p2_b,
      distance_m: parseFloat(distIn.value), snapped: wasSnapped,
    });
    try {
      await sendJSON(`/api/p/${project}/scale-references`, "PUT", { references: refs });
      resetPending(); distIn.value = ""; addBtn.disabled = true;
      render(); updateHint();
    } catch (e) { flash(count, e.message, false); }
  });

  clearBtn.addEventListener("click", async () => {
    refs = []; resetPending(); addBtn.disabled = true;
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
