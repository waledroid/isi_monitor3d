// Boards (AprilGrid) extrinsics — notebook-style, stage-by-stage.
//
// A READ-ONLY view layer over the extrinsic phase's already-captured data. It
// does NOT capture, solve, or write anything — the Capture / Solve controls at
// the top of the page (owned by capture.js) stay fully functional; this module
// only reads the existing shots + calibration and renders them as Jupyter-style
// cells when the extrinsic-method toggle is on "Boards":
//
//   cell ①: extrinsic pairs — galleries of the captured cam_a | cam_b AprilGrid
//           pairs (GET /api/p/{name}/shots/extrinsic/{cam}), count vs target.
//   cell ②: floor anchor   — per-camera ChArUco-on-floor world-anchor shots
//           (GET /api/p/{name}/floor-shots); "awaiting floor shots" if pending.
//   cell ③: result         — R/t + reprojection RMS per camera once solved
//           (GET /api/p/{name}/calibration-matrices).
//
// Selecting Boards on the toggle calls window.__boardsNotebook.refresh() (wired
// from capture_targetless.js's applyMethod), so galleries reflect the latest
// on-disk state each time the branch is shown.

import { getJSON } from "./api.js";

export function initBoards(root) {
  const project = root.dataset.project;
  const phase = root.dataset.phase;
  if (phase !== "extrinsic") return;
  const cameras = JSON.parse(root.dataset.cameras || "[]");
  const panel = document.getElementById("boards-panel");
  if (!panel) return;

  // Re-host the Cam A | Cam B live views at the TOP of the Boards panel, directly
  // above the heading (vertical order: [cam views] → title → notebook cells). The
  // views keep the same class-based wiring capture.js drives (streams/showOnly), so
  // this DOM move is transparent to the capture loop. Idempotent.
  const capViews = document.getElementById("cap-views");
  if (capViews && capViews.parentElement !== panel) {
    panel.insertBefore(capViews, panel.firstChild);
  }

  const refresh = () => {
    refreshPairs(project, cameras);
    refreshFloor(project, cameras);
    refreshResult(project);
  };
  // Expose so the method toggle can re-read on each show.
  window.__boardsNotebook = { refresh };
  refresh();
}

function revealCell(cellId) {
  const cell = document.getElementById(cellId);
  if (!cell) return;
  const motionOk = !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (motionOk) cell.classList.add("nb-reveal");
  cell.classList.add("nb-has-output");
}

// --- cell ①: captured synchronized AprilGrid pairs, both cameras -------------
function thumbHTML(project, cam, s, blurMin) {
  const sharp = s.blur_var >= blurMin * 1.5 ? "ok" : s.blur_var >= blurMin ? "warn" : "bad";
  return `<figure class="shot">
    <img loading="lazy" src="/shots/${project}/extrinsic/${cam}/${s.file}" alt="${s.file}">
    <figcaption><span class="badge">${s.corners} ⌗</span>
      <span class="dot ${sharp}" title="sharpness ${Math.round(s.blur_var)}"></span></figcaption>
  </figure>`;
}

async function refreshPairs(project, cameras) {
  const out = document.getElementById("boards-pairs-out");
  const status = document.getElementById("boards-pairs-status");
  if (!out) return;
  const results = await Promise.all(cameras.map(async (cam) => {
    try { return [cam, await getJSON(`/api/p/${project}/shots/extrinsic/${cam}`)]; }
    catch { return [cam, null]; }
  }));
  const valid = results.filter(([, r]) => r);
  if (!valid.length) {
    out.innerHTML = `<div class="nb-placeholder">no extrinsic pairs captured yet — use Start capture above</div>`;
    if (status) { status.textContent = "no pairs yet"; status.classList.remove("ok"); }
    return;
  }
  const target = valid[0][1].target || 0;
  const minCount = Math.min(...valid.map(([, r]) => r.count));
  out.innerHTML = valid.map(([cam, r]) => {
    const blurMin = r.blur_min_var || 80;
    const grid = r.shots.length
      ? `<div class="shot-grid">${r.shots.map((s) => thumbHTML(project, cam, s, blurMin)).join("")}</div>`
      : `<div class="nb-placeholder">no shots for ${cam}</div>`;
    return `<figure class="stage-fig">
      <figcaption>${cam} — ${r.count}/${r.target} pairs</figcaption>
      ${grid}
    </figure>`;
  }).join("");
  if (status) {
    const ok = target > 0 && minCount >= target;
    status.textContent = `${minCount}/${target} synchronized pairs${ok ? " ✓" : ""}`;
    status.classList.toggle("ok", ok);
  }
  revealCell("bcell-pairs");
}

// --- cell ②: per-camera floor-anchor ChArUco shots ---------------------------
async function refreshFloor(project, cameras) {
  const out = document.getElementById("boards-floor-out");
  const status = document.getElementById("boards-floor-status");
  if (!out) return;
  let cams = {};
  try { cams = (await getJSON(`/api/p/${project}/floor-shots`)).cameras || {}; }
  catch { return; }
  const present = cameras.filter((c) => cams[c]?.present);
  if (!present.length) {
    out.innerHTML = `<div class="nb-placeholder">awaiting floor pairs — press [FLOOR] above and lay the
      ChArUco FLAT on the floor in the overlap; synchronized pairs auto-snap for both cameras</div>`;
    if (status) { status.textContent = `0/${cameras.length} floor shots`; status.classList.remove("ok"); }
    return;
  }
  out.innerHTML = cameras.map((cam) => {
    const info = cams[cam];
    const files = info?.files || [];
    if (files.length) {
      const grid = files.map((f) => `<figure class="shot">
        <img loading="lazy" src="/floor-shot/${project}/${f}?t=${Date.now()}" alt="${cam} floor">
      </figure>`).join("");
      return `<figure class="stage-fig">
        <figcaption>${cam} — floor anchor ✓ (${files.length} placement${files.length > 1 ? "s" : ""})</figcaption>
        <div class="shot-grid">${grid}</div>
      </figure>`;
    }
    return `<figure class="stage-fig">
      <figcaption>${cam} — awaiting floor shot</figcaption>
      <div class="nb-placeholder">not captured yet</div>
    </figure>`;
  }).join("");
  if (status) {
    const ok = present.length === cameras.length;
    status.textContent = `${present.length}/${cameras.length} cameras with floor pairs${ok ? " ✓" : ""}`;
    status.classList.toggle("ok", ok);
  }
  revealCell("bcell-floor");
}

// --- cell ③: solved extrinsic R / t + reprojection RMS per camera ------------
async function refreshResult(project) {
  const pre = document.getElementById("boards-result-matrices");
  const status = document.getElementById("boards-result-status");
  const placeholder = document.getElementById("boards-result-placeholder");
  if (!pre) return;
  let m = null;
  try { m = (await getJSON(`/api/p/${project}/calibration-matrices`)).matrices; }
  catch { return; }
  if (!m || !m.cameras || !Object.keys(m.cameras).length) {
    if (status) { status.textContent = "not solved yet"; status.classList.remove("ok"); }
    if (placeholder) placeholder.hidden = false;
    pre.textContent = "";
    return;
  }
  pre.textContent = formatMatrices(m);
  if (placeholder) placeholder.hidden = true;
  if (status) { status.textContent = "solved ✓"; status.classList.add("ok"); }
  revealCell("bcell-result");
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
