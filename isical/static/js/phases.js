import { getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const board = document.getElementById("phase-board");
const project = board.dataset.project;
const logEl = document.getElementById("job-log");

// 3 phases. Intrinsic/Extrinsic have a capture page ("open ›") + a solve ("Run").
// Export installs the result.
const PHASES = [
  { key: "intrinsic", n: 1, title: "Intrinsic", capture: true,
    counts: (s) => Object.entries(s.intrinsic_counts || {})
      .map(([c, n]) => `${c}: ${n}/${s.targets.intrinsic}`).join(" · ") || "no shots",
    state: (s) => s.intrinsic_done ? "done"
      : s.intrinsic_captured ? "captured"
      : Object.values(s.intrinsic_counts || {}).some((n) => n > 0) ? "partial" : "todo",
    extra: (s) => rmsLine(s.intrinsic_done ? s.rms : null) },
  { key: "extrinsic", n: 2, title: "Extrinsic", capture: true,
    counts: (s) => {
      const pairs = Object.entries(s.extrinsic_counts || {}).map(([c, n]) => `${c}: ${n}`).join(" · ");
      const floors = Object.entries(s.floor || {}).filter(([, v]) => v).map(([c]) => c).join(",");
      return `${pairs || "no pairs"}${floors ? " · floor: " + floors : ""}`;
    },
    // "captured" (blue Solve-now) requires BOTH pairs at target AND floor shots —
    // run_extrinsic fails without floor anchors, so don't offer Solve until ready.
    state: (s) => s.extrinsic_done ? "done"
      : s.extrinsic_solve_ready ? "captured"
      : s.extrinsic_captured ? "needs_floor"
      : Object.values(s.extrinsic_counts || {}).some((n) => n > 0) ? "partial" : "todo",
    extra: (s) => s.extrinsic_done ? rmsLine(s.rms)
      : (s.extrinsic_captured && !s.extrinsic_floor_done)
        ? `Captures done · needs floor shots: ${(s.extrinsic_missing_floor || []).join(", ")}`
        : "" },
  { key: "export", n: 3, title: "Export", capture: false,
    counts: (s) => s.extrinsic_done
      ? (s.installed ? "calibration.json · installed ✓" : "calibration.json ready") : "—",
    state: (s) => s.installed ? "done" : "todo",
    extra: () => "" },
];

function rmsLine(rms) {
  if (!rms) return "";
  const parts = Object.entries(rms).map(([c, v]) =>
    `${c} ${v == null ? "?" : Number(v).toFixed(3)}px`).join(" · ");
  return parts ? `RMS: ${parts}` : "";
}

async function render() {
  let s = {};
  try { s = await getJSON(`/api/p/${project}/status`); } catch { /* keep */ }
  board.innerHTML = "";
  let prevDone = true;
  for (const ph of PHASES) {
    const st = ph.state(s);
    const locked = !prevDone && st !== "done";
    const glyph = locked ? "🔒" : (st === "done" || st === "captured") ? "✓"
      : (st === "partial" || st === "needs_floor") ? "◐" : "";
    const card = document.createElement("div");
    card.className = "phase-card" + (locked ? " locked" : "") +
      (st === "done" ? " done" : st === "captured" ? " captured"
        : (st === "partial" || st === "needs_floor") ? " partial" : "");
    const hint = st === "captured" ? `<div class="counts solve-hint">captured ✓ — Solve now ↓</div>`
      : st === "needs_floor" ? `<div class="counts solve-hint">pairs done — capture floor shots first ↗</div>` : "";
    card.innerHTML =
      `<div class="phase-head"><span class="phase-num">${ph.n}</span>
         <span class="phase-title">${ph.title}</span>
         <span class="phase-status">${glyph}</span></div>
       <div class="counts">${ph.counts(s)}</div>
       <div class="counts">${ph.extra(s)}</div>
       ${hint}
       <div class="phase-actions"></div>`;
    const actions = card.querySelector(".phase-actions");
    if (ph.capture) {
      const a = document.createElement("a");
      a.href = `/p/${project}/capture/${ph.key}`; a.textContent = "open capture ›";
      if (locked) { a.style.pointerEvents = "none"; a.style.opacity = "0.4"; }
      actions.appendChild(a);
    }
    // Export's install toggle
    let installCb = null;
    if (ph.key === "export") {
      const lbl = document.createElement("label");
      lbl.className = "chip"; lbl.style.marginRight = "8px";
      lbl.title = "Also copy to config/mode2/calibration.json (what the Backbone + dashboard load)";
      lbl.innerHTML = `<input type="checkbox" class="install-toggle" checked> install to live system`;
      actions.appendChild(lbl);
      installCb = lbl.querySelector("input");
    }
    const b = document.createElement("button");
    b.textContent = ph.key === "export" ? "Export"
      : st === "done" ? "Re-solve" : "Solve";
    if (locked) { b.disabled = true; b.title = "complete the previous phase first"; }
    else if (st === "needs_floor") {
      b.disabled = true;
      b.title = "capture a ChArUco floor shot for each camera (the world anchor) before solving";
    }
    b.onclick = async () => {
      const body = ph.key === "export" && installCb ? { install: installCb.checked } : {};
      try {
        const { job } = await sendJSON(`/api/p/${project}/run/${ph.key}`, "POST", body);
        watchJob(job.id, logEl, render);
      } catch (e) { logEl.textContent = `error: ${e.message}`; }
    };
    actions.appendChild(b);
    board.appendChild(card);
    prevDone = st === "done";
  }
}

// ---- calibration results + live stream-sync probe ----
function rmsBadge(v, gate) {
  if (v == null) return "<span class='msg'>?</span>";
  const ok = Number(v) <= gate;
  return `<b class="${ok ? "ok" : "bad"}">${Number(v).toFixed(3)} px ${ok ? "✓" : "✗"}</b>`;
}
async function loadResults() {
  const card = document.getElementById("results-card");
  if (!card) return;
  try {
    const { summary } = await getJSON(`/api/p/${project}/calibration-summary`);
    if (!summary) { card.hidden = true; return; }
    card.hidden = false;
    const rows = Object.entries(summary.cameras).map(([cid, c]) =>
      `<tr><td><b>${cid}</b></td>
         <td>${rmsBadge(c.reprojection_rms_px, summary.rms_gate_px)}</td>
         <td>${c.image_size}</td>
         <td>f=${c.focal_px.join(", ")}</td>
         <td>c=${c.principal_px.join(", ")}</td>
         <td>pos=[${c.position_m.join(", ")}] m</td></tr>`).join("");
    const baseline = summary.baseline_m != null
      ? `<p class="msg">Camera separation (baseline): <b>${summary.baseline_m} m</b> ·
         floor anchor: ${summary.floor_anchor || "?"} · mode: ${summary.calibration_mode || "?"}</p>` : "";
    document.getElementById("cal-summary").innerHTML =
      `<table class="cal-table"><thead><tr>
         <th>camera</th><th>reproj RMS (gate ${summary.rms_gate_px}px)</th><th>image</th>
         <th>focal px</th><th>principal px</th><th>world position</th></tr></thead>
       <tbody>${rows}</tbody></table>${baseline}`;
  } catch { card.hidden = true; }
}
document.getElementById("sync-probe-btn")?.addEventListener("click", async () => {
  const btn = document.getElementById("sync-probe-btn");
  const msg = document.getElementById("sync-msg");
  const out = document.getElementById("sync-result");
  btn.disabled = true; msg.textContent = "probing both cameras for 4 s…";
  try {
    const r = await getJSON(`/api/p/${project}/sync-probe?seconds=4`);
    const camRows = Object.entries(r.cameras).map(([cid, c]) => c.error
      ? `<tr><td><b>${cid}</b></td><td colspan="3" class="bad">${c.error}</td></tr>`
      : `<tr><td><b>${cid}</b></td><td>${c.fps} fps</td>
           <td>interval ${c.mean_interval_ms ?? "?"} ms</td>
           <td>jitter ±${c.jitter_ms ?? "?"} ms</td></tr>`).join("");
    let sync = "";
    if (r.sync && r.sync.pairs) {
      const s = r.sync;
      const ok = s.in_window_pct >= 80;
      sync = `<p class="msg">Inter-camera skew (${s.pair}): mean <b>${s.mean_skew_ms} ms</b> ·
        p95 ${s.p95_skew_ms} ms · max ${s.max_skew_ms} ms ·
        <b class="${ok ? "ok" : "bad"}">${s.in_window_pct}%</b> within the ${s.window_ms} ms
        sync window</p>`;
    } else if (r.sync) {
      sync = `<p class="msg">sync: not enough frames from one camera</p>`;
    }
    out.innerHTML = `<table class="cal-table"><tbody>${camRows}</tbody></table>${sync}`;
    msg.textContent = "";
  } catch (e) { msg.textContent = e.message; msg.className = "msg bad"; }
  finally { btn.disabled = false; }
});

document.getElementById("refresh-board")?.addEventListener("click", render);

// ---- intrinsic results panel (per-camera K + distortion + RMS) ----
function _fmtNum(v, dec) { return v == null ? "?" : Number(v).toFixed(dec); }

function _matrixHtml(K, dec = 2) {
  // matrix rendered as a monospace grid with bracket borders (K uses 2 dp;
  // the extrinsic [R|t] passes 4 dp so the near-±1 rotation stays readable).
  const rows = K.map((row) =>
    `<tr>${row.map((v) => `<td class="kmat-cell">${_fmtNum(v, dec)}</td>`).join("")}</tr>`
  ).join("");
  return `<div class="kmat-wrap">
    <span class="kmat-bracket">[</span>
    <table class="kmat"><tbody>${rows}</tbody></table>
    <span class="kmat-bracket">]</span>
  </div>`;
}

function _renderIntrinsicCamera(cam, gate) {
  const rmsOk = cam.rms != null && cam.rms <= gate;
  const rmsBadgeClass = cam.rms == null ? "msg" : (rmsOk ? "ok" : "bad");
  const rmsTxt = cam.rms == null
    ? "? (no sidecar — re-solve to populate)"
    : `${Number(cam.rms).toFixed(4)} px ${rmsOk ? "✓" : "✗"} (gate ${gate} px)`;
  const dist = (cam.dist || []).map((v) => _fmtNum(v, 6)).join(", ");
  const sz = cam.image_size ? `${cam.image_size[0]} × ${cam.image_size[1]}` : "?";
  return `
    <div class="intr-cam-body">
      <div class="intr-section">
        <div class="intr-label">Intrinsic matrix K</div>
        ${_matrixHtml(cam.K)}
      </div>
      <table class="cal-table intr-detail">
        <tbody>
          <tr><td class="intr-key">f<sub>x</sub> / f<sub>y</sub></td>
              <td>${_fmtNum(cam.fx, 2)} / ${_fmtNum(cam.fy, 2)} px</td></tr>
          <tr><td class="intr-key">c<sub>x</sub> / c<sub>y</sub></td>
              <td>${_fmtNum(cam.cx, 2)} / ${_fmtNum(cam.cy, 2)} px</td></tr>
          <tr><td class="intr-key">image size</td>
              <td>${sz}</td></tr>
          <tr><td class="intr-key">dist (k1,k2,p1,p2,k3)</td>
              <td><code>${dist}</code></td></tr>
          <tr><td class="intr-key">reproj RMS</td>
              <td><b class="${rmsBadgeClass}">${rmsTxt}</b></td></tr>
        </tbody>
      </table>
    </div>`;
}

let _intr_active_tab = null;

function _switchIntrTab(cid) {
  _intr_active_tab = cid;
  document.querySelectorAll(".intr-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.cam === cid);
  });
  document.querySelectorAll(".intr-panel").forEach((panel) => {
    panel.hidden = panel.dataset.cam !== cid;
  });
}

async function loadIntrinsicResults() {
  const card = document.getElementById("intrinsic-results-card");
  const tabsEl = document.getElementById("intr-tabs");
  const panelsEl = document.getElementById("intr-panels");
  if (!card) return;
  try {
    const data = await getJSON(`/api/p/${project}/intrinsic-summary`);
    const cams = data.cameras || {};
    const camKeys = Object.keys(cams);
    if (camKeys.length === 0) { card.hidden = true; return; }
    card.hidden = false;
    const gate = data.rms_gate_px;

    // build tabs
    tabsEl.innerHTML = camKeys.map((cid) =>
      `<button class="cam-tab intr-tab" data-cam="${cid}">${cid}</button>`
    ).join("");

    // build panels
    panelsEl.innerHTML = camKeys.map((cid) =>
      `<div class="intr-panel" data-cam="${cid}" hidden>${_renderIntrinsicCamera(cams[cid], gate)}</div>`
    ).join("");

    // wire tab clicks
    tabsEl.querySelectorAll(".intr-tab").forEach((btn) => {
      btn.addEventListener("click", () => _switchIntrTab(btn.dataset.cam));
    });

    // restore or default to first tab
    const toShow = (camKeys.includes(_intr_active_tab) ? _intr_active_tab : camKeys[0]);
    _switchIntrTab(toShow);
  } catch { card.hidden = true; }
}

// ---- extrinsic results panel (per-camera [R|t] pose — same format as K) ----
function _renderExtrinsicCamera(cam, gate, baseline) {
  const rmsOk = cam.rms != null && cam.rms <= gate;
  const rmsBadgeClass = cam.rms == null ? "msg" : (rmsOk ? "ok" : "bad");
  const rmsTxt = cam.rms == null
    ? "? (re-solve to populate)"
    : `${Number(cam.rms).toFixed(4)} px ${rmsOk ? "✓" : "✗"} (gate ${gate} px)`;
  // [R | t] as a 3×4 extrinsic matrix (rotation columns + translation column).
  const Rt = cam.R.map((row, i) => [...row, cam.t[i]]);
  const tTxt = (cam.t || []).map((v) => _fmtNum(v, 4)).join(", ");
  const baseTxt = baseline != null ? `${_fmtNum(baseline, 3)} m` : "—";
  return `
    <div class="intr-cam-body">
      <div class="intr-section">
        <div class="intr-label">Extrinsic matrix [R | t] <span class="msg">world → camera</span></div>
        ${_matrixHtml(Rt, 4)}
      </div>
      <table class="cal-table intr-detail">
        <tbody>
          <tr><td class="intr-key">translation t (m)</td>
              <td><code>${tTxt}</code></td></tr>
          <tr><td class="intr-key">camera baseline</td>
              <td>${baseTxt}</td></tr>
          <tr><td class="intr-key">reproj RMS</td>
              <td><b class="${rmsBadgeClass}">${rmsTxt}</b></td></tr>
        </tbody>
      </table>
    </div>`;
}

let _extr_active_tab = null;

function _switchExtrTab(cid) {
  _extr_active_tab = cid;
  document.querySelectorAll(".extr-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.cam === cid);
  });
  document.querySelectorAll(".extr-panel").forEach((panel) => {
    panel.hidden = panel.dataset.cam !== cid;
  });
}

async function loadExtrinsicResults() {
  const card = document.getElementById("extrinsic-results-card");
  const tabsEl = document.getElementById("extr-tabs");
  const panelsEl = document.getElementById("extr-panels");
  if (!card) return;
  try {
    const data = await getJSON(`/api/p/${project}/extrinsic-summary`);
    const cams = data.cameras || {};
    const camKeys = Object.keys(cams);
    if (camKeys.length === 0) { card.hidden = true; return; }
    card.hidden = false;
    const gate = data.rms_gate_px;
    const baseline = data.baseline_m;

    tabsEl.innerHTML = camKeys.map((cid) =>
      `<button class="cam-tab extr-tab" data-cam="${cid}">${cid}</button>`
    ).join("");

    panelsEl.innerHTML = camKeys.map((cid) =>
      `<div class="extr-panel" data-cam="${cid}" hidden>${_renderExtrinsicCamera(cams[cid], gate, baseline)}</div>`
    ).join("");

    tabsEl.querySelectorAll(".extr-tab").forEach((btn) => {
      btn.addEventListener("click", () => _switchExtrTab(btn.dataset.cam));
    });

    const toShow = (camKeys.includes(_extr_active_tab) ? _extr_active_tab : camKeys[0]);
    _switchExtrTab(toShow);
  } catch { card.hidden = true; }
}

// ---- cameras editor (always visible, pre-filled; add cam_b later / fix a URL) ----
const camsForm = document.getElementById("cams-form");
async function loadCams() {
  try {
    const c = await getJSON(`/api/p/${project}/cameras`);
    const set = (slot, spec) => {
      const t = spec?.type || "rtsp";
      camsForm.elements[`${slot}_type`].value = t;
      camsForm.elements[`${slot}_src`].value = spec ? (t === "usb" ? spec.device : spec.url) : "";
    };
    set("a", c.cam_a); set("b", c.cam_b);
  } catch { /* */ }
}
loadCams();          // pre-fill the saved links on page load (no button needed)
camsForm?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(camsForm);
  const cam = (type, src) => {
    const s = String(src || "").trim();
    return String(type) === "usb" ? { type: "usb", device: s } : { type: "rtsp", url: s };
  };
  const body = { cam_a: cam(f.get("a_type"), f.get("a_src")) };
  const bsrc = String(f.get("b_src") || "").trim();
  if (bsrc) body.cam_b = cam(f.get("b_type"), bsrc);
  const msg = document.getElementById("cams-msg");
  try {
    const r = await sendJSON(`/api/p/${project}/cameras`, "PUT", body);
    msg.textContent = `saved · ${r.cameras.join(", ")} (${r.mode2 ? "Mode 2" : "Mode 1"})`;
    setTimeout(() => location.reload(), 600);     // refresh chips + capture views
  } catch (e) { msg.textContent = e.message; msg.className = "msg bad"; }
});

render();
loadResults();
loadIntrinsicResults();
loadExtrinsicResults();
setInterval(() => {
  render(); loadResults(); loadIntrinsicResults(); loadExtrinsicResults();
}, 5000);
