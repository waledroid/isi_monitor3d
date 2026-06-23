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
      : Object.values(s.intrinsic_counts || {}).some((n) => n > 0) ? "partial" : "todo",
    extra: (s) => rmsLine(s.intrinsic_done ? s.rms : null) },
  { key: "extrinsic", n: 2, title: "Extrinsic", capture: true,
    counts: (s) => {
      const pairs = Object.entries(s.extrinsic_counts || {}).map(([c, n]) => `${c}: ${n}`).join(" · ");
      const floors = Object.entries(s.floor || {}).filter(([, v]) => v).map(([c]) => c).join(",");
      return `${pairs || "no pairs"}${floors ? " · floor: " + floors : ""}`;
    },
    state: (s) => s.extrinsic_done ? "done"
      : Object.values(s.extrinsic_counts || {}).some((n) => n > 0) ? "partial" : "todo",
    extra: (s) => rmsLine(s.extrinsic_done ? s.rms : null) },
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
    const glyph = locked ? "🔒" : st === "done" ? "✓" : st === "partial" ? "◐" : "";
    const card = document.createElement("div");
    card.className = "phase-card" + (locked ? " locked" : "") +
      (st === "done" ? " done" : st === "partial" ? " partial" : "");
    card.innerHTML =
      `<div class="phase-head"><span class="phase-num">${ph.n}</span>
         <span class="phase-title">${ph.title}</span>
         <span class="phase-status">${glyph}</span></div>
       <div class="counts">${ph.counts(s)}</div>
       <div class="counts">${ph.extra(s)}</div>
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
setInterval(() => { render(); loadResults(); }, 5000);
