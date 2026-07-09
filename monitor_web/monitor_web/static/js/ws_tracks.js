// WebSocket client for the Backbone bus.
//
// Subscribes to /ws/tracks, decodes JSON envelopes (Track2DMessage /
// Track3DMessage shapes), and publishes them on a custom event bus the
// other modules (floor_map, live_overlay, status panel) listen to.
//
// Also drives the green/red header dot and the STATUS sidebar via periodic
// polls to /api/status.

const STATE = window.__monitor_web ?? { wsUrl: "/ws/tracks" };

// In-memory snapshot of the latest tracks; the floor map and overlays read it.
window.__tracks = {
  byId2D: new Map(),     // track_id -> Track2DMessage
  byId3D: new Map(),     // track_id -> Track3DMessage
};

function ingest(msg) {
  if (!msg || !msg.type) return;
  if (msg.type === "track_2d") {
    window.__tracks.byId2D.set(msg.track_id, msg);
  } else if (msg.type === "track_3d") {
    window.__tracks.byId3D.set(msg.track_id, msg);
  } else {
    return;
  }
  document.dispatchEvent(new CustomEvent("tracks:update", { detail: msg }));
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}${STATE.wsUrl}`;
  let ws;
  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.warn("ws_tracks: connection failed", err);
    setTimeout(connect, 2000);
    return;
  }
  ws.addEventListener("message", (event) => {
    try {
      ingest(JSON.parse(event.data));
    } catch (err) {
      console.warn("ws_tracks: bad message", err);
    }
  });
  ws.addEventListener("close", () => setTimeout(connect, 1500));
  ws.addEventListener("error", () => ws.close());
}

// --- STATUS panel + green/red dot via /api/status polling ---

// Friendly names for the readiness checks, used to explain a red light.
const _CHECK_LABELS = {
  config_ok: "config",
  camera_live: "camera",
  model_ok: "model",
  calibration_ok: "calibration",
  sink_ok: "UDP sink",
};

function renderStatus(status) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  // 3-state light from the server (red = blocked / amber = ready / green = live).
  // Fall back to the old udp.fresh heuristic if an older backend omits readiness.
  const r = status.readiness || { light: status.udp.fresh ? "green" : "red", checks: {} };
  const dot = document.getElementById("status-dot");
  if (dot) {
    dot.classList.remove("status-dot-green", "status-dot-amber", "status-dot-red");
    dot.classList.add(`status-dot-${r.light}`);
  }
  // Per-camera liveness → the bigPanel store, so the UNIFIED tab can gate on the
  // cameras ACTUALLY streaming (not merely configured).
  const camsLive = r.cameras_live || {};
  const store = window.Alpine && window.Alpine.store && window.Alpine.store("bigPanel");
  if (store) store.camLive = camsLive;

  const target = document.getElementById("status-content");
  if (!target) return;
  // Human-readable status line + the reason when it's red.
  let stateText;
  if (r.light === "green") {
    stateText = strings.status_live || "Live";
  } else if (r.light === "amber" && r.degraded) {
    stateText = strings.status_degraded || "Degraded — running on one camera";
  } else if (r.light === "amber") {
    stateText = strings.status_ready || "Ready — press Start";
  } else if (status.backbone.state === "crashed") {
    stateText = strings.status_crashed || "Backbone crashed";
  } else {
    const missing = Object.keys(_CHECK_LABELS)
      .filter((k) => r.checks && r.checks[k] === false)
      .map((k) => _CHECK_LABELS[k]);
    stateText = (strings.status_blocked || "Blocked")
      + (missing.length ? `: ${missing.join(", ")}` : "");
  }
  const lat = status.udp.last_envelope_ts
    ? `${Math.max(0, (Date.now() / 1000 - status.udp.last_envelope_ts)).toFixed(2)} s ago`
    : "—";
  // Live KPI rows (real-time): capture→publish latency p95/p50, homography reproj.
  const k = status.kpis || {};
  const fmtLat = (v) => (v == null ? "—" : `${v} ms`);
  const latCls = k.latency_p95_ms == null
    ? "" : (k.latency_p95_ms < (k.latency_target_ms || 200) ? "kpi-ok" : "kpi-bad");
  const reproj = k.reproj_rms_px || {};
  const reprojRows = Object.entries(reproj).map(([cam, v]) => {
    const cls = v <= (k.reproj_target_px || 2) ? "kpi-ok" : "kpi-bad";
    return `<div class="kpi-row"><span class="kpi-key">${strings.reproj_label || "Reproj"} ${cam}</span>`
      + `<span class="kpi-val ${cls}">${v} px<span class="kpi-target">≤2</span></span></div>`;
  }).join("");
  // Per-camera capture FPS from the Backbone's diagnostics heartbeat (empty
  // while stopped/stale → rows hidden). Flagged red under 1 fps (camera
  // effectively stalled).
  const fpsByCam = k.fps_by_camera || {};
  // points mode: the per-camera rate is DETECTIONS/s (the perception tick),
  // not the camera capture fps — label it for what it is.
  const perCamLabel = k.points_mode
    ? (strings.det_rate_label || "Detections") : (strings.fps_label || "FPS");
  const perCamUnit = k.points_mode ? "/s" : " fps";
  const fpsRows = Object.entries(fpsByCam).map(([cam, v]) => {
    const cls = v < 1.0 ? "kpi-bad" : "";
    return `<div class="kpi-row"><span class="kpi-key">${perCamLabel} ${cam}</span>`
      + `<span class="kpi-val ${cls}">${Number(v).toFixed(1)}${perCamUnit}</span></div>`;
  }).join("");
  const pipelineFpsRow = (k.pipeline_fps == null) ? "" :
    `<div class="kpi-row"><span class="kpi-key">${strings.pipeline_fps_label || "FPS pipeline"}</span>`
    + `<span class="kpi-val">${Number(k.pipeline_fps).toFixed(1)} fps</span></div>`;
  // UI lag: how far the dashboard's own bus consumer runs behind the wire —
  // display sluggishness, NOT pipeline latency. Amber above 500 ms.
  const uiLagRow = (k.ui_lag_p50_ms == null) ? "" :
    `<div class="kpi-row"><span class="kpi-key">${strings.ui_lag_label || "UI lag p50"}</span>`
    + `<span class="kpi-val ${k.ui_lag_p50_ms > 500 ? "kpi-bad" : ""}">${fmtLat(k.ui_lag_p50_ms)}</span></div>`;
  const kpiHtml = `
    <div class="kpi-group">
      ${fpsRows}
      ${pipelineFpsRow}
      <div class="kpi-row">
        <span class="kpi-key">${strings.latency_p95 || "Engine latency p95"}</span>
        <span class="kpi-val ${latCls}">${fmtLat(k.latency_p95_ms)}<span class="kpi-target">&lt;200</span></span>
      </div>
      <div class="kpi-row">
        <span class="kpi-key">${strings.latency_p50 || "Engine latency p50"}</span>
        <span class="kpi-val">${fmtLat(k.latency_p50_ms)}</span>
      </div>
      ${uiLagRow}
      ${reprojRows}
    </div>`;
  // Live memory rows (used out of total): GPU VRAM + system RAM (+ GPU util %).
  const res = status.resources || {};
  const fmtMem = (u, t) => (u == null || t == null) ? "—" : `${u} / ${t} MB`;
  const memCls = (u, t) => (u != null && t > 0 && u / t > 0.9) ? "kpi-bad" : "";
  const resHtml = `
    <div class="kpi-group">
      <div class="kpi-row">
        <span class="kpi-key">${strings.vram_label || "VRAM (GPU)"}</span>
        <span class="kpi-val ${memCls(res.vram_used_mb, res.vram_total_mb)}">${fmtMem(res.vram_used_mb, res.vram_total_mb)}</span>
      </div>
      <div class="kpi-row">
        <span class="kpi-key">${strings.ram_label || "RAM (CPU)"}</span>
        <span class="kpi-val ${memCls(res.ram_used_mb, res.ram_total_mb)}">${fmtMem(res.ram_used_mb, res.ram_total_mb)}</span>
      </div>
      ${res.gpu_util_pct == null ? "" : `<div class="kpi-row"><span class="kpi-key">${strings.gpu_util_label || "GPU util"}</span><span class="kpi-val">${res.gpu_util_pct}%</span></div>`}
    </div>`;
  target.innerHTML = `
    <div class="status-row">
      <span class="status-key">${strings.status_label || "Status"}</span>
      <span class="status-value status-${r.light}">${stateText}</span>
    </div>
    <div class="status-row">
      <span class="status-key">${strings.mode_label || "Mode"}</span>
      <span class="status-value">${status.backbone.state}</span>
    </div>
    ${Object.keys(camsLive).length > 1 ? `
    <div class="status-row">
      <span class="status-key">${strings.cameras_label || "Cameras"}</span>
      <span class="status-value">${Object.entries(camsLive)
        .map(([c, v]) => `<span class="status-${v ? "green" : "red"}">${c} ${v ? "●" : "○"}</span>`)
        .join(" ")}</span>
    </div>` : ""}
    <div class="status-row">
      <span class="status-key">UDP</span>
      <span class="status-value">${status.udp.fresh ? "OK" : "stale"} (${status.udp.received})</span>
    </div>
    <div class="status-row">
      <span class="status-key">Tracks 2D</span>
      <span class="status-value">${status.tracks.active_2d}</span>
    </div>
    <div class="status-row">
      <span class="status-key">Tracks 3D</span>
      <span class="status-value">${status.tracks.active_3d}</span>
    </div>
    <div class="status-row">
      <span class="status-key">Last UDP</span>
      <span class="status-value">${lat}</span>
    </div>
    ${resHtml}
    ${kpiHtml}
  `;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) renderStatus(await res.json());
  } catch (err) {
    // ignore — fail quiet, next tick retries
  }
}

function startStatusLoop() {
  pollStatus();
  setInterval(pollStatus, 1000);
}

connect();
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", startStatusLoop);
} else {
  startStatusLoop();
}
