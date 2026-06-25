// comms_nodes.js — Warehouse-wide node status panel.
//
// Polls GET /api/gateway/nodes every 3 s and renders a node-per-row table
// into #comms-nodes-content.  Matches the style conventions of ws_tracks.js
// (kpi-row / kpi-key / kpi-val / status-dot CSS classes).
//
// Mode is implicit: NO gateway URL = on-prem (the default) — a calm local
// indicator, no nag. A gateway URL makes it "Online" and lists warehouse nodes.
// Rendering states:
//   configured:false  → calm "On-prem · local" (default; no gateway connected).
//   error             → amber warning line (gateway unreachable).
//   nodes[]           → "Online" header + one row per node (sorted; alive count).

const _POLL_INTERVAL_MS = 3000;

// Shorten mode names for the compact sidebar display.
function _shortMode(mode) {
  if (!mode) return "—";
  if (mode === "dual_cam_homography_triangulation") return "2-cam";
  if (mode === "single_cam_homography") return "1-cam";
  return mode;
}

function _renderNodes(data) {
  const target = document.getElementById("comms-nodes-content");
  if (!target) return;

  // No gateway URL → on-prem mode (the default). Calm local indicator, no nag;
  // online only engages once a Gateway URL is set in Settings → Communication.
  if (!data.configured) {
    target.innerHTML =
      '<div class="status-row comms-nodes-header">' +
      '<span class="status-dot status-dot-green" aria-hidden="true"></span>' +
      '<span class="kpi-key">On-prem · local</span></div>' +
      '<p class="comms-nodes-hint comms-nodes-subtle">Add a Gateway URL in ' +
      'Settings → Communication for the warehouse view.</p>';
    return;
  }

  // Gateway configured but unreachable.
  if (data.error) {
    target.innerHTML =
      `<div class="comms-nodes-error">Gateway unreachable: ${_esc(data.error)}</div>`;
    return;
  }

  const nodes = Array.isArray(data.nodes) ? data.nodes : [];
  if (nodes.length === 0) {
    target.innerHTML = '<p class="comms-nodes-hint">No nodes reporting yet.</p>';
    return;
  }

  // Sort by node_id (lexicographic).
  const sorted = [...nodes].sort((a, b) =>
    (a.node_id || "").localeCompare(b.node_id || ""),
  );
  const aliveCount = sorted.filter((n) => n.status === "alive").length;

  const header = `
    <div class="status-row comms-nodes-header">
      <span class="status-dot status-dot-green" aria-hidden="true"></span>
      <span class="kpi-key">Online · ${aliveCount}/${sorted.length} alive</span>
    </div>`;

  const rows = sorted.map((n) => {
    const alive = n.status === "alive";
    const dotCls = alive ? "status-dot status-dot-green" : "status-dot status-dot-grey";
    const latency = (n.latency_ms != null) ? `${n.latency_ms} ms` : "—";
    const fps = (n.fps != null) ? `${Number(n.fps).toFixed(1)} fps` : "—";
    const area = n.area ? `· ${_esc(n.area)}` : "";
    return `
      <div class="status-row comms-node-row">
        <span class="comms-node-id">${_esc(n.node_id)}</span>
        <span class="${dotCls}" aria-label="${alive ? "alive" : "stale"}"></span>
        <span class="kpi-key">${area} ${_esc(_shortMode(n.mode))}</span>
        <span class="kpi-val comms-node-kpis">${fps} · p95 ${latency}</span>
      </div>`;
  }).join("");

  target.innerHTML = header + rows;
}

function _esc(s) {
  // Minimal HTML-escape for values interpolated into innerHTML.
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function _pollNodes() {
  try {
    const res = await fetch("/api/gateway/nodes");
    if (res.ok) _renderNodes(await res.json());
  } catch (_err) {
    // Ignore — next tick retries; avoids console spam on hot-reload.
  }
}

function _startNodesLoop() {
  _pollNodes();
  setInterval(_pollNodes, _POLL_INTERVAL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _startNodesLoop);
} else {
  _startNodesLoop();
}
