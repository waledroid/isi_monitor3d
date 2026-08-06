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

// Below the node rows it renders one ZONE CARD per configured floor zone —
// a dashed outline in the zone's kind colour, dimmed/transparent when no live
// state is flowing, colour-filled (translucent) when the Backbone/broker are
// delivering the per-zone MQTT `zone_state` (palette present/vide + objects).
// State sources, merged per zone name: gateway /api/gateway/zones (the
// MQTT-retained multi-node view) wins; else the local UDP bus via
// /api/zones/state (fresh only). The card LIST comes from /api/zones
// (zones.yaml — offline-safe), so cards render even with everything down.

const _POLL_INTERVAL_MS = 3000;

// Shorten mode names for the compact sidebar display.
function _shortMode(mode) {
  if (!mode) return "—";
  if (mode === "dual_cam_homography_triangulation") return "2-cam";
  if (mode === "single_cam_homography") return "1-cam";
  return mode;
}

// Two stable children inside #comms-nodes-content so the node list and the
// zone cards render independently without clobbering each other.
function _panelDivs() {
  const target = document.getElementById("comms-nodes-content");
  if (!target) return null;
  let nodesDiv = document.getElementById("comms-nodes-list");
  let cardsDiv = document.getElementById("comms-zone-cards");
  if (!nodesDiv) {
    target.innerHTML = "";
    nodesDiv = document.createElement("div");
    nodesDiv.id = "comms-nodes-list";
    cardsDiv = document.createElement("div");
    cardsDiv.id = "comms-zone-cards";
    target.appendChild(nodesDiv);
    target.appendChild(cardsDiv);
  }
  return { nodesDiv, cardsDiv };
}

function _renderNodes(data) {
  const divs = _panelDivs();
  if (!divs) return;
  const target = divs.nodesDiv;

  // No gateway URL → on-prem mode (the default). Calm local indicator only —
  // no configuration hint (operators found it noisy; the Gateway URL field in
  // Settings → Communication is discoverable on its own).
  if (!data.configured) {
    target.innerHTML =
      '<div class="status-row comms-nodes-header">' +
      '<span class="status-dot status-dot-green" aria-hidden="true"></span>' +
      '<span class="kpi-key">On-prem · local</span></div>';
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

// ---- zone-status cards -----------------------------------------------------

// Same kind→colour convention as live_overlay.js / floor_map_3d.js.
const _KIND_COLORS = { danger: "#fca5a5", etagere: "#86efac", palette: "#9aa5b1" };

function _zoneColor(z) {
  return _KIND_COLORS[z.kind || z.type] || _KIND_COLORS.palette;
}

// Merge the two live sources for one zone: gateway (MQTT-retained) wins when
// it has a state (`objects != null`); else the local UDP bus, only while fresh.
function _stateForZone(name, gwByName, localState) {
  if (gwByName[name]) return gwByName[name];
  const local = localState && localState.fresh && localState.states
    ? localState.states[name] : null;
  return local || null;
}

// Same t() convention as floor_zones.js / zone_manager.js — the cards
// re-render every poll, so a language switch applies within one tick.
function t(key, fallback) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fallback;
}

// Human-readable palette summary (i18n via t()) for one zone STATE object.
// DUMB RENDERER: when the Backbone's PalletStateManager verdict is present
// (st.decision.palette_state) it is mapped 1:1 to text — no re-derivation.
// The objects heuristic below survives only as the fallback for decision-less
// sources (older Backbone payloads, per-camera zone-worker snapshots).
function _paletteLine(st) {
  const dec = st && st.decision;
  if (dec && dec.palette_state) {
    if (dec.palette_state === "no_palette") {
      return t("zone_no_palette", "There is no palette available");
    }
    if (dec.palette_state === "palette_empty") {
      return t("zone_palette_empty", "Palette present but empty");
    }
    if (dec.palette_state === "palette_loaded") {
      const what = (dec.content && dec.content.length)
        ? dec.content.join(` ${t("zone_and", "and")} `)
        : t("zone_an_object", "an object");
      return `${t("zone_palette_with", "Palette present with")} ${what}`;
    }
    if (dec.palette_state === "no_data") {
      // The manager has never stepped yet — honest "no live data" rather
      // than falling into the objects heuristic below (which would read an
      // empty objects list as "no palette", a claim no evidence supports).
      return t("zone_no_data", "no live data");
    }
    // Unknown future enum → fall through to the heuristic.
  }
  // Fallback heuristic. Detection quirks are resolved here, not displayed: a
  // single physical pallet can be detected twice with different occupancy
  // (palette_vide + palette_carton) — any LOADED reading wins and the carried
  // contents are unioned across readings.
  const objs = (st && st.objects) || [];
  const pals = objs.filter((o) => o.cls === "palette");
  if (!pals.length) return t("zone_no_palette", "There is no palette available");
  const contents = new Set();
  let loaded = false;
  for (const p of pals) {
    if (p.occupancy_state === "full") loaded = true;
    const c = p.occupancy_content;           // array (patch zones) or string (floor zones)
    if (Array.isArray(c)) c.forEach((x) => x && contents.add(x));
    else if (c) contents.add(c);
  }
  if (loaded || contents.size) {
    const what = contents.size
      ? [...contents].join(` ${t("zone_and", "and")} `)
      : t("zone_an_object", "an object");
    return `${t("zone_palette_with", "Palette present with")} ${what}`;
  }
  return t("zone_palette_empty", "Palette present but empty");
}

function _cardHtml(name, color, st) {
  const live = st != null;
  const cls = live ? "comms-zone-card is-live" : "comms-zone-card";
  // Live: translucent zone-colour fill; dim: outline only, no background.
  const style = `border-color:${color};${live ? `background:${color}2e;` : ""}`;
  const dot = live ? "status-dot status-dot-green" : "status-dot status-dot-grey";
  let body;
  if (live) {
    body = `<div class="comms-zone-line">${_esc(_paletteLine(st))}</div>`;
  } else {
    body = `<div class="comms-zone-line comms-nodes-subtle">${
      _esc(t("zone_no_data", "no live data"))}</div>`;
  }
  return `
    <div class="${cls}" style="${style}">
      <div class="comms-zone-head">
        <span class="${dot}" aria-hidden="true"></span>
        <span class="comms-zone-name">${_esc(name)}</span>
      </div>
      ${body}
    </div>`;
}

function _renderZoneCards(patches, patchStates, zonesList, localState, gwData) {
  const divs = _panelDivs();
  if (!divs) return;
  const host = divs.cardsDiv;

  // Operator CAMERA zones (zone patches — Zone 1/Zone 2 drawn on a cam).
  // Live state comes from the local zone workers via /api/zone-patches/state,
  // which works identically in Mode 1 and Mode 2. TWINS are skipped: they are
  // the same zone seen from the other camera (same name — listing them showed
  // every zone twice) and their contents are already merged into the base id.
  // Aggregated floor-zone state by NAME via gateway (MQTT) — built first:
  // both patch cards and floor cards consume it.
  const gwByName = {};
  if (gwData && gwData.configured && !gwData.error) {
    for (const z of gwData.zones || []) {
      if (z.objects != null) {
        // Gateway rows carry the PalletStateManager verdict FLAT
        // (palette_state/content); local bus rows nest it under `decision`.
        // Normalize to the nested shape here so _paletteLine renders both.
        gwByName[z.name] = {
          objects: z.objects, count: z.count, ts: z.state_ts,
          decision: z.palette_state
            ? { palette_state: z.palette_state, content: z.content || [] }
            : null,
        };
      }
    }
  }

  const userPatches = (patches || []).filter((p) => !p.twin_of);
  const patchCards = userPatches.map((p) => {
    // Prefer the Backbone's AGGREGATED zone state (cross-camera union +
    // occupancy voting — the same robust decision isicomms shows) over the
    // single-camera worker snapshot: one camera can miss the carton or vote
    // "empty" from its angle while the combined state correctly says "full".
    // The per-camera snapshot remains the fallback so cards stay live in
    // frames mode / pre-START preview when no aggregated state flows.
    const agg = _stateForZone(p.name || p.id, gwByName, localState);
    const st = agg
      || (patchStates && patchStates.states ? patchStates.states[p.id] : null);
    return _cardHtml(p.name || p.id, p.color || "#ff3b3b", st || null);
  });

  // Metric FLOOR zones (zones.yaml) — live via gateway (MQTT) or local bus.
  // Skip floor zones that share a name with a camera zone — one card per
  // physical zone, whichever system it was defined in.
  const patchNames = new Set(userPatches.map((p) => p.name || p.id));
  const floorCards = (zonesList || [])
    .filter((z) => !patchNames.has(z.name))
    .map((z) =>
      _cardHtml(z.name, _zoneColor(z), _stateForZone(z.name, gwByName, localState)));

  const cards = patchCards.concat(floorCards);
  if (!cards.length) {
    // No zones of either kind — say so instead of a silently empty panel.
    host.innerHTML =
      '<p class="comms-nodes-hint comms-nodes-subtle">No zones defined — ' +
      "draw one on a camera (ZONE panel) or in Settings → Zones.</p>";
    return;
  }
  host.innerHTML = cards.join("");
}

// ---- poll loop ---------------------------------------------------------------

async function _fetchJson(url) {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : null;
  } catch (_err) {
    return null;   // next tick retries; avoids console spam on hot-reload
  }
}

async function _pollNodes() {
  const [nodes, patches, patchStates, zones, localState, gwZones] = await Promise.all([
    _fetchJson("/api/gateway/nodes"),
    _fetchJson("/api/zone-patches"),        // operator camera zones (Zone 1/2…)
    _fetchJson("/api/zone-patches/state"),  // their live worker contents
    _fetchJson("/api/zones"),               // metric floor zones (zones.yaml)
    _fetchJson("/api/zones/state"),         // floor-zone state via local bus
    _fetchJson("/api/gateway/zones"),       // {configured:false} instantly when unset
  ]);
  if (nodes) _renderNodes(nodes);
  _renderZoneCards(
    patches ? patches.patches : [],
    patchStates,
    zones ? zones.zones : [],
    localState,
    gwZones,
  );
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
