// Zone manager overlay (S11).
//
// Wires the + button (#btn-add-zone) to a transparent overlay
// (#zone-manager) that lets the operator edit camera URLs and define up to
// six floor-zone polygons by clicking on the Pixi floor map.
//
// Persists via POST /api/config; refreshes the floor map's zone layer in
// place on success (camera URL changes need a Backbone restart).

import { cancelDraw } from "/static/js/draw_mode.js";
import { invalidateCalibrationCache } from "/static/js/draw_target_picker.js";
import { loadFloorZones } from "/static/js/floor_zones.js";

function t(key, fallback) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fallback;
}

function el(id) { return document.getElementById(id); }

// NOTE: the old metric-zone editor (map-click polygons, zones.yaml) was retired.
// Operator zones are drawn on a CAM and managed by zone_patch.js; this module no
// longer builds zone rows or sends `zones` in the /api/config payload.

let availableDevices = [];   // [{path, name}] from /api/cameras/available

async function loadDevices() {
  try {
    const res = await fetch("/api/cameras/available");
    if (res.ok) {
      const data = await res.json();
      availableDevices = data.devices || [];
    }
  } catch {
    availableDevices = [];
  }
}

// Fill a model <select> from an endpoint that returns {files:[{path,label}]}.
// Option value = absolute path (written to backbone.yaml, loads verbatim);
// option text = a friendly label. Keeps the leading placeholder option.
async function populateModelSelect(selId, endpoint) {
  const sel = el(selId);
  if (!sel) return [];
  let files = [];
  try {
    const res = await fetch(endpoint);
    files = res.ok ? ((await res.json()).files || []) : [];
  } catch {
    files = [];
  }
  const placeholder = sel.querySelector('option[value=""]');
  sel.replaceChildren();
  if (placeholder) sel.appendChild(placeholder);
  for (const f of files) {
    const o = document.createElement("option");
    o.value = f.path;             // absolute path → backbone.yaml
    o.textContent = f.label || f.path;
    sel.appendChild(o);
  }
  return files;
}

// Person-pose ONNX (pose *.onnx under runs/ + models/), newest first. Object
// models are picked PER ZONE (Settings ▸ Zones) — no global model picker here.
async function loadPoseOnnxFiles() {
  // Every trained export is selectable; ONE object model serves all zones.
  await populateModelSelect("zm-model-onnx", "/api/detection/onnx-files");
  await populateModelSelect("zm-model-pose-onnx", "/api/detection/pose-onnx-files");
}

// CPU deployment branch: SINGLE camera (Mode 1). The GPU line's Cam 2 slot
// is removed — backbone.yaml only ever carries cam_a.
const CAM1_DEFAULT_RTSP =
  "rtsp://admin:admin123@192.168.2.200/cam/realmonitor?channel=1&subtype=0";
const CAMERA_SLOTS = [
  { id: "cam_a", labelKey: "camera_1", labelFallback: "Cam 1" },
];

function buildCameraRow(slot, cam) {
  // cam: {name, url, device} from /api/config (or {} when unset).
  const isV4l2 = cam.name === "v4l2";

  const label = document.createElement("label");
  label.textContent = t(slot.labelKey, slot.labelFallback);

  // Camera-type selector: RTSP (IP) vs USB / V4L2.
  const typeSel = document.createElement("select");
  typeSel.className = "zm-cam-type";
  typeSel.id = `zm-type-${slot.id}`;
  typeSel.dataset.camId = slot.id;
  for (const [val, key, fb] of [
    ["rtsp", "type_rtsp", "RTSP (IP camera)"],
    ["v4l2", "type_usb", "USB / V4L2"],
  ]) {
    const o = document.createElement("option");
    o.value = val;
    o.textContent = t(key, fb);
    typeSel.appendChild(o);
  }
  typeSel.value = isV4l2 ? "v4l2" : "rtsp";

  // URL field (RTSP) — prefilled with the site default when unconfigured.
  const urlInput = document.createElement("input");
  urlInput.type = "text";
  urlInput.className = "zm-cam-url";
  urlInput.id = `zm-url-${slot.id}`;
  urlInput.placeholder = CAM1_DEFAULT_RTSP;
  urlInput.value = cam.url || (isV4l2 ? "" : CAM1_DEFAULT_RTSP);

  // Device field (USB) — text input backed by a datalist of detected devices,
  // so the operator can pick a detected /dev/video* OR type a path manually.
  const devInput = document.createElement("input");
  devInput.type = "text";
  devInput.className = "zm-cam-device";
  devInput.id = `zm-dev-${slot.id}`;
  devInput.placeholder = "/dev/video0";
  devInput.value = isV4l2 ? (cam.device || "") : "";
  const listId = `devices-${slot.id}`;
  devInput.setAttribute("list", listId);
  const datalist = document.createElement("datalist");
  datalist.id = listId;
  for (const d of availableDevices) {
    const o = document.createElement("option");
    o.value = d.path;
    o.textContent = d.name ? `${d.name}` : d.path;
    datalist.appendChild(o);
  }

  // Toggle which field is shown by the selected type.
  function applyType() {
    const usb = typeSel.value === "v4l2";
    urlInput.style.display = usb ? "none" : "";
    devInput.style.display = usb ? "" : "none";
  }
  typeSel.addEventListener("change", applyType);
  applyType();

  return { label, typeSel, urlInput, devInput, datalist };
}

function buildCameraInputs(cameras) {
  const host = el("zm-cameras");
  host.innerHTML = "";
  cameras = cameras || {};
  // Single slot — this branch is single-camera Mode 1 by construction.
  for (const slot of CAMERA_SLOTS) {
    const { label, typeSel, urlInput, devInput, datalist } =
      buildCameraRow(slot, cameras[slot.id] || {});
    host.appendChild(label);
    host.appendChild(typeSel);
    // url + device occupy the same grid cell; only one is visible at a time.
    const field = document.createElement("div");
    field.className = "zm-cam-field";
    field.appendChild(urlInput);
    field.appendChild(devInput);
    field.appendChild(datalist);
    host.appendChild(field);
  }
}

function collectMqttSink() {
  const nodeId = el("zm-comm-node-id")?.value.trim() || "";
  const host = el("zm-comm-host")?.value.trim() || "";
  const portRaw = parseInt(el("zm-comm-port")?.value || "1883", 10);
  const port = Number.isFinite(portRaw) && portRaw > 0 ? portRaw : 1883;
  const tls = !!(el("zm-comm-tls")?.checked);
  const caCert = el("zm-comm-ca-cert")?.value.trim() || "";
  const username = el("zm-comm-username")?.value.trim() || "";
  const password = el("zm-comm-password")?.value || "";
  // Default the prefix to isiMonitor3D/v1/<node_id> when the operator left it blank.
  const prefixRaw = el("zm-comm-prefix")?.value.trim() || "";
  const prefix = prefixRaw || (nodeId ? `isiMonitor3D/v1/${nodeId}` : "");
  return {
    node_id: nodeId,
    mqtt_sink: { host, port, tls, ca_cert: caCert, username, password, prefix },
  };
}

function fillCommSection(nodeId, mqttSink, uiSettings) {
  const set = (id, v) => { const e = el(id); if (e != null) e.value = v ?? ""; };
  // Gateway fields (ui-settings, top of the tab).
  set("zm-comm-gateway-url", uiSettings?.gateway_url || "");
  set("zm-comm-gateway-token", uiSettings?.gateway_token || "");
  // Node identity + MQTT broker (backbone.yaml).
  set("zm-comm-node-id", nodeId || "");
  set("zm-comm-host", mqttSink?.host || "");
  set("zm-comm-port", mqttSink?.port ?? 1883);
  const cbTls = el("zm-comm-tls");
  if (cbTls) cbTls.checked = !!(mqttSink?.tls);
  set("zm-comm-ca-cert", mqttSink?.ca_cert || "");
  set("zm-comm-username", mqttSink?.username || "");
  set("zm-comm-password", mqttSink?.password || "");
  set("zm-comm-prefix", mqttSink?.prefix || "");
  // Reflect TLS state in the Mode dropdown (best-effort; operator can override).
  const modeSel = el("zm-comm-mode");
  if (modeSel) modeSel.value = mqttSink?.tls ? "cloud" : "onprem";
}

function collectGatewayFields() {
  return {
    gateway_url: el("zm-comm-gateway-url")?.value.trim() || "",
    gateway_token: el("zm-comm-gateway-token")?.value || "",
  };
}

function wireCommMode() {
  const modeSel = el("zm-comm-mode");
  if (!modeSel || modeSel._commHook) return;
  modeSel._commHook = true;
  modeSel.addEventListener("change", () => {
    const cloud = modeSel.value === "cloud";
    const portEl = el("zm-comm-port");
    const tlsEl = el("zm-comm-tls");
    if (portEl) portEl.value = cloud ? 8883 : 1883;
    if (tlsEl) tlsEl.checked = cloud;
  });
}

function collectPayload() {
  const cameras = {};
  for (const slot of CAMERA_SLOTS) {
    const type = el(`zm-type-${slot.id}`)?.value || "rtsp";
    if (type === "v4l2") {
      const device = el(`zm-dev-${slot.id}`)?.value.trim() || "";
      if (device) cameras[slot.id] = { device };   // USB / V4L2
    } else {
      const url = el(`zm-url-${slot.id}`)?.value.trim() || "";
      if (url) cameras[slot.id] = { url };          // RTSP
    }
    // empty slot → omit (e.g. blank Cam 2 ⇒ Mode 1)
  }
  // No `zones` key: the metric-zone editor is gone, so Save leaves zones.yaml
  // untouched (the backend treats an omitted `zones` as "no change").
  const payload = { cameras };
  // Camera FPS — written to all cameras' source.capture_fps in backbone.yaml.
  const camFpsRaw = parseInt(el("zm-camera-fps")?.value || "20", 10);
  payload.camera_fps = Number.isFinite(camFpsRaw) && camFpsRaw > 0
    ? Math.max(1, Math.min(30, camFpsRaw)) : 20;
  // Pose model only — object models live per zone; the display toggles +
  // distance-line styles auto-save on change (see wireUiPrefSync), so the Save
  // button no longer carries a `detection` block.
  payload.pose = collectPose();
  // isistream perf toggles (default ON; a save applies them live).
  payload.motion_gate = el("zm-motion-gate")?.checked ?? true;
  // Detection quality: high = main stream, low = camera substream. Applied on
  // the fly (a save hot-restarts isistream ~4 s; the engine keeps running).
  payload.detect_substream = el("zm-detect-quality")?.value === "low";
  // trt_enabled: UI toggle retired (native .engine models supersede it);
  // the config default (true) governs .onnx paths, untouched by saves.
  // S16: distance lines — always send the field (empty list clears the file).
  payload.link_lines = collectLinkLines();
  // 3D localization — omit when the pane never populated (null) so a failed
  // load can't wipe subscriptions.yaml; [] explicitly clears every rule.
  // Communication — MQTT broker + node identity.
  const { node_id, mqtt_sink } = collectMqttSink();
  payload.node_id = node_id;
  payload.mqtt_sink = mqtt_sink;
  return payload;
}

// ---- S16: distance-lines section ----

// Class options shown in the row pickers — matches the avatars + the Backbone's
// emitted class names (lowercase). 'palette' covers both 'palette' and the
// legacy 'pallet' classes downstream.
const LINK_CLASSES = ["person", "palette", "forklift", "robot"];

function buildLinkRow(rule) {
  const node = document.createElement("div");
  node.className = "config-link-row";

  // FROM class
  const fromSel = document.createElement("select");
  fromSel.className = "zm-link-from";
  for (const c of LINK_CLASSES) {
    const o = document.createElement("option");
    o.value = c; o.textContent = c;
    fromSel.appendChild(o);
  }
  fromSel.value = rule?.from || "person";

  // TO classes — comma-separated text, accepting '*' for "all others".
  const toInput = document.createElement("input");
  toInput.type = "text";
  toInput.className = "zm-link-to";
  toInput.placeholder = "* or palette, forklift";
  toInput.value = Array.isArray(rule?.to) ? rule.to.join(", ") : "*";

  // max_distance_m
  const distInput = document.createElement("input");
  distInput.type = "number";
  distInput.step = "0.1";
  distInput.min = "0";
  distInput.className = "zm-link-dist";
  distInput.placeholder = "max m";
  distInput.value = (rule && Number.isFinite(rule.max_distance_m)) ? rule.max_distance_m : "";

  // color (optional)
  const colorInput = document.createElement("input");
  colorInput.type = "text";
  colorInput.className = "zm-link-color";
  colorInput.placeholder = "#fff";
  colorInput.value = rule?.color || "";

  // delete
  const del = document.createElement("button");
  del.type = "button";
  del.className = "glass-btn zm-iconbtn zm-link-del";
  del.title = t("clear", "Clear");
  del.setAttribute("aria-label", "Remove rule");
  del.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
  del.addEventListener("click", () => node.remove());

  node.appendChild(fromSel);
  node.appendChild(toInput);
  node.appendChild(distInput);
  node.appendChild(colorInput);
  node.appendChild(del);
  return node;
}

function buildLinkLines(rules) {
  const host = el("zm-link-lines");
  if (!host) return;
  host.innerHTML = "";
  for (const r of rules || []) host.appendChild(buildLinkRow(r));
}

function collectLinkLines() {
  const host = el("zm-link-lines");
  if (!host) return [];
  const out = [];
  for (const row of host.querySelectorAll(".config-link-row")) {
    const from = row.querySelector(".zm-link-from")?.value || "";
    const toRaw = row.querySelector(".zm-link-to")?.value || "";
    const dist = parseFloat(row.querySelector(".zm-link-dist")?.value);
    const color = (row.querySelector(".zm-link-color")?.value || "").trim();
    const to = toRaw.split(",").map((s) => s.trim()).filter(Boolean);
    if (!from || to.length === 0) continue;
    const rule = { from, to };
    if (Number.isFinite(dist) && dist > 0) rule.max_distance_m = dist;
    if (color) rule.color = color;
    out.push(rule);
  }
  return out;
}

// ---- detection model section ----
// Pose model only. Object-detection models are configured per zone (Settings ▸
// Zones); the display toggles + distance-line styles below auto-save on change.

function collectPose() {
  const v = parseFloat(el("zm-model-pose-conf")?.value);
  const num = (id, dflt) => {
    const x = parseFloat(el(id)?.value);
    return Number.isFinite(x) ? x : dflt;
  };
  // Settings ▸ Isistream: ONE object model + global knobs for ALL zones.
  return {
    pose_enabled: el("zm-pose-enabled")?.checked ?? true,
    pose_model_xml: el("zm-model-pose-onnx")?.value.trim() || "",   // "" = clear
    pose_confidence_threshold: Number.isFinite(v) ? v : 0.3,
    model_xml: el("zm-model-onnx")?.value.trim() || "",
    zone_imgsz: num("zm-model-zone-imgsz", 384),
    confidence_threshold: num("zm-model-conf", 0.25),
    sahi_enabled: el("zm-sahi-enabled")?.checked ?? false,
    sahi_tile: num("zm-sahi-tile", 0),
    sahi_overlap: num("zm-sahi-overlap", 0.2),
    enhance_enabled: el("zm-enh-enabled")?.checked ?? false,
    enhance_gamma: num("zm-enh-gamma", 1.0),
  };
}

// Auto-save a UI display preference. The server merges it into the UI-settings
// YAML atomically and the overlay re-reads prefs per frame, so the effect is
// immediate — no Save button, no stream reconnect.
function syncUiPref(patch) {
  fetch("/api/ui-settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  }).catch(() => showToast(t("save_failed", "Save failed"), true));
}

// Wire the change → POST hooks once (the modal markup is static).
function wireUiPrefSync() {
  // Zones FPS (Zones tab): the editable zone-worker / zone-patch rate. Persisted
  // (Cam-view pose now inherits the Camera FPS, so it ignores this value.)
  const hooks = [
    ["zm-model-show-masks", (e) => ({ show_masks: !!e.checked })],
    ["zm-model-show-boxes", (e) => ({ show_boxes: !!e.checked })],
    ["zm-show-floor-zones", (e) => ({ show_floor_zones: !!e.checked })],
    ["zm-show-zone-fill", (e) => ({ show_zone_fill: !!e.checked })],
    ["zm-model-dist-opacity",
      (e) => ({ distance_line_opacity: Math.max(0.05, Math.min(1, parseFloat(e.value) || 0.25)) })],
    ["zm-model-dist-color", (e) => ({ distance_line_color: e.value || "#ffffff" })],
    ["zm-model-dist-thickness",
      (e) => ({ distance_line_thickness: Math.max(1, Math.min(8, parseInt(e.value, 10) || 2)) })],
  ];
  for (const [id, toPatch] of hooks) {
    const e = el(id);
    if (e && !e._prefHook) {
      e._prefHook = true;
      e.addEventListener("change", () => syncUiPref(toPatch(e)));
    }
  }
}

// Select a configured path in a model dropdown. If it isn't among the discovered
// files (e.g. a custom path already in backbone.yaml), inject it as an option so
// it displays and round-trips through Save instead of being lost.
function selectModelOption(selId, path) {
  const sel = el(selId);
  if (!sel) return;
  if (path && !Array.from(sel.options).some((o) => o.value === path)) {
    const o = document.createElement("option");
    o.value = path;
    o.textContent = path;   // show the raw path for an off-list (custom) model
    sel.appendChild(o);
  }
  sel.value = path || "";
}

function fillModelSection(det, isis) {
  if (!det) return;
  isis = isis || {};
  const set = (id, v) => { const e = el(id); if (e != null) e.value = v ?? ""; };
  const poseEnabled = el("zm-pose-enabled");
  if (poseEnabled) poseEnabled.checked = det.pose_enabled !== false;
  // Global isistream object-model knobs (one model serves every zone).
  selectModelOption("zm-model-onnx", det.model_xml || "");
  set("zm-model-zone-imgsz", det.zone_imgsz ?? 384);
  set("zm-model-conf", det.confidence_threshold ?? 0.25);
  const sahiOn = el("zm-sahi-enabled");
  if (sahiOn) sahiOn.checked = det.sahi_enabled === true;
  set("zm-sahi-tile", det.sahi_tile ?? 0);
  set("zm-sahi-overlap", det.sahi_overlap ?? 0.2);
  const enhOn = el("zm-enh-enabled");
  if (enhOn) enhOn.checked = det.enhance_enabled === true;
  set("zm-enh-gamma", det.enhance_gamma ?? 1.0);
  const mg = el("zm-motion-gate");
  if (mg) mg.checked = isis.motion_gate !== false;
  const dq = el("zm-detect-quality");
  if (dq) {
    dq.value = isis.detect_substream ? "low" : "high";
    // "Low" needs a substream URL configured; if none, force + lock to High.
    const lowOpt = dq.querySelector('option[value="low"]');
    if (lowOpt) lowOpt.disabled = !isis.has_detect_source;
    if (!isis.has_detect_source) dq.value = "high";
  }
  // (TensorRT toggle retired — native .engine models supersede it.)
  selectModelOption("zm-model-pose-onnx", det.pose_model_xml || "");
  set("zm-model-pose-conf", det.pose_confidence_threshold ?? 0.3);
  // NOTE: global Detection FPS removed — zones run at the fixed DEFAULT_DETECTION_FPS (10).
  const cbNodes = el("zm-model-show-nodes");
  if (cbNodes) cbNodes.checked = det.show_nodes !== false;   // default true if undefined
  const cbMasks = el("zm-model-show-masks");
  if (cbMasks) cbMasks.checked = det.show_masks !== false;
  const cbFz = el("zm-show-floor-zones");
  if (cbFz) cbFz.checked = det.show_floor_zones === true;   // default OFF
  const cbZf = el("zm-show-zone-fill");
  if (cbZf) cbZf.checked = det.show_zone_fill === true;     // default OFF
  const cbBoxes = el("zm-model-show-boxes");
  if (cbBoxes) cbBoxes.checked = det.show_boxes !== false;
  const opEl = el("zm-model-dist-opacity");
  if (opEl) opEl.value = det.distance_line_opacity ?? 0.25;
  const colEl = el("zm-model-dist-color");
  if (colEl) colEl.value = det.distance_line_color ?? "#ffffff";
  const thEl = el("zm-model-dist-thickness");
  if (thEl) thEl.value = det.distance_line_thickness ?? 2;
  wireUiPrefSync();   // change → POST /api/ui-settings (idempotent wiring)
}

function showToast(message, isError) {
  const toast = el("zm-toast");
  toast.textContent = message;
  toast.classList.toggle("is-error", !!isError);
  toast.classList.remove("hidden");
  clearTimeout(showToast._h);
  showToast._h = setTimeout(() => toast.classList.add("hidden"), 5000);
}

async function revealMp4IfUnlocked() {
  // The MP4 (dev) source appears after Cam 2 only when the hidden viewer has
  // been unlocked (double-click logo + password → sessionStorage flag).
  const row = el("zm-mp4-row");
  const pick = el("mp4-pick");
  if (!row || !pick) return;
  if (sessionStorage.getItem("mp4_unlocked") !== "1") {
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  try {
    const res = await fetch("/api/media/mp4");
    if (!res.ok) return;
    const { files } = await res.json();
    const current = pick.value;
    pick.length = 1;                       // keep the "Choose…" option
    for (const f of files || []) {
      const o = document.createElement("option");
      o.value = f; o.textContent = f;
      pick.appendChild(o);
    }
    // Reflect the remembered selection (from the Alpine store) in the dropdown.
    const store = window.Alpine && Alpine.store("bigPanel");
    const want = (store && store.mp4Selected) || current;
    if (want) pick.value = want;
  } catch {
    /* ignore — fail quiet */
  }
}

async function open() {
  invalidateCalibrationCache();   // reload calibration status each time modal opens
  await loadDevices();   // populate the per-camera device dropdowns first
  await loadPoseOnnxFiles();   // populate the pose-model picker
  // Fetch config (backbone.yaml) and ui-settings in parallel.
  let configData = null, uiSettings = null;
  try {
    const [configRes, uiRes] = await Promise.all([
      fetch("/api/config"),
      fetch("/api/ui-settings"),
    ]);
    if (!configRes.ok) throw new Error(`/api/config status ${configRes.status}`);
    configData = await configRes.json();
    uiSettings = uiRes.ok ? await uiRes.json() : null;
  } catch (err) {
    console.warn("zone_manager: failed to load config", err);
  }
  loadFloorZones(configData);   // metric floor zones (Zones tab, drawn on a cam)
  if (configData) {
    buildCameraInputs(configData.cameras || {});
    // Camera FPS field (Cameras tab, backbone.yaml capture_fps).
    const camFpsEl = el("zm-camera-fps");
    if (camFpsEl) camFpsEl.value = configData.camera_fps ?? 20;
    fillModelSection(configData.detection, configData.isistream);
    buildLinkLines(configData.link_lines || []);
    fillCommSection(configData.node_id, configData.mqtt_sink, uiSettings);
  } else {
    buildCameraInputs({});
    const camFpsEl = el("zm-camera-fps");
    if (camFpsEl) camFpsEl.value = 20;
    buildLinkLines([]);
    fillCommSection("", null, uiSettings);
  }
  wireCommMode();
  await revealMp4IfUnlocked();
  resetTabs();
  el("zone-manager").classList.remove("hidden");
  el("zm-toast").classList.add("hidden");
}

function close() {
  cancelDraw();
  el("zone-manager").classList.add("hidden");
}

async function save() {
  const payload = collectPayload();
  // Persist gateway fields to ui-settings (they are dashboard-side, not backbone.yaml).
  const gatewayFields = collectGatewayFields();
  try {
    // Fire both writes; gateway fields go to ui-settings, everything else to /api/config.
    const [res] = await Promise.all([
      fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetch("/api/ui-settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(gatewayFields),
      }),
    ]);
    if (!res.ok) {
      const detail = await res.text();
      showToast(`${t("save_failed", "Save failed")}: ${detail}`, true);
      return;
    }
    showToast(t("restart_required", "Saved. Restart Backbone for camera changes to apply."), false);
    // Let big_panel re-evaluate CAM 2 visibility (revealed once cam_b is saved).
    document.dispatchEvent(new CustomEvent("config:saved"));
    if (window.__floor_map?.reloadZones) {
      await window.__floor_map.reloadZones();
    }
    // S16: refresh distance-line rules in the live floor map.
    if (window.__floor_map?.reloadLinkLines) {
      await window.__floor_map.reloadLinkLines();
    }
  } catch (err) {
    showToast(`${t("save_failed", "Save failed")}: ${err}`, true);
  }
}

function resetTabs() {
  const tabs = document.querySelectorAll(".settings-tab-btn");
  const contents = document.querySelectorAll(".settings-tab-content");
  tabs.forEach((tab, index) => {
    tab.classList.toggle("active", index === 0);
  });
  contents.forEach((content, index) => {
    content.classList.toggle("active", index === 0);
  });
}

function initTabs() {
  const tabs = document.querySelectorAll(".settings-tab-btn");
  const contents = document.querySelectorAll(".settings-tab-content");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      contents.forEach(c => c.classList.remove("active"));
      tab.classList.add("active");
      const targetId = `tab-${tab.dataset.tab}`;
      const content = el(targetId);
      if (content) content.classList.add("active");
    });
  });
}

function wire() {
  const btn = el("btn-add-zone");
  if (btn) {
    // Replace the old "append placeholder zone" handler from big_panel.js's
    // setupDynamicZones: clone the node to drop any prior listener.
    const fresh = btn.cloneNode(true);
    btn.parentNode.replaceChild(fresh, btn);
    fresh.addEventListener("click", open);
  }
  el("zm-close")?.addEventListener("click", close);
  el("zm-cancel")?.addEventListener("click", close);
  el("zm-save")?.addEventListener("click", save);
  // S16: + Add rule appends a blank distance-line row.
  el("zm-link-add")?.addEventListener("click", () => {
    el("zm-link-lines")?.appendChild(buildLinkRow(null));
  });
  // Click on backdrop (outside panel) closes too.
  el("zone-manager")?.addEventListener("click", (ev) => {
    if (ev.target.id === "zone-manager") close();
  });
  // MP4 picker → drive the big-panel store (jump to the MP4 tab) + close.
  el("mp4-pick")?.addEventListener("change", (ev) => {
    const store = window.Alpine && Alpine.store("bigPanel");
    if (store) store.selectMp4(ev.target.value);
    close();
  });
  initTabs();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
