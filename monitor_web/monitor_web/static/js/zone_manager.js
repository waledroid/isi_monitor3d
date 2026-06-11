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

// Detection ONNX (trainer/isidet/runs/**/*.onnx) and person-pose ONNX
// (pose *.onnx under runs/ + models/), newest first.
async function loadOnnxFiles() {
  await populateModelSelect("zm-model-onnx", "/api/detection/onnx-files");
  const sel = el("zm-model-onnx");
  if (sel && !sel._classHook) {        // refresh the displayed classes on model change
    sel._classHook = true;
    sel.addEventListener("change", refreshModelClasses);
  }
  const imgsz = el("zm-model-imgsz");
  if (imgsz && !imgsz._hook) {         // live px readout as the slider moves
    imgsz._hook = true;
    imgsz.addEventListener("input", updateImgszLabel);
  }
}
async function loadPoseOnnxFiles() {
  await populateModelSelect("zm-model-pose-onnx", "/api/detection/pose-onnx-files");
}

// Fixed two-camera layout. Display labels map cam_a/cam_b → "Cam 1"/"Cam 2";
// the backbone.yaml keys stay cam_a/cam_b.
const CAMERA_SLOTS = [
  { id: "cam_a", labelKey: "camera_1", labelFallback: "Cam 1" },
  { id: "cam_b", labelKey: "camera_2", labelFallback: "Cam 2" },
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

  // URL field (RTSP).
  const urlInput = document.createElement("input");
  urlInput.type = "text";
  urlInput.className = "zm-cam-url";
  urlInput.id = `zm-url-${slot.id}`;
  urlInput.placeholder = "rtsp://…";
  urlInput.value = cam.url || "";

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
  // Always render both slots so Cam 2 is always available to add (an empty
  // Cam 2 simply isn't saved → Mode 1).
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
  const detection = collectDetection();
  if (detection) payload.detection = detection;
  // S16: distance lines — always send the field (empty list clears the file).
  payload.link_lines = collectLinkLines();
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
// Backend is decided by the server from the host's hardware (GPU → yolo_onnx,
// CPU-only → yolo_openvino). The modal shows only the relevant model path.

let _modelBackend = "yolo_onnx";   // last backend reported by /api/config
let _modelClasses = [];            // class names auto-detected from the selected ONNX
const IMGSZ_STEPS = [320, 448, 512, 640, 1024];   // inference-size slider stops

function updateImgszLabel() {
  const v = IMGSZ_STEPS[parseInt(el("zm-model-imgsz")?.value ?? "4", 10)] ?? 1024;
  const span = el("zm-model-imgsz-val");
  if (span) span.textContent = v;
}

// The detector self-configures its classes from the ONNX metadata, so the modal
// DISPLAYS them read-only (no list to keep in sync). Refresh whenever the model
// changes; falls back to empty if the model carries none.
async function refreshModelClasses() {
  const path = el("zm-model-onnx")?.value || "";
  const field = el("zm-model-classes");
  const sl = el("zm-model-imgsz");
  const span = el("zm-model-imgsz-val");
  if (!path) { _modelClasses = []; if (field) field.value = ""; return; }
  let info = { classes: [], fixed_input: false, input_wh: null, family: "yolo" };
  try {
    const res = await fetch(`/api/detection/classes?path=${encodeURIComponent(path)}`);
    if (res.ok) info = await res.json();
  } catch { /* keep defaults */ }
  _modelClasses = info.classes || [];
  if (field) field.value = _modelClasses.join(", ");
  // The inference-size slider only applies to a DYNAMIC ONNX. A fixed-input model
  // (e.g. RF-DETR @432, or a YOLO exported with dynamic=False) ignores it — disable
  // the slider and show the model's own fixed size so it doesn't look broken.
  if (sl) {
    sl.disabled = !!info.fixed_input;
    if (info.fixed_input && info.input_wh) {
      if (span) span.textContent = `${info.input_wh[0]} (fixed by model)`;
    } else {
      updateImgszLabel();
    }
  }
}

function collectDetection() {
  const onnx = el("zm-model-onnx")?.value.trim() || "";
  const xml = el("zm-model-xml")?.value.trim() || "";
  const classes = _modelClasses.slice();   // auto-detected from the model metadata
  // Only send if the active backend's path + ≥1 class are set; an untouched
  // section is omitted (the server would otherwise 400/422).
  const activePath = _modelBackend === "yolo_openvino" ? xml : onnx;
  if (!activePath || classes.length === 0) return null;
  const conf = parseFloat(el("zm-model-conf")?.value);
  const pose = el("zm-model-pose-onnx")?.value.trim() || "";
  const imgsz = IMGSZ_STEPS[parseInt(el("zm-model-imgsz")?.value ?? "4", 10)] || 1024;
  return {
    onnx_path: onnx || null,
    model_xml: xml || null,
    pose_onnx_path: pose || null,
    pose_confidence_threshold: (() => {
      const v = parseFloat(el("zm-model-pose-conf")?.value);
      return Number.isFinite(v) ? v : 0.3;
    })(),
    class_names: classes,
    inference_imgsz: imgsz,
    confidence_threshold: Number.isFinite(conf) ? conf : 0.25,
    show_nodes: !!el("zm-model-show-nodes")?.checked,
    show_masks: !!el("zm-model-show-masks")?.checked,
    show_boxes: !!el("zm-model-show-boxes")?.checked,
    display_fps: Math.max(1, Math.min(30, parseInt(el("zm-model-display-fps")?.value, 10) || 10)),
    distance_line_opacity: Math.max(0.05, Math.min(1, parseFloat(el("zm-model-dist-opacity")?.value) || 0.25)),
    distance_line_color: el("zm-model-dist-color")?.value || "#ffffff",
    distance_line_thickness: Math.max(1, Math.min(8, parseInt(el("zm-model-dist-thickness")?.value, 10) || 2)),
  };
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

function fillModelSection(det) {
  if (!det) return;
  _modelBackend = det.backend || "yolo_onnx";
  const set = (id, v) => { const e = el(id); if (e != null) e.value = v ?? ""; };
  selectModelOption("zm-model-onnx", det.onnx_path || "");
  selectModelOption("zm-model-pose-onnx", det.pose_onnx_path || "");
  set("zm-model-xml", det.model_xml || "");
  // Class names are read-only and come from the model, not the config.
  refreshModelClasses();
  set("zm-model-conf", det.confidence_threshold ?? 0.25);
  set("zm-model-pose-conf", det.pose_confidence_threshold ?? 0.3);
  // Inference-size slider: map the configured px to its step index (default 1024).
  const sl = el("zm-model-imgsz");
  if (sl) {
    const idx = IMGSZ_STEPS.indexOf(det.inference_imgsz ?? 1024);
    sl.value = String(idx >= 0 ? idx : IMGSZ_STEPS.length - 1);
    updateImgszLabel();
  }
  const cbNodes = el("zm-model-show-nodes");
  if (cbNodes) cbNodes.checked = det.show_nodes !== false;   // default true if undefined
  const cbMasks = el("zm-model-show-masks");
  if (cbMasks) cbMasks.checked = det.show_masks !== false;
  const cbBoxes = el("zm-model-show-boxes");
  if (cbBoxes) cbBoxes.checked = det.show_boxes !== false;
  const fpsInput = el("zm-model-display-fps");
  if (fpsInput) fpsInput.value = det.display_fps ?? 10;
  const opEl = el("zm-model-dist-opacity");
  if (opEl) opEl.value = det.distance_line_opacity ?? 0.25;
  const colEl = el("zm-model-dist-color");
  if (colEl) colEl.value = det.distance_line_color ?? "#ffffff";
  const thEl = el("zm-model-dist-thickness");
  if (thEl) thEl.value = det.distance_line_thickness ?? 2;
  // Show only the path field for the detected backend; hide the other.
  const openvino = _modelBackend === "yolo_openvino";
  for (const [id, hidden] of [
    ["zm-model-onnx", openvino], ["zm-model-onnx-label", openvino],
    ["zm-model-xml", !openvino], ["zm-model-xml-label", !openvino],
  ]) {
    el(id)?.classList.toggle("hidden", hidden);
  }
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
  await loadOnnxFiles();   // populate the ONNX-path picker with trained exports
  await loadPoseOnnxFiles();   // populate the pose-model picker
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();
    buildCameraInputs(data.cameras || {});
    fillModelSection(data.detection);
    buildLinkLines(data.link_lines || []);
  } catch (err) {
    buildCameraInputs({});
    buildLinkLines([]);
    console.warn("zone_manager: failed to load config", err);
  }
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
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
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
