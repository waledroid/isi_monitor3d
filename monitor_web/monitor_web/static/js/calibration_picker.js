// Calibration picker (S18.A): wires #btn-calibrate to a small modal that
// dispatches to either the Mode 1 pallet sub-flow (calibration_mode1.js) or
// the Mode 2 Multical placeholder (S18.B drops the full UX in here).
//
// Mode is decided from /api/calibrate/status's configured_cameras list:
//   1 cam configured → Mode 1 pallet.
//   2 cams configured → Mode 2 Multical placeholder.
//   0 cams configured → toast / inline "configure a camera first".

import { mountMode1 } from "/static/js/calibration_mode1.js";

const PICKER_ID = "calibration-picker";

let _cleanupCurrentMode = null;
let _keyHandler = null;
let _backdropHandler = null;

function el(id) { return document.getElementById(id); }

function teardown() {
  if (_cleanupCurrentMode) { _cleanupCurrentMode(); _cleanupCurrentMode = null; }
  if (_keyHandler) {
    document.removeEventListener("keydown", _keyHandler);
    _keyHandler = null;
  }
  const root = el(PICKER_ID);
  if (root && _backdropHandler) {
    root.removeEventListener("click", _backdropHandler);
    _backdropHandler = null;
  }
  if (root) root.classList.add("hidden");
  // Reset all mode sub-panels + the result panel.
  el("cal-mode1")?.classList.add("hidden");
  el("cal-mode2")?.classList.add("hidden");
  el("cal-result")?.classList.add("hidden");
  const cancelBtn = el("cal-cancel");
  if (cancelBtn) cancelBtn.onclick = null;
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/calibrate/status");
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

async function open() {
  const root = el(PICKER_ID);
  if (!root) return;

  // Always teardown first so re-opening is clean.
  teardown();

  const status = await fetchStatus();
  const configured = (status && Array.isArray(status.configured_cameras))
    ? status.configured_cameras
    : [];

  // Decide which sub-flow to mount.
  if (configured.length === 1) {
    el("cal-mode1")?.classList.remove("hidden");
    _cleanupCurrentMode = mountMode1({
      cameraId: configured[0],
      onFinished: () => { /* button colour updates via calibration:saved event */ },
      onCancel: teardown,
    });
  } else if (configured.length === 2) {
    el("cal-mode2")?.classList.remove("hidden");
    // S18.B replaces this placeholder with a full Multical capture-and-run UX.
  } else {
    // No cameras configured yet — show the Mode 1 panel with a coaching hint.
    el("cal-mode1")?.classList.remove("hidden");
    showResultInline(false, "Configure a camera first via the + (Settings) button.");
  }

  // Cancel via button + ESC + backdrop click.
  const cancelBtn = el("cal-cancel");
  if (cancelBtn) cancelBtn.onclick = teardown;

  _keyHandler = (ev) => { if (ev.key === "Escape") teardown(); };
  document.addEventListener("keydown", _keyHandler);

  _backdropHandler = (ev) => { if (ev.target.id === PICKER_ID) teardown(); };
  root.addEventListener("click", _backdropHandler);

  root.classList.remove("hidden");
}

function showResultInline(ok, msg) {
  const panel = el("cal-result");
  const text = el("cal-result-msg");
  if (!panel || !text) return;
  panel.classList.remove("hidden", "is-success", "is-error");
  panel.classList.add(ok ? "is-success" : "is-error");
  text.textContent = msg;
}

// Clear the current mode's calibration (the green-button action). Feeds stop
// auto-warping and the button returns to white; the operator can recalibrate.
async function clearCalibration() {
  try {
    await fetch("/api/calibrate/clear", { method: "POST" });
  } catch {
    /* best-effort; status poll will reconcile */
  }
  document.dispatchEvent(new CustomEvent("calibration:cleared"));
}

function t(key, fb) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fb;
}

// One button, state-driven: GREEN (current mode calibrated) → clear it;
// WHITE (not calibrated) → open the calibration picker for the current mode.
// Clearing is destructive, and the green icon is easy to hit by mistake, so we
// confirm before deleting an existing calibration.
async function onCalibrateClick() {
  const status = await fetchStatus();
  if (status && status.is_fully_calibrated) {
    const ok = window.confirm(t("calibrate_clear_confirm",
      "Clear the current calibration? You'll need to recalibrate."));
    if (ok) await clearCalibration();
  } else {
    await open();
  }
}

function wire() {
  const btn = document.getElementById("btn-calibrate");
  if (btn) btn.addEventListener("click", onCalibrateClick);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}

export { open as openCalibrationPicker };
