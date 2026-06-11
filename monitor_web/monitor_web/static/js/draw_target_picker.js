// Draw-target picker (S17.1) — small modal that asks "draw on which camera?".
//
// Self-contained module. Consumers call `openPicker({onPick, onCancel})` and
// don't need to know about calibration status, DOM ids, or close lifecycle —
// the picker fetches /api/project/cameras itself, enables/disables CAM buttons
// accordingly, and reports back via the two callbacks.
//
// The picker is CAM-only by design (per S17.1): zones are conceptually
// anchored to a camera view, and pre-calibration the picker hard-blocks the
// flow with an inline message + only-Cancel-works. No MAP fallback.

const DOM_IDS = {
  root:    "zm-draw-picker",
  message: "zm-draw-picker-msg",
  cancel:  "zm-draw-picker-cancel",
};
const BTN_SELECTOR = ".zm-draw-target-btn";

// Cached calibrated-cameras response so a series of zone-row Draw clicks
// doesn't re-hit /api/project/cameras for every click. Reset via
// `invalidateCalibrationCache()` (called from zone_manager.open()).
let _calibratedCams = null;

export function invalidateCalibrationCache() {
  _calibratedCams = null;
}

async function fetchCalibratedCameraIds() {
  if (_calibratedCams) return _calibratedCams;
  try {
    const res = await fetch("/api/project/cameras");
    if (!res.ok) { _calibratedCams = []; return _calibratedCams; }
    const data = await res.json();
    _calibratedCams = Array.isArray(data.cameras) ? data.cameras : [];
  } catch {
    _calibratedCams = [];
  }
  return _calibratedCams;
}

// Active session — only one picker open at a time.
let _session = null;

function teardown() {
  if (!_session) return;
  const root = document.getElementById(DOM_IDS.root);
  if (root) {
    root.classList.add("hidden");
    for (const btn of root.querySelectorAll(BTN_SELECTOR)) btn.onclick = null;
    root.removeEventListener("click", _session.backdropHandler);
  }
  document.removeEventListener("keydown", _session.keyHandler);
  const cancelBtn = document.getElementById(DOM_IDS.cancel);
  if (cancelBtn) cancelBtn.onclick = null;
  _session = null;
}

/**
 * Open the picker.
 *
 *   await openPicker({
 *     onPick:   (target) => { ... start draw on target ... },
 *     onCancel: () => { ... operator dismissed ... },
 *   });
 *
 * Returns once the picker is shown. `onPick` / `onCancel` fire later, once
 * the operator picks a CAM or cancels. Only one picker can be open at a time;
 * a second `openPicker` call closes the first.
 *
 * Translation lookup (`t(key, fallback)`) is taken from the global string
 * bundle (`window.__monitor_web.strings`) — same convention as zone_manager.
 */
export async function openPicker({ onPick, onCancel } = {}) {
  if (_session) teardown();   // close any prior session first

  const root = document.getElementById(DOM_IDS.root);
  const msg  = document.getElementById(DOM_IDS.message);
  const cancelBtn = document.getElementById(DOM_IDS.cancel);
  if (!root) { console.warn("draw_target_picker: #zm-draw-picker missing"); return; }

  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  const t = (key, fb) => strings[key] || fb;

  const cams = await fetchCalibratedCameraIds();

  // Enable / disable each CAM button based on calibration. `data-target`
  // (cam_a / cam_b) on each button is the source of truth.
  for (const btn of root.querySelectorAll(BTN_SELECTOR)) {
    const target = btn.dataset.target;
    const calibrated = cams.includes(target);
    btn.disabled = !calibrated;
    btn.title = calibrated ? "" : t("draw_target_cam_uncalibrated",
      "Camera not calibrated yet — run `python -m calibration.calibrate` first.");
    btn.onclick = calibrated
      ? () => { teardown(); onPick?.(target); }
      : null;
  }

  // No calibrated camera at all → block with the inline message.
  if (msg) msg.classList.toggle("hidden", cams.length > 0);

  // Cancel via button + ESC + outside-click.
  const cancelAndTeardown = () => { teardown(); onCancel?.(); };
  if (cancelBtn) cancelBtn.onclick = cancelAndTeardown;
  const keyHandler = (ev) => { if (ev.key === "Escape") cancelAndTeardown(); };
  const backdropHandler = (ev) => { if (ev.target.id === DOM_IDS.root) cancelAndTeardown(); };
  document.addEventListener("keydown", keyHandler);
  root.addEventListener("click", backdropHandler);

  _session = { keyHandler, backdropHandler };
  root.classList.remove("hidden");
}

/** Force-close the picker without firing onPick / onCancel. */
export function closePicker() {
  teardown();
}
