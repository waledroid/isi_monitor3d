// Drives the colour state of #btn-calibrate based on /api/calibrate/status.
//
// State machine (two-state per S18 spec):
//   data-state="white" — no calibration, or it doesn't cover every configured camera.
//   data-state="green" — every configured camera has a calibration entry.
//
// Polled every 5 s as a safety net, and refreshed immediately on the
// `calibration:saved` custom event (fired by calibration_picker.js after a
// successful POST). Self-starts on DOMContentLoaded.

const POLL_INTERVAL_MS = 5000;
const BUTTON_ID = "btn-calibrate";

async function fetchStatus() {
  try {
    const res = await fetch("/api/calibrate/status");
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function applyState(status) {
  const btn = document.getElementById(BUTTON_ID);
  if (!btn) return;
  const fullyCalibrated = !!(status && status.is_fully_calibrated);
  btn.dataset.state = fullyCalibrated ? "green" : "white";
  // Tooltip lets the operator hover for a quick read of the current state.
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  if (fullyCalibrated) {
    btn.title = strings.calibrate_button_title_green
      || "All cameras calibrated. Click to clear the calibration.";
  } else {
    btn.title = strings.calibrate_button_title_white
      || "Not calibrated. Click to calibrate.";
  }
}

export async function refresh() {
  applyState(await fetchStatus());
}

document.addEventListener("calibration:saved", () => { refresh(); });
document.addEventListener("calibration:cleared", () => { refresh(); });

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    refresh();
    setInterval(refresh, POLL_INTERVAL_MS);
  });
} else {
  refresh();
  setInterval(refresh, POLL_INTERVAL_MS);
}
