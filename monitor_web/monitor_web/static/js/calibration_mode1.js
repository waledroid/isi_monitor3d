// Mode 1 calibration sub-flow (S18.A): single-camera pallet calibration.
//
// Operator types/keeps the pallet dimensions (defaults: EUR 1.2 × 0.8 m),
// clicks Start, then clicks the pallet's 4 corners on the live CAM image
// (TL → TR → BR → BL). Auto-finishes after the 4th click and POSTs to
// /api/calibrate/single-cam. Result is shown in the modal.

import { startDraw } from "/static/js/draw_mode.js";

function t(key, fb) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fb;
}
function el(id) { return document.getElementById(id); }

/**
 * Mount the Mode 1 sub-flow inside the calibration picker modal.
 *
 *   mountMode1({
 *     cameraId:   "cam_a",
 *     onFinished: ({ok, max_residual_m, error}) => { ... },
 *     onCancel:   () => { ... operator closed the picker ... },
 *   });
 *
 * Returns a `cleanup()` function the caller (calibration_picker) calls when
 * the picker closes — drops handlers so a re-mount doesn't double-fire.
 */
// Clamp the point-count input to [4, 8].
function clampCount(input) {
  let n = parseInt(input?.value, 10);
  if (!Number.isFinite(n)) n = 4;
  return Math.max(4, Math.min(8, n));
}

export function mountMode1({ cameraId, onFinished, onCancel }) {
  const targetSpan = el("cal-target-cam");
  if (targetSpan) targetSpan.textContent = cameraId;

  const widthInput = el("cal-pallet-width");
  const heightInput = el("cal-pallet-height");
  const countInput = el("cal-point-count");
  const extraBox = el("cal-extra-points");
  const startBtn = el("cal-start");

  if (!startBtn) return () => {};

  // Render (N-4) world-coordinate input rows for the extra floor points.
  function renderExtraRows() {
    if (!extraBox) return;
    const n = clampCount(countInput);
    extraBox.innerHTML = "";
    if (n <= 4) { extraBox.classList.add("hidden"); return; }
    extraBox.classList.remove("hidden");
    const hint = document.createElement("p");
    hint.className = "cal-hint";
    hint.textContent = t("calibrate_extra_hint",
      "For each extra point: enter its floor X,Y in metres (measured from the pallet TL corner = 0,0). Click them AFTER the 4 corners, in this order.");
    extraBox.appendChild(hint);
    for (let i = 5; i <= n; i++) {
      const row = document.createElement("div");
      row.className = "cal-extra-row";
      const lbl = document.createElement("label");
      lbl.textContent = `${t("calibrate_point", "Point")} ${i}`;
      const x = document.createElement("input");
      x.type = "number"; x.step = "0.01"; x.id = `cal-extra-x-${i}`; x.placeholder = "X (m)";
      const y = document.createElement("input");
      y.type = "number"; y.step = "0.01"; y.id = `cal-extra-y-${i}`; y.placeholder = "Y (m)";
      row.append(lbl, x, y);
      extraBox.appendChild(row);
    }
  }

  const countHandler = () => renderExtraRows();
  countInput?.addEventListener("input", countHandler);
  renderExtraRows();

  const startHandler = async () => {
    const w = parseFloat(widthInput?.value);
    const h = parseFloat(heightInput?.value);
    if (!Number.isFinite(w) || w <= 0 || !Number.isFinite(h) || h <= 0) {
      showResult(false, t("calibrate_bad_dims", "Pallet dimensions must be positive."));
      return;
    }
    const n = clampCount(countInput);
    // Collect + validate the extra points' world coords (points 5..N).
    const extraWorld = [];
    for (let i = 5; i <= n; i++) {
      const x = parseFloat(el(`cal-extra-x-${i}`)?.value);
      const y = parseFloat(el(`cal-extra-y-${i}`)?.value);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        showResult(false, t("calibrate_bad_extra", "Enter X,Y metres for every extra point."));
        return;
      }
      extraWorld.push([x, y]);
    }
    // Hide the modal during draw — the cam tab needs the full panel.
    el("calibration-picker")?.classList.add("hidden");
    // Switch the big-panel view to the target cam so the operator clicks on it.
    const store = window.Alpine && Alpine.store && Alpine.store("bigPanel");
    if (store && store.view !== cameraId) store.select(cameraId);

    const camLabel = cameraId.toUpperCase().replace("_", " ");
    const order = n > 4
      ? `TL → TR → BR → BL → P5${n > 5 ? "…P" + n : ""}`
      : "TL → TR → BR → BL";
    startDraw({
      target: cameraId,
      mode: "raw",                // store source-frame pixels, no projection
      minPoints: n,
      maxPoints: n,               // auto-finish at N clicks (no Done needed)
      label: `${t("calibrate_pallet_word", "Pallet")} · ${camLabel} · ${order}`,
      onDone: async (cornersUv) => {
        await submitCalibration({ cameraId, cornersUv, palletW: w, palletH: h, extraWorld, onFinished });
      },
      onCancel: () => {
        // Re-open the picker so the operator can retry or cancel.
        el("calibration-picker")?.classList.remove("hidden");
        showResult(false, t("calibrate_cancelled", "Calibration cancelled."));
      },
    });
  };

  startBtn.addEventListener("click", startHandler);
  return () => {
    startBtn.removeEventListener("click", startHandler);
    countInput?.removeEventListener("input", countHandler);
  };
}

async function submitCalibration({ cameraId, cornersUv, palletW, palletH, extraWorld, onFinished }) {
  // We need the camera's source-frame image size to send with the payload.
  // It's stable per-camera and small; read once from /api/project/cameras.
  const imageSize = await fetchImageSize(cameraId);
  const payload = {
    camera_id: cameraId,
    image_size: imageSize || [1920, 1080],   // safe default if metadata unavailable
    pallet_width_m: palletW,
    pallet_height_m: palletH,
    corners_uv: cornersUv,
  };
  if (extraWorld && extraWorld.length) payload.extra_world_xy = extraWorld;

  // Re-open the modal to show the result.
  el("calibration-picker")?.classList.remove("hidden");

  try {
    const res = await fetch("/api/calibrate/single-cam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text();
      showResult(false, t("calibrate_failed", "Calibration failed") + ": " + detail);
      onFinished?.({ ok: false, error: detail });
      return;
    }
    const data = await res.json();
    const mm = (data.max_residual_m * 1000).toFixed(1);
    showResult(true,
      t("calibrate_success", "Calibration saved.") +
      ` ${t("calibrate_residual", "Max residual")}: ${mm} mm`);
    // Tell the button + projection caches to refresh.
    document.dispatchEvent(new CustomEvent("calibration:saved"));
    onFinished?.({ ok: true, max_residual_m: data.max_residual_m });
  } catch (err) {
    showResult(false, t("calibrate_failed", "Calibration failed") + ": " + err);
    onFinished?.({ ok: false, error: String(err) });
  }
}

async function fetchImageSize(cameraId) {
  // The projection endpoint exposes per-camera native sizes for already-
  // calibrated cameras. Pre-calibration we have nothing to read from — fall
  // back to the natural size of the cam <img> if available.
  try {
    const res = await fetch("/api/project/cameras");
    if (res.ok) {
      const data = await res.json();
      const sizes = data.image_sizes || {};
      if (sizes[cameraId]) return sizes[cameraId];
    }
  } catch {
    /* fall through */
  }
  const img = document.getElementById(`${cameraId}-img`);
  if (img && img.naturalWidth && img.naturalHeight) {
    return [img.naturalWidth, img.naturalHeight];
  }
  return null;
}

function showResult(ok, msg) {
  const panel = el("cal-result");
  const text = el("cal-result-msg");
  if (!panel || !text) return;
  panel.classList.remove("hidden", "is-success", "is-error");
  panel.classList.add(ok ? "is-success" : "is-error");
  text.textContent = msg;
}
