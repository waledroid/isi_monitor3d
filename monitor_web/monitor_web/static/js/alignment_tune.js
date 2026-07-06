// Cross-camera alignment fine-tune (Settings ▸ Zones ▸ alignment section).
//
// Flow: [Pick 4 point pairs] → the Settings modal hides; for each of the 4
// physical floor spots the operator clicks it FIRST on CAM 1, then on CAM 2
// (the big panel switches automatically). The pairs go to
// POST /api/alignment/fit, which fits the rigid floor correction, writes
// calibration_refined.json and reports the residual. The checkbox toggles
// which calibration the system points at; [Refit] reuses the stored pairs
// after a new solve (staleness is reported by GET /api/alignment).

import { startDraw } from "/static/js/draw_mode.js";

const N_PAIRS = 4;

function el(id) { return document.getElementById(id); }

function t(key, fallback) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fallback;
}

// One raw click on one camera, as a promise. Resolves [u, v] (stream px) or
// null on cancel.
function clickOnce(cam, label) {
  return new Promise((resolve) => {
    const store = window.Alpine?.store?.("bigPanel");
    if (store && store.view !== cam) store.select(cam);
    setTimeout(() => startDraw({
      target: cam,
      mode: "raw",
      label,
      minPoints: 1,
      maxPoints: 1,               // auto-finishes on the click
      onDone: (pts) => resolve(pts && pts.length ? pts[0] : null),
      onCancel: () => resolve(null),
    }), 250);                     // let the cam view mount
  });
}

function frameWh(cam) {
  const img = el(`${cam}-img`);
  return img && img.naturalWidth ? [img.naturalWidth, img.naturalHeight] : null;
}

function reopenSettings() {
  el("btn-add-zone")?.click();
  setTimeout(() => {
    document.querySelector('.settings-tab-btn[data-tab="zones"]')?.click();
  }, 120);
}

async function pickPairs() {
  el("zone-manager")?.classList.add("hidden");   // the cams must be clickable
  const pairs = [];
  const wh = {};
  for (let i = 1; i <= N_PAIRS; i++) {
    const a = await clickOnce("cam_a",
      t("align_click_a", `Alignment point ${i}/${N_PAIRS} — click the floor spot on CAM 1`));
    if (!a) { reopenSettings(); return; }
    wh.cam_a = frameWh("cam_a") || wh.cam_a;
    const b = await clickOnce("cam_b",
      t("align_click_b", `Same spot ${i}/${N_PAIRS} — now click it on CAM 2`));
    if (!b) { reopenSettings(); return; }
    wh.cam_b = frameWh("cam_b") || wh.cam_b;
    pairs.push({ cam_a: a, cam_b: b });
  }
  try {
    const res = await fetch("/api/alignment/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pairs, frame_wh: wh }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      window.alert(data.detail || t("align_fit_failed", "Alignment fit failed"));
    } else {
      const f = data.fit || {};
      window.alert(
        `${t("align_fit_ok", "Alignment fitted")}: `
        + `residual ${((f.max_residual_m || 0) * 100).toFixed(1)} cm `
        + `(was ${(Math.max(...(data.before_error_m || [0])) * 100).toFixed(1)} cm). `
        + t("align_enable_hint", "Enable it with the checkbox, then restart the Backbone."));
    }
  } catch (e) {
    window.alert(`${t("align_fit_failed", "Alignment fit failed")}: ${e}`);
  }
  reopenSettings();
}

export async function loadAlignment() {
  const status = el("zm-align-status");
  const enabled = el("zm-align-enabled");
  const refit = el("zm-align-refit");
  const clear = el("zm-align-clear");
  if (!status) return;
  let st = null;
  try {
    const res = await fetch("/api/alignment");
    if (res.ok) st = await res.json();
  } catch { /* leave defaults */ }
  const fit = st && st.fit;
  if (!fit) {
    status.textContent = t("align_none", "No fine-tune yet — pick 4 point pairs.");
  } else {
    const parts = [
      `${t("align_fitted", "fitted")}: ${fit.theta_deg.toFixed(2)}°, `
      + `(${(fit.tx_m * 100).toFixed(1)}, ${(fit.ty_m * 100).toFixed(1)}) cm, `
      + `${t("align_residual", "residual")} ${(fit.max_residual_m * 100).toFixed(1)} cm`,
    ];
    if (st.stale) parts.push(t("align_stale", "⚠ calibration re-solved since the fit — Refit"));
    status.textContent = parts.join(" · ");
  }
  if (enabled) {
    enabled.disabled = !fit;
    enabled.checked = !!(st && st.enabled);
  }
  if (refit) refit.hidden = !(st && st.stale && st.pairs && st.pairs.length >= 3);
  if (clear) clear.hidden = !fit;
}

async function setEnabled(on) {
  try {
    const res = await fetch("/api/alignment/enable", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: on }),
    });
    if (!res.ok) window.alert((await res.json()).detail || "toggle failed");
    else document.dispatchEvent(new CustomEvent("config:saved"));  // refresh overlays
  } catch { /* next open re-syncs */ }
  await loadAlignment();
}

async function doRefit() {
  try {
    const res = await fetch("/api/alignment/refit", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) window.alert(data.detail || "refit failed");
    else document.dispatchEvent(new CustomEvent("config:saved"));
  } catch { /* */ }
  await loadAlignment();
}

async function doClear() {
  if (!window.confirm(t("align_clear_confirm",
      "Forget the alignment fine-tune and go back to the base calibration?"))) return;
  try { await fetch("/api/alignment", { method: "DELETE" }); } catch { /* */ }
  document.dispatchEvent(new CustomEvent("config:saved"));
  await loadAlignment();
}

function wire() {
  el("zm-align-pick")?.addEventListener("click", pickPairs);
  el("zm-align-enabled")?.addEventListener("change", (ev) => setEnabled(ev.target.checked));
  el("zm-align-refit")?.addEventListener("click", doRefit);
  el("zm-align-clear")?.addEventListener("click", doClear);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
