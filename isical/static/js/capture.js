import { flash, getJSON, sendJSON } from "/static/js/api.js";

const root = document.getElementById("cap-root");
const project = root.dataset.project;
const phase = root.dataset.phase;
const msg = document.getElementById("cap-msg");
const startBtn = document.getElementById("cap-start");      // extrinsic only
const stopBtn = document.getElementById("cap-stop");
const restartBtn = document.getElementById("cap-restart");
const camSelect = document.getElementById("cam-select");    // intrinsic only
const targetInput = document.getElementById("extrinsic-target");  // extrinsic only
const tagLenInput = document.getElementById("tag-length-cm");      // extrinsic only
const tagGapInput = document.getElementById("tag-gap-cm");         // extrinsic only
const boardDerived = document.getElementById("board-derived");     // extrinsic only
let statusTimer = null;
const shownGallery = new Set();   // cams already swapped from live → gallery (one-shot)
let floorPromptShown = false;     // extrinsic: prompt revealed once captures complete (one-shot)

// ---- ingested-shot gallery (intrinsic only) ----
function coverageSVG(shots) {
  const W = 160, H = 90;
  const dots = shots.filter((s) => s.centroid).map((s) => {
    const [x, y] = s.centroid;
    const col = s.corners >= 16 ? "#1c7a3f" : s.corners >= 8 ? "#95680f" : "#b00020";
    return `<circle cx="${(x * W).toFixed(1)}" cy="${(y * H).toFixed(1)}" r="3" `
         + `fill="${col}" fill-opacity="0.8"/>`;
  }).join("");
  return `<svg class="cov-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`
       + `<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" stroke="#cbd5e1"/>`
       + `${dots}</svg>`;
}

function thumbHTML(cam, s, blurMin) {
  const sharp = s.blur_var >= blurMin * 1.5 ? "ok" : s.blur_var >= blurMin ? "warn" : "bad";
  return `<figure class="shot">
    <img loading="lazy" src="/shots/${project}/intrinsic/${cam}/${s.file}" alt="${s.file}">
    <figcaption><span class="badge">${s.corners} ⌗</span>
      <span class="dot ${sharp}" title="sharpness ${Math.round(s.blur_var)}"></span></figcaption>
  </figure>`;
}

async function showGallery(cam) {
  const fig = document.querySelector(`.cap-figure[data-cam="${cam}"]`);
  if (!fig) return;
  const gal = fig.querySelector(".shot-gallery");
  try {
    const r = await getJSON(`/api/p/${project}/shots/intrinsic/${cam}`);
    const blurMin = r.blur_min_var || 80;
    gal.innerHTML =
      `<div class="coverage">${coverageSVG(r.shots)}
         <span class="msg">${r.count}/${r.target} shots · board coverage</span></div>
       <div class="shot-grid">${r.shots.map((s) => thumbHTML(cam, s, blurMin)).join("")}</div>`;
    fig.querySelector(".canvas-wrap").style.display = "none";
    gal.hidden = false;
  } catch { /* keep live view on error */ }
}

function showLive(cam) {
  const fig = document.querySelector(`.cap-figure[data-cam="${cam}"]`);
  if (!fig) return;
  fig.querySelector(".canvas-wrap").style.display = "";
  const gal = fig.querySelector(".shot-gallery");
  gal.hidden = true; gal.innerHTML = "";
}

async function switchCam(cam) {
  document.querySelectorAll(".cam-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.cam === cam));
  showOnly(cam);
  let done = false;
  try {
    const st = await getJSON(`/api/p/${project}/status`);
    done = (st.intrinsic_counts?.[cam] || 0) >= (st.targets?.intrinsic || Infinity);
  } catch { /* fall through to live */ }
  if (done) { showLive(cam); await showGallery(cam); }
  else { shownGallery.delete(cam); showLive(cam); startCapture(false); }
}

// intrinsic captures ONE selected camera; extrinsic captures all (sync pairs).
function activeCam() { return camSelect ? camSelect.value : null; }
function camQuery() { return activeCam() ? `?cam=${activeCam()}` : ""; }

function showOnly(cam) {
  document.querySelectorAll(".cap-figure").forEach((fig) => {
    fig.style.display = (!cam || fig.dataset.cam === cam) ? "" : "none";
  });
}

function streams(on) {
  document.querySelectorAll(".cap-stream").forEach((img) => {
    const cam = img.dataset.cam;
    const visible = !activeCam() || cam === activeCam();
    img.src = (on && visible) ? `/stream/${project}/${cam}?t=${Date.now()}` : "";
  });
}

async function pollStatus() {
  try {
    const s = await getJSON(`/api/p/${project}/capture/status`);
    if (!s.active) return;
    for (const [cam, c] of Object.entries(s.cameras || {})) {
      const el = document.querySelector(`.counts[data-cam="${cam}"]`);
      // Live per-frame tag/corner count — instant feedback on whether the board
      // placement/size is being detected (extrinsic: "tags", intrinsic: "corners").
      const unit = phase === "extrinsic" ? "tags" : "corners";
      if (el) {
        el.textContent = `${c.count}/${c.target} · ${c.status} · ${c.detections} ${unit}`;
        el.classList.toggle("no-det", (c.detections || 0) === 0);
      }
      if (phase === "intrinsic" && cam === activeCam() && c.count >= c.target && !shownGallery.has(cam)) {
        shownGallery.add(cam);
        await stopCapture();
        showGallery(cam);
      }
    }
    // Extrinsic: once EVERY camera reached the pair target, stop and prompt the
    // operator for the floor-anchor shots (run_extrinsic needs them to solve).
    if (phase === "extrinsic" && !floorPromptShown) {
      const cams = Object.values(s.cameras || {});
      const allDone = cams.length > 0 && cams.every((c) => c.count >= c.target);
      if (allDone) {
        floorPromptShown = true;
        await stopCapture();
        await revealFloorPrompt();
      }
    }
  } catch { /* studio busy */ }
}

// ---- floor-prompt state (extrinsic) ----
async function refreshFloorState() {
  // Mark each floor button ✓ from the on-disk floor booleans (phase_status);
  // when both shots exist, show the "ready to Solve" confirmation.
  const prompt = document.getElementById("floor-prompt");
  if (!prompt) return;
  let floors = {};
  try {
    const st = await getJSON(`/api/p/${project}/status`);
    floors = st.floor || {};
  } catch { return; }
  document.querySelectorAll(".floor-btn").forEach((btn) => {
    const cam = btn.dataset.cam;
    const stEl = document.querySelector(`.floor-status[data-cam="${cam}"]`);
    if (floors[cam] && stEl && !stEl.textContent.includes("corners")) {
      stEl.textContent = "✓ captured";
      stEl.className = "msg floor-status ok";
    }
  });
  const cams = Object.keys(floors);
  const allFloor = cams.length > 0 && cams.every((c) => floors[c]);
  const doneEl = document.getElementById("floor-prompt-done");
  if (doneEl) doneEl.hidden = !allFloor;
}

async function revealFloorPrompt() {
  const prompt = document.getElementById("floor-prompt");
  if (!prompt) return;
  prompt.hidden = false;
  await refreshFloorState();
  const motionOk = !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  prompt.scrollIntoView({ behavior: motionOk ? "smooth" : "auto", block: "center" });
}

function running(on) {
  if (startBtn) startBtn.disabled = on;
  stopBtn.disabled = !on;
  if (camSelect) camSelect.disabled = on && phase === "intrinsic" ? false : camSelect.disabled;
}

// Persist the operator-chosen extrinsic pair count before (re)starting capture.
async function saveExtrinsicTarget() {
  if (phase !== "extrinsic" || !targetInput) return;
  const v = parseInt(targetInput.value, 10);
  if (!Number.isFinite(v)) return;
  const floor = parseInt(targetInput.min, 10) || 4;
  const target = Math.max(floor, v);
  targetInput.value = target;
  try { await sendJSON(`/api/p/${project}/capture-config`, "PUT", { extrinsic_target: target }); }
  catch { /* keep going with the stored value */ }
}

// ---- AprilGrid board measurements (operator measures the printed tag in cm) ----
// The operator enters tag length + inter-tag gap in cm; isical derives the Kalibr
// board config: tag_length_m = length/100, tag_spacing = gap/length (the ratio).
function deriveBoard() {
  if (!tagLenInput || !tagGapInput) return null;
  const len = parseFloat(tagLenInput.value);
  const gap = parseFloat(tagGapInput.value);
  if (!Number.isFinite(len) || len <= 0 || !Number.isFinite(gap) || gap < 0) {
    if (boardDerived) { boardDerived.textContent = "↛ enter valid cm values"; boardDerived.className = "msg board-derived bad"; }
    return null;
  }
  const tag_length_m = len / 100;
  const tag_spacing = gap / len;
  if (boardDerived) {
    boardDerived.textContent = `→ tag_length_m ${tag_length_m.toFixed(2)}, tag_spacing ${tag_spacing.toFixed(2)}`;
    boardDerived.className = "msg board-derived";
  }
  return { tag_length_cm: len, tag_gap_cm: gap };
}

let _boardSaveTimer = null;
async function saveBoardConfig() {
  const body = deriveBoard();
  if (!body) return;
  try { await sendJSON(`/api/p/${project}/board-config`, "PUT", body); }
  catch { /* keep the in-form values; capture still uses the stored config */ }
}

if (tagLenInput && tagGapInput) {
  deriveBoard();   // show the derived config on load
  for (const el of [tagLenInput, tagGapInput]) {
    el.addEventListener("input", () => {
      deriveBoard();
      clearTimeout(_boardSaveTimer);
      _boardSaveTimer = setTimeout(saveBoardConfig, 500);   // debounced persist
    });
  }
}

async function startCapture(restart = false) {
  await saveExtrinsicTarget();
  await saveBoardConfig();
  const verb = restart ? "restart" : "start";
  try {
    await sendJSON(`/api/p/${project}/capture/${phase}/${verb}${camQuery()}`, "POST", {});
    flash(msg, restart ? "wiped + recapturing" : "capturing — auto-snap on a good board", true);
    running(true);
    showOnly(activeCam());
    streams(true);
    if (!statusTimer) statusTimer = setInterval(pollStatus, 700);
  } catch (e) { flash(msg, e.message, false); }
}

async function stopCapture() {
  try { await sendJSON(`/api/p/${project}/capture/${phase}/stop`, "POST", {}); } catch { /* */ }
  clearInterval(statusTimer); statusTimer = null;
  streams(false);
  running(false);
  flash(msg, "stopped — go to the board to Solve", true);
}

if (startBtn) startBtn.onclick = () => startCapture(false);
stopBtn.onclick = stopCapture;
restartBtn.onclick = async () => {
  if (!confirm(`Restart ${phase}${activeCam() ? " for " + activeCam() : ""}? This DELETES the captured `
               + `images and starts again.`)) return;
  await startCapture(true);
};

// Intrinsic: tabs drive the (hidden) select; switching decides live vs gallery.
if (camSelect) {
  document.querySelectorAll(".cam-tab").forEach((btn) => {
    btn.onclick = async () => {
      if (btn.dataset.cam === camSelect.value && !stopBtn.disabled) return; // already live here
      await stopCapture();
      camSelect.value = btn.dataset.cam;
      switchCam(btn.dataset.cam);
    };
  });
  switchCam(camSelect.value);     // auto-open the first camera on load
}

window.addEventListener("beforeunload", () => {
  navigator.sendBeacon?.(`/api/p/${project}/capture/${phase}/stop`);
});

// ---- floor anchor shots (extrinsic only; needs capture stopped) ----
document.querySelectorAll(".floor-btn").forEach((btn) => {
  btn.onclick = async () => {
    const cam = btn.dataset.cam;
    const st = document.querySelector(`.floor-status[data-cam="${cam}"]`);
    btn.disabled = true;
    if (st) { st.textContent = "grabbing…"; st.className = "msg floor-status"; }
    try {
      const r = await sendJSON(`/api/p/${project}/floor/${cam}`, "POST", {});
      if (st) { st.textContent = `✓ ${r.corners} corners`; st.className = "msg floor-status ok"; }
      await refreshFloorState();
    } catch (e) {
      if (st) { st.textContent = e.message; st.className = "msg floor-status bad"; }
    } finally { btn.disabled = false; }
  };
});

// On load (extrinsic), reflect any floor shots already on disk.
if (phase === "extrinsic") refreshFloorState();
