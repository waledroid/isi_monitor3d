import { flash, getJSON, sendJSON } from "/static/js/api.js";

const root = document.getElementById("cap-root");
const project = root.dataset.project;
const phase = root.dataset.phase;
const msg = document.getElementById("cap-msg");
const startBtn = document.getElementById("cap-start");      // extrinsic only
const stopBtn = document.getElementById("cap-stop");
const restartBtn = document.getElementById("cap-restart");
const camSelect = document.getElementById("cam-select");    // intrinsic only
let statusTimer = null;

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
      if (el) el.textContent = `${c.count}/${c.target} · ${c.status} · ${c.detections} det`;
    }
  } catch { /* studio busy */ }
}

function running(on) {
  if (startBtn) startBtn.disabled = on;
  stopBtn.disabled = !on;
  if (camSelect) camSelect.disabled = on && phase === "intrinsic" ? false : camSelect.disabled;
}

async function startCapture(restart = false) {
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

// Intrinsic: auto-start the selected camera, and re-start on change.
if (camSelect) {
  showOnly(camSelect.value);
  camSelect.addEventListener("change", async () => {
    await stopCapture();
    startCapture(false);
  });
  startCapture(false);     // auto-start on page load
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
    } catch (e) {
      if (st) { st.textContent = e.message; st.className = "msg floor-status bad"; }
    } finally { btn.disabled = false; }
  };
});
