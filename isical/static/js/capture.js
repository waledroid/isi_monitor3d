import { flash, getJSON, sendJSON } from "/static/js/api.js";

const root = document.getElementById("cap-root");
const project = root.dataset.project;
const phase = root.dataset.phase;
const msg = document.getElementById("cap-msg");
const startBtn = document.getElementById("cap-start");
const stopBtn = document.getElementById("cap-stop");
let statusTimer = null;

function streams(on) {
  document.querySelectorAll(".cap-stream").forEach((img) => {
    const cam = img.dataset.cam;
    img.src = on ? `/stream/${project}/${cam}?t=${Date.now()}` : "";
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

startBtn.onclick = async () => {
  try {
    const { status } = await sendJSON(`/api/p/${project}/capture/${phase}/start`, "POST", {});
    flash(msg, "capturing — auto-snap on a good board", true);
    startBtn.disabled = true; stopBtn.disabled = false;
    streams(true);
    statusTimer = setInterval(pollStatus, 700);
    pollStatus(status);
  } catch (e) { flash(msg, e.message, false); }
};

stopBtn.onclick = async () => {
  try { await sendJSON(`/api/p/${project}/capture/${phase}/stop`, "POST", {}); } catch { /* */ }
  clearInterval(statusTimer); statusTimer = null;
  streams(false);
  startBtn.disabled = false; stopBtn.disabled = true;
  flash(msg, "stopped — go to the board to Solve", true);
};

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

window.addEventListener("beforeunload", () => {
  navigator.sendBeacon?.(`/api/p/${project}/capture/${phase}/stop`);
});
