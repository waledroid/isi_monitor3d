import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const gallery = document.getElementById("mint-gallery");
const showMask = document.getElementById("show-mask");
const strengthInput = document.getElementById("mint-strength");
const strengthVal = document.getElementById("strength-val");
const mintMsg = document.getElementById("mint-msg");

let synth = [];

async function load() {
  let records = [];
  try { records = (await getJSON(`/api/p/${project}/records`)).records; }
  catch (e) { gallery.textContent = `error: ${e.message}`; return; }
  synth = records.filter((r) => r.synthetic && !r.excluded);
  render();
}

function render() {
  const masks = showMask.checked;
  document.getElementById("mint-count").textContent = `${synth.length} minted`;
  gallery.innerHTML = synth.length ? "" : "<p class='msg'>nothing minted yet — run phase 7</p>";
  for (const r of synth) {
    const kind = masks && r.mask ? "mask" : "image";
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML =
      `<img src="/media/${project}/${kind}/${r.id}" loading="lazy" alt="${r.id}">
       <div class="meta">${r.id}</div>`;
    gallery.appendChild(tile);
  }
}

showMask.addEventListener("change", render);

// ── Phase-7 generate controls (strength slider shown only for the inpaint path) ──
async function loadGenSettings() {
  try {
    const g = await getJSON(`/api/p/${project}/generation`);
    const label = document.getElementById("strength-label");
    const help = document.getElementById("strength-help");
    const testBtn = document.getElementById("strength-test");
    if (label) label.hidden = help.hidden = !g.is_inpaint;
    if (testBtn) testBtn.hidden = !g.is_inpaint;
    if (strengthInput && document.activeElement !== strengthInput) {
      strengthInput.value = g.strength;
      if (strengthVal) strengthVal.textContent = Number(g.strength).toFixed(2);
    }
  } catch { /* leave defaults */ }
}

const testBtn = document.getElementById("strength-test");
if (testBtn) testBtn.onclick = async () => {
  testBtn.disabled = true;
  flash(mintMsg, "sweeping strengths 0.1–0.7 (3 samples each)… this runs several mints", true);
  try {
    const { job } = await sendJSON(`/api/p/${project}/run/strength_test`, "POST", {});
    watchJob(job.id, null, (j) => {
      testBtn.disabled = false;
      flash(mintMsg, j.state === "done"
        ? "strength sweep done → montage on the project board (above the Job log)"
        : j.error, j.state === "done");
    });
  } catch (e) { testBtn.disabled = false; flash(mintMsg, e.message, false); }
};
strengthInput?.addEventListener("input", () => {
  if (strengthVal) strengthVal.textContent = Number(strengthInput.value).toFixed(2);
});

document.getElementById("mint-run").onclick = async () => {
  const body = strengthInput ? { strength: parseFloat(strengthInput.value) } : {};
  try {
    const { job } = await sendJSON(`/api/p/${project}/run/generate`, "POST", body);
    flash(mintMsg, "generating…", true);
    watchJob(job.id, null, (j) => {
      flash(mintMsg, j.state === "done" ? "done" : j.error, j.state === "done");
      load();
    });
  } catch (e) { flash(mintMsg, e.message, false); }
};

load();
loadGenSettings();
