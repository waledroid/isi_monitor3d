import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const host = document.getElementById("lora-runs");
const stepsInput = document.getElementById("lora-steps");
const msg = document.getElementById("lora-msg");

async function load() {
  let data = { runs: [], max_steps: 2000 };
  try { data = await getJSON(`/api/p/${project}/lora-runs`); }
  catch (e) { host.textContent = `error: ${e.message}`; return; }
  if (document.activeElement !== stepsInput) stepsInput.value = data.max_steps ?? 2000;
  const runs = data.runs;
  host.innerHTML = runs.length ? "" : "<p class='msg'>no training runs yet — set steps and Train LoRA</p>";
  for (const r of runs) {
    const block = document.createElement("div");
    block.className = "lora-run";
    const plot = r.has_plot
      ? `<img class="lora-plot" src="/media/${project}/lora/${r.run}/plot" alt="loss curve">`
      : `<p class="msg">(no loss curve — trained before plots were added)</p>`;
    block.innerHTML =
      `<h3>${r.run}${r.has_weights ? "" : " <span class='msg'>(no weights)</span>"}</h3>
       ${plot}
       <pre class="lora-report">${(r.report || "(no report)").replace(/</g, "&lt;")}</pre>`;
    host.appendChild(block);
  }
}

document.getElementById("lora-train").onclick = async () => {
  const steps = parseInt(stepsInput.value, 10);
  if (!steps || steps < 1) { flash(msg, "enter a step count", false); return; }
  if (!confirm(`Train LoRA for ${steps} steps?\nThis is a long GPU job (≈ ${Math.round(steps * 7 / 60)} min at ~7s/step).`)) return;
  try {
    const { job } = await sendJSON(`/api/p/${project}/run/lora`, "POST", { max_steps: steps });
    flash(msg, `training started (${steps} steps)…`, true);
    watchJob(job.id, null, () => { flash(msg, "training done", true); load(); });
  } catch (e) { flash(msg, e.message, false); }
};

load();
