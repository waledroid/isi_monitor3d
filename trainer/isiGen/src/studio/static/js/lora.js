import { getJSON } from "/static/js/api.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const host = document.getElementById("lora-runs");

async function load() {
  let runs = [];
  try { runs = (await getJSON(`/api/p/${project}/lora-runs`)).runs; }
  catch (e) { host.textContent = `error: ${e.message}`; return; }
  host.innerHTML = runs.length ? "" : "<p class='msg'>no training runs yet — run phase 5</p>";
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

load();
