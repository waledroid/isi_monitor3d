import { getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const board = document.getElementById("phase-board");
const project = board.dataset.project;
const logEl = document.getElementById("job-log");

const PHASES = [
  { key: "curate",   n: 1, title: "Curate real images",  page: "curate",
    counts: (s) => `${s.records - s.excluded} active / ${s.records} total` },
  { key: "maps",     n: 2, title: "Control maps",        page: "maps", run: "maps",
    counts: (s) => `depth ${s.depth} · canny ${s.canny}` },
  { key: "masks",    n: 2, title: "Ground-truth masks",  page: "maps", run: "masks",
    counts: (s) => `${s.masked} masked · ${s.prompted} prompted · ${s.needs_review} review` },
  { key: "captions", n: 3, title: "Anti-bleed captions", page: "captions", run: "captions",
    counts: (s) => `${s.captioned} written · ${s.caption_edited} edited` },
  { key: "lora",     n: 4, title: "LoRA training (SD3.5 QLoRA)", run: "lora",
    counts: () => "hours-long GPU job" },
  { key: "scaffolds",n: 6, title: "Synthetic scaffolds", run: "scaffolds",
    counts: (s) => { const c = s.scaffolds || {}; return `${c.total ?? 0} pairs · ${c.pending ?? 0} pending`; } },
  { key: "generate", n: 7, title: "Mint synthetics (SD3.5)", run: "generate",
    counts: (s) => { const c = s.scaffolds || {}; return `${s.synthetic ?? 0} minted · ${c.pending ?? 0} queued`; } },
  { key: "export",   n: 8, title: "Filter + export",     run: "export",
    counts: (s) => `${s.clip_scored ?? 0} scored · yolo_seg ${s.exported ? "OK" : "—"}` },
];

async function render() {
  let s = {};
  try { s = await getJSON(`/api/p/${project}/status`); } catch { /* keep zeros */ }
  board.innerHTML = "";
  for (const ph of PHASES) {
    const card = document.createElement("div");
    card.className = "phase-card" + (ph.stub ? " stub" : "");
    card.innerHTML = `<h3>P${ph.n} · ${ph.title}</h3>
      <div class="counts">${ph.stub ? "lands next session" : (ph.counts?.(s) ?? "")}</div>`;
    if (ph.page) {
      const a = document.createElement("a");
      a.href = `/p/${project}/${ph.page}`; a.textContent = "open ›";
      card.appendChild(a);
    }
    if (ph.run) {
      const b = document.createElement("button");
      b.textContent = "Run";
      b.onclick = async () => {
        try {
          const { job } = await sendJSON(`/api/p/${project}/run/${ph.run}`, "POST", {});
          watchJob(job.id, logEl, render);
        } catch (e) { logEl.textContent = `error: ${e.message}`; }
      };
      card.appendChild(b);
    }
    board.appendChild(card);
  }
}

render();
setInterval(render, 5000);
