import { getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const board = document.getElementById("phase-board");
const project = board.dataset.project;
const logEl = document.getElementById("job-log");

// state(s) → "done" (light green) | "partial" (amber) | "todo". `active` is the
// number of non-excluded records; a phase is done when it covers all of them.
const active = (s) => (s.records ?? 0) - (s.excluded ?? 0);

const PHASES = [
  { key: "curate",   n: 1, title: "Curate real images",  page: "curate",
    counts: (s) => `${active(s)} active / ${s.records} total`,
    state: (s) => active(s) > 0 ? "done" : "todo" },
  { key: "maps",     n: 2, title: "Control maps",        page: "maps", run: "maps",
    counts: (s) => `depth ${s.depth} · canny ${s.canny}`,
    state: (s) => { const a = active(s); if (!a) return "todo";
      if (s.depth >= a && s.canny >= a) return "done";
      return (s.depth || s.canny) ? "partial" : "todo"; } },
  { key: "masks",    n: 2, title: "Ground-truth masks",  page: "maps", run: "masks",
    counts: (s) => `${s.masked} masked · ${s.prompted} prompted · ${s.needs_review} review`,
    state: (s) => { const a = active(s); if (!a) return "todo";
      if (s.masked >= a && !s.needs_review) return "done";
      return s.masked ? "partial" : "todo"; } },
  { key: "captions", n: 3, title: "Anti-bleed captions", page: "captions", run: "captions",
    counts: (s) => `${s.captioned} written · ${s.caption_edited} edited`,
    state: (s) => { const a = active(s); if (!a) return "todo";
      if (s.captioned >= a) return "done";
      return s.captioned ? "partial" : "todo"; } },
  { key: "lora",     n: 4, title: "LoRA training (SDXL LoRA)", run: "lora",
    counts: () => "hours-long GPU job",
    state: (s) => s.lora_trained ? "done" : "todo" },
  { key: "scaffolds",n: 6, title: "Synthetic scaffolds", run: "scaffolds",
    counts: (s) => { const c = s.scaffolds || {}; return `${c.total ?? 0} pairs · ${c.pending ?? 0} pending`; },
    state: (s) => (s.scaffolds?.total ?? 0) > 0 ? "done" : "todo" },
  { key: "generate", n: 7, title: "Mint synthetics (SDXL)", run: "generate",
    counts: (s) => { const c = s.scaffolds || {}; return `${s.synthetic ?? 0} minted · ${c.pending ?? 0} queued`; },
    state: (s) => { const minted = s.synthetic ?? 0, pending = s.scaffolds?.pending ?? 0;
      if (minted > 0 && !pending) return "done";
      return minted ? "partial" : "todo"; } },
  { key: "export",   n: 8, title: "Filter + export",     run: "export",
    counts: (s) => `${s.clip_scored ?? 0} scored · yolo_seg ${s.exported ? "OK" : "—"}`,
    state: (s) => s.exported ? "done" : "todo" },
];

async function render() {
  let s = {};
  try { s = await getJSON(`/api/p/${project}/status`); } catch { /* keep zeros */ }
  board.innerHTML = "";
  for (const ph of PHASES) {
    const st = ph.stub ? "todo" : (ph.state?.(s) ?? "todo");
    const card = document.createElement("div");
    card.className = "phase-card" + (ph.stub ? " stub" : "") +
      (st === "done" ? " done" : st === "partial" ? " partial" : "");
    card.innerHTML = `<h3>P${ph.n} · ${ph.title}</h3>
      <div class="counts">${ph.stub ? "lands next session" : (ph.counts?.(s) ?? "")}</div>`;
    if (ph.page) {
      const a = document.createElement("a");
      a.href = `/p/${project}/${ph.page}`; a.textContent = "open ›";
      card.appendChild(a);
    }
    if (ph.run) {
      const b = document.createElement("button");
      b.textContent = st === "done" ? "Re-run" : "Run";
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

document.getElementById("refresh-board")?.addEventListener("click", render);

render();
setInterval(render, 5000);
