import { getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const board = document.getElementById("phase-board");
const project = board.dataset.project;
const logEl = document.getElementById("job-log");

// state(s) → "done" (light green) | "partial" (amber) | "todo".
// Phases 1-4 act on REAL curated images, so they measure against `real` (active
// non-synthetic records) — minted/synthetic records must not block them.
const real = (s) => s.real ?? Math.max(0, (s.records ?? 0) - (s.excluded ?? 0));

const PHASES = [
  { key: "curate",   n: 1, title: "Curate real images",  page: "curate",
    counts: (s) => `${real(s)} images / ${s.records} total`,
    state: (s) => real(s) > 0 ? "done" : "todo" },
  { key: "maps",     n: 2, title: "Control maps",        page: "maps", run: "maps",
    counts: (s) => `depth ${s.depth} · canny ${s.canny}`,
    state: (s) => { const a = real(s); if (!a) return "todo";
      if (s.depth >= a && s.canny >= a) return "done";
      return (s.depth || s.canny) ? "partial" : "todo"; } },
  { key: "masks",    n: 3, title: "Ground-truth masks",  page: "masks", run: "masks",
    counts: (s) => `${s.masked} masked · ${s.prompted} prompted · ${s.needs_review} review`,
    state: (s) => { const a = real(s); if (!a) return "todo";
      if (s.masked >= a) return "done";          // green once all are masked; review is optional
      return s.masked ? "partial" : "todo"; } },
  { key: "captions", n: 4, title: "Anti-bleed captions", page: "captions", run: "captions",
    counts: (s) => `${s.captioned} written · ${s.caption_edited} edited`,
    state: (s) => { const a = real(s); if (!a) return "todo";
      if (s.captioned >= a) return "done";
      return s.captioned ? "partial" : "todo"; } },
  { key: "lora",     n: 5, title: "LoRA training", page: "lora", run: "lora",
    counts: () => "hours-long GPU job",
    state: (s) => s.lora_trained ? "done" : "todo" },
  { key: "scaffolds",n: 6, title: "Synthetic scaffolds", page: "scaffolds", run: "scaffolds",
    counts: (s) => { const c = s.scaffolds || {}; return `${c.total ?? 0} pairs · ${c.generated ?? 0} minted`; },
    state: (s) => (s.scaffolds?.total ?? 0) > 0 ? "done" : "todo" },
  { key: "generate", n: 7, title: "Mint synthetics", page: "mint", run: "generate",
    counts: (s) => { const c = s.scaffolds || {}; return `${s.synthetic ?? 0} minted · ${c.pending ?? 0} queued`; },
    state: (s) => (s.synthetic ?? 0) > 0 ? "done" : "todo" },   // green once anything is minted
  { key: "export",   n: 8, title: "Filter + export",     run: "export",
    counts: (s) => `${s.clip_scored ?? 0} scored · yolo_seg ${s.exported ? "OK" : "—"}`,
    state: (s) => s.exported ? "done" : "todo" },
];

// Labeling-mode projects skip the synthetic half — the board is just
// curate → SAM2 masks → LabelMe export (export unlocks right after masks).
const LABEL_PHASES = [
  { ...PHASES.find((p) => p.key === "curate"), n: 1 },
  { ...PHASES.find((p) => p.key === "masks"),  n: 2 },
  { key: "export", n: 3, title: "Export (LabelMe)", run: "export",
    counts: (s) => `${s.masked ?? 0} masked${s.backgrounds ? ` · ${s.backgrounds} bg` : ""} · labelme ${s.exported ? "OK" : "—"}`,
    state: (s) => s.exported ? "done" : "todo" },
];

async function render() {
  let s = {};
  try { s = await getJSON(`/api/p/${project}/status`); } catch { /* keep zeros */ }
  board.innerHTML = "";
  // Labeling projects show only curate → masks → export; generation is hidden.
  const phases = s.mode === "label" ? LABEL_PHASES : PHASES;
  // Sequential chain: a not-yet-done phase is locked until its predecessor is
  // done. Already-done phases stay re-runnable. `prev` carries the previous
  // phase's state (and number, for the hint); the first phase is never locked.
  let prevState = "done";
  let prevPhase = null;
  for (const ph of phases) {
    const st = ph.stub ? "todo" : (ph.state?.(s) ?? "todo");
    const locked = prevState !== "done" && st !== "done";
    const glyph = locked ? "🔒" : st === "done" ? "✓" : st === "partial" ? "◐" : "";
    const card = document.createElement("div");
    card.className = "phase-card" + (ph.stub ? " stub" : "") + (locked ? " locked" : "") +
      (st === "done" ? " done" : st === "partial" ? " partial" : "");
    card.innerHTML =
      `<div class="phase-head">
         <span class="phase-num">${ph.n}</span>
         <span class="phase-title">${ph.title}</span>
         <span class="phase-status">${glyph}</span>
       </div>
       <div class="counts">${ph.stub ? "lands next session" : (ph.counts?.(s) ?? "")}</div>
       <div class="phase-actions"></div>`;
    const actions = card.querySelector(".phase-actions");
    if (ph.page) {
      const a = document.createElement("a");
      a.href = `/p/${project}/${ph.page}`; a.textContent = "open ›";
      actions.appendChild(a);
    }
    if (ph.run) {
      const b = document.createElement("button");
      b.textContent = st === "done" ? "Re-run" : "Run";
      if (locked) {
        b.disabled = true;
        b.title = `Complete phase ${prevPhase?.n ?? ""} first`;
      }
      b.onclick = async () => {
        try {
          const { job } = await sendJSON(`/api/p/${project}/run/${ph.run}`, "POST", {});
          watchJob(job.id, logEl, render);
        } catch (e) { logEl.textContent = `error: ${e.message}`; }
      };
      actions.appendChild(b);
      // Reset: wipe this phase's outputs so it can be re-run cleanly.
      if (st !== "todo") {
        const rb = document.createElement("button");
        rb.className = "reset-btn";
        rb.textContent = "Reset";
        rb.title = `Delete ${ph.title} outputs`;
        rb.onclick = async () => {
          if (!confirm(`Reset "${ph.title}"?\nThis deletes this phase's outputs so you can re-run it cleanly.`)) return;
          try {
            const r = await sendJSON(`/api/p/${project}/reset/${ph.run}`, "POST", {});
            logEl.textContent = `reset ${ph.run}: ${JSON.stringify(r.reset)}`;
            render();
          } catch (e) { logEl.textContent = `error: ${e.message}`; }
        };
        actions.appendChild(rb);
      }
    }
    board.appendChild(card);
    prevState = st;
    prevPhase = ph;
  }
}

document.getElementById("refresh-board")?.addEventListener("click", render);

// Show the latest strength-sweep montage (from the Mint page's "Test strengths")
// just above the Job log — hidden until one exists.
function loadStrengthMontage() {
  const card = document.getElementById("strength-montage-card");
  const img = document.getElementById("strength-montage");
  const link = document.getElementById("strength-montage-link");
  if (!card || !img) return;
  const url = `/api/p/${project}/strength-montage?n=${Date.now()}`;   // cache-bust
  img.onload = () => { card.hidden = false; };
  img.onerror = () => { card.hidden = true; };
  if (link) link.href = url;                          // click → full-res in new tab
  img.src = url;
}

render();
setInterval(render, 5000);
loadStrengthMontage();
