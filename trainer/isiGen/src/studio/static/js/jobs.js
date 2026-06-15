// Topbar job indicator — polls /api/jobs; pages can also import watchJob().
import { getJSON } from "/static/js/api.js";

const bar = document.getElementById("jobbar");

async function tick() {
  try {
    const { jobs } = await getJSON("/api/jobs");
    const j = jobs[0];
    if (!j) { bar.textContent = ""; bar.className = "jobbar"; return; }
    bar.textContent = `${j.project} · ${j.phase} · ${j.state}${fmtPct(j.progress)}`;
    bar.className = "jobbar " + (j.state === "running" ? "running"
                                : j.state === "failed" ? "failed" : "");
  } catch { /* studio restarting */ }
}
setInterval(tick, 2000);
tick();

function fmtPct(p) {
  return p && p.total ? ` · ${Math.round(100 * p.done / p.total)}% (${p.done}/${p.total})` : "";
}

// Render a <progress> bar into #job-progress (if the page has one).
function renderProgress(job) {
  const host = document.getElementById("job-progress");
  if (!host) return;
  const p = job.progress;
  if (job.state === "running" && p && p.total) {
    host.innerHTML = `<progress value="${p.done}" max="${p.total}"></progress>` +
      `<span class="msg">${p.label || job.phase} ${p.done}/${p.total} · ` +
      `${Math.round(100 * p.done / p.total)}%</span>`;
    host.style.display = "flex";
  } else if (job.state === "running") {
    host.innerHTML = `<progress></progress><span class="msg">${job.phase}…</span>`;
    host.style.display = "flex";
  } else {
    host.style.display = "none";
  }
}

// Poll one job until it finishes; stream its log into `logEl`; resolve with the job.
export function watchJob(jobId, logEl, onDone) {
  const t = setInterval(async () => {
    try {
      const { job, log } = await getJSON(`/api/jobs/${jobId}/log`);
      if (logEl) { logEl.textContent = log.join("\n") || "…"; logEl.scrollTop = logEl.scrollHeight; }
      renderProgress(job);
      if (job.state === "done" || job.state === "failed") {
        clearInterval(t);
        renderProgress(job);
        if (onDone) onDone(job);
      }
    } catch { clearInterval(t); }
  }, 1000);
}
