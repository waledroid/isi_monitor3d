// Topbar job indicator — polls /api/jobs; pages can also import watchJob().
import { getJSON } from "/static/js/api.js";

const bar = document.getElementById("jobbar");

async function tick() {
  try {
    const { jobs } = await getJSON("/api/jobs");
    const j = jobs[0];
    if (!j) { bar.textContent = ""; bar.className = "jobbar"; return; }
    bar.textContent = `${j.project} · ${j.phase} · ${j.state}`;
    bar.className = "jobbar " + (j.state === "running" ? "running"
                                : j.state === "failed" ? "failed" : "");
  } catch { /* studio restarting */ }
}
setInterval(tick, 2000);
tick();

// Poll one job until it finishes; stream its log into `logEl`; resolve with the job.
export function watchJob(jobId, logEl, onDone) {
  const t = setInterval(async () => {
    try {
      const { job, log } = await getJSON(`/api/jobs/${jobId}/log`);
      if (logEl) { logEl.textContent = log.join("\n") || "…"; logEl.scrollTop = logEl.scrollHeight; }
      if (job.state === "done" || job.state === "failed") {
        clearInterval(t);
        if (onDone) onDone(job);
      }
    } catch { clearInterval(t); }
  }, 1000);
}
