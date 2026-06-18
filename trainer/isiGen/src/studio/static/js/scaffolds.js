import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const gallery = document.getElementById("scaffold-gallery");
const pasteLabel = document.getElementById("paste-label");
const help = document.getElementById("paste-help");
const pasteSel = document.getElementById("paste-count");
const placeSel = document.getElementById("placement");
const placeLabel = document.getElementById("placement-label");
const countInput = document.getElementById("scaffold-count");
const msg = document.getElementById("scaffold-msg");
let usesCopyPaste = false;       // set in load(); the run handler reads it

async function load() {
  let resp = { scaffolds: [], sources: [], paste_count: 1, count: 500 };
  try { resp = await getJSON(`/api/p/${project}/scaffolds`); }
  catch (e) { gallery.textContent = `error: ${e.message}`; return; }
  const items = resp.scaffolds;
  // Count applies to ALL sources; the objects-per-background toggle is copy_paste-only.
  usesCopyPaste = (resp.sources || []).includes("copy_paste");
  pasteLabel.hidden = help.hidden = !usesCopyPaste;
  if (placeLabel) placeLabel.hidden = !usesCopyPaste;
  if (placeSel && document.activeElement !== placeSel) {
    placeSel.value = resp.placement || "random";   // default random
  }
  // Resolved synthesis path badge (mode + why).
  const badge = document.getElementById("synthesis-badge");
  if (badge && resp.synthesis) {
    const s = resp.synthesis;
    badge.textContent = `▶ ${s.path}  ·  mode: ${s.mode}, ${s.bg_count} bg`;
  }
  if (document.activeElement !== countInput) countInput.value = resp.count ?? 500;
  if (usesCopyPaste && document.activeElement !== pasteSel) {
    pasteSel.value = Array.isArray(resp.paste_count) ? "1-2" : "1";
  }
  gallery.innerHTML = items.length ? "" : "<p class='msg'>no scaffolds yet — run phase 6</p>";
  let minted = 0;
  for (const e of items) {
    const cls = (e.classes || []).join(", ");
    const tag = e.status === "generated" ? " · ✓ minted" : "";
    if (e.status === "generated") minted++;
    const tile = document.createElement("div");
    tile.className = "tile scaffold-tile";
    tile.innerHTML =
      `<div class="pair">
         <figure><img src="/media/${project}/scaffold/${e.id}/control" loading="lazy" alt="control"><figcaption>control</figcaption></figure>
         <figure><img src="/media/${project}/scaffold/${e.id}/mask" loading="lazy" alt="mask"><figcaption>mask</figcaption></figure>
       </div>
       <div class="meta">${e.id}${tag}${cls ? " · " + cls : ""}</div>`;
    gallery.appendChild(tile);
  }
  const head = document.querySelector("#gal-root .msg");
  if (head) head.textContent = `${items.length} scaffolds · ${minted} minted. Each pair is a control map (depth) + its ground-truth mask.`;
}

document.getElementById("scaffold-run").onclick = async () => {
  const body = {};
  const n = parseInt(countInput.value, 10);
  if (Number.isFinite(n) && n > 0) body.count = n;   // else → project default (500)
  if (usesCopyPaste) {
    body.paste_count = pasteSel.value === "1-2" ? [1, 2] : 1;
    if (placeSel) body.placement = placeSel.value;   // original / random
  }
  try {
    const { job } = await sendJSON(`/api/p/${project}/run/scaffolds`, "POST", body);
    flash(msg, "generating scaffolds…", true);
    watchJob(job.id, null, () => { flash(msg, "scaffolds done", true); load(); });
  } catch (e) { flash(msg, e.message, false); }
};

load();
