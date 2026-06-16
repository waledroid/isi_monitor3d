import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const gallery = document.getElementById("scaffold-gallery");
const toolbar = document.getElementById("paste-toolbar");
const help = document.getElementById("paste-help");
const pasteSel = document.getElementById("paste-count");
const msg = document.getElementById("scaffold-msg");

async function load() {
  let resp = { scaffolds: [], sources: [], paste_count: 1 };
  try { resp = await getJSON(`/api/p/${project}/scaffolds`); }
  catch (e) { gallery.textContent = `error: ${e.message}`; return; }
  const items = resp.scaffolds;
  // The objects-per-background toggle only applies to the copy_paste source.
  const usesCopyPaste = (resp.sources || []).includes("copy_paste");
  toolbar.hidden = help.hidden = !usesCopyPaste;
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
  const paste_count = pasteSel.value === "1-2" ? [1, 2] : 1;
  try {
    const { job } = await sendJSON(`/api/p/${project}/run/scaffolds`, "POST", { paste_count });
    flash(msg, "generating scaffolds…", true);
    watchJob(job.id, null, () => { flash(msg, "scaffolds done", true); load(); });
  } catch (e) { flash(msg, e.message, false); }
};

load();
