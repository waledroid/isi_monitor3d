import { getJSON } from "/static/js/api.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const gallery = document.getElementById("scaffold-gallery");

async function load() {
  let items = [];
  try { items = (await getJSON(`/api/p/${project}/scaffolds`)).scaffolds; }
  catch (e) { gallery.textContent = `error: ${e.message}`; return; }
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

load();
