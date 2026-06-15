import { getJSON } from "/static/js/api.js";

const root = document.getElementById("gal-root");
const project = root.dataset.project;
const gallery = document.getElementById("mint-gallery");
const showMask = document.getElementById("show-mask");

let synth = [];

async function load() {
  let records = [];
  try { records = (await getJSON(`/api/p/${project}/records`)).records; }
  catch (e) { gallery.textContent = `error: ${e.message}`; return; }
  synth = records.filter((r) => r.synthetic && !r.excluded);
  render();
}

function render() {
  const masks = showMask.checked;
  document.getElementById("mint-count").textContent = `${synth.length} minted`;
  gallery.innerHTML = synth.length ? "" : "<p class='msg'>nothing minted yet — run phase 7</p>";
  for (const r of synth) {
    const kind = masks && r.mask ? "mask" : "image";
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML =
      `<img src="/media/${project}/${kind}/${r.id}" loading="lazy" alt="${r.id}">
       <div class="meta">${r.id}</div>`;
    gallery.appendChild(tile);
  }
}

showMask.addEventListener("change", render);
load();
