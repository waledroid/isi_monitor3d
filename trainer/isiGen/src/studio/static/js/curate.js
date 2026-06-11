import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const gallery = document.getElementById("gallery");
const project = gallery.dataset.project;
const classes = JSON.parse(gallery.dataset.classes);
let records = [];
let clsFilter = "";

async function load() {
  try {
    records = (await getJSON(`/api/p/${project}/records`)).records;
    render();
  } catch (e) { gallery.textContent = `error: ${e.message}`; }
}

function render() {
  const showEx = document.getElementById("show-excluded").checked;
  const subset = records.filter((r) =>
    (showEx || !r.excluded) && (!clsFilter || r.class_name === clsFilter));
  gallery.innerHTML = subset.length ? "" : "<p class='msg'>no images — ingest a folder above</p>";
  for (const r of subset) {
    const tile = document.createElement("div");
    tile.className = "tile" + (r.excluded ? " excluded" : "");
    const opts = classes.map((c) =>
      `<option value="${c.name}" ${c.name === r.class_name ? "selected" : ""}>${c.name}</option>`).join("");
    tile.innerHTML = `<img src="/media/${project}/thumb/${r.id}" loading="lazy" alt="${r.id}">
      <div class="meta"><select>${opts}</select>
        <label class="ex"><input type="checkbox" ${r.excluded ? "checked" : ""}> excl</label></div>`;
    tile.querySelector("select").addEventListener("change", async (e) => {
      await sendJSON(`/api/p/${project}/records/${r.id}`, "PATCH", { class_name: e.target.value });
      r.class_name = e.target.value;
    });
    tile.querySelector("input").addEventListener("change", async (e) => {
      await sendJSON(`/api/p/${project}/records/${r.id}`, "PATCH", { excluded: e.target.checked });
      r.excluded = e.target.checked;
      render();
    });
    gallery.appendChild(tile);
  }
}

document.getElementById("class-filters").addEventListener("click", (e) => {
  const b = e.target.closest("button.chip");
  if (!b) return;
  clsFilter = b.dataset.cls;
  document.querySelectorAll("#class-filters button.chip").forEach((x) =>
    x.classList.toggle("active", x === b));
  render();
});
document.getElementById("show-excluded").addEventListener("change", render);

document.getElementById("ingest-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const cls = String(f.get("class_name") || "");
  try {
    const { job } = await sendJSON(`/api/p/${project}/ingest`, "POST", {
      source: f.get("source"),
      class_name: cls || null,
      auto_class: !cls,
    });
    flash(document.getElementById("ingest-msg"), "ingest started…", true);
    watchJob(job.id, null, (j) => {
      flash(document.getElementById("ingest-msg"),
            j.state === "done" ? `done: ${JSON.stringify(j.result)}` : j.error, j.state === "done");
      load();
    });
  } catch (e) { flash(document.getElementById("ingest-msg"), e.message, false); }
});

load();
