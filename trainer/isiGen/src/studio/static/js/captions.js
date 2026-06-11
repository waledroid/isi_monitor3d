import { flash, getJSON, sendJSON } from "/static/js/api.js";
import { watchJob } from "/static/js/jobs.js";

const root = document.getElementById("captions-root");
const project = root.dataset.project;
const list = document.getElementById("caption-list");

async function load() {
  const { records } = await getJSON(`/api/p/${project}/records`);
  const active = records.filter((r) => !r.excluded);
  list.innerHTML = active.length ? "" : "<p class='msg'>no images yet</p>";
  for (const r of active) {
    const row = document.createElement("div");
    row.className = "caption-row";
    row.innerHTML = `<img src="/media/${project}/thumb/${r.id}" loading="lazy">
      <textarea placeholder="(no caption yet — Generate missing)"></textarea>
      <div><button>Save</button><div class="edited">${r.caption_edited ? "edited ✔" : ""}</div></div>`;
    const ta = row.querySelector("textarea");
    getJSON(`/api/p/${project}/records/${r.id}/caption`)
      .then((d) => { ta.value = d.caption; })
      .catch(() => {});
    row.querySelector("button").onclick = async () => {
      await sendJSON(`/api/p/${project}/records/${r.id}/caption`, "PUT", { caption: ta.value });
      row.querySelector(".edited").textContent = "edited ✔";
    };
    list.appendChild(row);
  }
}

document.getElementById("run-captions").onclick = async () => {
  try {
    const { job } = await sendJSON(`/api/p/${project}/run/captions`, "POST", {});
    flash(document.getElementById("cap-msg"), "running…", true);
    watchJob(job.id, null, (j) => {
      flash(document.getElementById("cap-msg"),
            j.state === "done" ? "done" : j.error, j.state === "done");
      load();
    });
  } catch (e) { flash(document.getElementById("cap-msg"), e.message, false); }
};

load();
