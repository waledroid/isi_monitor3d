import { flash, getJSON, sendJSON } from "/static/js/api.js";

const PALETTE = [[220,40,40],[40,200,40],[40,90,230],[240,180,30],[160,60,220],[30,200,200]];

async function render() {
  const host = document.getElementById("project-list");
  try {
    const { projects } = await getJSON("/api/projects");
    host.innerHTML = projects.length ? "" : "<p class='msg'>no projects yet — create one below</p>";
    for (const p of projects) {
      const row = document.createElement("div");
      row.className = "project-row";
      const chips = p.classes.map((c) =>
        `<span class="chip" style="--c: rgb(${c.color.join(",")})">${c.name}</span>`).join("");
      row.innerHTML = `<a href="/p/${p.name}">${p.name}</a> ${chips}
                       <span class="msg">${p.records} image(s)</span>`;
      const del = document.createElement("button");
      del.className = "project-del"; del.textContent = "✕";
      del.title = `Delete ${p.name} and all its files`;
      del.onclick = () => deleteProject(p.name);
      row.appendChild(del);
      host.appendChild(row);
    }
  } catch (e) { host.textContent = `error: ${e.message}`; }
}

document.getElementById("create-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const classes = String(f.get("classes")).split(",").map((s) => s.trim()).filter(Boolean)
    .map((spec, i) => {
      const [name, trigger] = spec.split(":").map((x) => x.trim());
      return { name, trigger: trigger || `ISI_${name.toUpperCase()}`,
               color: PALETTE[i % PALETTE.length] };
    });
  try {
    await sendJSON("/api/projects", "POST", { name: f.get("name"), classes });
    flash(document.getElementById("create-msg"), "created", true);
    render();
  } catch (e) { flash(document.getElementById("create-msg"), e.message, false); }
});

async function deleteProject(name) {
  if (!confirm(`Delete project "${name}" and ALL its data + trained LoRA?\nThis cannot be undone.`)) return;
  try { await sendJSON(`/api/projects/${name}`, "DELETE"); render(); }
  catch (e) { alert(`delete failed: ${e.message}`); }
}

render();
