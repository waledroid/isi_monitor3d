import { flash, getJSON, sendJSON } from "/static/js/api.js";

async function render() {
  const host = document.getElementById("project-list");
  try {
    const { projects } = await getJSON("/api/projects");
    if (!projects.length) { host.innerHTML = "<p class='msg'>no calibrations yet — create one below.</p>"; return; }
    host.innerHTML = "";
    for (const p of projects) {
      const row = document.createElement("div");
      row.className = "project-row";
      const mode = p.mode2 ? "Mode 2 (2 cams)" : "Mode 1 (1 cam)";
      row.innerHTML = `<a href="/p/${p.name}"><b>${p.name}</b></a>
        <span class="msg">${(p.cameras || []).join(", ") || "no cameras"} · ${mode}</span>
        <button class="reset-btn" data-name="${p.name}">✕ delete</button>`;
      row.querySelector("button").onclick = async () => {
        if (!confirm(`Delete calibration "${p.name}" and all its captures?`)) return;
        await sendJSON(`/api/projects/${p.name}`, "DELETE", {});
        render();
      };
      host.appendChild(row);
    }
  } catch (e) { host.textContent = `error: ${e.message}`; }
}

document.getElementById("create-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const cam = (type, src) => {
    const t = String(type || "rtsp");
    const s = String(src || "").trim();
    return t === "usb" ? { type: "usb", device: s } : { type: "rtsp", url: s };
  };
  const body = { name: f.get("name"), cam_a: cam(f.get("a_type"), f.get("a_src")) };
  const bsrc = String(f.get("b_src") || "").trim();
  if (bsrc) body.cam_b = cam(f.get("b_type"), bsrc);
  try {
    await sendJSON("/api/projects", "POST", body);
    flash(document.getElementById("create-msg"), "created", true);
    ev.target.reset();
    render();
  } catch (e) { flash(document.getElementById("create-msg"), e.message, false); }
});

render();
