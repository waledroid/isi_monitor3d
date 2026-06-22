// Tiny fetch helpers shared by every Studio page.
export async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
export async function sendJSON(url, method, body) {
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" },
                               body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
export function flash(el, text, ok = true) {
  if (!el) return;
  el.textContent = text;
  el.className = "msg " + (ok ? "ok" : "bad");
  setTimeout(() => { el.textContent = ""; }, 6000);
}
