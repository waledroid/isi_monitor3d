// Hidden MP4 dev tab (S12.2).
//
// Double-click the Isitec logo → password prompt → POST /api/unlock. On success
// the "MP4" big-panel tab is revealed and its picker is populated from
// /api/media/mp4. Unlock state is kept in sessionStorage so it survives reloads
// within the same browser session (re-locks in a new session).
//
// This is security-by-obscurity for a localhost dev tool, not real auth.

const UNLOCK_KEY = "mp4_unlocked";

function t(key, fallback) {
  const strings = (window.__monitor_web && window.__monitor_web.strings) || {};
  return strings[key] || fallback;
}

function bigPanelStore() {
  return (window.Alpine && Alpine.store("bigPanel")) || null;
}

async function populatePicker() {
  const pick = document.getElementById("mp4-pick");
  if (!pick) return;
  try {
    const res = await fetch("/api/media/mp4");
    if (!res.ok) return;
    const { files } = await res.json();
    // Keep the first ("Choose…") option, drop the rest, re-add.
    pick.length = 1;
    for (const f of files || []) {
      const o = document.createElement("option");
      o.value = f;
      o.textContent = f;
      pick.appendChild(o);
    }
    const want = bigPanelStore()?.mp4Selected;   // restore selection
    if (want) pick.value = want;
  } catch {
    /* ignore — fail quiet */
  }
}

function revealTab() {
  // The MP4 tab is shown reactively via x-show="$store.bigPanel.mp4Unlocked".
  const store = bigPanelStore();
  if (store) store.mp4Unlocked = true;
  populatePicker();
}

async function tryUnlock() {
  const pwd = window.prompt(t("unlock_prompt", "Enter password to unlock the MP4 view:"));
  if (pwd == null) return;   // cancelled
  try {
    const res = await fetch("/api/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pwd }),
    });
    const { ok } = await res.json();
    if (ok) {
      sessionStorage.setItem(UNLOCK_KEY, "1");   // store seeds mp4Unlocked from this on init
      revealTab();
    } else {
      window.alert(t("unlock_failed", "Incorrect password."));
    }
  } catch {
    window.alert(t("unlock_failed", "Incorrect password."));
  }
}

function wire() {
  const logo = document.querySelector(".header-logo");
  if (logo) logo.addEventListener("dblclick", tryUnlock);
  // Already unlocked this session? The store seeds mp4Unlocked from sessionStorage;
  // just populate the picker.
  if (sessionStorage.getItem(UNLOCK_KEY) === "1") populatePicker();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
