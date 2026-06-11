// GB/FR language toggle. Swaps the text content of every [data-i18n] element
// from the JSON bundle at /static/i18n/{lang}.json. Persists choice in
// localStorage so reloads remember it.

const STORAGE_KEY = "monitor_web.lang";
const STATE = window.__monitor_web ?? { lang: "fr", availableLangs: ["en", "fr"], strings: {} };

async function fetchBundle(lang) {
  const res = await fetch(`/static/i18n/${lang}.json`);
  if (!res.ok) throw new Error(`failed to load ${lang}`);
  return res.json();
}

function applyBundle(strings) {
  for (const el of document.querySelectorAll("[data-i18n]")) {
    const key = el.getAttribute("data-i18n");
    if (key in strings) {
      el.textContent = strings[key];
    }
  }
}

function setActiveButton(lang) {
  for (const btn of document.querySelectorAll(".lang-btn")) {
    btn.setAttribute("aria-pressed", btn.dataset.lang === lang ? "true" : "false");
  }
}

async function setLang(lang) {
  let strings;
  try {
    strings = await fetchBundle(lang);
  } catch (err) {
    console.warn(err);
    return;
  }
  applyBundle(strings);
  setActiveButton(lang);
  document.documentElement.setAttribute("lang", lang);
  localStorage.setItem(STORAGE_KEY, lang);
  STATE.lang = lang;
  STATE.strings = strings;
}

function init() {
  for (const btn of document.querySelectorAll(".lang-btn")) {
    btn.addEventListener("click", () => setLang(btn.dataset.lang));
  }
  const stored = localStorage.getItem(STORAGE_KEY);
  const lang = stored && STATE.availableLangs.includes(stored) ? stored : STATE.lang;
  setLang(lang);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
