// GB/FR language toggle. Swaps the text content of every [data-i18n] element
// from the JSON bundle at /static/i18n/{lang}.json. Persists choice in
// localStorage so reloads remember it.

const STORAGE_KEY = "monitor_web.lang";
const STATE = window.__monitor_web ?? { lang: "fr", availableLangs: ["en", "fr"], strings: {} };

// Custom, self-contained inline SVG definitions for the flags
const FR_FLAG = `
<svg class="flag-icon" viewBox="0 0 3 2" width="24" height="16" style="display: block; border-radius: 2px;">
  <rect width="1" height="2" fill="#00209F"/>
  <rect x="1" width="1" height="2" fill="#F4F9FF"/>
  <rect x="2" width="1" height="2" fill="#C8102E"/>
</svg>
`;

const GB_FLAG = `
<svg class="flag-icon" viewBox="0 0 60 30" width="24" height="12" style="display: block; border-radius: 2px;">
  <clipPath id="flag-gb-clip">
    <path d="M0,0 L30,15 L30,0 Z M0,30 L30,15 L0,15 Z M60,30 L30,15 L30,30 Z M60,0 L30,15 L60,15 Z"/>
  </clipPath>
  <rect width="60" height="30" fill="#012169"/>
  <path d="M0,0 L60,30 M60,0 L0,30" stroke="#fff" stroke-width="6"/>
  <path d="M0,0 L60,30 M60,0 L0,30" stroke="#C8102E" stroke-width="4" clip-path="url(#flag-gb-clip)"/>
  <path d="M30,0 V30 M0,15 H60" stroke="#fff" stroke-width="10"/>
  <path d="M30,0 V30 M0,15 H60" stroke="#C8102E" stroke-width="6"/>
</svg>
`;

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

function updateLangButton(currentLang) {
  const btn = document.querySelector(".lang-btn");
  if (!btn) return;

  if (currentLang === "en") {
    // When the page is in English, show French flag and target switching to French
    btn.dataset.lang = "fr";
    btn.innerHTML = FR_FLAG;
    btn.setAttribute("title", "Passer en français");
    btn.setAttribute("aria-label", "Switch to French");
  } else {
    // When the page is in French, show UK flag and target switching to English
    btn.dataset.lang = "en";
    btn.innerHTML = GB_FLAG;
    btn.setAttribute("title", "Switch to English");
    btn.setAttribute("aria-label", "Passer en anglais");
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
  updateLangButton(lang);
  document.documentElement.setAttribute("lang", lang);
  localStorage.setItem(STORAGE_KEY, lang);
  STATE.lang = lang;
  STATE.strings = strings;
}

function init() {
  const btn = document.querySelector(".lang-btn");
  if (btn) {
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
