// Client-side cache + lookup for /api/link-lines (S16).
//
// The floor map calls shouldLink(a_cls, b_cls) on every pair each frame, so we
// keep the rule list in-memory and only re-fetch on explicit reload(). The
// zone-manager save flow calls reload() after a successful POST /api/config.

let _rules = [];          // [{from, to: [...], max_distance_m, color}]
let _loaded = false;

function _normaliseRule(raw) {
  if (!raw || typeof raw !== "object") return null;
  const from = String(raw.from || "").trim();
  const to = Array.isArray(raw.to)
    ? raw.to.map((s) => String(s || "").trim()).filter(Boolean)
    : [];
  if (!from || to.length === 0) return null;
  const max = raw.max_distance_m;
  return {
    from,
    to,
    max_distance_m: (typeof max === "number" && Number.isFinite(max) && max > 0) ? max : null,
    color: typeof raw.color === "string" && raw.color ? raw.color : null,
  };
}

export async function reload() {
  try {
    const res = await fetch("/api/link-lines");
    if (!res.ok) {
      _rules = [];
    } else {
      const data = await res.json();
      _rules = (Array.isArray(data.rules) ? data.rules : [])
        .map(_normaliseRule)
        .filter(Boolean);
    }
  } catch {
    _rules = [];
  }
  _loaded = true;
  return _rules;
}

export function isLoaded() { return _loaded; }
export function rules() { return _rules; }

/** Does any rule link (a_cls, b_cls) — undirected? */
function _matches(rule, fromCls, toCls) {
  if (fromCls === toCls) return false;
  if (rule.from !== fromCls) return false;
  if (rule.to.indexOf("*") !== -1) return true;
  return rule.to.indexOf(toCls) !== -1;
}

export function shouldLink(aCls, bCls) {
  for (const r of _rules) {
    if (_matches(r, aCls, bCls) || _matches(r, bCls, aCls)) return true;
  }
  return false;
}

/**
 * Return the FIRST rule that matches the pair (a_cls, b_cls) — used to pick
 * an effective max_distance_m + color override. Returns null if no match.
 */
export function ruleFor(aCls, bCls) {
  for (const r of _rules) {
    if (_matches(r, aCls, bCls) || _matches(r, bCls, aCls)) return r;
  }
  return null;
}
