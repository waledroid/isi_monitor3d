// Camera-zone projection cache (S17).
//
// World zones live in zones.yaml as floor metres. For each tracked camera we
// POST every zone's polygon to /api/project/floor-to-pixel, which returns the
// equivalent pixel polygon in the camera's *source* frame size (e.g. 1920×1080).
// live_overlay.js then maps source-px → display-px (object-fit: cover) and
// strokes the polygons on top of the MJPEG <img>.
//
// State is invalidated by listening for `config:saved` so the operator sees
// edits land instantly without a page refresh. Calibration absence is a
// silent no-op — the CAM tab simply shows no zone overlay.

let _cache = {};   // {camera_id: {zones: [{name, kind, severity, polygon: [[u,v],...]}, ...], imageSize: [w,h]}}
let _calibratedCameras = null;   // [...] or null when unknown / not yet probed
let _loaded = false;

export function isLoaded() { return _loaded; }
export function get(camId) { return _cache[camId] || null; }
export function calibratedCameras() { return _calibratedCameras; }

async function fetchCalibratedCameras() {
  try {
    const res = await fetch("/api/project/cameras");
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.cameras) ? data.cameras : [];
  } catch {
    return [];
  }
}

async function fetchZones() {
  try {
    const res = await fetch("/api/zones");
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.zones) ? data.zones : [];
  } catch {
    return [];
  }
}

async function projectZonesFor(camId, zones) {
  // One round-trip per zone — small set (≤6) so an extra request is fine.
  // A future tweak could batch into a single endpoint call.
  const out = { zones: [], imageSize: null };
  for (const z of zones) {
    if (!Array.isArray(z.polygon) || z.polygon.length < 3) continue;
    try {
      const res = await fetch("/api/project/floor-to-pixel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: camId, polygon: z.polygon }),
      });
      if (!res.ok) continue;        // 503 (no calibration) / 404 → skip silently
      const data = await res.json();
      out.zones.push({
        name: z.name,
        kind: z.kind,
        severity: z.severity,
        polygon: data.points,        // pixels in SOURCE frame coords
      });
      out.imageSize = data.image_size;
    } catch {
      /* network blip — skip this zone for now */
    }
  }
  return out;
}

export async function refresh() {
  _calibratedCameras = await fetchCalibratedCameras();
  if (_calibratedCameras.length === 0) {
    _cache = {};
    _loaded = true;
    return;
  }
  const zones = await fetchZones();
  const next = {};
  for (const camId of _calibratedCameras) {
    next[camId] = await projectZonesFor(camId, zones);
  }
  _cache = next;
  _loaded = true;
}

// Initial fetch + react to saves from the Settings modal.
refresh();
document.addEventListener("config:saved", () => { refresh(); });
