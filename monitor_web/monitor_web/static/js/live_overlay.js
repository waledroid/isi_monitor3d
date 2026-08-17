// Zone-patch overlay on the live camera <img> elements: the operator's dashed
// ROI polygons + their cross-camera ghost outlines, mapped from source-frame
// pixels to displayed pixels (object-fit: contain formula). Detections, masks
// and floor-zone outlines are drawn SERVER-side into the frames themselves.

import { renderActiveCamPreview } from "/static/js/draw_mode.js";
import { getGhosts, getPatches } from "/static/js/zone_patch.js";
import { getStates, getZones, isEditing } from "/static/js/etagere.js";

const OVERLAYS = [
  { camId: "cam_a", canvasId: "cam_a-overlay", imgId: "cam_a-img" },
  { camId: "cam_b", canvasId: "cam_b-overlay", imgId: "cam_b-img" },
];

// Map source-frame pixel (u, v) → displayed pixel for an <img> rendered with
// object-fit: contain. Image is scaled by min(displayW/natW, displayH/natH) and
// centred (letterboxed). The cam wrapper's CSS overscan transform then scales this
// canvas with the image, so overlays ride the overscan automatically — these are
// pre-transform (clientWidth) coords by design.
// Source-frame dimensions for a cam view. Normally the <img>'s natural size;
// when the compressed-video passthrough is active the <img> has no src
// (naturalWidth 0) — fall back to the decoded frame size it reports.
function naturalSize(img, camId) {
  if (img.naturalWidth && img.naturalHeight) return [img.naturalWidth, img.naturalHeight];
  const pt = window.__passthrough;
  return (pt && pt.frameSize && pt.frameSize(camId)) || [0, 0];
}
// Exposed for modules that must not import this one (e.g. etagere.js, whose
// browser-only absolute-path imports make it un-importable from live_overlay)
// but still need the SAME "naturalWidth, or the passthrough player's decoded
// frame size" fallback — see __displayToSource just below for the sibling
// export convention.
window.__naturalSize = naturalSize;

// `box` is the overlay canvas's OWN layout box {w, h} — never the <img>'s.
// The img's layout transiently collapses during view switches / expand
// animations / src swaps; sizing or mapping against it shrank the canvas
// backing store while its CSS box stayed panel-sized, and the browser
// upscaled the raster into the giant blurry "Zone N" label artifact.
function sourceToDisplay(box, su, sv, natW, natH) {
  const dw = box.w, dh = box.h;
  const scale = Math.min(dw / natW, dh / natH);
  const renderedW = natW * scale, renderedH = natH * scale;
  const offsetX = (dw - renderedW) / 2;
  const offsetY = (dh - renderedH) / 2;
  return [su * scale + offsetX, sv * scale + offsetY];
}

// Inverse of sourceToDisplay, PLUS the natural→frame_wh rescale drawEtagere
// applies going the other way (see its `sx`/`sy`) — so callers get back
// coordinates in the SAME frame_wh pixel space a zone's `cells[].rect` is
// stored in, ready to feed into etagere.js's `hitTest`/`applyDrag`.
// `dx, dy` are canvas-local pixels (e.g. a mouse event's offsetX/offsetY).
function displayToSourceForCam(canvas, camId, dx, dy, frameWh) {
  const img = document.getElementById(`${camId}-img`);
  const [natW, natH] = img ? naturalSize(img, camId) : [0, 0];
  if (!natW || !natH) return [0, 0];
  const box = { w: canvas.clientWidth, h: canvas.clientHeight };
  const scale = Math.min(box.w / natW, box.h / natH);
  const offsetX = (box.w - natW * scale) / 2;
  const offsetY = (box.h - natH * scale) / 2;
  const nx = (dx - offsetX) / scale;
  const ny = (dy - offsetY) / scale;
  const fw = (frameWh && frameWh[0]) || natW;
  const fh = (frameWh && frameWh[1]) || natH;
  const sx = natW / fw, sy = natH / fh;
  return [nx / sx, ny / sy];
}
window.__displayToSource = displayToSourceForCam;

// Pixel-space zone-patch ROIs — bold red POLYGONS (no calibration). The polygon is
// the drawn shape (its bounding rect is what gets cropped for detection); legacy
// rect-only patches fall back to their box. Coords are source px at the size drawn
// (frame_wh); rescale to the current natural size, then map to display.
function drawZonePatches(ctx, box, img, camId) {
  const ps = getPatches(camId);
  if (!ps.length) return;
  const [natW, natH] = naturalSize(img, camId);
  if (!natW || !natH) return;
  ctx.lineWidth = 3;
  ctx.font = "bold 12px monospace";
  for (const p of ps) {
    const fw = (p.frame_wh && p.frame_wh[0]) || natW;
    const fh = (p.frame_wh && p.frame_wh[1]) || natH;
    const sx = natW / fw, sy = natH / fh;
    // Polygon when present (>=3 pts), else the rectangle as a 4-point polygon.
    const poly = (Array.isArray(p.polygon) && p.polygon.length >= 3)
      ? p.polygon
      : [[p.rect[0], p.rect[1]], [p.rect[2], p.rect[1]],
         [p.rect[2], p.rect[3]], [p.rect[0], p.rect[3]]];
    ctx.strokeStyle = p.color || "#ff3b3b";   // per-zone outline colour (Settings)
    ctx.setLineDash([9, 6]);          // dashed outline, no fill
    ctx.beginPath();
    // Label anchor: the polygon VERTEX nearest its top-right (max x−y score) —
    // ON the zone outline itself, not the floating bounding-box corner.
    let vx = 0, vy = 0, best = -Infinity;
    for (let i = 0; i < poly.length; i++) {
      const [dx, dy] = sourceToDisplay(box, poly[i][0] * sx, poly[i][1] * sy, natW, natH);
      if (i === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy);
      if (dx - dy > best) { best = dx - dy; vx = dx; vy = dy; }
    }
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);              // reset so other overlays stay solid
    const label = p.name || "";
    const tw = ctx.measureText(label).width;
    // Tag hugs that corner: box right-aligned at the vertex, just above it;
    // clamped so it never slips off the canvas at the image's top/right edge.
    const bx = Math.max(0, Math.min(vx - tw - 8, ctx.canvas.width - tw - 8));
    const by = Math.max(0, vy - 17);
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(bx, by, tw + 8, 15);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, bx + 4, by + 11);
  }
}

// Cross-camera GHOSTS: patches drawn on the OTHER camera, projected through the
// floor into this one (server-computed, Mode-2 calibration). Finely dashed +
// translucent so they read as "defined elsewhere"; labelled "(from cam_x)".
// Ghost polygon coords are in the ghost camera's CALIBRATION frame (image_wh).
function drawPatchGhosts(ctx, box, img, camId) {
  const ghosts = getGhosts(camId);
  if (!ghosts.length) return;
  const [natW, natH] = naturalSize(img, camId);
  if (!natW || !natH) return;
  ctx.lineWidth = 2;
  ctx.font = "11px monospace";
  for (const g of ghosts) {
    const gw = (g.image_wh && g.image_wh[0]) || natW;
    const gh = (g.image_wh && g.image_wh[1]) || natH;
    const sx = natW / gw, sy = natH / gh;
    ctx.strokeStyle = g.color || "#ff3b3b";
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([3, 7]);
    ctx.beginPath();
    let lx = 0, ly = 0;
    for (let i = 0; i < g.polygon.length; i++) {
      const [dx, dy] = sourceToDisplay(box, g.polygon[i][0] * sx, g.polygon[i][1] * sy, natW, natH);
      if (i === 0) { ctx.moveTo(dx, dy); lx = dx; ly = dy; } else ctx.lineTo(dx, dy);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
    const label = `${g.name} (from ${g.from})`;
    const tw = ctx.measureText(label).width;
    const bx = Math.max(0, Math.min(lx, ctx.canvas.width - tw - 8));
    const by = Math.max(0, ly - 16);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(bx, by, tw + 8, 14);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText(label, bx + 4, by + 11);
    ctx.globalAlpha = 1;
  }
}

// Étagère (bin-rack) cell outlines + live fill state — analogous to
// drawZonePatches but rectangular cells, colour-by-state, and (while the
// Settings "Adjust cells" drag session for this zone is open) corner
// handles. Coords are source px at frame_wh; rescale to natural, then map
// to display — same convention as drawZonePatches/drawPatchGhosts above.
function drawEtagere(ctx, box, img, camId) {
  const zones = getZones(camId);
  if (!zones.length) return;
  const [natW, natH] = naturalSize(img, camId);
  if (!natW || !natH) return;
  const allStates = getStates();
  ctx.lineWidth = 2;
  ctx.font = "bold 10px monospace";
  for (const z of zones) {
    const fw = (z.frame_wh && z.frame_wh[0]) || natW;
    const fh = (z.frame_wh && z.frame_wh[1]) || natH;
    const sx = natW / fw, sy = natH / fh;
    const st = allStates[z.id];
    const matrix = st && st.matrix;
    const editingThis = isEditing(z.id);
    for (const cell of z.cells || []) {
      const [x0, y0, x1, y1] = cell.rect;
      const [dx0, dy0] = sourceToDisplay(box, x0 * sx, y0 * sy, natW, natH);
      const [dx1, dy1] = sourceToDisplay(box, x1 * sx, y1 * sy, natW, natH);
      const rx0 = Math.min(dx0, dx1), ry0 = Math.min(dy0, dy1);
      const rw = Math.abs(dx1 - dx0), rh = Math.abs(dy1 - dy0);

      let state = "unknown";
      if (matrix && matrix[cell.r - 1] && matrix[cell.r - 1][cell.c - 1] != null) {
        state = matrix[cell.r - 1][cell.c - 1];
      }
      let stroke = "#9aa0a6", fill = "rgba(154,160,166,0.25)", dash = [];
      if (state === "filled") { stroke = "#2ea043"; fill = "rgba(46,160,67,0.25)"; }
      else if (state === "empty") { stroke = "#9aa0a6"; fill = "rgba(154,160,166,0.25)"; }
      else { stroke = "#f0a028"; fill = "rgba(240,160,40,0.2)"; dash = [4, 3]; }

      ctx.setLineDash(dash);
      ctx.fillStyle = fill;
      ctx.fillRect(rx0, ry0, rw, rh);
      ctx.strokeStyle = stroke;
      ctx.strokeRect(rx0, ry0, rw, rh);
      ctx.setLineDash([]);

      const label = `r${cell.r}c${cell.c}`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(rx0 + 2, ry0 + 2, tw + 6, 13);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rx0 + 5, ry0 + 12);

      if (editingThis) {
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 1;
        for (const [hx, hy] of [[rx0, ry0], [rx0 + rw, ry0], [rx0 + rw, ry0 + rh], [rx0, ry0 + rh]]) {
          ctx.fillRect(hx - 3, hy - 3, 6, 6);
          ctx.strokeRect(hx - 3, hy - 3, 6, 6);
        }
        ctx.lineWidth = 2;
      }
    }
  }
}

function drawForCam(canvas, img, camId) {
  if (canvas.classList.contains("hidden")) return;
  // Size the backing store from the CANVAS's own CSS box (pinned to 100% of
  // the panel), never the <img>'s layout — see the sourceToDisplay note. A
  // degenerate box (mid-transition, hidden ancestor) draws nothing rather
  // than rendering labels tiny for the browser to blow up.
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w < 40 || h < 40) return;
  const ctx = canvas.getContext("2d");
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  ctx.clearRect(0, 0, w, h);
  const box = { w, h };

  // Server-side floor-zone outlines are the Settings 'Show floor zones'
  // toggle (routes_video); this canvas draws only the operator's patch layer.
  drawZonePatches(ctx, box, img, camId);         // pixel-space red ROI watch boxes
  drawPatchGhosts(ctx, box, img, camId);         // cross-camera ghost outlines (Mode 2)
  drawEtagere(ctx, box, img, camId);             // bin-rack cells, colour-by-state
  renderActiveCamPreview(camId);                 // keep calibration dots over the live frame
}

function tick() {
  for (const o of OVERLAYS) {
    const canvas = document.getElementById(o.canvasId);
    const img = document.getElementById(o.imgId);
    if (canvas && img) drawForCam(canvas, img, o.camId);
  }
  requestAnimationFrame(tick);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", tick);
} else {
  tick();
}
