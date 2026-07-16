// Zone-patch overlay on the live camera <img> elements: the operator's dashed
// ROI polygons + their cross-camera ghost outlines, mapped from source-frame
// pixels to displayed pixels (object-fit: contain formula). Detections, masks
// and floor-zone outlines are drawn SERVER-side into the frames themselves.

import { renderActiveCamPreview } from "/static/js/draw_mode.js";
import { getGhosts, getPatches } from "/static/js/zone_patch.js";

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
