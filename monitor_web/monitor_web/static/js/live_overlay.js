// Bbox overlay on the live camera <img> elements.
//
// The MJPEG <img> renders the raw frame; this canvas, layered on top, draws
// (a) a small per-camera Track2D legend, and (b) the world-zone polygons
// projected onto the camera's pixel space (S17). The projection uses the
// camera's calibration via /api/project/floor-to-pixel; pixel coords come
// back in source-frame coordinates, which we then map to displayed pixels
// using the `object-fit: cover` formula.

import * as cameraZones from "/static/js/camera_zones.js";
import { renderActiveCamPreview } from "/static/js/draw_mode.js";
import { getGhosts, getPatches } from "/static/js/zone_patch.js";

const OVERLAYS = [
  { camId: "cam_a", canvasId: "cam_a-overlay", imgId: "cam_a-img" },
  { camId: "cam_b", canvasId: "cam_b-overlay", imgId: "cam_b-img" },
];

function colorForClass(cls) {
  switch (cls) {
    case "person":   return "#ffd54f";
    case "forklift": return "#ff7043";
    case "pallet":   return "#4fc3f7";
    default:         return "#ffffff";
  }
}

function colorForZone(zone) {
  const k = zone.kind || zone.type || "palette";
  if (k === "danger")  return "#fca5a5";   // light red
  if (k === "etagere") return "#86efac";   // light green
  return "#9aa5b1";                         // palette — neutral
}

// Map source-frame pixel (u, v) → displayed pixel for an <img> rendered with
// object-fit: contain. Image is scaled by min(displayW/natW, displayH/natH) and
// centred (letterboxed). The cam wrapper's CSS overscan transform then scales this
// canvas with the image, so overlays ride the overscan automatically — these are
// pre-transform (clientWidth) coords by design.
function sourceToDisplay(img, su, sv, natW, natH) {
  const dw = img.clientWidth, dh = img.clientHeight;
  const scale = Math.min(dw / natW, dh / natH);
  const renderedW = natW * scale, renderedH = natH * scale;
  const offsetX = (dw - renderedW) / 2;
  const offsetY = (dh - renderedH) / 2;
  return [su * scale + offsetX, sv * scale + offsetY];
}

function drawZones(ctx, img, camZones) {
  if (!camZones || !camZones.imageSize || !Array.isArray(camZones.zones)) return;
  const [natW, natH] = camZones.imageSize;
  ctx.lineWidth = 2;
  ctx.font = "11px monospace";
  for (const z of camZones.zones) {
    if (!Array.isArray(z.polygon) || z.polygon.length < 3) continue;
    const color = colorForZone(z);
    ctx.strokeStyle = color;
    ctx.fillStyle = color + "33";   // ~20% alpha
    ctx.beginPath();
    for (let i = 0; i < z.polygon.length; i++) {
      const [su, sv] = z.polygon[i];
      const [dx, dy] = sourceToDisplay(img, su, sv, natW, natH);
      if (i === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    // Label backdrop + text at the first vertex.
    const [su0, sv0] = z.polygon[0];
    const [lx, ly] = sourceToDisplay(img, su0, sv0, natW, natH);
    const label = z.name || "";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(lx + 4, ly - 16, tw + 8, 14);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, lx + 8, ly - 5);
  }
}

function drawTrackLegend(ctx, camId) {
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = "12px monospace";
  let row = 16;
  for (const tr of window.__tracks.byId2D.values()) {
    if (!tr.cameras_seeing || !tr.cameras_seeing.includes(camId)) continue;
    ctx.fillStyle = colorForClass(tr.cls);
    ctx.fillRect(8, row - 12, 12, 12);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText(
      `#${tr.track_id} ${tr.cls} @ (${tr.xy_m[0].toFixed(2)}, ${tr.xy_m[1].toFixed(2)}) m`,
      26, row,
    );
    row += 16;
  }
}

// Pixel-space zone-patch ROIs — bold red POLYGONS (no calibration). The polygon is
// the drawn shape (its bounding rect is what gets cropped for detection); legacy
// rect-only patches fall back to their box. Coords are source px at the size drawn
// (frame_wh); rescale to the current natural size, then map to display.
function drawZonePatches(ctx, img, camId) {
  const ps = getPatches(camId);
  if (!ps.length) return;
  const natW = img.naturalWidth, natH = img.naturalHeight;
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
      const [dx, dy] = sourceToDisplay(img, poly[i][0] * sx, poly[i][1] * sy, natW, natH);
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
function drawPatchGhosts(ctx, img, camId) {
  const ghosts = getGhosts(camId);
  if (!ghosts.length) return;
  const natW = img.naturalWidth, natH = img.naturalHeight;
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
      const [dx, dy] = sourceToDisplay(img, g.polygon[i][0] * sx, g.polygon[i][1] * sy, natW, natH);
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
  const ctx = canvas.getContext("2d");
  const w = img.clientWidth, h = img.clientHeight;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  // Projected FLOOR zones are deliberately NOT drawn on the cam views: floor
  // zones are auto-derived from the zone patches (floor_zone_sync), so the
  // projection duplicated every outline + name label right next to the patch
  // the operator drew. The patch polygons below are the single zone layer;
  // the floor geometry stays an engine-side concept (map + Backbone only).
  // drawZones(ctx, img, cameraZones.get(camId));
  drawZonePatches(ctx, img, camId);              // pixel-space red ROI watch boxes
  drawPatchGhosts(ctx, img, camId);              // cross-camera ghost outlines (Mode 2)
  // Top-left per-track UDP readout disabled for now (to be surfaced elsewhere
  // later). Re-enable by uncommenting; drawTrackLegend() is kept intact below.
  // drawTrackLegend(ctx, camId);
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
