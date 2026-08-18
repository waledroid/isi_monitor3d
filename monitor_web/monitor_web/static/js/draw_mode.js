// Polygon-draw lifecycle on the floor map (MAP target) or on a live camera
// frame (CAM target — S17). S18 adds a `mode` flag for calibration use.
//
// startDraw({ target, mode, label, onDone, onCancel, minPoints, maxPoints })
//   target ∈ {'map', 'cam_a', 'cam_b'}. Default 'map' (back-compat).
//   mode   ∈ {'project', 'raw'}.       Default 'project' (back-compat).
//   minPoints — minimum to allow Done. Default 3.
//   maxPoints — auto-finish on this click count. Default Infinity (no auto).
//
// Modes:
//   'project' (default) — each click is converted to world (X, Y) metres.
//      • MAP target: raycast the click onto the 3D floor via the Three.js map's
//        `transform.screenToWorld(sx, sy)`; the rubber-band preview is drawn in 3D
//        (`fm.previewPolygon` / `fm.clearPreview`).
//      • CAM target: POST display→source-px to `/api/project/pixel-to-floor`
//        and store the returned world coords.
//      Used by S17 zone authoring — onDone receives world (X, Y) m.
//
//   'raw' — CAM target only. Each click stores SOURCE-FRAME PIXEL coords
//      directly (no backend call). Used by S18 pallet calibration where
//      calibration doesn't exist yet, so pixel→world projection isn't
//      possible. onDone receives [[u, v], ...] in source pixel coords.

// Two opposite corners → axis-aligned rectangle as 4 world points (TL,TR,BR,BL).

// Blocking zone-name entry: loops until the operator gives a non-empty name
// unique among `taken` (case-insensitive), or cancels (returns null). The
// suggestion prefills the field so Enter-through stays fast. Shared by the
// camera zone patches (zone_patch.js) and the metric floor zones
// (floor_zones.js) — one naming contract everywhere.
export function promptZoneName(taken, suggestion, current) {
  const lowered = taken
    .filter((n) => n && n !== current)
    .map((n) => String(n).trim().toLowerCase());
  let val = suggestion || "";
  for (;;) {
    val = window.prompt("Zone name (required, must be unique):", val);
    if (val === null) return null;             // cancelled
    val = val.trim();
    if (!val) continue;                        // empty -> ask again
    if (lowered.includes(val.toLowerCase())) {
      window.alert(`"${val}" is already used by another zone — pick a unique name.`);
      continue;
    }
    return val;
  }
}

export function rectFromCorners([x1, y1], [x2, y2]) {
  const xa = Math.min(x1, x2), xb = Math.max(x1, x2);
  const ya = Math.min(y1, y2), yb = Math.max(y1, y2);
  return [[xa, ya], [xb, ya], [xb, yb], [xa, yb]];
}

// Snap a world point to a metric grid (e.g. 0.1 m). step<=0 → no snap.
export function snapToGrid([x, y], step) {
  if (!step || step <= 0) return [x, y];
  const r = (v) => Math.round(v / step) * step;
  // round to the grid then to 4 dp to kill FP dust (2.04→2.0, 1.07→1.1)
  return [Number(r(x).toFixed(4)), Number(r(y).toFixed(4))];
}

let active = null;

function ensureToolbar() {
  return {
    bar:    document.getElementById("draw-toolbar"),
    label:  document.getElementById("draw-zone-label"),
    count:  document.getElementById("draw-count"),
    undo:   document.getElementById("draw-undo"),
    done:   document.getElementById("draw-done"),
    cancel: document.getElementById("draw-cancel"),
  };
}

// ---- MAP target ----

function renderMapPreview() {
  if (!active || active.target !== "map") return;
  const { fm, points } = active;
  // rect-snap rubber-band: one fixed corner + the live cursor → preview rectangle.
  if (active.mode === "rectSnap" && points.length === 1 && active.ghost) {
    fm.previewPolygon(rectFromCorners(points[0], active.ghost), { close: true });
    return;
  }
  if (points.length === 0) { fm.clearPreview(); return; }
  fm.previewPolygon(points, { close: points.length >= 3 });
}

function onMapPointerDown(ev) {
  if (!active || active.target !== "map") return;
  const { fm, canvas, points } = active;
  const rect = canvas.getBoundingClientRect();
  // Raycast the click onto the 3D floor → world (X, Y) metres.
  let [x, y] = fm.transform.screenToWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  if (active.mode === "rectSnap") {
    [x, y] = snapToGrid([x, y], active.gridStep || 0);
    points.push([x, y]);
    if (points.length === 2) {                       // 2 corners → finish a rectangle
      const r = rectFromCorners(points[0], points[1]);
      const onDone = active.onDone;
      cleanup();
      onDone?.(r);
      return;
    }
  } else {
    points.push([x, y]);
  }
  active.toolbar.count.textContent = String(points.length);
  renderMapPreview();
}

// Live cursor tracking for the rect-snap rubber-band (after the first corner).
function onMapPointerMove(ev) {
  if (!active || active.target !== "map" || active.mode !== "rectSnap") return;
  if (active.points.length !== 1) return;
  const { fm, canvas } = active;
  const rect = canvas.getBoundingClientRect();
  const [x, y] = fm.transform.screenToWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  active.ghost = snapToGrid([x, y], active.gridStep || 0);
  renderMapPreview();
}

// ---- CAM target ----

// display-px on the <img> → source-frame px (object-fit: contain formula). Reads
// the post-transform rect; only used while drawing, when the cam wrapper is forced
// to scale 1 (body.cam-drawing), so there's no overscan term to undo.
function displayToSource(img, dx, dy) {
  const rect = img.getBoundingClientRect();
  const localX = dx - rect.left;
  const localY = dy - rect.top;
  const dw = rect.width, dh = rect.height;
  const nw = img.naturalWidth || dw, nh = img.naturalHeight || dh;
  const scale = Math.min(dw / nw, dh / nh);
  const offsetX = (dw - nw * scale) / 2;
  const offsetY = (dh - nh * scale) / 2;
  const srcX = (localX - offsetX) / scale;
  const srcY = (localY - offsetY) / scale;
  return [Math.max(0, Math.min(nw - 1, srcX)), Math.max(0, Math.min(nh - 1, srcY))];
}

function renderCamPreview() {
  if (!active || (active.target !== "cam_a" && active.target !== "cam_b")) return;
  const { canvas, img, displayPoints } = active;
  // Make sure the canvas matches the img's displayed size.
  const w = img.clientWidth, h = img.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  // live_overlay.js shares this canvas; it'll re-clear next frame. We layer
  // our preview as a separate save/restore so a stale half-poly never lingers
  // after Done. (Re-cleared each render here too.)
  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#2ecc71";
  ctx.fillStyle = "#2ecc7155";
  if (displayPoints.length > 0) {
    ctx.beginPath();
    ctx.moveTo(displayPoints[0][0], displayPoints[0][1]);
    for (let i = 1; i < displayPoints.length; i++) {
      ctx.lineTo(displayPoints[i][0], displayPoints[i][1]);
    }
    if (displayPoints.length >= 3) {
      ctx.lineTo(displayPoints[0][0], displayPoints[0][1]);
      ctx.fill();
    }
    ctx.stroke();
    // Point dots — numbered in click order (calibration: TL→TR→BR→BL) so the
    // operator can see which corner is which and verify the sequence.
    displayPoints.forEach(([dx, dy], i) => {
      ctx.beginPath();
      ctx.arc(dx, dy, 7, 0, Math.PI * 2);
      ctx.fillStyle = "#2ecc71";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#04361b";
      ctx.stroke();
      ctx.fillStyle = "#04361b";
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), dx, dy);
    });
  }
  ctx.restore();
}

async function onCamPointerDown(ev) {
  if (!active || (active.target !== "cam_a" && active.target !== "cam_b")) return;
  const { img, points, displayPoints, toolbar, target, mode } = active;
  // Where on the displayed image did the operator click?
  const rect = img.getBoundingClientRect();
  const dispX = ev.clientX - rect.left;
  const dispY = ev.clientY - rect.top;
  const [srcX, srcY] = displayToSource(img, ev.clientX, ev.clientY);

  if (mode === "raw") {
    // Calibration path: store raw source-frame pixel coords directly. No
    // backend call — calibration doesn't exist yet, so pixel→world is
    // mathematically impossible.
    points.push([srcX, srcY]);
    displayPoints.push([dispX, dispY]);
    toolbar.count.textContent = String(points.length);
    renderCamPreview();
    maybeAutoFinish();
    return;
  }

  // 'project' mode (default): convert to world (X, Y) m via the camera's
  // geometry — on the plane z = zM (a raised platform/shelf zone's own base
  // height, passed by the caller; 0 = the floor path, unchanged).
  try {
    const res = await fetch("/api/project/pixel-to-floor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // frame_wh: the stream can be a downscaled copy of the calibrated
      // sensor — the server rescales the click into the calibration frame.
      body: JSON.stringify({ camera_id: target, points: [[srcX, srcY]],
                             frame_wh: [img.naturalWidth, img.naturalHeight],
                             z_m: active.zM || 0 }),
    });
    if (!res.ok) {
      console.warn("draw_mode: pixel-to-floor failed", await res.text());
      toolbar.count.textContent = "calibration?";
      return;
    }
    const data = await res.json();
    const [X_m, Y_m] = data.points[0];
    points.push([X_m, Y_m]);
    displayPoints.push([dispX, dispY]);
    toolbar.count.textContent = String(points.length);
    renderCamPreview();
    maybeAutoFinish();
  } catch (err) {
    console.warn("draw_mode: pixel-to-floor network error", err);
  }
}

// If the session has a maxPoints cap and we've hit it, finish automatically —
// useful for fixed-count flows (e.g. calibration's 4 pallet corners).
function maybeAutoFinish() {
  if (!active) return;
  if (Number.isFinite(active.maxPoints) && active.points.length >= active.maxPoints) {
    const pts = active.points.slice();
    const onDone = active.onDone;
    cleanup();
    onDone?.(pts);
  }
}

// ---- common lifecycle ----

function cleanup() {
  if (!active) return;
  if (active.target === "map") {
    active.canvas.removeEventListener("pointerdown", onMapPointerDown);
    active.canvas.removeEventListener("pointermove", onMapPointerMove);
    active.fm.clearPreview();
    if (active.fm.controls) active.fm.controls.enabled = true;   // re-enable orbit
  } else {
    active.img.removeEventListener("pointerdown", onCamPointerDown);
    active.img.style.cursor = active._oldCursor || "";
    document.body.classList.remove("cam-drawing");   // restore the cam overscan
    // Wipe the preview from the cam overlay; live_overlay.js will redraw next frame.
    const ctx = active.canvas.getContext("2d");
    ctx.clearRect(0, 0, active.canvas.width, active.canvas.height);
  }
  active.toolbar.bar.classList.add("hidden");
  active.toolbar.undo.onclick = null;
  active.toolbar.done.onclick = null;
  active.toolbar.cancel.onclick = null;
  active = null;
}

function rerenderPreview() {
  if (!active) return;
  if (active.target === "map") renderMapPreview();
  else renderCamPreview();
}

export function startDraw({
  target = "map",
  mode = "project",
  label,
  onDone,
  onCancel,
  minPoints = 3,
  maxPoints = Infinity,
  zM = 0,   // 'project' mode: decode clicks onto the plane z = zM metres
            // (a raised zone's base height) instead of the floor
}) {
  const toolbar = ensureToolbar();
  if (!toolbar.bar) return;
  if (active) cleanup();
  if (mode === "raw" && target === "map") {
    console.warn("draw_mode: mode='raw' is only meaningful for cam targets");
    return;
  }

  // Resolve the target surface.
  if (target === "map") {
    const fm = window.__floor_map;
    if (!fm || !fm.transform || typeof fm.transform.screenToWorld !== "function") {
      console.warn("draw_mode: 3D floor map not ready");
      return;
    }
    const canvas = fm.app.canvas;
    if (fm.controls) fm.controls.enabled = false;   // don't orbit the camera while drawing
    active = {
      target: "map", mode, fm, canvas,
      points: [], toolbar, onDone, minPoints, maxPoints,
      gridStep: (arguments[0] && arguments[0].gridStep) || 0,
    };
    canvas.addEventListener("pointerdown", onMapPointerDown);
    if (mode === "rectSnap") canvas.addEventListener("pointermove", onMapPointerMove);
  } else if (target === "cam_a" || target === "cam_b") {
    const imgId = `${target}-img`;
    const canvasId = `${target}-overlay`;
    const img = document.getElementById(imgId);
    const canvas = document.getElementById(canvasId);
    if (!img || !canvas) {
      console.warn(`draw_mode: ${target} elements not found`);
      return;
    }
    active = {
      target, mode, img, canvas, zM,
      points: [],          // 'project': world (X, Y) m; 'raw': source-frame (u, v) px
      displayPoints: [],   // display-px (for the preview overlay only)
      toolbar, onDone, minPoints, maxPoints,
      _oldCursor: img.style.cursor,
    };
    img.style.cursor = "crosshair";
    img.addEventListener("pointerdown", onCamPointerDown);
    // Drop the cam overscan to scale 1 so click capture + preview map in plain
    // contain space and the operator can draw across the full frame.
    document.body.classList.add("cam-drawing");
  } else {
    console.warn(`draw_mode: unknown target ${target}`);
    return;
  }

  toolbar.label.textContent = label ?? "Zone";   // "" = intentionally blank
  toolbar.count.textContent = "0";
  toolbar.bar.classList.remove("hidden");

  toolbar.undo.onclick = () => {
    if (!active || !active.points.length) return;
    active.points.pop();
    if (active.displayPoints) active.displayPoints.pop();
    active.toolbar.count.textContent = String(active.points.length);
    rerenderPreview();
  };
  toolbar.done.onclick = () => {
    if (!active) return;
    if (active.points.length < active.minPoints) {
      toolbar.count.textContent = `≥${active.minPoints} required`;
      return;
    }
    const pts = active.points.slice();
    cleanup();
    onDone?.(pts);
  };
  toolbar.cancel.onclick = () => {
    cleanup();
    onCancel?.();
  };
}

export function cancelDraw() {
  if (active) cleanup();
}

// Re-draw the in-progress cam draw (calibration dots + polygon) for `camId`.
// live_overlay.js clears the shared overlay canvas every animation frame, which
// would otherwise wipe the calibration preview the instant it's drawn — so the
// live loop calls this each tick to keep the clicked points visible.
export function renderActiveCamPreview(camId) {
  if (!active || active.target !== camId) return;
  renderCamPreview();
}
