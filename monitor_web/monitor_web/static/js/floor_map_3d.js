// 3D floor map (Three.js) — replaces the Pixi 2.5D map.
//
// Renders the warehouse digital twin as a genuinely 3D, orbit-able scene:
// a tilted perspective camera + OrbitControls (zoom / pan / constrained orbit),
// extruded translucent zones, and per-class volumetric track objects that glide
// to their live metric positions. Reads the SAME data the Pixi map did
// (window.__tracks.byId2D/byId3D, /api/zones) and exposes the SAME
// `window.__floor_map` contract so draw_mode / layout_manager / live_overlay /
// zone_manager keep working.
//
// Coordinate frame: world metres → Three units 1:1. The floor is the X–Z plane;
// world (X, Y) maps to Three (x = X, y = height, z = -Y). Track2D sits at y = 0;
// Track3D height (if present) lifts it.
//
// "Not heavy": low-poly primitives, no shadows / post-FX, and the render loop
// PAUSES whenever the MAP view is hidden (the host has zero layout box).
//
// Phase 1 (this file): scene + camera + zones + live tracks + the shim.
// Phase 2 (todo): proximity rings + breach arrows, link-lines, the
// drawLayout/drawOutline extrusions, and the draw_mode raycast 3D preview.
// Those methods are present below as safe no-ops so nothing crashes meanwhile.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const HOST_ID = "floor-map-host";

// ---- palette (mirrors floor_map.js so the two read identically) ----
const COLORS = {
  bg: 0x101418,
  grid: 0x1f2933,
  gridCenter: 0x2c3a47,
  floor: 0x0c1014,
  text: 0xe6edf3,
  zone_danger: 0xfca5a5,
  zone_etagere: 0x86efac,
  zone_palette: 0x9aa5b1,
  cls_person: 0xffd54f,
  cls_forklift: 0xff7043,
  cls_pallet: 0x4fc3f7,
  cls_robot: 0x9b59b6,
  cls_unknown: 0xbdc3c7,
};

const LERP = 0.18;          // per-frame position smoothing (matches Pixi map)
const GC_MS = 5000;         // drop a track unseen for >5 s
const ZONE_DEPTH = 0.12;    // metres of extrusion for a flat zone prism
// Default framing: how much of the panel the scene fills. Bounds now HUG the
// content (recomputeBounds), so a mild overscan fills the panel without
// cropping — the old 1.5 was compensating for the fixed ±5 m floor.
const FILL = 1.15;
const TILT_DEG = 58;        // camera elevation above the floor (higher = more top-down → floor fills more)

function colorForClass(cls) {
  if (cls === "person") return COLORS.cls_person;
  if (cls === "forklift") return COLORS.cls_forklift;
  if (cls === "pallet" || cls === "palette" || cls === "palette_vide") return COLORS.cls_pallet;
  if (cls === "custom-robot" || cls === "robot") return COLORS.cls_robot;
  return COLORS.cls_unknown;
}
function colorForZone(zone) {
  const k = zone.kind || zone.type || "palette";
  if (k === "danger") return COLORS.zone_danger;
  if (k === "etagere") return COLORS.zone_etagere;
  return COLORS.zone_palette;
}
function occupancyText(msg) {
  if (!msg || msg.occupancy_state == null) return "";
  return msg.occupancy_state === "empty" ? " · empty" : ` · ${msg.occupancy_content || "full"}`;
}
function occupancyColor(msg) {
  if (!msg || msg.occupancy_state == null) return null;
  if (msg.occupancy_state === "empty") return 0x2ed573;
  if (msg.occupancy_content === "carton") return 0xf5ab35;
  if (msg.occupancy_content === "polybag") return 0x4fc3f7;
  return 0xe6edf3;
}

// world (X, Y[, height]) → Three.Vector3
function w2t(x, y, h = 0) { return new THREE.Vector3(x, h, -y); }

async function init() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;

  // ---- renderer / scene / camera ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth || 800, host.clientHeight || 600, false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);

  const camera = new THREE.PerspectiveCamera(
    55, (host.clientWidth || 800) / (host.clientHeight || 600), 0.1, 2000,
  );

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = THREE.MathUtils.degToRad(85);  // never go under the floor
  controls.minDistance = 1;
  controls.maxDistance = 500;

  // ---- lights (cheap, no shadows) ----
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dir = new THREE.DirectionalLight(0xffffff, 0.55);
  dir.position.set(0.4, 1, 0.6);
  scene.add(dir);

  // ---- static groups ----
  const groundGroup = new THREE.Group();
  const layoutGroup = new THREE.Group();   // static warehouse twin (racks / walls / outline)
  const zoneGroup = new THREE.Group();
  const objectGroup = new THREE.Group();
  const previewGroup = new THREE.Group();  // draw-mode rubber-band (zones / racks)
  scene.add(groundGroup, layoutGroup, zoneGroup, objectGroup, previewGroup);

  // ---- state ----
  let bounds = { minX: -6, maxX: 6, minY: -6, maxY: 6 };
  let zones = [];
  let twinPolys = [];     // zone-patch outlines (floor metres) from /api/map/twin
  let perClass = {};
  const views = new Map();          // track_id -> { group, mesh, label, target, ... }
  const zoneMeshes = [];            // { mesh, pulse } for the per-frame pulse
  let grid = null;

  // ---------- ground ----------
  function rebuildGround() {
    groundGroup.clear();
    const w = bounds.maxX - bounds.minX;
    const h = bounds.maxY - bounds.minY;
    const cx = (bounds.minX + bounds.maxX) / 2;
    const cy = (bounds.minY + bounds.maxY) / 2;
    const size = Math.max(w, h) + 4;

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      new THREE.MeshStandardMaterial({ color: COLORS.floor, roughness: 1, metalness: 0 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(cx, -0.02, -cy);
    groundGroup.add(floor);

    grid = new THREE.GridHelper(size, Math.round(size), COLORS.gridCenter, COLORS.grid);
    grid.position.set(cx, 0, -cy);
    grid.material.opacity = 0.5;
    grid.material.transparent = true;
    groundGroup.add(grid);
  }

  // ---------- zones (extruded translucent prisms) ----------
  function rebuildZones() {
    for (const z of zoneMeshes) zoneGroup.remove(z.mesh);
    zoneMeshes.length = 0;
    for (const zone of zones) {
      const poly = zone.polygon;
      if (!poly || poly.length < 3) continue;
      const shape = new THREE.Shape();
      shape.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) shape.lineTo(poly[i][0], poly[i][1]);
      shape.closePath();
      const geo = new THREE.ExtrudeGeometry(shape, { depth: ZONE_DEPTH, bevelEnabled: false });
      // shape (sx, sy) extruded +Z → rotate -90° about X lays it on the floor as
      // (sx, gz, -sy) == our (x, height, -y) convention.
      geo.rotateX(-Math.PI / 2);
      const color = colorForZone(zone);
      const kind = zone.kind || zone.type || "palette";
      const baseAlpha = kind === "danger" ? 0.24 : 0.16;
      const mat = new THREE.MeshStandardMaterial({
        color, transparent: true, opacity: baseAlpha,
        roughness: 0.9, side: THREE.DoubleSide,
        emissive: color, emissiveIntensity: kind === "danger" ? 0.25 : 0.12,
      });
      const mesh = new THREE.Mesh(geo, mat);
      // outline
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
      );
      mesh.add(edges);
      // label at the first vertex
      const label = makeLabel(zone.name || kind, color);
      label.position.copy(w2t(poly[0][0], poly[0][1], ZONE_DEPTH + 0.3));
      mesh.add(label);
      zoneGroup.add(mesh);

      const sev = zone.severity;
      const pulse = kind === "danger" && (sev === "warning" || sev === "critical")
        ? { base: baseAlpha, amp: 0.14, freq: 0.4 } : null;
      zoneMeshes.push({ mesh, mat, pulse });
    }
  }

  // ---------- text billboards (canvas-texture sprites) ----------
  // Default look = dark tag. Pass {bg, radius} for the WHITE ROUNDED BADGE the
  // cam-view distance lines use (mirrors overlay.py::_draw_distance's badge).
  function makeLabel(text, color = 0xffffff, { bg = "rgba(5,8,11,0.72)", radius = 0 } = {}) {
    const pad = radius > 0 ? 14 : 8, fs = 48;
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d");
    ctx.font = `500 ${fs}px monospace`;
    const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
    c.width = w; c.height = fs + pad * 2;
    ctx.font = `500 ${fs}px monospace`;
    ctx.fillStyle = bg;
    if (radius > 0 && ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(0, 0, c.width, c.height, radius);
      ctx.fill();
    } else {
      ctx.fillRect(0, 0, c.width, c.height);
    }
    ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
    ctx.textBaseline = "middle";
    ctx.fillText(text, pad, c.height / 2);
    const tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    // Small tags, clearly subordinate to the bodies (a 0.4 m label dwarfed the
    // 0.42 m pallets and the tags overlapped each other); keep aspect.
    const hM = 0.22;
    spr.scale.set(hM * (c.width / c.height), hM, 1);
    spr.userData.tex = tex;
    return spr;
  }

  // ---------- track objects ----------
  function geometryForClass(cls) {
    // Compact low-poly primitives (~55% smaller than life-size) — small bodies with
    // generous space between them so the operator can read the scene at a glance.
    // h = label height, y = floor baseline.
    if (cls === "person") return { geo: new THREE.CapsuleGeometry(0.10, 0.38, 4, 10), h: 0.62, y: 0.3 };
    if (cls === "forklift") return { geo: new THREE.BoxGeometry(0.42, 0.45, 0.75), h: 0.45, y: 0.225 };
    if (cls === "custom-robot" || cls === "robot") return { geo: new THREE.CylinderGeometry(0.14, 0.14, 0.34, 16), h: 0.34, y: 0.17 };
    // pallets / unknown → flat box
    return { geo: new THREE.BoxGeometry(0.42, 0.13, 0.52), h: 0.13, y: 0.065 };
  }

  function buildView(msg) {
    const cls = msg.cls;
    const { geo, h, y } = geometryForClass(cls);
    const color = colorForClass(cls);
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = y;
    const group = new THREE.Group();
    group.add(mesh);
    const label = makeLabel(msg.label || `#${msg.track_id}${occupancyText(msg)}`,
                            occupancyColor(msg) ?? COLORS.text);
    label.position.y = h + 0.18;
    group.add(label);
    // velocity arrow
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.05, 0), 0.01, color, 0.25, 0.15);
    group.add(arrow);
    objectGroup.add(group);
    const start = w2t(msg.xy_m[0], msg.xy_m[1]);
    group.position.copy(start);
    return {
      group, mesh, mat, label, arrow, cls, topH: h, baseY: y, color,
      target: start.clone(), lastSeen: performance.now(),
    };
  }

  function setLabel(view, text, color) {
    const old = view.label;
    const next = makeLabel(text, color);
    next.position.copy(old.position);
    view.group.remove(old);
    old.material.map.dispose(); old.material.dispose();
    view.group.add(next);
    view.label = next;
  }

  // ---------- detection twin (the realtime mirror of the cam view) ----------
  // Polls /api/map/twin: the dashboard's OWN detections (zone worker objects +
  // full-frame pose people) projected to floor metres, plus the drawn zone-patch
  // polygons as floor outlines. Mode 1 mirrors CAM 1; Mode 2 unions both cameras
  // (the unified view). While the twin is AVAILABLE (calibration + worker), it is
  // the map's object source and the Backbone-track layer stays off — one truth.
  let twinAvailable = false;
  let twinAt = 0;                       // performance.now() of the last good poll
  let twinSeq = 0;
  let twinTracks = [];                  // [{id, cls, x, y, conf}] — matched frame-to-frame
  let twinZoneSig = "";                 // change detector for the zone outlines
  const twinZoneGroup = new THREE.Group();
  scene.add(twinZoneGroup);

  // Greedy nearest-neighbour id matching so twin objects keep a stable identity
  // (and therefore smooth lerp motion) across polls — same class within 1.5 m.
  function matchTwin(dets) {
    const used = new Set();
    const out = [];
    for (const d of dets) {
      let best = null, bestD = 1.5;
      for (const t of twinTracks) {
        if (used.has(t.id) || t.cls !== d.cls) continue;
        const dist = Math.hypot(t.x - d.xy_m[0], t.y - d.xy_m[1]);
        if (dist < bestD) { best = t; bestD = dist; }
      }
      const id = best ? best.id : `tw${++twinSeq}`;
      used.add(id);
      out.push({ id, cls: d.cls, x: d.xy_m[0], y: d.xy_m[1], conf: d.conf ?? 0 });
    }
    twinTracks = out;
    return out;
  }

  function rebuildTwinZones(zs) {
    const sig = JSON.stringify(zs);
    if (sig === twinZoneSig) return;
    twinZoneSig = sig;
    while (twinZoneGroup.children.length) {
      const c = twinZoneGroup.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }
    for (const z of zs || []) {
      const poly = z.polygon_m;
      if (!Array.isArray(poly) || poly.length < 3) continue;
      const pts = poly.map(([x, y]) => w2t(x, y, 0.03));
      pts.push(pts[0].clone());
      const color = new THREE.Color(z.color || "#ff3b3b");
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(geo, new THREE.LineDashedMaterial({
        color, dashSize: 0.35, gapSize: 0.2, transparent: true, opacity: 0.9,
      }));
      line.computeLineDistances();
      twinZoneGroup.add(line);
      const label = makeLabel(z.name || z.id, color.getHex());
      label.position.copy(w2t(poly[0][0], poly[0][1], 0.22));
      twinZoneGroup.add(label);
    }
    // Zone outlines define where the action is — fold them into the framing.
    if ((zs || []).length) {
      twinPolys = zs.map((z) => z.polygon_m);
      recomputeBounds();
      rebuildGround();
      fitCamera();
    }
  }

  async function pollTwin() {
    // Skip while the MAP view is hidden (zero box) — no work off-screen.
    if (!host.clientWidth || host.offsetParent === null) return;
    try {
      const r = await fetch("/api/map/twin");
      const d = await r.json();
      twinAvailable = !!d.available;
      if (!twinAvailable) return;
      twinAt = performance.now();
      rebuildTwinZones(d.zones || []);
      const dets = [
        ...(d.objects || []),
        ...(d.people || []).map((p) => ({ cls: "person", xy_m: p.xy_m, conf: p.conf })),
      ];
      const now = performance.now();
      const seen = new Set();
      for (const t of matchTwin(dets)) {
        const id = `twin:${t.id}`;
        seen.add(id);
        // Class-only tag: confidence jitters every poll (97↔98%) and rebuilding
        // the label texture each time churned + read noisy. Clean, stable text.
        const msg = { track_id: id, cls: t.cls, xy_m: [t.x, t.y], label: t.cls };
        let view = views.get(id);
        if (!view) { view = buildView(msg); views.set(id, view); }
        else if (t.cls !== view.cls) rebuildBody(view, t.cls);
        view.target = w2t(t.x, t.y);
        view.lastSeen = now;
        if (view._lbl !== msg.label) { setLabel(view, msg.label, COLORS.text); view._lbl = msg.label; }
        view.arrow.visible = false;
      }
      // Drop twin objects that vanished (faster GC than tracks — detections flicker).
      for (const [id, view] of views) {
        if (id.startsWith?.("twin:") && !seen.has(id) && now - view.lastSeen > 1200) {
          objectGroup.remove(view.group);
          view.mesh.geometry.dispose(); view.mat.dispose();
          view.label.material.map.dispose(); view.label.material.dispose();
          views.delete(id);
        }
      }
    } catch { /* network blip — keep last state */ }
  }
  setInterval(pollTwin, 300);

  function syncFromTracks() {
    // While the detection twin drives the map, the Backbone-track layer is off —
    // otherwise every pallet would render twice (track + twin). Crucially, PURGE
    // any track views built before the twin came up (an early return alone left
    // stale "#1 · empty" ghosts next to the twin objects forever).
    if (twinAvailable && performance.now() - twinAt < 2000) {
      for (const [id, view] of views) {
        if (typeof id === "string" && id.startsWith("twin:")) continue;
        removeGizmo(view);
        objectGroup.remove(view.group);
        view.mesh.geometry.dispose(); view.mat.dispose();
        view.label.material.map.dispose(); view.label.material.dispose();
        views.delete(id);
      }
      return;
    }
    const now = performance.now();
    const live = window.__tracks?.byId2D;
    if (!live) return;
    for (const msg of live.values()) {
      let view = views.get(msg.track_id);
      if (!view) {
        view = buildView(msg);
        views.set(msg.track_id, view);
      } else {
        if (msg.cls !== view.cls) { rebuildBody(view, msg.cls); }
      }
      view.lastSeen = now;
      view.lastMsg = msg;

      // Triangulated base height (Mode-2 proof): a REAL 2-view Track3D lifts
      // the body off the floor by xyz_m[2] and grows an XYZ axis gizmo whose
      // Z arrow carries the height badge (the label no longer repeats it).
      // single_view Z is floor-pinned by design — stays flat, no gizmo.
      const t3 = window.__tracks?.byId3D?.get(msg.track_id);
      const sv = !!(t3 && t3.single_view);
      // Fresh = the 3D fix keeps pace with the 2D track (same capture clock);
      // a stale byId3D leftover (subscription lapsed) must not keep the gizmo.
      const fresh3d = !!(t3 && !sv && (msg.ts - t3.ts) < 1.5);
      const z = fresh3d ? Math.max(0, t3.xyz_m?.[2] ?? 0) : 0;
      view.target = w2t(msg.xy_m[0], msg.xy_m[1], z);
      updateGizmo(view, fresh3d, z);
      const lbl = `#${msg.track_id}${occupancyText(msg)}${sv ? "  ·1cam" : ""}`;
      if (view._lbl !== lbl) { setLabel(view, lbl, occupancyColor(msg) ?? COLORS.text); view._lbl = lbl; }
      view.mat.opacity = sv ? 0.5 : 1.0;
      view.mat.transparent = sv;

      // velocity arrow
      const vx = msg.vxy_m?.[0] ?? 0, vy = msg.vxy_m?.[1] ?? 0;
      const speed = Math.hypot(vx, vy);
      if (speed > 0.05) {
        view.arrow.visible = true;
        view.arrow.setDirection(new THREE.Vector3(vx, 0, -vy).normalize());
        view.arrow.setLength(Math.min(1.5, speed) + 0.15, 0.25, 0.15);
      } else {
        view.arrow.visible = false;
      }
    }
    // GC
    for (const [id, view] of views) {
      if (!live.has(id) && now - view.lastSeen > GC_MS) {
        removeGizmo(view);
        objectGroup.remove(view.group);
        view.mesh.geometry.dispose(); view.mat.dispose();
        view.label.material.map.dispose(); view.label.material.dispose();
        views.delete(id);
      }
    }
  }

  function rebuildBody(view, cls) {
    view.group.remove(view.mesh);
    view.mesh.geometry.dispose(); view.mat.dispose();
    const { geo, h, y } = geometryForClass(cls);
    const color = colorForClass(cls);
    view.mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
    view.mesh = new THREE.Mesh(geo, view.mat);
    view.mesh.position.y = y;
    view.group.add(view.mesh);
    view.cls = cls; view.topH = h; view.baseY = y; view.color = color;
    view.label.position.y = h + 0.18;
  }

  // ---------- 3D-localization axis gizmo ----------
  // A classic 3-arrow XYZ marker at the object's base while the track has a
  // FRESH 2-view Track3D fix: X (red) / Y (green) in the floor plane, Z (blue)
  // up, with the measured height on the Z tip in the same white rounded badge
  // the cam-view distance lines use. Built once per view; transforms + badge
  // text update only when the height moves > 1 cm.
  const GIZMO_AXIS = 0.45;   // metres — X/Y arrow length
  const GIZMO_HEAD = 0.10, GIZMO_HEAD_W = 0.05;
  function updateGizmo(view, fresh3d, z) {
    if (!fresh3d) { removeGizmo(view); return; }
    if (!view.gizmo) {
      const g = new THREE.Group();
      const o = new THREE.Vector3(0, 0.01, 0);
      const ax = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), o, GIZMO_AXIS, 0xef4444, GIZMO_HEAD, GIZMO_HEAD_W);   // world +X
      const ay = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), o, GIZMO_AXIS, 0x22c55e, GIZMO_HEAD, GIZMO_HEAD_W);  // world +Y (three -z)
      const az = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), o, GIZMO_AXIS, 0x3b82f6, GIZMO_HEAD, GIZMO_HEAD_W);   // up
      g.add(ax, ay, az);
      view.group.add(g);
      view.gizmo = { group: g, arrows: [ax, ay, az], zArrow: az, label: null, h: NaN };
    }
    const gz = view.gizmo;
    if (!(Math.abs(z - gz.h) > 0.01)) return;
    gz.h = z;
    const zLen = Math.max(0.3, Math.min(2.5, z));   // visible even at z≈0
    gz.zArrow.setLength(zLen, GIZMO_HEAD, GIZMO_HEAD_W);
    const next = makeLabel(`${z.toFixed(2)} m`, 0x0b0e11, { bg: "#ffffff", radius: 18 });
    next.position.set(0, zLen + 0.14, 0);
    if (gz.label) {
      gz.group.remove(gz.label);
      gz.label.material.map.dispose(); gz.label.material.dispose();
    }
    gz.group.add(next);
    gz.label = next;
  }

  function removeGizmo(view) {
    const gz = view.gizmo;
    if (!gz) return;
    view.group.remove(gz.group);
    // ArrowHelper geometries are shared module-level in three — dispose only
    // the per-instance materials (same spirit as the view-purge pattern).
    for (const a of gz.arrows) { a.line.material.dispose(); a.cone.material.dispose(); }
    if (gz.label) { gz.label.material.map.dispose(); gz.label.material.dispose(); }
    view.gizmo = null;
  }

  // ---------- camera framing (wide default, ~80% fill) ----------
  function fitCamera() {
    const cx = (bounds.minX + bounds.maxX) / 2;
    const cy = (bounds.minY + bounds.maxY) / 2;
    const w = Math.max(2, bounds.maxX - bounds.minX);
    const h = Math.max(2, bounds.maxY - bounds.minY);
    const R = 0.5 * Math.hypot(w, h);
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const distV = (R / FILL) / Math.tan(vFov / 2);
    const distH = (R / FILL) / (Math.tan(vFov / 2) * camera.aspect);
    const dist = Math.max(distV, distH);
    controls.target.set(cx, 0, -cy);
    const tilt = THREE.MathUtils.degToRad(TILT_DEG);
    camera.position.set(cx, Math.sin(tilt) * dist, -cy + Math.cos(tilt) * dist);
    camera.near = Math.max(0.1, dist / 200);
    camera.far = dist * 12;
    camera.updateProjectionMatrix();
    controls.update();
  }

  // ---------- data loads ----------
  async function reloadZones() {
    try {
      const r = await fetch("/api/zones");
      const d = await r.json();
      zones = d.zones || [];
    } catch { zones = []; }
    recomputeBounds();
    rebuildGround();
    rebuildZones();
    fitCamera();
    fm.zones = zones;
  }
  async function loadPerClass() {
    try {
      const r = await fetch("/api/danger-zones-object");
      const d = await r.json();
      perClass = d.classes || {};
    } catch { perClass = {}; }
    fm.perClass = perClass;
  }

  // ---------- static warehouse twin: 3-level white racks (+ optional walls) ----------
  // Pallets and people are TRACKED objects (rendered live in objectGroup); this is
  // only the fixed scenery they move around — the shelving racks (and an optional
  // perimeter wall / floor outline).
  let layoutElements = [];
  let layoutOutline = null;
  const RACK_WHITE = 0xe9edf2;
  const WALL_GRAY = 0x6b7682;
  const _rackFrameMat = new THREE.MeshStandardMaterial({ color: RACK_WHITE, roughness: 0.7, metalness: 0.04 });
  const _rackShelfMat = new THREE.MeshStandardMaterial({ color: RACK_WHITE, roughness: 0.85, side: THREE.DoubleSide });

  function _bbox(fp) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of fp) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    return { minX, maxX, minY, maxY, w: maxX - minX, d: maxY - minY,
             cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
  }

  // Parametric white shelving rack: 4 corner uprights + `levels` shelf planes.
  function buildRack(el) {
    const fp = el.footprint;
    if (!Array.isArray(fp) || fp.length < 3) return null;
    const b = _bbox(fp);
    const h = Math.max(0.3, Number(el.height_m) || 2.0);
    const levels = Math.max(1, Math.min(8, Math.round(Number(el.levels) || 3)));
    const g = new THREE.Group();
    const post = 0.08;
    for (const [sx, sy] of [[b.minX, b.minY], [b.maxX, b.minY], [b.maxX, b.maxY], [b.minX, b.maxY]]) {
      const up = new THREE.Mesh(new THREE.BoxGeometry(post, h, post), _rackFrameMat);
      up.position.copy(w2t(sx, sy, h / 2));
      g.add(up);
    }
    for (let i = 0; i < levels; i++) {
      const ly = h * ((i + 1) / levels);            // top shelf at full height
      const shelf = new THREE.Mesh(new THREE.BoxGeometry(Math.max(0.1, b.w), 0.04, Math.max(0.1, b.d)), _rackShelfMat);
      shelf.position.copy(w2t(b.cx, b.cy, ly - 0.02));
      g.add(shelf);
    }
    if (el.rotation_deg) {
      g.position.copy(w2t(b.cx, b.cy, 0));
      g.children.forEach((c) => c.position.sub(w2t(b.cx, b.cy, 0)));
      g.rotation.y = -THREE.MathUtils.degToRad(el.rotation_deg);
    }
    const label = makeLabel(el.label || "rack", 0xcfe8ff);
    label.position.copy(w2t(b.cx, b.maxY, h + 0.25));
    g.add(label);
    g.userData.kind = "el";
    return g;
  }

  // Optional opaque vertical slab wall (footprint extruded to height_m).
  function buildWall(el) {
    const fp = el.footprint;
    if (!Array.isArray(fp) || fp.length < 3) return null;
    const h = Math.max(0.3, Number(el.height_m) || 2.5);
    const shape = new THREE.Shape();
    shape.moveTo(fp[0][0], fp[0][1]);
    for (let i = 1; i < fp.length; i++) shape.lineTo(fp[i][0], fp[i][1]);
    const geo = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
    geo.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: WALL_GRAY, roughness: 0.95, side: THREE.DoubleSide }));
    mesh.userData.kind = "el";
    return mesh;
  }

  function drawLayout(elements) {
    layoutElements = Array.isArray(elements) ? elements : [];
    for (const c of [...layoutGroup.children]) {
      if (c.userData.kind === "el") layoutGroup.remove(c);   // keep the outline
    }
    for (const el of layoutElements) {
      const mesh = (el.type === "wall") ? buildWall(el) : buildRack(el);  // rack = default
      if (mesh) layoutGroup.add(mesh);
    }
  }

  function drawOutline(outline) {
    layoutOutline = outline || null;
    const prev = layoutGroup.getObjectByName("__outline");
    if (prev) layoutGroup.remove(prev);
    const fp = outline && outline.footprint;
    if (!Array.isArray(fp) || fp.length < 3) return;
    const pts = fp.map(([x, y]) => w2t(x, y, 0.02));
    pts.push(pts[0].clone());
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x00d9ff, transparent: true, opacity: 0.85 }),
    );
    line.name = "__outline";
    layoutGroup.add(line);
  }

  async function reloadLayout() {
    try {
      const r = await fetch("/api/warehouse-map");
      const d = await r.json();
      drawLayout(d.elements || []);
      drawOutline(d.outline || null);
    } catch { drawLayout([]); drawOutline(null); }
    recomputeBounds();
    rebuildGround();
    fitCamera();
  }

  // ---------- draw-mode preview (engine-agnostic API for draw_mode.js) ----------
  // Replaces the old Pixi `drawLayer` graphics: draws the in-progress zone/rack
  // rubber-band as a green line loop + vertex dots ON THE FLOOR, so it rides the
  // camera (orbit/zoom) naturally. `worldPts` are [[x,y], ...] in metres.
  function clearPreview() {
    while (previewGroup.children.length) {
      const c = previewGroup.children.pop();
      previewGroup.remove(c);
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }
  }

  function previewPolygon(worldPts, { close = false, color = 0x2ecc71 } = {}) {
    clearPreview();
    if (!Array.isArray(worldPts) || worldPts.length === 0) return;
    const pts = worldPts.map(([x, y]) => w2t(x, y, 0.04));
    if (close && pts.length >= 2) pts.push(pts[0].clone());
    if (pts.length >= 2) {
      previewGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color })));
    }
    const dotGeo = new THREE.SphereGeometry(0.08, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color });
    for (const [x, y] of worldPts) {
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(w2t(x, y, 0.06));
      previewGroup.add(dot);
    }
  }

  function recomputeBounds() {
    // Bounds HUG the actual content (zones / twin outlines / layout) so the
    // workspace fills the panel — the old fixed ±5 m floor made a ~2 m scene
    // shrink into a dot in the middle. Fixed default only when nothing exists.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const polys = [
      ...(zones || []).map((z) => z.polygon),
      ...(twinPolys || []),
      ...(layoutElements || []).map((e) => e.footprint),
      layoutOutline && layoutOutline.footprint,
    ].filter(Boolean);
    for (const p of polys) {
      for (const [x, y] of p) {
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
    if (!Number.isFinite(minX)) {           // empty scene → neutral default
      bounds = { minX: -5, maxX: 5, minY: -5, maxY: 5 };
      return;
    }
    const pad = 0.8;                        // breathing room around the content
    minX -= pad; maxX += pad; minY -= pad; maxY += pad;
    // Enforce a minimum workspace (don't over-zoom a single small zone).
    const MIN_SPAN = 3.5;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const halfW = Math.max((maxX - minX) / 2, MIN_SPAN / 2);
    const halfH = Math.max((maxY - minY) / 2, MIN_SPAN / 2);
    bounds = { minX: cx - halfW, maxX: cx + halfW, minY: cy - halfH, maxY: cy + halfH };
  }

  // ---------- transform shim (raycast onto the floor) ----------
  const raycaster = new THREE.Raycaster();
  const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  let _px = 0, _py = 0;   // last pointer position over the canvas (CSS px)
  function trackPointer(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    _px = e.clientX - rect.left; _py = e.clientY - rect.top;
  }
  renderer.domElement.addEventListener("pointermove", trackPointer);
  renderer.domElement.addEventListener("pointerdown", trackPointer);

  function screenToWorld(sx, sy) {
    const rect = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2((sx / rect.width) * 2 - 1, -((sy / rect.height) * 2 - 1));
    raycaster.setFromCamera(ndc, camera);
    const hit = new THREE.Vector3();
    if (!raycaster.ray.intersectPlane(floorPlane, hit)) return [0, 0];
    return [hit.x, -hit.z];
  }
  function worldToScreen(x, y) {
    const rect = renderer.domElement.getBoundingClientRect();
    const v = w2t(x, y).project(camera);
    return [(v.x * 0.5 + 0.5) * rect.width, (-v.y * 0.5 + 0.5) * rect.height];
  }
  const transform = {
    // draw_mode passes (sx, sy) from the same pointer event that updated
    // (_px,_py); we raycast the cached pointer so X/Y stay on one ray.
    stageToWorldX: () => screenToWorld(_px, _py)[0],
    stageToWorldY: () => screenToWorld(_px, _py)[1],
    screenToWorld,
    worldToStageX: (x) => worldToScreen(x, transform._lastY ?? 0)[0],
    worldToStageY: (y) => { transform._lastY = y; return worldToScreen(0, y)[1]; },
    get scale() { return transform.meters(1); },
    meters: (m) => {
      const a = worldToScreen(0, 0); const b = worldToScreen(m, 0);
      return Math.hypot(b[0] - a[0], b[1] - a[1]);
    },
    get renderBounds() { return { ...bounds }; },
  };

  // ---------- resize / hide-pause ----------
  function onResize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;            // hidden — skip
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    fitCamera();                     // re-frame wide on resize/expand
  }
  const ro = new ResizeObserver(onResize);
  ro.observe(host);
  window.addEventListener("resize", onResize);

  // ---------- render loop (pauses when hidden) ----------
  let _frame = 0;
  function loop() {
    requestAnimationFrame(loop);
    // hidden (x-show display:none) → zero box → skip all work.
    if (!host.clientWidth || !host.clientHeight || host.offsetParent === null) return;
    _frame++;
    syncFromTracks();
    // smooth motion
    for (const view of views.values()) {
      view.group.position.lerp(view.target, LERP);
    }
    // zone danger pulse
    const t = performance.now() * 0.001;
    for (const z of zoneMeshes) {
      if (!z.pulse) continue;
      z.mat.opacity = z.pulse.base + z.pulse.amp * (0.5 + 0.5 * Math.sin(t * 2 * Math.PI * z.pulse.freq));
    }
    controls.update();
    renderer.render(scene, camera);
  }

  // ---------- the public contract ----------
  const fm = {
    app: { canvas: renderer.domElement },
    scene, camera, controls, renderer,
    views, zones, perClass,
    get transform() { return transform; },
    // Draw-mode preview API (used by draw_mode.js's MAP branch) — 3D rubber-band.
    previewPolygon,
    clearPreview,
    reloadZones,
    reloadLinkLines: async () => {},          // Phase 2
    setUnderlay() {},                          // Phase 2 (textured floor)
    setUnderlayOpacity() {},
    fitToUnderlay() {},
    fitCroppedUnderlay() {},
    drawLayout,                                // real 3D racks/walls
    drawOutline,
    reloadLayout,
    fitCamera,
  };
  window.__floor_map = Object.assign(window.__floor_map || {}, fm);

  // events the Pixi map honored
  document.addEventListener("zone-patches:saved", reloadZones);
  document.addEventListener("config:saved", () => { reloadZones(); });
  document.addEventListener("layout:changed", () => { reloadLayout(); });

  // ---------- go ----------
  rebuildGround();
  await reloadZones();
  await reloadLayout();
  loadPerClass();
  fitCamera();
  loop();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
