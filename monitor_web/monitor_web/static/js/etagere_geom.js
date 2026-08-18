// Étagère (bin-rack) pure geometry helpers — cell drag-adjust math + hit
// testing. Deliberately DOM-free / import-free so it loads cleanly under
// Node for the unit test (tests/test_etagere_js.mjs) as well as in the
// browser via etagere.js, which re-exports these two functions.

const HANDLE_PX = 8;

// Move/resize a [x0, y0, x1, y1] rect by (dx, dy) via one of the drag
// handles. Corner drags only move that corner and never invert past a 4 px
// minimum cell size.
export function applyDrag(rect, handle, dx, dy) {
  let [x0, y0, x1, y1] = rect;
  if (handle === "move") return [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
  if (handle === "tl") { x0 = Math.min(x0 + dx, x1 - 4); y0 = Math.min(y0 + dy, y1 - 4); }
  if (handle === "tr") { x1 = Math.max(x1 + dx, x0 + 4); y0 = Math.min(y0 + dy, y1 - 4); }
  if (handle === "br") { x1 = Math.max(x1 + dx, x0 + 4); y1 = Math.max(y1 + dy, y0 + 4); }
  if (handle === "bl") { x0 = Math.min(x0 + dx, x1 - 4); y1 = Math.max(y1 + dy, y0 + 4); }
  return [x0, y0, x1, y1];
}

// Hit-test a point (x, y, in the zone's frame_wh pixel space) against a
// zone's cells. Corner handles (within `tol` px) win over a plain move;
// later cells (drawn on top) win ties. Returns {cellIdx: -1, handle: null}
// on a miss.
// [w, h] if both are positive integers, else null — a 0x0 (or missing)
// frame size must never reach a saved zone: EtagereDetector._crop() divides
// the actual frame's width/height BY frame_wh, so a 0 there is a
// ZeroDivisionError that kills every cell on every tick. Callers should
// alert the operator and abort the draw when this returns null.
export function frameWhOrNull(w, h) {
  if (!Number.isFinite(w) || !Number.isFinite(h) || w < 1 || h < 1) return null;
  return [Math.round(w), Math.round(h)];
}

// --- rotated cells -----------------------------------------------------------
// A cell is its axis-aligned `rect` rotated by `angle_deg` about the rect's
// centre (positive = clockwise on screen, y down). Editing works in the cell's
// LOCAL (unrotated) frame: pointer positions/deltas are un-rotated into it,
// then the plain axis-aligned `applyDrag`/containment logic applies.
export function cellCentre(rect) {
  return [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2];
}

export function rotateVec(dx, dy, deg) {
  const rad = (deg * Math.PI) / 180;
  const c = Math.cos(rad), s = Math.sin(rad);
  return [dx * c - dy * s, dx * s + dy * c];
}

// Screen-space corners TL,TR,BR,BL of a (possibly rotated) cell.
export function cellCorners(cell) {
  const [x0, y0, x1, y1] = cell.rect;
  const a = cell.angle_deg || 0;
  const [cx, cy] = cellCentre(cell.rect);
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => {
    const [rx, ry] = rotateVec(x - cx, y - cy, a);
    return [cx + rx, cy + ry];
  });
}

// Screen point → the cell's local (unrotated) frame.
export function toCellLocal(cell, x, y) {
  const [cx, cy] = cellCentre(cell.rect);
  const [lx, ly] = rotateVec(x - cx, y - cy, -(cell.angle_deg || 0));
  return [cx + lx, cy + ly];
}

export function hitTest(zone, x, y, tol = HANDLE_PX) {
  const cells = zone.cells || [];
  for (let i = cells.length - 1; i >= 0; i--) {          // top-most first
    const [lx, ly] = toCellLocal(cells[i], x, y);
    const [x0, y0, x1, y1] = cells[i].rect;
    const corners = { tl: [x0, y0], tr: [x1, y0], br: [x1, y1], bl: [x0, y1] };
    for (const [h, [cx, cy]] of Object.entries(corners)) {
      if (Math.abs(cx - lx) <= tol && Math.abs(cy - ly) <= tol) return { cellIdx: i, handle: h };
    }
  }
  for (let i = cells.length - 1; i >= 0; i--) {
    const [lx, ly] = toCellLocal(cells[i], x, y);
    const [x0, y0, x1, y1] = cells[i].rect;
    if (lx >= x0 && lx <= x1 && ly >= y0 && ly <= y1) return { cellIdx: i, handle: "move" };
  }
  return { cellIdx: -1, handle: null };
}

// Apply a screen-space drag delta to a (possibly rotated) cell: "move"
// translates the rect as-is; corner handles resize in the local frame, so a
// rotated cell's corner follows the pointer along its own tilted axes.
export function applyCellDrag(cell, handle, dx, dy) {
  if (handle === "move") return { ...cell, rect: applyDrag(cell.rect, "move", dx, dy) };
  const [ldx, ldy] = rotateVec(dx, dy, -(cell.angle_deg || 0));
  return { ...cell, rect: applyDrag(cell.rect, handle, ldx, ldy) };
}

// Rotate the outer quad (4 [x,y] corners) around its centroid by `deg`
// (positive = clockwise on screen, y down). Cells are re-derived from the
// rotated corners by the server's auto-split, so this rotates the whole grid.
export function rotateCorners(corners, deg) {
  const cx = corners.reduce((a, p) => a + p[0], 0) / corners.length;
  const cy = corners.reduce((a, p) => a + p[1], 0) / corners.length;
  const rad = (deg * Math.PI) / 180;
  const c = Math.cos(rad), s = Math.sin(rad);
  return corners.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    return [cx + dx * c - dy * s, cy + dx * s + dy * c];
  });
}
