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
export function hitTest(zone, x, y, tol = HANDLE_PX) {
  const cells = zone.cells || [];
  for (let i = cells.length - 1; i >= 0; i--) {          // top-most first
    const [x0, y0, x1, y1] = cells[i].rect;
    const corners = { tl: [x0, y0], tr: [x1, y0], br: [x1, y1], bl: [x0, y1] };
    for (const [h, [cx, cy]] of Object.entries(corners)) {
      if (Math.abs(cx - x) <= tol && Math.abs(cy - y) <= tol) return { cellIdx: i, handle: h };
    }
  }
  for (let i = cells.length - 1; i >= 0; i--) {
    const [x0, y0, x1, y1] = cells[i].rect;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return { cellIdx: i, handle: "move" };
  }
  return { cellIdx: -1, handle: null };
}
