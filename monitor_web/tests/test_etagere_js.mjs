// Pure-geometry test for the étagère drag-adjust helpers (no DOM).
//
// NOTE (deviation from the task-9 brief): the brief's snippet imports
// `applyDrag`/`hitTest` from "../monitor_web/static/js/etagere.js". That file
// imports `startDraw` from the absolute browser path "/static/js/draw_mode.js",
// which Node's ESM resolver cannot load (not a relative/bare specifier) —
// importing etagere.js here would fail before a single assertion runs. The
// two pure helpers live in the DOM-free sibling module etagere_geom.js
// instead; etagere.js re-exports them for the browser. Import from there.
import assert from "node:assert/strict";
import { applyDrag, frameWhOrNull, hitTest } from "../monitor_web/static/js/etagere_geom.js";

// frameWhOrNull: the I2 guard against a 0x0 frame_wh (would ZeroDivisionError
// EtagereDetector._crop on every tick). Valid sizes round-trip (rounded);
// zero, negative, missing or non-finite dimensions all reject.
assert.deepEqual(frameWhOrNull(1920, 1080), [1920, 1080]);
assert.deepEqual(frameWhOrNull(1920.4, 1080.6), [1920, 1081]);
assert.equal(frameWhOrNull(0, 1080), null);
assert.equal(frameWhOrNull(1920, 0), null);
assert.equal(frameWhOrNull(0, 0), null);
assert.equal(frameWhOrNull(-1, 100), null);
assert.equal(frameWhOrNull(undefined, undefined), null);
assert.equal(frameWhOrNull(NaN, 100), null);

// move: whole rect translates
assert.deepEqual(applyDrag([10, 10, 50, 50], "move", 5, -5), [15, 5, 55, 45]);
// corner drags: only that corner moves; never inverted (min 4 px)
assert.deepEqual(applyDrag([10, 10, 50, 50], "br", 10, 10), [10, 10, 60, 60]);
assert.deepEqual(applyDrag([10, 10, 50, 50], "tl", 100, 100), [46, 46, 50, 50]);
// tr / bl: same never-inverted contract, the other two corners.
assert.deepEqual(applyDrag([10, 10, 50, 50], "tr", -5, 5), [10, 15, 45, 50]);
assert.deepEqual(applyDrag([10, 10, 50, 50], "tr", 100, 100), [10, 46, 150, 50]);
assert.deepEqual(applyDrag([10, 10, 50, 50], "bl", 5, -5), [15, 10, 50, 45]);
assert.deepEqual(applyDrag([10, 10, 50, 50], "bl", 100, -100), [46, 10, 50, 14]);

// hit-test: corner handle within 8 px wins over move
const zone = { cells: [{ r: 1, c: 1, rect: [10, 10, 50, 50] }, { r: 1, c: 2, rect: [60, 10, 100, 50] }] };
assert.deepEqual(hitTest(zone, 49, 49, 8), { cellIdx: 0, handle: "br" });
assert.deepEqual(hitTest(zone, 30, 30, 8), { cellIdx: 0, handle: "move" });
assert.deepEqual(hitTest(zone, 80, 30, 8), { cellIdx: 1, handle: "move" });
assert.deepEqual(hitTest(zone, 200, 200, 8), { cellIdx: -1, handle: null });

// hit-test: overlapping cells — the later (top-most, drawn-on-top) cell wins
// a move-hit in the overlap region; a cell entirely outside the top cell's
// box still resolves via its own bounds.
const overlap = { cells: [
  { r: 1, c: 1, rect: [0, 0, 100, 100] },
  { r: 2, c: 1, rect: [20, 20, 60, 60] },
] };
assert.deepEqual(hitTest(overlap, 30, 30, 8), { cellIdx: 1, handle: "move" });
assert.deepEqual(hitTest(overlap, 90, 90, 8), { cellIdx: 0, handle: "move" });
// corner-priority holds even when the corner belongs to the TOP cell but
// falls inside the underneath cell's body too: (60,60) is cells[1]'s br AND
// well inside cells[0]'s 0..100 box — the corner still wins over any move.
assert.deepEqual(hitTest(overlap, 60, 60, 8), { cellIdx: 1, handle: "br" });

// hit-test: corner-priority ALSO holds the other way — a corner belonging to
// the UNDERNEATH cell, sitting inside the TOP cell's body, still wins over a
// move-hit on the top cell (corners are checked across every cell before any
// move-check runs at all).
const cornerBeatsMove = { cells: [
  { r: 1, c: 1, rect: [0, 0, 50, 50] },     // underneath; br corner = (50,50)
  { r: 1, c: 2, rect: [30, 30, 70, 70] },   // on top; (50,50) is deep inside its body, not a corner
] };
assert.deepEqual(hitTest(cornerBeatsMove, 50, 50, 8), { cellIdx: 0, handle: "br" });

console.log("etagere.js helpers OK");

// rotateCorners: rotates around the centroid; 90° cw maps TL→TR on screen (y down);
// 0° is identity; ±deg round-trips.
{
  const { rotateCorners } = await import("../monitor_web/static/js/etagere_geom.js");
  const sq = [[0, 0], [10, 0], [10, 10], [0, 10]];   // TL,TR,BR,BL, centre (5,5)
  const same = rotateCorners(sq, 0);
  assert.deepEqual(same.map((p) => p.map(Math.round)), sq);
  const cw = rotateCorners(sq, 90).map((p) => p.map((v) => Math.round(v) + 0));   // +0 folds -0
  assert.deepEqual(cw, [[10, 0], [10, 10], [0, 10], [0, 0]]);   // TL moved to TR's spot
  const back = rotateCorners(rotateCorners(sq, 7), -7).map((p) => p.map((v) => Math.round(v * 1000) / 1000));
  assert.deepEqual(back, sq);
  // centroid preserved
  const r = rotateCorners(sq, 33);
  const cx = r.reduce((a, p) => a + p[0], 0) / 4, cy = r.reduce((a, p) => a + p[1], 0) / 4;
  assert.ok(Math.abs(cx - 5) < 1e-9 && Math.abs(cy - 5) < 1e-9);
  console.log("rotateCorners OK");
}
