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
import { applyDrag, hitTest } from "../monitor_web/static/js/etagere_geom.js";

// move: whole rect translates
assert.deepEqual(applyDrag([10, 10, 50, 50], "move", 5, -5), [15, 5, 55, 45]);
// corner drags: only that corner moves; never inverted (min 4 px)
assert.deepEqual(applyDrag([10, 10, 50, 50], "br", 10, 10), [10, 10, 60, 60]);
assert.deepEqual(applyDrag([10, 10, 50, 50], "tl", 100, 100), [46, 46, 50, 50]);
// hit-test: corner handle within 8 px wins over move
const zone = { cells: [{ r: 1, c: 1, rect: [10, 10, 50, 50] }, { r: 1, c: 2, rect: [60, 10, 100, 50] }] };
assert.deepEqual(hitTest(zone, 49, 49, 8), { cellIdx: 0, handle: "br" });
assert.deepEqual(hitTest(zone, 30, 30, 8), { cellIdx: 0, handle: "move" });
assert.deepEqual(hitTest(zone, 80, 30, 8), { cellIdx: 1, handle: "move" });
assert.deepEqual(hitTest(zone, 200, 200, 8), { cellIdx: -1, handle: null });
console.log("etagere.js helpers OK");
