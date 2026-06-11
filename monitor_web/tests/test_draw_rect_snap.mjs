// Pure-geometry test for the rectangle-snap helper (no DOM/Pixi).
import assert from "node:assert";
import { rectFromCorners, snapToGrid } from "../monitor_web/static/js/draw_mode.js";

// two opposite corners → 4 axis-aligned corners (TL, TR, BR, BL order)
const r = rectFromCorners([2.0, 1.0], [3.5, 2.0]);
assert.deepStrictEqual(r, [[2.0, 1.0], [3.5, 1.0], [3.5, 2.0], [2.0, 2.0]]);

// grid snap to 0.1 m
assert.deepStrictEqual(snapToGrid([2.04, 1.07], 0.1), [2.0, 1.1]);
console.log("ok");
