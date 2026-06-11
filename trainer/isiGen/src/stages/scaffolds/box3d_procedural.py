"""Procedural 3D-box scaffolds — Phase 6.

Renders an ANALYTIC depth map (DepthAnythingV2 convention: brighter = closer)
plus a perfectly aligned class-color ground-truth mask, with zero models:
a receding floor gradient and 1-4 "stacks" (pallet slab + cartons, sometimes a
polybag-wrapped load) placed back-to-front so occlusion comes free.

The ControlNet only needs plausible geometry — the base model invents all
appearance; the mask is the free label.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .base import SCAFFOLD_SOURCES, ScaffoldSource

if TYPE_CHECKING:
    from ...core.project import ProjectConfig


@SCAFFOLD_SOURCES.register("box3d_procedural")
class Box3dProceduralScaffolds(ScaffoldSource):
    def __init__(self, width: int = 1024, height: int = 1024, seed: int | None = None,
                 max_stacks: int = 4, **cfg) -> None:
        super().__init__(width=width, height=height, seed=seed,
                         max_stacks=max_stacks, **cfg)
        self.width = int(width)
        self.height = int(height)
        self.seed = seed
        self.max_stacks = int(max_stacks)

    def generate(self, project: ProjectConfig, count: int
                 ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
        rng = random.Random(self.seed)
        names = project.class_names()
        colors = {c.name: c.color for c in project.classes}
        for i in range(count):
            yield self._scene(rng, names, colors, i)

    # ---- scene construction ----

    def _scene(self, rng: random.Random, names: list[str],
               colors: dict[str, list[int]], idx: int):
        w, h = self.width, self.height
        horizon = int(h * rng.uniform(0.30, 0.45))
        depth = np.zeros((h, w), dtype=np.uint8)
        # Sky/wall above the horizon: far (dark); floor below: near at the bottom.
        depth[:horizon] = rng.randint(8, 25)
        floor_rows = np.linspace(40, 215, h - horizon).astype(np.uint8)
        depth[horizon:] = floor_rows[:, None]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        present: set[str] = set()
        stacks = rng.randint(1, self.max_stacks)
        # Back-to-front: sort base lines top→bottom so near stacks overdraw far ones.
        bases = sorted(rng.uniform(0.15, 0.95) for _ in range(stacks))
        for t in bases:
            y_base = int(horizon + t * (h - horizon - 4))
            near = float(depth[min(y_base, h - 1), w // 2])     # floor depth there
            scale = 0.25 + 0.75 * t                              # nearer = bigger
            sw = int(w * rng.uniform(0.18, 0.34) * scale)
            x0 = rng.randint(0, max(1, w - sw - 1))
            self._stack(rng, depth, mask, names, colors, present,
                        x0, y_base, sw, near, scale)
        if not present:                # degenerate roll — guarantee one pallet
            present.add(names[0])
        meta = {"classes": sorted(present), "source": "box3d_procedural", "index": idx}
        return depth, mask, meta

    def _stack(self, rng, depth, mask, names, colors, present,
               x0, y_base, sw, near, scale) -> None:
        h_img = depth.shape[0]

        def paint(x0_, y0_, x1_, y1_, value, cls):
            x0_, x1_ = max(0, x0_), min(depth.shape[1], x1_)
            y0_, y1_ = max(0, y0_), min(h_img, y1_)
            if x1_ <= x0_ or y1_ <= y0_:
                return
            # Slight vertical depth gradient so faces read as 3D volumes.
            grad = np.clip(np.linspace(value + 6, value - 6, y1_ - y0_), 0, 255)
            depth[y0_:y1_, x0_:x1_] = grad.astype(np.uint8)[:, None]
            r, g, b = colors[cls]
            mask[y0_:y1_, x0_:x1_] = (b, g, r)                  # BGR canvas
            present.add(cls)

        # Pallet slab (always, if the project has one).
        pallet = next((n for n in names if "pal" in n.lower()), names[0])
        ph = max(6, int(28 * scale))
        paint(x0, y_base - ph, x0 + sw, y_base, near, pallet)
        top = y_base - ph

        # 0-3 cartons stacked on the pallet, slightly inset; sometimes the load
        # reads as a polybag-wrapped unit instead.
        carton = next((n for n in names if "cart" in n.lower() or "box" in n.lower()), None)
        polybag = next((n for n in names if "poly" in n.lower() or "bag" in n.lower()), None)
        for _ in range(rng.randint(0, 3)):
            if carton is None:
                break
            ch = int(sw * rng.uniform(0.25, 0.55))
            cw = int(sw * rng.uniform(0.55, 0.95))
            cx0 = x0 + rng.randint(0, max(1, sw - cw))
            cls = polybag if (polybag is not None and rng.random() < 0.3) else carton
            paint(cx0, top - ch, cx0 + cw, top, near + rng.randint(2, 10), cls)
            top -= ch
            if top < int(h_img * 0.05):
                break
        # Soften depth edges a touch (real depth maps aren't razor-edged).
        cv2.GaussianBlur(depth, (5, 5), 0, dst=depth)
