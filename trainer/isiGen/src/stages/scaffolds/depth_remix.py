"""Depth-remix scaffolds — Phase 6.

Recombines the project's REAL phase-2 artifacts into new layouts: a random
(depth, mask) pair gets a PAIRED affine jitter (hflip / rotate / scale /
translate) — geometry and label transform in lockstep, so every remix is a new
plausible scene with a perfect label, anchored in real-world depth statistics.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ...core.manifest import Manifest
from .base import SCAFFOLD_SOURCES, ScaffoldSource

if TYPE_CHECKING:
    from ...core.project import ProjectConfig


@SCAFFOLD_SOURCES.register("depth_remix")
class DepthRemixScaffolds(ScaffoldSource):
    def __init__(self, project_dir: str | None = None, seed: int | None = None,
                 max_rotate_deg: float = 8.0, scale_range: tuple = (0.85, 1.15),
                 max_translate: float = 0.10, **cfg) -> None:
        super().__init__(project_dir=project_dir, seed=seed,
                         max_rotate_deg=max_rotate_deg, scale_range=scale_range,
                         max_translate=max_translate, **cfg)
        self.project_dir = project_dir            # injected by the runner
        self.seed = seed
        self.max_rotate_deg = float(max_rotate_deg)
        self.scale_range = tuple(scale_range)
        self.max_translate = float(max_translate)

    def generate(self, project: ProjectConfig, count: int
                 ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
        if not self.project_dir:
            raise ValueError("depth_remix needs project_dir (set by the runner)")
        pdir = Path(self.project_dir)
        manifest = Manifest.load(pdir)
        pairs = [(r, pdir / r.depth_map, pdir / r.mask)
                 for r in manifest.active()
                 if r.depth_map and r.mask and not getattr(r, "synthetic", False)]
        pairs = [(r, d, m) for r, d, m in pairs if d.exists() and m.exists()]
        if not pairs:
            raise ValueError(
                "depth_remix: no records with BOTH a depth map and a mask — "
                "run phase 2 (maps + masks) first")
        color_to_class = {tuple(c.color): c.name for c in project.classes}
        rng = random.Random(self.seed)
        for i in range(count):
            rec, dpath, mpath = rng.choice(pairs)
            depth = cv2.imread(str(dpath), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mpath))
            depth, mask = self._jitter(rng, depth, mask)
            classes = self._classes_in(mask, color_to_class)
            yield depth, mask, {"classes": classes or [rec.class_name],
                                "source": "depth_remix", "from": rec.id, "index": i}

    # ---- paired transforms ----

    def _jitter(self, rng: random.Random, depth: np.ndarray, mask: np.ndarray):
        h, w = depth.shape[:2]
        if rng.random() < 0.5:
            depth = cv2.flip(depth, 1)
            mask = cv2.flip(mask, 1)
        angle = rng.uniform(-self.max_rotate_deg, self.max_rotate_deg)
        scale = rng.uniform(*self.scale_range)
        tx = rng.uniform(-self.max_translate, self.max_translate) * w
        ty = rng.uniform(-self.max_translate, self.max_translate) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        # Depth: replicate the border (continues the floor); LINEAR keeps it smooth.
        depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        # Mask: NEAREST + black border — class colors must stay EXACT.
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return depth, mask

    @staticmethod
    def _classes_in(mask_bgr: np.ndarray, color_to_class: dict) -> list[str]:
        present = []
        for (r, g, b), name in color_to_class.items():
            if (np.all(mask_bgr == (b, g, r), axis=2)).any():
                present.append(name)
        return sorted(present)
