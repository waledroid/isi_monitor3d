"""Copy-paste scaffolds — Phase 6, the "paste" half of paste-then-harmonize.

Cuts a real object (via its mask) from one curated record and pastes it onto
another real image at a **depth-aware** location + scale, then emits everything
the inpaint generator needs to harmonize it onto that REAL background:

  - control : composite **depth** (bg depth + the pasted object's depth) — the
              depth ControlNet still drives the object's geometry.
  - mask    : the LABEL — the background's own object mask PLUS the pasted region,
              class-colored (so every object in the frame is labeled, no false
              negatives).
  - meta.base    : the composite **RGB** (real background + pasted object pixels)
                   — the image the inpaint generator edits.
  - meta.inpaint : the (dilated) pasted region — the ONLY pixels the generator
                   regenerates; the real background stays pixel-exact.

The depth ControlNet path (depth_remix / box3d) repaints whole scenes; this path
keeps real backgrounds and only harmonizes the pasted object.
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


@SCAFFOLD_SOURCES.register("copy_paste")
class CopyPasteScaffolds(ScaffoldSource):
    def __init__(self, project_dir: str | None = None, seed: int | None = None,
                 scale_jitter: tuple = (0.8, 1.2), min_frac: float = 0.25,
                 max_frac: float = 0.85, depth_scale: bool = True,
                 avoid_overlap: bool = True, placement_tries: int = 30,
                 dilate: int = 9, **cfg) -> None:
        super().__init__(project_dir=project_dir, seed=seed, scale_jitter=scale_jitter,
                         min_frac=min_frac, max_frac=max_frac, depth_scale=depth_scale,
                         avoid_overlap=avoid_overlap, placement_tries=placement_tries,
                         dilate=dilate, **cfg)
        self.project_dir = project_dir            # injected by the runner
        self.seed = seed
        self.scale_jitter = tuple(scale_jitter)   # jitter around the object's REAL size
        self.min_frac = float(min_frac)           # min/max object height as frac of frame
        self.max_frac = float(max_frac)
        self.depth_scale = bool(depth_scale)
        self.avoid_overlap = bool(avoid_overlap)  # place clear of the bg's own object
        self.placement_tries = int(placement_tries)
        self.dilate = int(dilate)

    def generate(self, project: ProjectConfig, count: int
                 ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
        if not self.project_dir:
            raise ValueError("copy_paste needs project_dir (set by the runner)")
        pdir = Path(self.project_dir)
        manifest = Manifest.load(pdir)
        recs = [r for r in manifest.active()
                if r.image and r.mask and r.depth_map
                and not getattr(r, "synthetic", False)
                and (pdir / r.image).exists() and (pdir / r.mask).exists()
                and (pdir / r.depth_map).exists()]
        if not recs:
            raise ValueError("copy_paste: no records with image + mask + depth — "
                             "run phases 1-3 first")
        colors = {c.name: tuple(c.color) for c in project.classes}
        rng = random.Random(self.seed)
        for i in range(count):
            bg = rng.choice(recs)
            obj = rng.choice(recs)
            out = self._compose(pdir, bg, obj, colors, rng)
            if out is None:
                continue
            depth, label, base, inpaint, classes = out
            yield depth, label, {"classes": classes, "source": "copy_paste",
                                 "base": base, "inpaint": inpaint,
                                 "from_bg": bg.id, "from_obj": obj.id, "index": i}

    def _compose(self, pdir, bg, obj, colors, rng):
        bg_img = cv2.imread(str(pdir / bg.image))                  # BGR
        bg_depth = cv2.imread(str(pdir / bg.depth_map), cv2.IMREAD_GRAYSCALE)
        bg_mask = cv2.imread(str(pdir / bg.mask))                  # color label
        obj_img = cv2.imread(str(pdir / obj.image))
        obj_depth = cv2.imread(str(pdir / obj.depth_map), cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(str(pdir / obj.mask))
        if any(x is None for x in (bg_img, bg_depth, bg_mask, obj_img, obj_depth, obj_mask)):
            return None
        H, W = bg_img.shape[:2]

        # --- cut the object to its mask bbox ---
        obj_bin = obj_mask.any(axis=2)
        ys, xs = np.nonzero(obj_bin)
        if xs.size == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop_rgb = obj_img[y0:y1, x0:x1]
        crop_dep = obj_depth[y0:y1, x0:x1]
        crop_bin = obj_bin[y0:y1, x0:x1]
        bh, bw = crop_bin.shape

        # --- target size: anchor to the object's REAL frame-fraction (not an
        #     arbitrary fraction → no tiny pastes), jitter + clamp to [min,max] ---
        native_frac = bh / float(obj_img.shape[0])               # object height ÷ its frame
        frac = native_frac * rng.uniform(*self.scale_jitter)
        if self.depth_scale:                                     # mild near⇒bigger nudge
            cx0 = rng.randint(int(0.2 * W), int(0.8 * W))
            cy0 = rng.randint(int(0.4 * H), int(0.9 * H))
            frac *= 0.85 + 0.30 * (float(bg_depth[cy0, cx0]) / 255.0)
        frac = float(np.clip(frac, self.min_frac, self.max_frac))
        th = max(8, int(frac * H))
        tw = max(8, int(bw * th / bh))
        if tw >= W or th >= H:                                   # fit to frame
            f = min(W / tw, H / th) * 0.9
            tw, th = max(8, int(tw * f)), max(8, int(th * f))
        rgb_r = cv2.resize(crop_rgb, (tw, th), interpolation=cv2.INTER_AREA)
        dep_r = cv2.resize(crop_dep, (tw, th), interpolation=cv2.INTER_AREA)
        bin_r = cv2.resize(crop_bin.astype(np.uint8), (tw, th),
                           interpolation=cv2.INTER_NEAREST).astype(bool)

        # --- placement: pick a spot CLEAR of the bg's own object (clean doubles) ---
        bg_obj = bg_mask.any(axis=2)
        px, py = self._place(rng, W, H, tw, th, bin_r, bg_obj)

        # --- composite RGB + depth (paste real pixels where the object is) ---
        base = bg_img.copy()
        comp_depth = bg_depth.copy()
        base[py:py + th, px:px + tw][bin_r] = rgb_r[bin_r]
        comp_depth[py:py + th, px:px + tw][bin_r] = dep_r[bin_r]

        # --- pasted region on the full frame ---
        paste = np.zeros((H, W), dtype=bool)
        paste[py:py + th, px:px + tw][bin_r] = True

        # --- LABEL = background's own objects + the pasted object, class-colored ---
        label = bg_mask.copy()
        r, g, b = colors.get(obj.class_name, (255, 255, 255))
        label[paste] = (b, g, r)                                 # BGR

        # --- inpaint mask = dilated pasted region (harmonize just the seam/object) ---
        k = max(1, self.dilate)
        inpaint = cv2.dilate(paste.astype(np.uint8) * 255,
                             np.ones((k, k), np.uint8), iterations=1)

        present = sorted({obj.class_name} | self._classes_in(bg_mask, colors))
        return comp_depth, label, base, inpaint, present

    def _place(self, rng, W, H, tw, th, bin_r, bg_obj):
        """Top-left (px,py) for the paste. When avoid_overlap, try several random
        spots and keep the one whose object pixels least overlap the background's
        own object (→ clean, separate doubles); else just random/centered."""
        def at(cx, cy):
            return (int(np.clip(cx - tw // 2, 0, W - tw)),
                    int(np.clip(cy - th // 2, 0, H - th)))

        if not self.avoid_overlap or not bg_obj.any():
            return at(rng.randint(int(0.2 * W), int(0.8 * W)),
                      rng.randint(int(0.4 * H), int(0.9 * H)))
        best, best_ov = None, 1e9
        for _ in range(max(1, self.placement_tries)):
            px, py = at(rng.randint(tw // 2, W - tw // 2),
                        rng.randint(th // 2, H - th // 2))
            region = bg_obj[py:py + th, px:px + tw]
            ov = int(np.logical_and(region, bin_r).sum())         # object-on-object px
            if ov == 0:
                return px, py
            if ov < best_ov:
                best, best_ov = (px, py), ov
        return best

    @staticmethod
    def _classes_in(mask_bgr, colors) -> set[str]:
        out = set()
        for name, (r, g, b) in colors.items():
            if np.all(mask_bgr == (b, g, r), axis=2).any():
                out.add(name)
        return out
