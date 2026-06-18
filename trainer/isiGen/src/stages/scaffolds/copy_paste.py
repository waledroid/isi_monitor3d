"""Copy-paste scaffolds — Phase 6, the "paste" half of paste-then-harmonize.

Cuts one or more real objects (via their masks) from curated records and pastes
them onto a real **background** at a **depth-aware** location + scale, then emits
everything the inpaint generator needs to harmonize them onto that REAL background:

The preferred background is an **empty-scene** record (``background=True``, no
object) — pasting onto it can never overlap an existing object, so the label is
exact. If no background-only images were ingested, it falls back to pasting onto
an object image (placement then avoids that image's own object). ``paste_count``
(an int or ``[lo, hi]``) controls how many objects land per scene (clean doubles).


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
                 dilate: int = 9, paste_count: int | list | tuple = 1,
                 placement: str = "random", **cfg) -> None:
        super().__init__(project_dir=project_dir, seed=seed, scale_jitter=scale_jitter,
                         min_frac=min_frac, max_frac=max_frac, depth_scale=depth_scale,
                         avoid_overlap=avoid_overlap, placement_tries=placement_tries,
                         dilate=dilate, paste_count=paste_count, placement=placement, **cfg)
        self.project_dir = project_dir            # injected by the runner
        self.seed = seed
        self.scale_jitter = tuple(scale_jitter)   # jitter around the object's REAL size
        self.min_frac = float(min_frac)           # min/max object height as frac of frame
        self.max_frac = float(max_frac)
        self.depth_scale = bool(depth_scale)
        self.avoid_overlap = bool(avoid_overlap)  # place clear of existing objects
        self.placement_tries = int(placement_tries)
        self.dilate = int(dilate)
        # "original": paste each object at its SOURCE location + native scale (bg and
        # object frames share the fixed camera, so the real position transfers exactly
        # → on the belt, full, never cut). "random": random location + scale jitter.
        self.placement = str(placement)
        # objects pasted per scene: an int (exactly N) or [lo, hi] (random per scene)
        self.paste_count = (list(paste_count) if isinstance(paste_count, (list, tuple))
                            else int(paste_count))

    def generate(self, project: ProjectConfig, count: int
                 ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
        if not self.project_dir:
            raise ValueError("copy_paste needs project_dir (set by the runner)")
        pdir = Path(self.project_dir)
        manifest = Manifest.load(pdir)
        active = [r for r in manifest.active()
                  if not getattr(r, "synthetic", False)]

        # OBJECTS to cut & paste: real records with image + mask + depth.
        obj_pool = [r for r in active
                    if r.image and r.mask and r.depth_map and not r.background
                    and (pdir / r.image).exists() and (pdir / r.mask).exists()
                    and (pdir / r.depth_map).exists()]
        if not obj_pool:
            raise ValueError("copy_paste: no records with image + mask + depth — "
                             "run phases 1-3 first")
        # BACKGROUNDS to paste onto: empty-scene records (image + depth, no object).
        # Pasting onto these guarantees NO overlap with an existing object. If none
        # were ingested, fall back to pasting onto object images (least-overlap).
        bg_pool = [r for r in active
                   if r.background and r.image and r.depth_map
                   and (pdir / r.image).exists() and (pdir / r.depth_map).exists()]
        if not bg_pool:
            bg_pool = obj_pool

        colors = {c.name: tuple(c.color) for c in project.classes}
        rng = random.Random(self.seed)
        # Cycle a reshuffled background list so every background is reused ~evenly
        # (20 bgs over 500 mints → ~25 each), not lumpily random.
        bg_cycle = self._even_cycle(bg_pool, rng)
        for i in range(count):
            bg = next(bg_cycle)
            n = self._draw_count(rng)
            objs = [rng.choice(obj_pool) for _ in range(n)]
            out = self._compose(pdir, bg, objs, colors, rng)
            if out is None:
                continue
            depth, label, base, inpaint, classes = out
            yield depth, label, {"classes": classes, "source": "copy_paste",
                                 "base": base, "inpaint": inpaint,
                                 "from_bg": bg.id, "from_obj": [o.id for o in objs],
                                 "index": i}

    def _draw_count(self, rng) -> int:
        if isinstance(self.paste_count, list):
            lo, hi = int(self.paste_count[0]), int(self.paste_count[-1])
            return rng.randint(min(lo, hi), max(lo, hi))
        return max(1, int(self.paste_count))

    @staticmethod
    def _even_cycle(items, rng):
        """Yield items forever, reshuffling each pass → even reuse across a run."""
        while True:
            order = list(items)
            rng.shuffle(order)
            yield from order

    def _compose(self, pdir, bg, objs, colors, rng):
        bg_img = cv2.imread(str(pdir / bg.image))                  # BGR
        bg_depth = cv2.imread(str(pdir / bg.depth_map), cv2.IMREAD_GRAYSCALE)
        if bg_img is None or bg_depth is None:
            return None
        H, W = bg_img.shape[:2]
        # A true background has no object mask → start from a BLANK label; an
        # object-image fallback keeps its own objects in the label + avoid region.
        bg_mask = cv2.imread(str(pdir / bg.mask)) if bg.mask else None
        label = bg_mask.copy() if bg_mask is not None else np.zeros((H, W, 3), np.uint8)
        occupied = (bg_mask.any(axis=2) if bg_mask is not None
                    else np.zeros((H, W), dtype=bool))            # avoid region (grows)

        base = bg_img.copy()
        comp_depth = bg_depth.copy()
        paste_all = np.zeros((H, W), dtype=bool)
        pasted_classes: set[str] = set()
        for obj in objs:
            if self.placement == "original":
                # Paste at the object's real position + native size (same camera).
                prep = self._prep_object_original(pdir, obj, H, W)
                if prep is None:
                    continue
                rgb_r, dep_r, bin_r, px, py, tw, th = prep
            else:
                prep = self._prep_object(pdir, obj, H, W, bg_depth, rng)
                if prep is None:
                    continue
                rgb_r, dep_r, bin_r, tw, th = prep
                px, py = self._place(rng, W, H, tw, th, bin_r, occupied)
                if px is None:
                    continue
            region = (py, py + th, px, px + tw)
            y0, y1, x0, x1 = region
            base[y0:y1, x0:x1][bin_r] = rgb_r[bin_r]
            comp_depth[y0:y1, x0:x1][bin_r] = dep_r[bin_r]
            paste = np.zeros((H, W), dtype=bool)
            paste[y0:y1, x0:x1][bin_r] = True
            r, g, b = colors.get(obj.class_name, (255, 255, 255))
            label[paste] = (b, g, r)                              # BGR
            paste_all |= paste
            occupied |= paste                                     # next paste avoids this one
            pasted_classes.add(obj.class_name)

        if not paste_all.any():                                   # nothing placed
            return None

        # --- inpaint mask = dilated pasted region (harmonize just the seam/object) ---
        k = max(1, self.dilate)
        inpaint = cv2.dilate(paste_all.astype(np.uint8) * 255,
                             np.ones((k, k), np.uint8), iterations=1)

        bg_classes = self._classes_in(bg_mask, colors) if bg_mask is not None else set()
        present = sorted(pasted_classes | bg_classes)
        return comp_depth, label, base, inpaint, present

    def _prep_object(self, pdir, obj, H, W, bg_depth, rng):
        """Cut obj to its mask bbox + resize to a realistic size for this frame.
        Returns (rgb_r, dep_r, bin_r, tw, th) or None."""
        obj_img = cv2.imread(str(pdir / obj.image))
        obj_depth = cv2.imread(str(pdir / obj.depth_map), cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(str(pdir / obj.mask))
        if any(x is None for x in (obj_img, obj_depth, obj_mask)):
            return None
        obj_bin = obj_mask.any(axis=2)
        ys, xs = np.nonzero(obj_bin)
        if xs.size == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop_rgb = obj_img[y0:y1, x0:x1]
        crop_dep = obj_depth[y0:y1, x0:x1]
        crop_bin = obj_bin[y0:y1, x0:x1]
        bh, bw = crop_bin.shape

        # target size: anchor to the object's REAL frame-fraction (not an arbitrary
        # fraction → no tiny pastes), jitter + clamp to [min,max].
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
        return rgb_r, dep_r, bin_r, tw, th

    def _prep_object_original(self, pdir, obj, H, W):
        """Cut obj to its mask bbox at NATIVE scale and return it with the SOURCE
        top-left, so the object lands exactly where it was photographed.

        Returns (rgb, dep, binc, px, py, w, h) or None. Object and background come
        from the same fixed camera, so the position transfers directly; if their
        resolutions differ the bbox is scaled by the size ratio. The paste is
        clipped into the frame defensively (a source bbox is already in-frame)."""
        obj_img = cv2.imread(str(pdir / obj.image))
        obj_depth = cv2.imread(str(pdir / obj.depth_map), cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(str(pdir / obj.mask))
        if any(x is None for x in (obj_img, obj_depth, obj_mask)):
            return None
        oh, ow = obj_img.shape[:2]
        obj_bin = obj_mask.any(axis=2)
        ys, xs = np.nonzero(obj_bin)
        if xs.size == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgb = obj_img[y0:y1, x0:x1]
        dep = obj_depth[y0:y1, x0:x1]
        binc = obj_bin[y0:y1, x0:x1]
        bh, bw = binc.shape
        px, py = int(x0), int(y0)
        if (ow, oh) != (W, H):                       # different resolution → scale pos+size
            sx, sy = W / ow, H / oh
            bw, bh = max(1, round(bw * sx)), max(1, round(bh * sy))
            rgb = cv2.resize(rgb, (bw, bh), interpolation=cv2.INTER_AREA)
            dep = cv2.resize(dep, (bw, bh), interpolation=cv2.INTER_AREA)
            binc = cv2.resize(binc.astype(np.uint8), (bw, bh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            px, py = round(x0 * sx), round(y0 * sy)
        px = max(0, min(int(px), W - bw))
        py = max(0, min(int(py), H - bh))
        return rgb, dep, binc, px, py, bw, bh

    def _place(self, rng, W, H, tw, th, bin_r, occupied):
        """Top-left (px,py) for the paste. When avoid_overlap, try several random
        spots and keep the one whose object pixels least overlap already-occupied
        regions (the bg's own object + earlier pastes) → clean, separate doubles."""
        def at(cx, cy):
            return (int(np.clip(cx - tw // 2, 0, W - tw)),
                    int(np.clip(cy - th // 2, 0, H - th)))

        if not self.avoid_overlap or not occupied.any():
            return at(rng.randint(int(0.2 * W), int(0.8 * W)),
                      rng.randint(int(0.4 * H), int(0.9 * H)))
        best, best_ov = None, 1e9
        for _ in range(max(1, self.placement_tries)):
            px, py = at(rng.randint(tw // 2, W - tw // 2),
                        rng.randint(th // 2, H - th // 2))
            region = occupied[py:py + th, px:px + tw]
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
