"""Post-processing for the etagere bin dataset built by grid_click.py.

  bg       add background crops (regions of the annotated frames that contain
           no drawn box) as images with EMPTY label files -> teaches "no bin".
  augment  offline augmentation of every dataset image (bins + backgrounds):
             _fx  horizontal flip (polygon x -> 1 - x)
             _bc  brightness / contrast jitter          (labels copied)
             _bl  gaussian blur (handheld motion blur)   (labels copied)
           Originals are kept; augmented files carry the suffix so a later
           split can group every variant of a frame into the same split.

  split    train/val split of dataset/ into dataset_split/ (copies):
             - grouped by SOURCE FRAME (all crops / backgrounds / augmented
               variants of a frame land on the same side -> no leakage),
             - stratified by source video (each video contributes ~val-frac),
             - val keeps ORIGINALS ONLY (augmented variants of val frames are
               dropped, so val metrics reflect real images),
             - writes dataset_split/data.yaml for Ultralytics,
             - --task detect (default) converts the 4-point box polygons to
               YOLO detection labels "cls cx cy w h" (exact: the polygons are
               axis-aligned rectangles); --task segment keeps the polygons.

Usage (from trainer/isidet):
  python scripts/etagere_dataset.py bg      --root data/etagere --frames data/etagere/200_filtered
  python scripts/etagere_dataset.py augment --root data/etagere
  python scripts/etagere_dataset.py split   --root data/etagere --val-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

OUT_SIZE = 320
PAD_COLOR = (114, 114, 114)
BG_PER_FRAME = 1
BG_TARGET_FRAC = 0.10        # backgrounds ~10% of the object crops
BG_MAX_OVERLAP = 0.05        # bg crop may overlap drawn boxes by at most 5% of its area
BG_SIZE_FRAC = (0.15, 0.35)  # bg crop w/h as fraction of frame w/h


def letterbox(img: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


# --- bg ---------------------------------------------------------------------

def _overlap_frac(rect: tuple, boxes: list) -> float:
    x0, y0, x1, y1 = rect
    area = (x1 - x0) * (y1 - y0)
    inter = 0.0
    for bx0, by0, bx1, by1, _ in boxes:
        iw = max(0.0, min(x1, bx1) - max(x0, bx0))
        ih = max(0.0, min(y1, by1) - max(y0, by0))
        inter += iw * ih
    return inter / max(area, 1e-6)


def cmd_bg(root: Path, frames_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    img_dir, lbl_dir = root / "dataset" / "images", root / "dataset" / "labels"
    n_obj = len([p for p in img_dir.glob("*.jpg") if "_bg" not in p.stem and "_b" in p.stem])
    target = max(1, int(n_obj * BG_TARGET_FRAC))
    sidecars = sorted((root / "mask_vis").glob("*.boxes.json"))
    rng.shuffle(sidecars)
    written = 0
    for side in sidecars:
        if written >= target:
            break
        stem = side.name[: -len(".boxes.json")]
        src = frames_dir / f"{stem}.jpg"
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        boxes = [tuple(b) for b in json.loads(side.read_text())]
        made = 0
        for _ in range(60):  # rejection sampling
            cw = int(w * rng.uniform(*BG_SIZE_FRAC))
            ch = int(h * rng.uniform(*BG_SIZE_FRAC))
            x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
            rect = (x0, y0, x0 + cw, y0 + ch)
            if _overlap_frac(rect, boxes) > BG_MAX_OVERLAP:
                continue
            made += 1
            name = f"{stem}_bg{made:02d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), letterbox(img[y0:y0 + ch, x0:x0 + cw]))
            (lbl_dir / f"{name}.txt").write_text("")
            written += 1
            if made >= BG_PER_FRAME or written >= target:
                break
    print(f"backgrounds written: {written} (objects: {n_obj}, target {BG_TARGET_FRAC:.0%})")


# --- augment ----------------------------------------------------------------

def _read_label(p: Path) -> list[tuple[int, np.ndarray]]:
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        out.append((int(parts[0]), np.array(parts[1:], dtype=np.float64).reshape(-1, 2)))
    return out


def _write_label(p: Path, shapes: list[tuple[int, np.ndarray]]) -> None:
    lines = [f"{c} " + " ".join(f"{v:.6f}" for v in pts.flatten()) for c, pts in shapes]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))


def aug_flip_x(img, shapes):
    return cv2.flip(img, 1), [(c, np.column_stack([1.0 - pts[:, 0], pts[:, 1]])) for c, pts in shapes]


def aug_bright_contrast(img, shapes, rng):
    alpha, beta = rng.uniform(0.7, 1.3), rng.uniform(-30, 30)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta), shapes


def aug_blur(img, shapes, rng):
    k = rng.choice((3, 5))
    return cv2.GaussianBlur(img, (k, k), 0), shapes


AUGS = ("fx", "bc", "bl")


def cmd_augment(root: Path, seed: int) -> None:
    rng = random.Random(seed)
    img_dir, lbl_dir = root / "dataset" / "images", root / "dataset" / "labels"
    originals = [p for p in sorted(img_dir.glob("*.jpg"))
                 if not any(p.stem.endswith(f"_{a}") for a in AUGS)]
    n = 0
    for p in originals:
        img = cv2.imread(str(p))
        lbl = lbl_dir / f"{p.stem}.txt"
        shapes = _read_label(lbl) if lbl.exists() else []
        variants = {
            "fx": aug_flip_x(img, shapes),
            "bc": aug_bright_contrast(img, shapes, rng),
            "bl": aug_blur(img, shapes, rng),
        }
        for suffix, (aimg, ashapes) in variants.items():
            cv2.imwrite(str(img_dir / f"{p.stem}_{suffix}.jpg"), aimg)
            _write_label(lbl_dir / f"{p.stem}_{suffix}.txt", ashapes)
            n += 1
    print(f"augmented: {len(originals)} originals -> +{n} variants "
          f"(total images {len(list(img_dir.glob('*.jpg')))})")


# --- split ------------------------------------------------------------------

def _frame_of(stem: str) -> str:
    """'vid596_0334_b03_fx' / 'vid596_0334_bg01' -> 'vid596_0334'."""
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if i and (part.startswith("bg") or part.startswith("b")) and part[1:].lstrip("g").isdigit():
            return "_".join(parts[:i])
    return stem


def _is_augmented(stem: str) -> bool:
    return any(stem.endswith(f"_{a}") for a in AUGS)


def _to_detect(text: str) -> str:
    """Polygon lines 'cls x1 y1 ... xn yn' -> 'cls cx cy w h' (normalised)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:          # already a box
            out.append(line)
            continue
        if len(parts) < 7:
            continue
        pts = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        out.append(f"{parts[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} "
                   f"{x1 - x0:.6f} {y1 - y0:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def cmd_split(root: Path, val_frac: float, seed: int, task: str = "detect") -> None:
    import shutil
    rng = random.Random(seed)
    src_img, src_lbl = root / "dataset" / "images", root / "dataset" / "labels"
    out = root / "dataset_split"
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    files = sorted(src_img.glob("*.jpg"))
    frames: dict[str, list[Path]] = {}
    for f in files:
        frames.setdefault(_frame_of(f.stem), []).append(f)

    # stratify by video prefix (vid584 / vid585 / still ...)
    by_video: dict[str, list[str]] = {}
    for fr in frames:
        by_video.setdefault(fr.split("_")[0], []).append(fr)
    val_frames: set[str] = set()
    for frs in by_video.values():
        frs = sorted(frs)
        rng.shuffle(frs)
        n_val = max(1, round(len(frs) * val_frac)) if len(frs) > 1 else 0
        val_frames.update(frs[:n_val])

    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}  # per split: empty, filled, background
    n_img = {"train": 0, "val": 0}
    for fr, fl in frames.items():
        split = "val" if fr in val_frames else "train"
        for f in fl:
            if split == "val" and _is_augmented(f.stem):
                continue
            lbl = src_lbl / f"{f.stem}.txt"
            shutil.copy2(f, out / "images" / split / f.name)
            text = lbl.read_text() if lbl.exists() else ""
            if task == "detect":
                text = _to_detect(text)
            (out / "labels" / split / f"{f.stem}.txt").write_text(text)
            n_img[split] += 1
            if not text.strip():
                counts[split][2] += 1
            for line in text.splitlines():
                counts[split][int(line.split()[0])] += 1

    names = (root / "dataset" / "classes.txt").read_text().split()
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )
    print(f"task: {task}   frames: {len(frames)} (val {len(val_frames)})   val-frac {val_frac}")
    for split in ("train", "val"):
        e, f_, bgn = counts[split]
        print(f"  {split:5s} images={n_img[split]:5d}  empty_box={e:5d}  "
              f"filled_box={f_:5d}  background={bgn:4d}")
    print(f"data.yaml -> {out / 'data.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("bg", "augment", "split"))
    ap.add_argument("--root", type=Path, default=Path("data/etagere"))
    ap.add_argument("--frames", type=Path, help="(bg) folder of the annotated source frames")
    ap.add_argument("--val-frac", type=float, default=0.15, help="(split) fraction of frames")
    ap.add_argument("--task", choices=("detect", "segment"), default="detect",
                    help="(split) label format to emit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "bg":
        cmd_bg(args.root, args.frames or args.root / "200_filtered", args.seed)
    elif args.cmd == "augment":
        cmd_augment(args.root, args.seed)
    else:
        cmd_split(args.root, args.val_frac, args.seed, args.task)


if __name__ == "__main__":
    main()
