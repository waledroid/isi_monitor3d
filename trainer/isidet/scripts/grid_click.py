"""Drag-a-box labeler for the etagere bins: box -> crop -> 320 -> YOLO-seg dataset.

For each image, DRAG a rectangle around every visible bin (mouse down, drag,
release). Before dragging, pick the class with 1 / 2. Each box becomes:

  * a crop = the box + MARGIN, letterboxed to 320x320  -> dataset/images/<stem>_bNN.jpg
  * a YOLO-seg label: the box as a 4-point polygon in
    320-image coordinates, normalised                    -> dataset/labels/<stem>_bNN.txt
  * an overlay of the source image with all boxes drawn  -> mask_vis/<stem>.jpg
    (+ mask_vis/<stem>.boxes.json sidecar for resume / re-export)

Classes: 0 = empty_box, 1 = filled_box (dataset/classes.txt is written).

Keys:
  drag        add a box (class = current class)
  1 / 2       current class = empty_box / filled_box
  u           undo last box
  r           reset boxes on this image
  s / SPACE   save and next image
  n           skip image (nothing written)
  b           back to previous image
  q / ESC     quit

Images already present in mask_vis/ are skipped on start (resume-friendly);
pass --redo to revisit everything.  --export (no GUI) rebuilds dataset/ from
the sidecars.

Run with a GUI-capable OpenCV (the isi-train env; monitor3d's cv2 is headless):
  cd trainer/isidet
  /home/aatanda/miniforge3/envs/isi-train/bin/python scripts/grid_click.py \
      --input data/etagere/200_filtered
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["empty_box", "filled_box"]
CLASS_COLORS = [(0, 0, 255), (0, 200, 0)]  # BGR per class
OUT_SIZE = 320
PAD_COLOR = (114, 114, 114)
MARGIN = 0.08          # crop margin as a fraction of the box's w / h
MIN_DRAG_PX = 6        # smaller drags are ignored (accidental clicks)
WINDOW = "grid_click"


# --- geometry -----------------------------------------------------------------

def letterbox(img: np.ndarray, size: int = OUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize onto a size x size canvas -> (canvas, scale, dx, dy)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, r, x0, y0


def crop_rect(box: tuple, w: int, h: int) -> tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1) + MARGIN, clipped to the image."""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    return (max(int(x0 - mx), 0), max(int(y0 - my), 0),
            min(int(x1 + mx), w), min(int(y1 + my), h))


def export_box(img: np.ndarray, box: tuple, cls: int, name: str,
               img_dir: Path, lbl_dir: Path) -> bool:
    """One box -> 320 crop + YOLO-seg polygon label. Returns False if degenerate."""
    h, w = img.shape[:2]
    cx0, cy0, cx1, cy1 = crop_rect(box, w, h)
    if cx1 - cx0 < 10 or cy1 - cy0 < 10:
        return False
    canvas, r, dx, dy = letterbox(img[cy0:cy1, cx0:cx1])
    x0, y0, x1, y1 = box
    # box corners -> crop coords -> 320 coords -> normalised
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    pts = (pts - [cx0, cy0]) * r + [dx, dy]
    pts = np.clip(pts / OUT_SIZE, 0.0, 1.0)
    cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
    coords = " ".join(f"{v:.6f}" for v in pts.flatten())
    (lbl_dir / f"{name}.txt").write_text(f"{cls} {coords}\n")
    return True


def draw_boxes(img: np.ndarray, boxes: list[tuple], live: tuple | None = None) -> np.ndarray:
    out = img.copy()
    for k, (x0, y0, x1, y1, cls) in enumerate(boxes, 1):
        col = CLASS_COLORS[cls]
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
        cv2.putText(out, f"b{k:02d} {CLASSES[cls]}", (int(x0) + 3, int(y0) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    if live is not None:
        x0, y0, x1, y1, cls = live
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), CLASS_COLORS[cls], 1)
    return out


def export_image(img: np.ndarray, boxes: list[tuple], stem: str,
                 img_dir: Path, lbl_dir: Path) -> int:
    n = 0
    for k, (x0, y0, x1, y1, cls) in enumerate(boxes, 1):
        n += export_box(img, (x0, y0, x1, y1), cls, f"{stem}_b{k:02d}", img_dir, lbl_dir)
    return n


# --- UI -----------------------------------------------------------------------

def annotate_status(canvas: np.ndarray, name: str, pos: str, boxes: list, cls: int) -> None:
    lines = (f"{name}  [{pos}]",
             f"class: {CLASSES[cls]} (1/2)   boxes: {len(boxes)}" + ("   s=save" if boxes else ""),
             "drag=box  u=undo  r=reset  s=save  n=skip  b=back  q=quit")
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="folder of images")
    ap.add_argument("--out", type=Path, help="output root (default <input>/..)")
    ap.add_argument("--redo", action="store_true", help="also revisit images already in mask_vis")
    ap.add_argument("--export", action="store_true",
                    help="no GUI: rebuild dataset/ from mask_vis/*.boxes.json sidecars")
    ap.add_argument("--max-width", type=int, default=1280, help="display width cap")
    args = ap.parse_args()

    root = args.out or args.input.parent
    vis_dir = root / "mask_vis"
    img_dir, lbl_dir = root / "dataset" / "images", root / "dataset" / "labels"
    for d in (vis_dir, img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    files = sorted(p for p in args.input.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    if args.export:
        total = 0
        for f in files:
            side = vis_dir / f"{f.stem}.boxes.json"
            if not side.exists():
                continue
            boxes = [tuple(b) for b in json.loads(side.read_text())]
            total += export_image(cv2.imread(str(f)), boxes, f.stem, img_dir, lbl_dir)
        print(f"exported {total} crops -> {img_dir.parent}")
        return

    if not args.redo:
        files = [f for f in files if not (vis_dir / f.name).exists()]
    if not files:
        print("nothing to do (all images already in mask_vis/; use --redo to revisit)")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    state = {"scale": 1.0, "cls": 0, "drag": None, "boxes": []}

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        px, py = x / state["scale"], y / state["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (px, py, px, py)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            x0, y0, _, _ = state["drag"]
            state["drag"] = (x0, y0, px, py)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
            x0, y0, _, _ = state["drag"]
            state["drag"] = None
            bx0, bx1 = sorted((x0, px))
            by0, by1 = sorted((y0, py))
            if bx1 - bx0 >= MIN_DRAG_PX and by1 - by0 >= MIN_DRAG_PX:
                state["boxes"].append((bx0, by0, bx1, by1, state["cls"]))

    cv2.setMouseCallback(WINDOW, on_mouse)

    saved = skipped = 0
    i = 0
    while 0 <= i < len(files):
        f = files[i]
        img = cv2.imread(str(f))
        if img is None:
            i += 1
            continue
        h, w = img.shape[:2]
        state["scale"] = min(1.0, args.max_width / w)
        state["boxes"] = []
        state["drag"] = None
        # --redo: preload the previous boxes so they can be edited
        side = vis_dir / f"{f.stem}.boxes.json"
        if args.redo and side.exists():
            state["boxes"] = [tuple(b) for b in json.loads(side.read_text())]

        while True:
            boxes = state["boxes"]
            live = None
            if state["drag"] is not None:
                x0, y0, x1, y1 = state["drag"]
                live = (x0, y0, x1, y1, state["cls"])
            frame = draw_boxes(img, boxes, live)
            sc = state["scale"]
            canvas = cv2.resize(frame, None, fx=sc, fy=sc) if sc < 1.0 else frame
            annotate_status(canvas, f.name, f"{i + 1}/{len(files)}", boxes, state["cls"])
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"saved={saved} skipped={skipped}")
                return
            if key == ord("1"):
                state["cls"] = 0
            elif key == ord("2"):
                state["cls"] = 1
            elif key == ord("u") and boxes:
                boxes.pop()
            elif key == ord("r"):
                boxes.clear()
            elif key == ord("n"):
                skipped += 1
                i += 1
                break
            elif key == ord("b"):
                i = max(0, i - 1)
                break
            elif key in (ord("s"), ord(" ")) and boxes:
                # clip boxes to the image before persisting
                clean = [(max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1), c)
                         for x0, y0, x1, y1, c in boxes]
                cv2.imwrite(str(vis_dir / f.name), draw_boxes(img, clean))
                side.write_text(json.dumps(clean))
                n = export_image(img, clean, f.stem, img_dir, lbl_dir)
                counts = {c: sum(1 for b in clean if b[4] == k) for k, c in enumerate(CLASSES)}
                print(f"{f.name}: {n} crops {counts}")
                saved += 1
                i += 1
                break

    cv2.destroyAllWindows()
    print(f"saved={saved} skipped={skipped}")


if __name__ == "__main__":
    main()
