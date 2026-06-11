"""Run a YOLO-pose ONNX over an image or folder and save annotated outputs.

The ONNX-side counterpart of run_test.sh: a quick sanity check that an exported
person-pose model produces sensible keypoints + foot nodes on real frames (e.g.
site captures from the RTSP camera). Draws the COCO skeleton and the yellow foot
node (ankle midpoint) the homography layer will project.

    conda activate isi-train
    python scripts/pose_infer.py --model path/to/yolo11n-pose.onnx \
        --source path/to/img_or_folder --out runs/pose_infer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ISIDET = Path(__file__).resolve().parents[1]
if str(_ISIDET) not in sys.path:
    sys.path.insert(0, str(_ISIDET))

from src.inference.pose_onnx_inferencer import (  # noqa: E402
    LEFT_ANKLE,
    RIGHT_ANKLE,
    PoseOnnxInferencer,
)

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in _IMG_EXT)
    return [source]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a YOLO-pose ONNX on image(s).")
    ap.add_argument("--model", required=True, help="path to the pose .onnx")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default="runs/pose_infer", help="output dir for annotated images")
    ap.add_argument("--conf", type=float, default=0.25, help="box confidence threshold")
    ap.add_argument("--device", default=None, help="'cpu' to force CPU (default: CUDA if available)")
    args = ap.parse_args()

    eng = PoseOnnxInferencer(args.model, conf_threshold=args.conf, device=args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _images(Path(args.source))
    if not images:
        print(f"no images under {args.source!r}")
        return 1

    total = 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  skip (unreadable): {img_path}")
            continue
        poses = eng.predict(frame)
        total += len(poses)
        ankle_ok = sum(
            1 for p in poses
            if p.keypoints[LEFT_ANKLE, 2] >= eng.kpt_conf or p.keypoints[RIGHT_ANKLE, 2] >= eng.kpt_conf
        )
        cv2.imwrite(str(out_dir / img_path.name), eng.draw(frame, poses))
        print(f"  {img_path.name}: {len(poses)} person(s), {ankle_ok} with a visible ankle")

    print(f"\n{len(images)} image(s), {total} person detection(s). Annotated → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
