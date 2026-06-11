"""Annotate a recorded video with pose (people) + pallet boxes → watchable MP4.

Runs two models per frame and burns the result into an output video:
  • person POSE — skeleton + keypoints, plus the yellow FOOT NODE (ankle midpoint)
    that the homography layer projects to the floor;
  • pallet DETECT — bounding box(es).

A quick "does it see what we want" check on real site footage. Uses Ultralytics
(GPU if available) — run in the isi-train env.

Examples
--------
    conda activate isi-train
    # first 60 s of one chunk → annotated.mp4 beside it
    python tools/visualize_video.py --video recordings/20260601_171448/cam_20260601_171450.mp4 --seconds 60
    # whole chunk, custom output
    python tools/visualize_video.py --video <chunk.mp4> --out viz.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

_REPO = Path(__file__).resolve().parents[1]
_ISIDET = _REPO / "trainer" / "isidet"
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def _default_pallet_model() -> str | None:
    runs = _ISIDET / "runs" / "detect"
    pts = list(runs.glob("**/weights/best.pt")) if runs.exists() else []
    return str(max(pts, key=lambda p: p.stat().st_mtime)) if pts else None


def _default_pose_model() -> str:
    local = _ISIDET / "yolo11n-pose.pt"
    return str(local) if local.exists() else "yolo11n-pose.pt"  # else ultralytics fetches


def main() -> int:
    ap = argparse.ArgumentParser(description="Annotate a video with pose + pallet boxes.")
    ap.add_argument("--video", required=True, help="input video (one recorded chunk)")
    ap.add_argument("--out", default=None, help="output mp4 (default: <video>.viz.mp4)")
    ap.add_argument("--pose-model", default=_default_pose_model())
    ap.add_argument("--pallet-model", default=_default_pallet_model())
    ap.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    ap.add_argument("--start", type=float, default=0.0, help="start at this second of the video")
    ap.add_argument("--seconds", type=float, default=0.0, help="duration from --start (0=to end)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}")
        return 1
    if not args.pallet_model:
        print("no pallet model found under trainer/isidet/runs/detect — pass --pallet-model")
        return 1
    out = Path(args.out) if args.out else video.with_suffix(".viz.mp4")

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"pose  : {args.pose_model}")
    print(f"pallet: {args.pallet_model}")
    print(f"device: {device}")
    pose_model = YOLO(args.pose_model)
    pallet_model = YOLO(args.pallet_model)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(fps * args.start) if args.start > 0 else 0
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = start_frame + int(fps * args.seconds) if args.seconds > 0 else n_total
    out_fps = fps / max(args.stride, 1)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    bar = tqdm(total=((limit - start_frame) // max(args.stride, 1)) or None,
               unit="frame", desc="annotating")
    idx = start_frame
    written = 0
    while True:
        if not cap.grab() or (limit and idx >= limit):
            break
        if idx % max(args.stride, 1) == 0:
            ok, frame = cap.retrieve()
            if ok:
                pr = pose_model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
                dr = pallet_model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
                ann = pr.plot()   # skeleton + keypoints + person boxes (BGR)
                # pallet boxes (orange) over the pose overlay
                if dr.boxes is not None:
                    for b, c in zip(dr.boxes.xyxy.cpu().numpy(),
                                    dr.boxes.conf.cpu().numpy(), strict=True):
                        x1, y1, x2, y2 = b.astype(int)
                        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 140, 255), 3)
                        cv2.putText(ann, f"pallet {c:.2f}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
                # foot node = ankle midpoint (what the homography projects)
                if pr.keypoints is not None and pr.keypoints.data.numel():
                    for person in pr.keypoints.data.cpu().numpy():   # [17,3]
                        vis = [person[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if person[i, 2] >= 0.3]
                        if vis:
                            fx, fy = np.mean([k[:2] for k in vis], axis=0).astype(int)
                            cv2.circle(ann, (int(fx), int(fy)), 7, (0, 255, 255), -1)
                writer.write(ann)
                written += 1
                bar.update(1)
        idx += 1
    bar.close()
    cap.release()
    writer.release()
    print(f"\nwrote {written} annotated frames → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
