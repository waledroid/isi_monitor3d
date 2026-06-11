"""Extract frames containing people from a recording, ready for annotation.

Samples a video (or a folder of recorded chunks) at a target FPS, runs the
person-pose model on each sampled frame, and saves only the frames where a
person is found — turning a 2-hour capture into a compact, person-only folder.

Built on the exported YOLO-pose ONNX + the verified ``PoseOnnxInferencer``, so it
needs only onnxruntime + OpenCV (runs in the ``monitor3d`` env — no torch). Because
it uses the pose model, ``--require-ankle`` additionally keeps only frames where an
ankle is visible — a direct check of whether your camera angle yields the foot
nodes the homography needs.

Examples
--------
    conda activate monitor3d
    # whole recording session (folder of chunks) → person frames at 1 fps
    python tools/extract_person_frames.py --source recordings/20260601_171448 --out person_frames
    # only frames with a visible ankle, 2 fps, save annotated previews too
    python tools/extract_person_frames.py --source rec.mp4 --out out --fps 2 --require-ankle --annotate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parents[1]
# PoseOnnxInferencer lives in the trainer package.
_ISIDET = _REPO / "trainer" / "isidet"
if str(_ISIDET) not in sys.path:
    sys.path.insert(0, str(_ISIDET))

from src.inference.pose_onnx_inferencer import (  # noqa: E402
    LEFT_ANKLE,
    RIGHT_ANKLE,
    PoseOnnxInferencer,
)

_VID_EXT = {".mp4", ".mkv", ".mov", ".avi", ".m4v"}


def _find_pose_onnx() -> str | None:
    """Newest pose ``*.onnx`` (path contains "pose") under trainer runs / models."""
    roots = [_ISIDET, _REPO / "models"]
    files = [
        p for root in roots if root.exists()
        for p in root.glob("**/*.onnx")
        if p.is_file() and "pose" in str(p).lower()
    ]
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime))


def _videos(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in _VID_EXT)
    return [source] if source.suffix.lower() in _VID_EXT else []


def _has_visible_ankle(pose, kpt_conf: float) -> bool:
    return bool(pose.keypoints[LEFT_ANKLE, 2] >= kpt_conf
                or pose.keypoints[RIGHT_ANKLE, 2] >= kpt_conf)


class _PlainBar:
    """Minimal tqdm fallback (prints a % line every ~2 s) when tqdm is absent."""

    def __init__(self, total: int) -> None:
        self.total, self.n, self._post = total, 0, ""

    def update(self, k: int = 1) -> None:
        self.n += k
        if self.total and self.n % 200 == 0:
            pct = 100 * self.n // self.total
            print(f"  {pct:3d}%  {self.n}/{self.total} frames  {self._post}", flush=True)

    def set_postfix_str(self, s: str) -> None:
        self._post = s

    def write(self, s: str) -> None:
        print(s, flush=True)

    def close(self) -> None:
        pass


def _make_bar(total: int):
    try:
        from tqdm import tqdm

        return tqdm(total=total or None, unit="frame", dynamic_ncols=True,
                    desc="scanning")
    except ImportError:
        return _PlainBar(total)


def _plan_video(vid: Path, target_fps: float) -> tuple[int, int]:
    """Return (stride, estimated_sampled_frames) for a video — a cheap metadata
    read (no decode) so the progress bar gets a total to count toward."""
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return 0, 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    stride = max(1, round(src_fps / max(target_fps, 1e-6)))
    return stride, (nframes // stride if nframes > 0 else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract person frames from a recording.")
    ap.add_argument("--source", required=True, help="video file OR folder of recorded chunks")
    ap.add_argument("--out", default="person_frames", help="output dir for kept frames")
    ap.add_argument("--model", default=None,
                    help="pose .onnx (default: newest pose model under trainer/)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames to sample per second")
    ap.add_argument("--conf", type=float, default=0.4, help="person confidence threshold")
    ap.add_argument("--require-ankle", action="store_true",
                    help="keep only frames with a visible ankle (foot-node validation)")
    ap.add_argument("--annotate", action="store_true",
                    help="also save annotated previews to <out>/_preview/")
    ap.add_argument("--max", type=int, default=0, help="stop after N kept frames (0 = no limit)")
    ap.add_argument("--device", default=None, help="'cpu' to force CPU")
    args = ap.parse_args()

    model = args.model or _find_pose_onnx()
    if not model:
        print("No pose .onnx found. Export one (configs/train_pose.yaml) or pass --model.",
              file=sys.stderr)
        return 1

    videos = _videos(Path(args.source))
    if not videos:
        print(f"No videos found at {args.source!r}", file=sys.stderr)
        return 1

    eng = PoseOnnxInferencer(model, conf_threshold=args.conf, device=args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "_preview"
    if args.annotate:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={model}")
    print(f"{len(videos)} video(s) | sample={args.fps} fps | conf={args.conf}"
          f"{' | require-ankle' if args.require_ankle else ''} → {out_dir}")

    # Plan first (cheap metadata read) so the bar has a total to count toward.
    plans = [(vid, *_plan_video(vid, args.fps)) for vid in videos]
    total_est = sum(est for _, _, est in plans)
    bar = _make_bar(total_est)

    sampled = kept = 0
    try:
        for vid, stride, _est in plans:
            if stride == 0:
                bar.write(f"  skip (unreadable): {vid}")
                continue
            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                bar.write(f"  skip (unreadable): {vid}")
                continue
            idx = vid_kept = 0
            # grab() is cheap (no decode); only retrieve() (decode) sampled frames.
            while True:
                if not cap.grab():
                    break
                if idx % stride == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        sampled += 1
                        poses = eng.predict(frame)
                        if args.require_ankle:
                            poses = [p for p in poses if _has_visible_ankle(p, eng.kpt_conf)]
                        if poses:
                            name = f"{vid.stem}_f{idx:07d}.jpg"
                            cv2.imwrite(str(out_dir / name), frame)
                            if args.annotate:
                                cv2.imwrite(str(preview_dir / name), eng.draw(frame, poses))
                            kept += 1
                            vid_kept += 1
                        bar.update(1)
                        bar.set_postfix_str(f"kept {kept}")
                        if args.max and kept >= args.max:
                            cap.release()
                            bar.write(f"reached --max {args.max}; stopping.")
                            _summary(sampled, kept, out_dir)
                            return 0
                idx += 1
            cap.release()
            bar.write(f"  {vid.name}: kept {vid_kept} person frame(s)")
    finally:
        bar.close()

    _summary(sampled, kept, out_dir)
    return 0


def _summary(sampled: int, kept: int, out_dir: Path) -> None:
    print(f"\nscanned {sampled} sampled frame(s) → kept {kept} with people. → {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
