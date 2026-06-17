"""Merge every MP4 in a directory into one MP4.

Point it at a folder of `.mp4` files; it first **verifies** each is a readable
MP4 (right container + a video stream) and, if any file is bad / not MP4, prints
exactly which ones and stops without writing anything. When all are valid it
concatenates them in filename order:

  * **stream-copy** (`-c copy`) when every clip shares the same codec + resolution
    — fast and lossless (the usual case for footage from one camera);
  * **re-encode** (libx264, scaled/padded to a common size, video-only) when the
    clips differ, or with ``--reencode`` — slower but always produces a clean file.

Usage:
    python scripts/merge_mp4.py /path/to/clips                 # → /path/to/clips/merged.mp4
    python scripts/merge_mp4.py /path/to/clips -o all.mp4
    python scripts/merge_mp4.py /path/to/clips --reencode      # force re-encode

Only the Python stdlib + ffmpeg/ffprobe are needed (no project deps).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MP4_SUFFIXES = {".mp4", ".m4v", ".mov"}


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        sys.exit(f"❌ {tool} not found. Install ffmpeg:  sudo apt install ffmpeg -y")
    return path


def probe(ffprobe: str, path: Path) -> dict:
    """Return {ok, format_name, vcodec, width, height, fps} or {ok: False, error}."""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=format_name",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "error": (exc.stderr or "ffprobe failed").strip().splitlines()[-1]}
    data = json.loads(out or "{}")
    fmt = (data.get("format") or {}).get("format_name", "")
    vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if "mp4" not in fmt:
        return {"ok": False, "error": f"not an MP4 container (format: {fmt or 'unknown'})"}
    if vstream is None:
        return {"ok": False, "error": "no video stream"}
    num, _, den = vstream.get("r_frame_rate", "0/1").partition("/")
    fps = (float(num) / float(den)) if den and float(den) else 0.0
    return {"ok": True, "format_name": fmt, "vcodec": vstream.get("codec_name"),
            "width": int(vstream.get("width", 0)), "height": int(vstream.get("height", 0)),
            "fps": fps}


def find_clips(directory: Path, output: Path) -> list[Path]:
    """MP4-ish files in `directory` (sorted), excluding the output file itself."""
    out_resolved = output.resolve()
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in MP4_SUFFIXES and p.resolve() != out_resolved
    )


def merge_copy(ffmpeg: str, clips: list[Path], output: Path) -> None:
    """Lossless concat via the concat demuxer (requires uniform codec/params)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for c in clips:
            fh.write(f"file '{c.resolve().as_posix()}'\n")
        list_path = fh.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", "-movflags", "+faststart", str(output)],
            check=True,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


def merge_reencode(ffmpeg: str, clips: list[Path], infos: list[dict], output: Path) -> None:
    """Robust concat for mismatched clips: scale/pad each to a common size +
    fps, then concat (video-only — audio is dropped in this path)."""
    w = max(i["width"] for i in infos)
    h = max(i["height"] for i in infos)
    fps = next((i["fps"] for i in infos if i["fps"] > 0), 25.0)
    cmd = [ffmpeg, "-y"]
    for c in clips:
        cmd += ["-i", str(c)]
    parts, labels = [], []
    for idx in range(len(clips)):
        parts.append(
            f"[{idx}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:g}[v{idx}]"
        )
        labels.append(f"[v{idx}]")
    filtergraph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(clips)}:v=1:a=0[v]"
    cmd += ["-filter_complex", filtergraph, "-map", "[v]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="folder containing the .mp4 files")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output path (default: <directory>/merged.mp4)")
    parser.add_argument("--reencode", action="store_true",
                        help="force a re-encode even if the clips look uniform")
    args = parser.parse_args(argv)

    ffmpeg, ffprobe = _require("ffmpeg"), _require("ffprobe")
    directory = args.directory
    if not directory.is_dir():
        sys.exit(f"❌ not a directory: {directory}")
    output = args.output or (directory / "merged.mp4")

    clips = find_clips(directory, output)
    if not clips:
        sys.exit(f"❌ no .mp4 files found in {directory}")

    # ---- verify every clip is a readable MP4 ----
    print(f"🔎 verifying {len(clips)} file(s) in {directory}")
    infos, bad = [], []
    for c in clips:
        info = probe(ffprobe, c)
        if info["ok"]:
            infos.append(info)
            print(f"   ✅ {c.name}  ({info['width']}x{info['height']}, "
                  f"{info['vcodec']}, {info['fps']:g} fps)")
        else:
            bad.append((c, info["error"]))
            print(f"   ❌ {c.name}  — {info['error']}")
    if bad:
        print(f"\n❌ {len(bad)} bad/non-MP4 file(s); nothing merged. Fix or remove:")
        for c, reason in bad:
            print(f"   • {c.name}: {reason}")
        return 1
    if len(clips) == 1:
        print(f"\nonly one valid MP4 ({clips[0].name}) — nothing to merge.")
        return 0

    # ---- merge ----
    uniform = len({(i["vcodec"], i["width"], i["height"]) for i in infos}) == 1
    output.parent.mkdir(parents=True, exist_ok=True)
    if uniform and not args.reencode:
        print(f"\n🔗 merging {len(clips)} clips (stream-copy, lossless) → {output}")
        merge_copy(ffmpeg, clips, output)
    else:
        why = "forced" if args.reencode else "clips differ in codec/resolution"
        print(f"\n🔁 merging {len(clips)} clips (re-encode, {why}; video-only) → {output}")
        merge_reencode(ffmpeg, clips, infos, output)

    res = probe(ffprobe, output)
    if not res["ok"]:
        sys.exit(f"❌ merge produced an unreadable file: {res['error']}")
    size_mb = output.stat().st_size / 1e6
    print(f"\n✅ wrote {output}  ({res['width']}x{res['height']}, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
