import argparse
import json
import multiprocessing
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


class VideoFrameExtractor:
    def __init__(self, output_dir, fps=2, quality=2, width=1280, max_workers=None):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.quality = quality
        self.width = width
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.results = []
        
        # Check for ffmpeg on initialization
        self.ffmpeg_path = self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Check if ffmpeg is available and return its path"""
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            print("\n❌ ERROR: ffmpeg not found!")
            print("\nPlease install ffmpeg:")
            print("  sudo apt update")
            print("  sudo apt install ffmpeg -y")
            print("\nAfter installation, run this script again.")
            sys.exit(1)
        
        # Also check ffprobe
        ffprobe_path = shutil.which('ffprobe')
        if ffprobe_path is None:
            print("\n⚠️  Warning: ffprobe not found (will affect progress estimation)")
        
        return ffmpeg_path
        
    def extract_single(self, video_path):
        """Extract frames from a single video"""
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Create video-specific output directory
        video_output = self.output_dir / video_name
        video_output.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Build optimized ffmpeg command with full path
        cmd = [
            self.ffmpeg_path,  # Use full path to ffmpeg
            "-y",
            "-hwaccel", "auto",
            "-i", str(video_path),
            # width<=0 → keep native resolution (no scaling)
            "-vf", (f"fps={self.fps},scale={self.width}:-2" if self.width and self.width > 0
                    else f"fps={self.fps}"),
            "-q:v", str(self.quality),
            "-preset", "fast",
            "-threads", "0",
            "-progress", "pipe:1",
            "-nostats",
            "-loglevel", "error",
            # source-prefixed so frames from different videos never collide when
            # collated into one folder (e.g. cam_20260617_091915_000123.jpg)
            str(video_output / f"{video_name}_%06d.jpg")
        ]
        
        # Get estimated frame count
        estimated_frames = self._estimate_frames(video_path)
        
        # Create progress bar
        desc = f"{video_name[:30]}{'...' if len(video_name) > 30 else ''}"
        pbar = tqdm(
            total=estimated_frames,
            desc=desc.ljust(35),
            unit='frames',
            position=None,
            leave=False
        )
        
        try:
            # Run ffmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Monitor progress
            frames_extracted = 0
            for line in process.stdout:
                if line.startswith("frame="):
                    try:
                        current = int(line.split('=')[1].strip())
                        pbar.update(current - frames_extracted)
                        frames_extracted = current
                    except Exception:
                        pass
            
            process.wait()
            
        except Exception as e:
            pbar.close()
            return {
                'video': str(video_path),
                'name': video_name,
                'frames': 0,
                'time': round(time.time() - start_time, 2),
                'success': False,
                'output': str(video_output),
                'error': str(e)
            }
        
        pbar.close()
        elapsed = time.time() - start_time
        
        # Count actual frames
        actual_frames = len(list(video_output.glob("*.jpg")))
        
        result = {
            'video': str(video_path),
            'name': video_name,
            'frames': actual_frames,
            'time': round(elapsed, 2),
            'success': process.returncode == 0,
            'output': str(video_output)
        }
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            result['error'] = stderr[:200] if stderr else "Unknown error"
        
        return result
    
    def _estimate_frames(self, video_path):
        """Try to estimate number of frames that will be extracted"""
        ffprobe_path = shutil.which('ffprobe')
        if not ffprobe_path:
            return None
            
        try:
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            if len(lines) >= 2 and lines[0].isdigit():
                total_frames = int(lines[0])
                fps_str = lines[1]
                
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps_val = num / den
                else:
                    fps_val = float(fps_str)
                
                return int(total_frames * self.fps / fps_val)
        except Exception:
            pass
        return None
    
    def process_many(self, video_paths):
        """Process multiple videos in parallel"""
        # Validate inputs
        valid_videos = []
        for path in video_paths:
            path = Path(path)
            if path.exists():
                valid_videos.append(path)
            else:
                print(f"⚠️  Skipping {path} (not found)")
        
        if not valid_videos:
            print("❌ No valid videos found!")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'timestamp': datetime.now().isoformat(),
            'fps': self.fps,
            'quality': self.quality,
            'width': self.width,
            'videos': [str(v) for v in valid_videos]
        }
        
        with open(self.output_dir / 'extraction_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*60}")
        print("🎬 Video Frame Extractor")
        print(f"{'='*60}")
        print(f"📁 Output: {self.output_dir.absolute()}")
        print(f"⚙️  Settings: {self.fps} fps, quality={self.quality}, width={self.width}px")
        print(f"🔄 Parallel workers: {self.max_workers}")
        print(f"📹 Videos: {len(valid_videos)}")
        print(f"🔧 ffmpeg: {self.ffmpeg_path}")
        print(f"{'='*60}\n")
        
        # Process videos in parallel
        start_total = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.extract_single, video): video 
                for video in valid_videos
            }
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    # Print result
                    status = "✅" if result['success'] else "❌"
                    error_msg = f" - {result.get('error', '')}" if not result['success'] else ""
                    print(f"{status} {result['name']}: {result['frames']} frames in {result['time']}s{error_msg}")
                    
                except Exception as e:
                    video = futures[future]
                    print(f"❌ Error processing {video.name}: {e}")
        
        # Final summary
        total_time = time.time() - start_total
        successful = sum(1 for r in self.results if r['success'])
        total_frames = sum(r['frames'] for r in self.results if r['success'])
        
        print(f"\n{'='*60}")
        print("📊 SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}/{len(valid_videos)}")
        print(f"📸 Total frames: {total_frames}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"⚡ Avg speed: {total_frames/total_time:.1f} frames/sec" if total_time > 0 else "⚡ Avg speed: N/A")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        
        # Save results
        with open(self.output_dir / 'extraction_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_videos': len(valid_videos),
                    'successful': successful,
                    'total_frames': total_frames,
                    'total_time': total_time
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📄 Results saved to: {self.output_dir}/extraction_results.json")

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract frames from a video (or every video in a folder). "
                    "Frames are written as <video_stem>_NNNNNN.jpg under "
                    "<output>/<video_stem>/.")
    ap.add_argument("video", help="a video file, or a directory of videos")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="frames per second to extract (default: 1)")
    ap.add_argument("--width", type=int, default=0,
                    help="scale to this width in px; 0 = keep native resolution (default)")
    ap.add_argument("--quality", type=int, default=2, help="JPEG quality, 2=best (default: 2)")
    ap.add_argument("-o", "--output", default=None,
                    help="output directory (default: alongside the video / inside the folder)")
    ap.add_argument("--workers", type=int, default=3, help="parallel workers for a folder")
    args = ap.parse_args(argv)

    src = Path(args.video).expanduser()
    if not src.exists():
        sys.exit(f"❌ not found: {src}")

    if src.is_dir():
        videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            sys.exit(f"❌ no videos ({'/'.join(sorted(VIDEO_EXTS))}) in {src}")
        out = args.output or str(src)
        VideoFrameExtractor(output_dir=out, fps=args.fps, quality=args.quality,
                            width=args.width, max_workers=args.workers).process_many(videos)
    else:
        out = args.output or str(src.parent)
        ex = VideoFrameExtractor(output_dir=out, fps=args.fps, quality=args.quality,
                                 width=args.width)
        res = ex.extract_single(src)
        if res.get("success"):
            print(f"✅ {res['frames']} frames in {res['time']}s → {res['output']}")
        else:
            sys.exit(f"❌ extraction failed: {res.get('error', 'unknown')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
