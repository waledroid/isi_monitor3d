"""Live stream-sync probe — measures FPS + inter-camera timing (NOT a Multical output).

Opens the configured cameras for a few seconds, collects each decoded frame's
``capture_ts`` (the same clock the Backbone's FrameSynchronizer pairs on), and
reports per-camera FPS/jitter + the inter-camera skew (how far apart the nearest
frames from the two cameras land in time). This is a *streaming* health check —
the calibration solve says nothing about lag or sync; this does.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .session import _open_source

_SYNC_WINDOW_MS = 33.0     # the Backbone FrameSynchronizer's default max_skew_ms


def _collect(cam_spec, camera_id: str, seconds: float, out: dict, source_factory) -> None:
    ts: list[float] = []
    try:
        source = source_factory(cam_spec, camera_id)
        source.start()
    except Exception as exc:
        out[camera_id] = {"error": str(exc), "ts": []}
        return
    deadline = time.time() + seconds
    try:
        for frame in source.frames():
            ts.append(float(frame.capture_ts))
            if time.time() >= deadline:
                break
    except Exception:
        pass
    finally:
        try:
            source.stop()
        except Exception:
            pass
    out[camera_id] = {"ts": ts}


def _per_camera(ts: list[float], seconds: float) -> dict:
    if len(ts) < 2:
        return {"frames": len(ts), "fps": round(len(ts) / seconds, 2),
                "mean_interval_ms": None, "jitter_ms": None}
    arr = np.asarray(sorted(ts))
    iv = np.diff(arr) * 1000.0
    return {"frames": len(ts), "fps": round(len(ts) / seconds, 2),
            "mean_interval_ms": round(float(iv.mean()), 1),
            "jitter_ms": round(float(iv.std()), 1)}


def _skew(a: list[float], b: list[float]) -> dict:
    """Nearest-frame skew between two capture-ts streams (ms): mean/p95/max + %in-window."""
    if len(a) < 2 or len(b) < 2:
        return {"pairs": 0}
    a_arr, b_arr = np.asarray(sorted(a)), np.asarray(sorted(b))
    idx = np.searchsorted(b_arr, a_arr)
    idx = np.clip(idx, 1, len(b_arr) - 1)
    left, right = b_arr[idx - 1], b_arr[idx]
    nearest = np.where(np.abs(a_arr - left) <= np.abs(a_arr - right), left, right)
    skew_ms = np.abs(a_arr - nearest) * 1000.0
    return {"pairs": len(skew_ms),
            "mean_skew_ms": round(float(skew_ms.mean()), 1),
            "p95_skew_ms": round(float(np.percentile(skew_ms, 95)), 1),
            "max_skew_ms": round(float(skew_ms.max()), 1),
            "in_window_pct": round(100.0 * float((skew_ms <= _SYNC_WINDOW_MS).mean()), 1),
            "window_ms": _SYNC_WINDOW_MS}


def probe_streams(project_dir: Path, cfg, *, seconds: float = 4.0,
                  source_factory=_open_source) -> dict:
    """Probe all configured cameras for ``seconds`` → FPS + inter-camera sync stats."""
    cams = cfg.configured_cameras()
    if not cams:
        raise ValueError("no cameras configured")
    seconds = max(1.0, min(float(seconds), 15.0))
    out: dict = {}
    threads = [threading.Thread(target=_collect,
                                args=(cfg.cameras[c], c, seconds, out, source_factory))
               for c in cams]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 8.0)

    cameras = {}
    for c in cams:
        rec = out.get(c, {"ts": []})
        if "error" in rec:
            cameras[c] = {"error": rec["error"]}
        else:
            cameras[c] = _per_camera(rec["ts"], seconds)
    result = {"seconds": seconds, "cameras": cameras}
    if len(cams) == 2:
        a, b = cams
        ta, tb = out.get(a, {}).get("ts", []), out.get(b, {}).get("ts", [])
        result["sync"] = {"pair": f"{a}↔{b}", **_skew(ta, tb)}
    return result
