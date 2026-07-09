"""Perception/metric config fingerprint — drift detection across processes.

In Direction 1 the metric engine can no longer verify what it isn't shown:
a producer running a different model, stale zones, or a mismatched
calibration produces confidently-wrong geometry. Both sides compute this
fingerprint from the SAME backbone.yaml-derived facts; the producer stamps it
on every ``DetectionSetMessage`` and the engine warns (never drops) on
mismatch.

Deliberately coarse: paths + zones/calibration file mtimes, not content
hashes — cheap, and any real drift (retrain, redraw, recalibrate) touches an
mtime.
"""

from __future__ import annotations

import hashlib
import os


def config_fingerprint(cfg: dict) -> str:
    """12-hex-char fingerprint of the perception-relevant config facts."""
    det = cfg.get("detection", {})
    parts = [
        str(det.get("onnx_path", "")),
        str(det.get("pose_onnx_path", "")),
        str(cfg.get("calibration_path", "")),
        str(cfg.get("zones_path", "")),
    ]
    for path_key in ("calibration_path", "zones_path"):
        path = cfg.get(path_key)
        try:
            parts.append(str(int(os.path.getmtime(path))) if path else "-")
        except OSError:
            parts.append("-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
