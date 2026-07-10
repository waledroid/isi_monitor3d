"""Mtime-cached YAML reads for the HOT display path.

``_ui_pref`` used to ``yaml.safe_load`` the whole UI-settings file on EVERY
call, and the video path asks it 4-6 questions PER FRAME PER STREAM
(nodes/masks/boxes/distances/occupancy/fps). Measured on the rig: 18.5 ms per
parse of the 299-line file ⇒ ~370% of a core of **GIL-held** work at 25 fps x
2 streams. The asyncio loop never got scheduled: every endpoint hung, Settings
saves failed with "Failed to fetch", the frame-bus poll lagged past its
staleness window and flapped.

This module parses a YAML file only when its (mtime_ns, size) changes — a
single ``os.stat`` per call otherwise — and uses libyaml's C loader when
available (~20x faster than the pure-Python one). Writers are atomic
(tempfile + os.replace), so an mtime change always means a complete new file.
"""

from __future__ import annotations

import os
import threading
import time

import yaml

try:                                     # libyaml — present in the conda env
    from yaml import CSafeLoader as _Loader
except ImportError:                      # pragma: no cover
    from yaml import SafeLoader as _Loader

_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int, int], dict]] = {}

# A cached parse is trusted only once the file has been quiet this long.
# Filesystem mtime granularity can be coarser than the edits (WSL/drvfs), and
# an in-place rewrite of the same byte-length would otherwise be invisible:
# the operator's Settings save would be silently ignored. Files change
# rarely, so this costs a re-parse only in the second after a write.
_SETTLE_S = 1.0


def load_yaml_cached(path) -> dict:
    """Parsed YAML mapping for ``path`` ({} when missing/unreadable/not a map).

    Re-parses when the file's (mtime_ns, size, inode) changed, or while the
    file is younger than ``_SETTLE_S`` (see above).
    """
    key = str(path)
    try:
        st = os.stat(key)
    except OSError:
        with _lock:
            _cache.pop(key, None)
        return {}
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    settled = (time.time() - st.st_mtime) > _SETTLE_S
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == stamp and settled:
            return hit[1]
    try:
        with open(key) as fh:
            data = yaml.load(fh, Loader=_Loader) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    with _lock:
        _cache[key] = (stamp, data)
    return data


def invalidate(path=None) -> None:
    """Drop cached parses (all, or one path). Writers don't need this — the
    mtime check catches them — it exists for tests and hot-reload paths."""
    with _lock:
        if path is None:
            _cache.clear()
        else:
            _cache.pop(str(path), None)
