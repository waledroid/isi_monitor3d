"""Per-shot metadata sidecar written next to each captured jpg."""

from __future__ import annotations

import json

from isical.capture.detect import Detection
from isical.capture.session import _write_shot_meta


def test_write_shot_meta_writes_sidecar(tmp_path):
    jpg = tmp_path / "cam_a_000.jpg"
    jpg.write_bytes(b"not-a-real-jpg")
    _write_shot_meta(jpg, Detection(n=18, centroid=(0.4, 0.6), blur_var=123.4))
    meta = json.loads((tmp_path / "cam_a_000.json").read_text())
    assert meta == {"corners": 18, "centroid": [0.4, 0.6], "blur_var": 123.4}


def test_write_shot_meta_handles_no_board(tmp_path):
    jpg = tmp_path / "cam_a_001.jpg"
    jpg.write_bytes(b"x")
    _write_shot_meta(jpg, Detection())          # n=0, centroid=None, blur_var=0.0
    meta = json.loads((tmp_path / "cam_a_001.json").read_text())
    assert meta == {"corners": 0, "centroid": None, "blur_var": 0.0}
