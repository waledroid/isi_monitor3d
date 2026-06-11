"""``ReplayFrameSource`` — in-memory + plugin registration contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backbone.core.interfaces import frame_source_registry
from backbone.ingestion.replay import ReplayFrameSource


def _img(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_plugin_registered_under_replay() -> None:
    assert "replay" in frame_source_registry


def test_construct_via_registry() -> None:
    src = frame_source_registry.create(
        "replay",
        camera_id="cam_a",
        frames=[(_img(0), 0.0)],
    )
    assert isinstance(src, ReplayFrameSource)
    assert src.camera_id == "cam_a"


def test_emits_frames_in_order() -> None:
    pairs = [(_img(i), float(i) * 0.033) for i in range(3)]
    src = ReplayFrameSource(camera_id="cam_a", frames=pairs)
    out = list(src.frames())
    assert len(out) == 3
    assert [f.capture_ts for f in out] == [0.0, 0.033, 0.066]
    assert [f.frame_idx for f in out] == [0, 1, 2]
    assert all(f.camera_id == "cam_a" for f in out)


def test_stop_breaks_out_of_iterator() -> None:
    pairs = [(_img(i), float(i) * 0.033) for i in range(10)]
    src = ReplayFrameSource(camera_id="cam_a", frames=pairs)
    consumed = []
    for f in src.frames():
        consumed.append(f)
        if len(consumed) >= 2:
            src.stop()
    assert len(consumed) == 2


def test_construction_without_source_rejected() -> None:
    with pytest.raises(ValueError, match=r"frames=|mp4_path="):
        ReplayFrameSource(camera_id="cam_a")


def test_construction_with_only_mp4_path_rejected() -> None:
    with pytest.raises(ValueError, match=r"frames=|mp4_path="):
        ReplayFrameSource(camera_id="cam_a", mp4_path="/x.mp4")


def test_mp4_missing_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ReplayFrameSource(
            camera_id="cam_a",
            mp4_path=tmp_path / "missing.mp4",
            timestamps_json=tmp_path / "missing.json",
        )


def test_mp4_timestamp_length_mismatch_raises(tmp_path: Path) -> None:
    """If timestamps and decoded frames don't line up, we refuse silently."""
    import cv2

    # Write a tiny 3-frame MP4.
    mp4 = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4), fourcc, 30.0, (8, 8))
    try:
        for _ in range(3):
            writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    finally:
        writer.release()

    # Sidecar JSON with the wrong number of timestamps.
    ts_file = tmp_path / "tiny.timestamps.json"
    ts_file.write_text(json.dumps([0.0, 0.033]))   # 2 instead of 3

    with pytest.raises(ValueError, match="timestamps"):
        ReplayFrameSource(camera_id="cam_a", mp4_path=mp4, timestamps_json=ts_file)
