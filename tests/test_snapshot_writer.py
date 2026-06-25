"""``SnapshotWriter`` — unit tests (hermetic, no I/O to external services)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from backbone.shared.snapshot_writer import SnapshotWriter


def _black_image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_write_creates_file_and_returns_url(tmp_path: Path) -> None:
    """write() saves a JPEG and returns a URL ending with the filename."""
    writer = SnapshotWriter(out_dir=str(tmp_path), url_base="file://", jpeg_quality=85)
    url = writer.write(_black_image(), track_id=7, zone="B3D", ts=1700000000.0)

    assert url is not None
    # Derive expected filename from the same formula used by the writer.
    ts = 1700000000.0
    expected_name = f"{int(ts * 1000)}_B3D_7.jpg"
    assert url.endswith(expected_name), f"URL {url!r} does not end with {expected_name!r}"
    assert (tmp_path / expected_name).exists()


def test_write_url_base_trailing_slash_stripped(tmp_path: Path) -> None:
    """Trailing slash on url_base must not produce double slashes in the URL."""
    writer = SnapshotWriter(
        out_dir=str(tmp_path),
        url_base="http://storage.local/snapshots/",
        jpeg_quality=85,
    )
    url = writer.write(_black_image(), track_id=1, zone="Z1", ts=1.0)
    assert url is not None
    assert "//" not in url.replace("http://", "")


def test_write_sanitises_zone_name(tmp_path: Path) -> None:
    """Zone names with slashes produce a safe filename."""
    writer = SnapshotWriter(out_dir=str(tmp_path), url_base="file://")
    url = writer.write(_black_image(), track_id=0, zone="zone/A+B", ts=2.0)
    assert url is not None
    # The URL / filename must NOT contain a raw slash inside the zone segment.
    filename = url.split("/")[-1]
    assert "zone" in filename
    assert "/" not in filename


def test_write_creates_out_dir_on_construction(tmp_path: Path) -> None:
    """SnapshotWriter creates out_dir (and parents) at construction time."""
    new_dir = tmp_path / "a" / "b" / "c"
    assert not new_dir.exists()
    SnapshotWriter(out_dir=str(new_dir))
    assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------

def test_write_returns_none_when_imwrite_fails(tmp_path: Path) -> None:
    """write() returns None (and does not raise) when cv2.imwrite returns False."""
    writer = SnapshotWriter(out_dir=str(tmp_path), url_base="file://")
    with patch("backbone.shared.snapshot_writer.cv2.imwrite", return_value=False):
        result = writer.write(_black_image(), track_id=42, zone="X", ts=3.0)
    assert result is None
    # No file should have been created.
    assert list(tmp_path.iterdir()) == []


def test_write_returns_none_on_exception(tmp_path: Path) -> None:
    """write() swallows any exception from cv2.imwrite and returns None."""
    writer = SnapshotWriter(out_dir=str(tmp_path), url_base="file://")
    with patch(
        "backbone.shared.snapshot_writer.cv2.imwrite",
        side_effect=OSError("disk full"),
    ):
        result = writer.write(_black_image(), track_id=1, zone="Y", ts=4.0)
    assert result is None
