"""Shared frame bus — writer/reader round-trip, staleness, torn reads."""

from __future__ import annotations

import os
import time

import numpy as np

from backbone.shared.frame_shm import FrameShmReader, FrameShmWriter, shm_path


def _img(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_round_trip_latest_frame(tmp_path):
    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    r = FrameShmReader("cam_t", directory=str(tmp_path))
    assert r.latest() is None                      # nothing yet
    a, b = _img(seed=1), _img(seed=2)
    ts = time.time()
    w.write(a, ts)
    got, got_ts = r.latest()
    assert got_ts == ts and np.array_equal(got, a)
    w.write(b, ts + 0.04)
    got, got_ts = r.latest()
    assert got_ts == ts + 0.04 and np.array_equal(got, b)
    # The returned frame is a COPY — later writes must not mutate it.
    w.write(a, ts + 0.08)
    assert np.array_equal(got, b)
    w.close()
    r.close()


def test_stale_bus_reads_none(tmp_path):
    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    r = FrameShmReader("cam_t", directory=str(tmp_path), max_age_s=0.5)
    w.write(_img(), time.time() - 10.0)            # written long "ago"
    assert r.latest() is None
    assert r.fresh() is False
    w.close()


def test_resolution_change_recreates(tmp_path):
    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    r = FrameShmReader("cam_t", directory=str(tmp_path))
    w.write(_img(64, 48), time.time())
    assert r.latest()[0].shape == (48, 64, 3)
    w.write(_img(32, 24), time.time())
    # First read after a resize remaps; the next read delivers.
    r.latest()
    got = r.latest()
    assert got is not None and got[0].shape == (24, 32, 3)
    w.close()
    r.close()


def test_unlink_makes_bus_absent(tmp_path):
    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    r = FrameShmReader("cam_t", directory=str(tmp_path))
    w.write(_img(), time.time())
    assert r.latest() is not None
    w.unlink()
    assert r.latest() is None


def test_mid_write_seq_is_rejected(tmp_path):
    import struct

    from backbone.shared.frame_shm import _HEADER, _SLOT_HDR

    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    w.write(_img(), time.time())
    # Corrupt the latest slot's seq to ODD (mid-write) directly in the file.
    path = shm_path("cam_t", str(tmp_path))
    with open(path, "r+b") as fh:
        data = bytearray(fh.read())
        _, _, fw, fh_, c, latest = _HEADER.unpack_from(data, 0)
        slot_span = _SLOT_HDR.size + fw * fh_ * c
        off = _HEADER.size + latest * slot_span
        seq, _ts = _SLOT_HDR.unpack_from(data, off)
        struct.pack_into("<Q", data, off, seq + 1)   # odd
        fh.seek(0)
        fh.write(data)
    r = FrameShmReader("cam_t", directory=str(tmp_path))
    assert r.latest() is None
    w.close()


def test_reader_returns_writable_frames(tmp_path):
    """Display consumers draw overlays IN PLACE (cv2.addWeighted dst=image).
    A read-only view would crash them — the reader must hand out a writable
    copy, and mutating it must never touch the bus."""
    import cv2

    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    r = FrameShmReader("cam_t", directory=str(tmp_path))
    src = _img(seed=7)
    w.write(src, time.time())

    frame, _ts = r.latest()
    assert frame.flags.writeable, "frame bus handed out a read-only array"
    # The exact call that died in production must succeed.
    overlay = frame.copy()
    cv2.rectangle(overlay, (2, 2), (20, 20), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

    # Mutating the returned frame must not corrupt the bus for the next reader.
    again, _ = r.latest()
    assert np.array_equal(again, src)
    w.close()
    r.close()


def test_peek_ts_is_cheap_and_matches_latest(tmp_path):
    """Pollers must be able to ask 'is there a new frame?' without copying."""
    w = FrameShmWriter("cam_t", directory=str(tmp_path))
    r = FrameShmReader("cam_t", directory=str(tmp_path))
    assert r.peek_ts() is None                      # nothing written yet
    ts = time.time()
    w.write(_img(), ts)
    assert r.peek_ts() == ts
    frame, got_ts = r.latest()
    assert got_ts == ts and frame is not None
    w.write(_img(seed=3), ts + 0.04)
    assert r.peek_ts() == ts + 0.04
    w.close()
    r.close()


# ---- overlapping-restart unlink race ----------------------------------------


def test_old_writers_unlink_spares_the_successors_bus(tmp_path, monkeypatch):
    """An old instance's shutdown unlink must NOT remove the file a NEW
    instance already (re)created — the successor would keep publishing into a
    nameless inode while every reader sees 'absent' (the double-RTSP
    fallback observed on the rig)."""
    monkeypatch.setenv("ISI3D_SHM_DIR", str(tmp_path))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    old = FrameShmWriter("cam_x")
    old.write(img, time.time())
    # Successor starts: recreates the path (new inode via unlink+create is not
    # guaranteed — simulate the racy case where the path was removed and the
    # successor made a genuinely new file).
    os.unlink(shm_path("cam_x"))
    new = FrameShmWriter("cam_x")
    ts2 = time.time()
    new.write(img, ts2)
    old.unlink()                    # old instance's late cleanup fires
    assert os.path.exists(shm_path("cam_x")), \
        "old writer's unlink deleted the successor's live bus"
    r = FrameShmReader("cam_x", max_age_s=1e9)
    got = r.latest()
    assert got is not None and got[1] == ts2


def test_writer_self_heals_after_external_unlink(tmp_path, monkeypatch):
    """If the bus file vanishes (racing cleanup, operator rm), the writer
    recreates it within _RECHECK_EVERY writes instead of publishing into a
    nameless inode forever."""
    monkeypatch.setenv("ISI3D_SHM_DIR", str(tmp_path))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    w = FrameShmWriter("cam_y")
    w.write(img, time.time())
    os.unlink(shm_path("cam_y"))
    last = 0.0
    for _ in range(FrameShmWriter._RECHECK_EVERY + 1):
        last = time.time()
        w.write(img, last)
    assert os.path.exists(shm_path("cam_y")), "writer never recreated the bus"
    got = FrameShmReader("cam_y", max_age_s=1e9).latest()
    assert got is not None and got[1] == last


def test_unlink_still_removes_own_file(tmp_path, monkeypatch):
    """The normal shutdown path keeps its behavior: the owner's unlink removes
    the bus so readers see 'absent' instantly."""
    monkeypatch.setenv("ISI3D_SHM_DIR", str(tmp_path))
    w = FrameShmWriter("cam_z")
    w.write(np.zeros((8, 8, 3), dtype=np.uint8), time.time())
    w.unlink()
    assert not os.path.exists(shm_path("cam_z"))
