"""Shared frame bus — writer/reader round-trip, staleness, torn reads."""

from __future__ import annotations

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
