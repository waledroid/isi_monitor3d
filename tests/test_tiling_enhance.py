"""SAHI tiling, crop enhancement, and TensorRT batch bucketing."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from backbone.core.types import Detection, Frame, FramePair
from backbone.detection.enhance import enhance_bgr
from backbone.detection.tiling import merge_tiled, tile_boxes
from backbone.detection.zone_scope import ZoneScopedDetector


def _det(cls, box, conf=0.9):
    return Detection(camera_id="c", capture_ts=0.0, cls=cls, confidence=conf,
                     bbox_xyxy=box, foot_uv=((box[0] + box[2]) / 2, box[3]))


# ---- tiling geometry ----

def test_small_crop_is_a_single_tile():
    assert tile_boxes(300, 200, 384, 0.2) == [(0, 0, 300, 200)]


def test_large_crop_tiles_with_overlap_and_full_cover():
    rects = tile_boxes(800, 500, 384, 0.25)
    assert len(rects) > 1
    assert min(r[0] for r in rects) == 0 and min(r[1] for r in rects) == 0
    assert max(r[2] for r in rects) == 800 and max(r[3] for r in rects) == 500
    # Overlap: consecutive tiles on a row must not be disjoint.
    xs = sorted({(r[0], r[2]) for r in rects})
    assert any(a[1] > b[0] for a, b in pairwise(xs))


# ---- tile merge ----

def test_merge_absorbs_a_clipped_copy_into_the_whole_object():
    """The tile overlap must exceed the largest object, so SOME tile sees it
    whole; the clipped copy from the neighbouring tile is absorbed (its box is
    contained in the full one). Two adjacent pallets must NOT fuse — see the
    next test — which is why the rule is containment-based, not gap-based."""
    whole = _det("palette", (0.0, 0.0, 200.0, 100.0), 0.9)
    clipped = _det("palette", (120.0, 0.0, 200.0, 100.0), 0.7)
    merged = merge_tiled([whole, clipped])
    assert len(merged) == 1
    assert merged[0].bbox_xyxy == (0.0, 0.0, 200.0, 100.0)
    assert merged[0].confidence == 0.9                       # best member wins


def test_merge_never_fuses_two_adjacent_objects():
    a = _det("palette", (0.0, 0.0, 100.0, 100.0), 0.9)
    b = _det("palette", (104.0, 0.0, 204.0, 100.0), 0.9)     # pallet beside pallet
    assert len(merge_tiled([a, b])) == 2


def test_merge_keeps_distinct_objects_and_classes():
    a = _det("palette", (0.0, 0.0, 50.0, 50.0))
    b = _det("palette", (500.0, 500.0, 550.0, 550.0))
    c = _det("carton", (0.0, 0.0, 50.0, 50.0))       # same place, other class
    assert len(merge_tiled([a, b, c])) == 3


# ---- enhancement ----

def test_enhance_lifts_contrast_without_resizing():
    img = np.full((60, 80, 3), 40, dtype=np.uint8)
    img[20:40, 20:60] = 60                           # faint object
    out = enhance_bgr(img, clip_limit=3.0)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert out.std() > img.std()


# ---- batch bucketing (TensorRT compiles one engine per shape) ----

class _RecordingDetector:
    input_size = (384, 384)

    def __init__(self):
        self.batch_sizes = []

    def detect(self, pair):
        self.batch_sizes.append(len(pair.frames))
        return {sid: [_det("palette", (1.0, 1.0, 20.0, 20.0))] for sid in pair.frames}


def _pair(n_px=400):
    img = np.zeros((n_px, n_px, 3), dtype=np.uint8)
    return FramePair(capture_ts=1.0, frame_idx=0,
                     frames={"cam_a": Frame(camera_id="cam_a", capture_ts=1.0,
                                            frame_idx=0, image=img)})


def _boxes(n):
    return {"cam_a": [(f"Z{i}", (0, 0, 100, 100)) for i in range(n)]}


def test_batch_is_padded_to_a_bucket_and_padding_never_leaks():
    det = _RecordingDetector()
    zs = ZoneScopedDetector(det, _boxes(3), {"cam_a": (400, 400)},
                            batch_buckets=(1, 2, 4, 8))
    out = zs.detect(_pair())
    assert det.batch_sizes == [4], "3 crops must be padded up to the bucket of 4"
    assert len(out["cam_a"]) == 3, "padded frames' detections must be discarded"


def test_no_buckets_means_no_padding():
    det = _RecordingDetector()
    zs = ZoneScopedDetector(det, _boxes(3), {"cam_a": (400, 400)})
    zs.detect(_pair())
    assert det.batch_sizes == [3]


def test_sahi_tiles_ride_the_same_batched_call():
    det = _RecordingDetector()
    zs = ZoneScopedDetector(det, {"cam_a": [("Z0", (0, 0, 800, 800))]},
                            {"cam_a": (800, 800)},
                            sahi={"enabled": True, "tile": 300, "overlap": 0.2})
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    pair = FramePair(capture_ts=1.0, frame_idx=0,
                     frames={"cam_a": Frame(camera_id="cam_a", capture_ts=1.0,
                                            frame_idx=0, image=img)})
    out = zs.detect(pair)
    assert det.batch_sizes and det.batch_sizes[0] > 1, "tiles must batch together"
    assert len(det.batch_sizes) == 1, "ONE inference call for all tiles"
    # Every returned detection is in FRAME coordinates.
    for d in out["cam_a"]:
        assert 0 <= d.bbox_xyxy[0] < 800 and 0 <= d.bbox_xyxy[3] <= 800
