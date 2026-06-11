"""Pallet occupancy — the A+B fusion, end-to-end association, and temporal vote.

Hermetic: synthetic ``Detection``s + a fake projector (pixel to metres, scaled),
so the geometry is analytic. No model, no calibration file.
"""

from __future__ import annotations

from backbone.core.types import Detection, Track2D
from backbone.homography.pallet_occupancy import (
    OccupancyStabilizer,
    PalletOccupancy,
    _Verdict,
)


class _FakeProjector:
    """foot_uv (px) to floor metres, scaled; raises if `fail`."""

    def __init__(self, scale: float = 0.01, fail: bool = False) -> None:
        self.scale, self.fail = scale, fail

    def project(self, det: Detection):
        if self.fail:
            raise RuntimeError("no calibration")
        u, v = det.foot_uv
        return (u * self.scale, v * self.scale)


def _det(cls, bbox, conf=0.9):
    x1, _y1, x2, y2 = bbox
    return Detection(camera_id="cam_a", capture_ts=0.0, cls=cls, confidence=conf,
                     bbox_xyxy=bbox, foot_uv=((x1 + x2) / 2.0, y2))


def _pallet_track(track_id, xy_m):
    return Track2D(track_id=track_id, cls="palette", capture_ts=0.0, xy_m=xy_m,
                   vxy_m=(0.0, 0.0), confidence=0.9, cameras_seeing=("cam_a",))


# ---------- fusion / fallback (the redundancy) ----------

def test_fuse_agree():
    occ = PalletOccupancy(_FakeProjector())
    assert occ._fuse(_Verdict(0, 0.9), _Verdict(0, 0.8)) == 0


def test_fuse_disagree_b_near_edge_trusts_a():
    occ = PalletOccupancy(_FakeProjector())
    assert occ._fuse(_Verdict(0, 0.9), _Verdict(1, 0.3, uncertain=True)) == 0


def test_fuse_disagree_a_ambiguous_trusts_b():
    occ = PalletOccupancy(_FakeProjector())
    assert occ._fuse(_Verdict(0, 0.5, ambiguous=True), _Verdict(1, 0.8)) == 1


def test_fuse_only_a_available():
    occ = PalletOccupancy(_FakeProjector())
    assert occ._fuse(_Verdict(0, 0.9), _Verdict(None, 0.0, available=False)) == 0


def test_fuse_only_b_available():
    occ = PalletOccupancy(_FakeProjector())
    assert occ._fuse(_Verdict(None, 0.0), _Verdict(1, 0.7)) == 1


def test_fuse_neither():
    occ = PalletOccupancy(_FakeProjector())
    assert occ._fuse(_Verdict(None, 0.0), _Verdict(None, 0.0)) is None


# ---------- end-to-end association ----------

def test_carton_above_pallet_marks_full_carton():
    occ = PalletOccupancy(_FakeProjector())
    pallet = _det("palette", (100, 300, 300, 360))   # foot (200, 360) → (2.0, 3.6) m
    carton = _det("carton", (150, 250, 250, 300))    # sits on top, aligned
    tracks = [_pallet_track(1, (2.0, 3.6))]
    occ.enrich(tracks, {"cam_a": [pallet, carton]})
    assert tracks[0].occupancy_state == "full"
    assert tracks[0].occupancy_content == "carton"


def test_object_beside_pallet_marks_empty():
    occ = PalletOccupancy(_FakeProjector())
    pallet = _det("palette", (100, 300, 300, 360))
    far = _det("polybag", (600, 250, 700, 300))      # not above + far in metres
    tracks = [_pallet_track(1, (2.0, 3.6))]
    occ.enrich(tracks, {"cam_a": [pallet, far]})
    assert tracks[0].occupancy_state == "empty"
    assert tracks[0].occupancy_content is None


def test_polybag_dominates_when_larger():
    occ = PalletOccupancy(_FakeProjector())
    pallet = _det("palette", (100, 300, 300, 360))
    carton = _det("carton", (150, 270, 200, 300))    # small
    polybag = _det("polybag", (140, 240, 260, 300))  # bigger → dominant
    tracks = [_pallet_track(1, (2.0, 3.6))]
    occ.enrich(tracks, {"cam_a": [pallet, carton, polybag]})
    assert tracks[0].occupancy_state == "full"
    assert tracks[0].occupancy_content == "polybag"


# ---------- temporal vote ----------

def test_stabilizer_votes_majority_and_smooths_flicker():
    s = OccupancyStabilizer(window=5)
    s.vote(7, "empty", None)
    s.vote(7, "full", "carton")
    s.vote(7, "full", "carton")
    state, content = s.vote(7, "full", "carton")   # 3x full vs 1x empty
    assert state == "full" and content == "carton"


def test_stabilizer_last_without_new_observation():
    s = OccupancyStabilizer()
    s.vote(7, "full", "polybag")
    s.vote(7, "full", "polybag")
    assert s.last(7) == ("full", "polybag")
    assert s.last(99) is None
