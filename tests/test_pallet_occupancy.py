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


# ---------- metric occupancy fallback (cross-camera decision) ----------


def test_split_view_carton_in_other_camera_marks_full():
    """cam_a sees only the pallet, cam_b only the carton sitting on it — the
    per-camera pass says "empty", the cross-camera metric fallback (pooled
    objects vs the FUSED track position) flips it to full."""
    occ = PalletOccupancy(_FakeProjector())
    pallet = _det("palette", (80.0, 80.0, 120.0, 100.0))     # foot (100,100) → (1.0, 1.0)
    carton = _det("carton", (100.0, 80.0, 120.0, 105.0))     # foot (110,105) → (1.1, 1.05)
    t = _pallet_track(1, (1.0, 1.0))
    occ.enrich([t], {"cam_a": [pallet], "cam_b": [carton]})
    assert t.occupancy_state == "full"
    assert t.occupancy_content == "carton"


def test_cross_camera_object_far_from_track_stays_empty():
    occ = PalletOccupancy(_FakeProjector())
    pallet = _det("palette", (80.0, 80.0, 120.0, 100.0))
    carton = _det("carton", (280.0, 280.0, 320.0, 300.0))    # foot (300,300) → (3.0, 3.0)
    t = _pallet_track(1, (1.0, 1.0))
    occ.enrich([t], {"cam_a": [pallet], "cam_b": [carton]})
    assert t.occupancy_state == "empty"


def test_pooled_object_attaches_only_to_nearest_track():
    """Winner-takes-all: one carton within radius of two pallet tracks loads
    only the nearer one — no cross-view double counting."""
    occ = PalletOccupancy(_FakeProjector())
    carton = _det("carton", (100.0, 80.0, 120.0, 105.0))     # foot → (1.1, 1.05)
    near = _pallet_track(1, (1.0, 1.0))
    far = _pallet_track(2, (1.6, 1.0))                       # also within 0.7 m of the carton
    occ.enrich([near, far], {"cam_b": [carton]})
    assert near.occupancy_state == "full"
    assert far.occupancy_state != "full"


def test_occluded_pallet_with_visible_carton_marks_full():
    """No camera detected the pallet this frame (track alive from history) but
    a carton sits at its position — positive evidence still lands."""
    occ = PalletOccupancy(_FakeProjector())
    carton = _det("carton", (100.0, 80.0, 120.0, 105.0))
    t = _pallet_track(1, (1.0, 1.0))
    occ.enrich([t], {"cam_b": [carton]})
    assert t.occupancy_state == "full"
    assert t.occupancy_content == "carton"


# ---------- flip hysteresis (2026-08-06: live full/empty flapping) ----------


def test_stabilizer_sticky_state_resists_marginal_oscillation():
    """Live regression: a carton sitting near both estimators' thresholds
    produced alternating per-frame verdicts, and the plain 5-frame majority
    flipped the PUBLISHED state every few seconds. Once a state is held, a
    50/50 oscillation must NOT flip it — only a supermajority may."""
    s = OccupancyStabilizer(window=6, flip_ratio=0.7)
    for _ in range(6):                       # establish "full" solidly
        s.vote(7, "full", "carton")
    # marginal oscillation: alternating empty/full → challenger never reaches
    # 70% of the window → the held state must survive throughout
    for i in range(12):
        state, _ = s.vote(7, "empty" if i % 2 == 0 else "full", "carton")
        assert state == "full"


def test_stabilizer_supermajority_flips_within_a_window():
    """A REAL unload (sustained 'empty') must still flip in ~one window."""
    s = OccupancyStabilizer(window=6, flip_ratio=0.7)
    for _ in range(6):
        s.vote(7, "full", "carton")
    flipped_at = None
    for i in range(8):
        state, content = s.vote(7, "empty", None)
        if state == "empty":
            flipped_at = i
            break
    assert flipped_at is not None and flipped_at < 6
    assert content is None


def test_stabilizer_startup_behaves_like_majority():
    """Before any state is held, the first vote establishes it immediately —
    no hysteresis deadband at startup."""
    s = OccupancyStabilizer(window=6, flip_ratio=0.7)
    state, content = s.vote(7, "full", "carton")
    assert state == "full" and content == "carton"
