"""PalletStateManager — detection-evidence zone decisions.

Hermetic: synthetic ``Detection``s + a fake projector (pixel to metres,
scaled) + a small in-memory ``ZoneRegistry``, following the pattern in
``tests/test_pallet_occupancy.py``. No model, no calibration file.
"""

from __future__ import annotations

import numpy as np

from backbone.core.types import Detection
from backbone.homography.foot_projector import FootProjector
from backbone.homography.pallet_occupancy import PalletOccupancy
from backbone.homography.pallet_state_manager import PalletStateManager, ZoneDecision
from backbone.shared.geometry import floor_homography_from_K_R_t
from backbone.shared.zones import Zone, ZoneRegistry


class _FakeProjector:
    """foot_uv (px) to floor metres, scaled; raises if `fail`."""

    def __init__(self, scale: float = 0.01, fail: bool = False) -> None:
        self.scale, self.fail = scale, fail

    def project(self, det: Detection):
        if self.fail:
            raise RuntimeError("no calibration")
        u, v = det.foot_uv
        return (u * self.scale, v * self.scale)


def _det(cls, bbox, camera_id: str = "cam_a", conf: float = 0.9) -> Detection:
    x1, _y1, x2, y2 = bbox
    return Detection(camera_id=camera_id, capture_ts=0.0, cls=cls, confidence=conf,
                     bbox_xyxy=bbox, foot_uv=((x1 + x2) / 2.0, y2))


def _zone_registry(x0=0.0, y0=0.0, x1=5.0, y1=5.0,
                   name="Zone A", zone_id="zone_a") -> ZoneRegistry:
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return ZoneRegistry([Zone(name=name, type="palette", polygon=poly, id=zone_id)])


def _manager(zones: ZoneRegistry | None = None, **kwargs) -> PalletStateManager:
    zones = zones or _zone_registry()
    projector = _FakeProjector()
    occupancy = PalletOccupancy(projector)
    return PalletStateManager(zones, projector, occupancy, **kwargs)


def _zone_dec(decisions: list[ZoneDecision], zone_id: str = "zone_a") -> ZoneDecision:
    return next(d for d in decisions if d.zone_id == zone_id)


_CAMS = ("cam_a", "cam_b")


# ---------- presence = detection evidence (decision 2) ----------


def test_single_camera_evidence_is_proof():
    """cam_b-only palette evidence is proof of presence — the live regression."""
    mgr = _manager()
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_b")  # foot (200,360)->(2.0,3.6)
    decisions = []
    for _ in range(2):  # default enter_after=2
        decisions = mgr.step({"cam_b": [pallet]})
    assert _zone_dec(decisions).palette_state == "palette_empty"


def test_presence_survives_one_camera_losing_it():
    """cam_a's evidence disappears but cam_b's persists ⇒ state unchanged every frame."""
    mgr = _manager()
    pallet_a = _det("palette", (100, 300, 300, 360), camera_id="cam_a")   # (2.0, 3.6)
    pallet_b = _det("palette", (110, 300, 310, 358), camera_id="cam_b")   # (2.1, 3.58)
    for _ in range(2):
        mgr.step({"cam_a": [pallet_a], "cam_b": [pallet_b]})
    prev = None
    for _ in range(5):
        dec = _zone_dec(mgr.step({"cam_b": [pallet_b]}))  # cam_a's evidence gone
        assert dec.palette_state == "palette_empty"
        if prev is not None:
            assert dec == prev
        prev = dec


# ---------- occupancy: full wins across cameras (decision 3) ----------


def test_full_wins_across_cameras_for_occupancy():
    """cam_a sees the pallet empty; cam_b sees the same pallet with a carton
    on it (its own A-estimator overlap) ⇒ the zone reads loaded."""
    mgr = _manager()
    pallet_a = _det("palette", (100, 300, 300, 360), camera_id="cam_a")   # (2.0, 3.6)
    pallet_b = _det("palette", (110, 300, 310, 360), camera_id="cam_b")   # (2.1, 3.6)
    carton_b = _det("carton", (150, 250, 250, 300), camera_id="cam_b")    # above pallet_b, aligned
    decisions = []
    for _ in range(2):
        decisions = mgr.step({"cam_a": [pallet_a], "cam_b": [pallet_b, carton_b]})
    dec = _zone_dec(decisions)
    assert dec.palette_state == "palette_loaded"
    assert dec.content == ("carton",)


# ---------- count = max across cameras, never sum (decision 4) ----------


def test_count_is_max_not_sum():
    mgr = _manager()
    pallet_a1 = _det("palette", (50, 300, 150, 360), camera_id="cam_a")    # (1.0, 3.6)
    pallet_a2 = _det("palette", (250, 300, 350, 360), camera_id="cam_a")   # (3.0, 3.6)
    pallet_b1 = _det("palette", (150, 300, 250, 360), camera_id="cam_b")   # (2.0, 3.6)
    decisions = mgr.step({"cam_a": [pallet_a1, pallet_a2], "cam_b": [pallet_b1]})
    assert _zone_dec(decisions).counts["palette"] == 2


# ---------- presence hysteresis (decision 5): 2-in / 15-out ----------


def test_presence_hysteresis_two_in_fifteen_out():
    mgr = _manager()
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")     # (2.0, 3.6)
    evidence_frame = {"cam_a": [pallet]}
    no_dets = {"cam_a": []}

    assert _zone_dec(mgr.step(evidence_frame)).palette_state == "no_palette"     # 1st: not yet
    assert _zone_dec(mgr.step(evidence_frame)).palette_state == "palette_empty"  # 2nd: present

    for _ in range(14):
        assert _zone_dec(mgr.step(no_dets)).palette_state == "palette_empty"     # still present

    assert _zone_dec(mgr.step(no_dets)).palette_state == "no_palette"            # 15th: exits


# ---------- carton without a palette (decision 5 enum) ----------


def test_carton_alone_no_palette():
    mgr = _manager()
    carton = _det("carton", (150, 250, 250, 300), camera_id="cam_a")   # foot (200,300)->(2.0,3.0)
    dec = _zone_dec(mgr.step({"cam_a": [carton]}))
    assert dec.palette_state == "no_palette"
    assert dec.counts["carton"] == 1


# ---------- zone-bucketing tolerance (reuses build_zone_membership_filter mechanics) ----------


def test_tolerance_catches_edge_detection():
    mgr = _manager()  # default tol_m=0.15; zone spans x in [0, 5]
    # foot at x=5.1 (0.1 m past the zone's right edge)
    pallet = _det("palette", (500, 240, 520, 260), camera_id="cam_a")  # foot (510,260)->(5.1,2.6)
    decisions = []
    for _ in range(2):
        decisions = mgr.step({"cam_a": [pallet]})
    dec = _zone_dec(decisions)
    assert dec.palette_state == "palette_empty"
    assert dec.counts.get("palette") == 1


# ---------- stability over repeated identical frames ----------


def test_decision_is_stable_over_repeated_identical_frames():
    mgr = _manager()
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")  # (2.0, 3.6)
    seen = set()
    for i in range(30):
        dec = _zone_dec(mgr.step({"cam_a": [pallet]}))
        if i >= 1:  # frame 0 is the hysteresis warm-up (enter_after=2)
            seen.add((dec.palette_state, dec.content, tuple(sorted(dec.counts.items()))))
    assert len(seen) == 1


# ---------- camera-loss gate (Finding 2): partial frames must not read as
# ---------- evidence of absence ----------


def test_partial_frame_absence_does_not_exit_presence_over_30_frames():
    """cam_a goes dark (only cam_b reports, and it has nothing) — 30 straight
    partial frames must NOT exit presence (vs. the pre-fix 15-out default,
    which would have exited by frame 15)."""
    mgr = _manager(camera_ids=_CAMS)
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")  # (2.0, 3.6)
    for _ in range(2):  # enter_after=2, both cameras reporting
        mgr.step({"cam_a": [pallet], "cam_b": []}, reporting_cameras=_CAMS)
    for _ in range(30):
        dec = _zone_dec(mgr.step({"cam_b": []}, reporting_cameras=("cam_b",)))
        assert dec.palette_state == "palette_empty"


def test_full_frame_absence_still_exits_after_15():
    """Both configured cameras reporting (not partial) + genuine absence still
    exits presence at the documented 15th absent frame — the gate must not
    disable exiting altogether."""
    mgr = _manager(camera_ids=_CAMS)
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")
    for _ in range(2):
        mgr.step({"cam_a": [pallet], "cam_b": []}, reporting_cameras=_CAMS)
    for _ in range(14):
        dec = _zone_dec(mgr.step({"cam_a": [], "cam_b": []}, reporting_cameras=_CAMS))
        assert dec.palette_state == "palette_empty"
    dec = _zone_dec(mgr.step({"cam_a": [], "cam_b": []}, reporting_cameras=_CAMS))
    assert dec.palette_state == "no_palette"


def test_entering_evidence_on_partial_frame_still_enters():
    """A partial frame's SURVIVING camera seeing a brand-new palette is real
    evidence — entering must not be blocked, only exiting."""
    mgr = _manager(camera_ids=_CAMS)
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_b")
    decisions = []
    for _ in range(2):  # enter_after=2; cam_a is down the whole time
        decisions = mgr.step({"cam_b": [pallet]}, reporting_cameras=("cam_b",))
    assert _zone_dec(decisions).palette_state == "palette_empty"


def test_no_camera_loss_gate_when_camera_ids_unset():
    """Back-compat: a manager built without camera_ids (most tests above)
    gates nothing — partial-looking frames behave exactly as before Finding 2."""
    mgr = _manager()  # camera_ids defaults to () — gate inert
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")
    for _ in range(2):
        mgr.step({"cam_a": [pallet]})
    for _ in range(14):
        dec = _zone_dec(mgr.step({"cam_a": []}))
        assert dec.palette_state == "palette_empty"
    dec = _zone_dec(mgr.step({"cam_a": []}))
    assert dec.palette_state == "no_palette"


# ---------- stale occupancy history (Finding 5) ----------


def test_stale_occupancy_forgotten_when_presence_exits():
    """A loaded palette leaves (15 absent frames exits presence); a NEW,
    unrelated palette then enters ⇒ the zone reads palette_empty immediately
    — it must not inherit the departed pallet's "loaded" vote history."""
    mgr = _manager()
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")  # (2.0, 3.6)
    carton = _det("carton", (150, 250, 250, 300), camera_id="cam_a")   # loads it
    dec = None
    for _ in range(2):
        dec = _zone_dec(mgr.step({"cam_a": [pallet, carton]}))
    assert dec.palette_state == "palette_loaded"

    # The palette leaves entirely — 15 consecutive absent frames exits presence.
    for _ in range(15):
        dec = _zone_dec(mgr.step({"cam_a": []}))
    assert dec.palette_state == "no_palette"

    # A NEW, empty palette enters at the same spot — must read empty right
    # away, not resurrect the departed pallet's "loaded" verdict.
    new_pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")
    for _ in range(2):  # enter_after=2
        dec = _zone_dec(mgr.step({"cam_a": [new_pallet]}))
    assert dec.palette_state == "palette_empty"


# ---------- plane-aware bucketing (zone-base-height, decision 5) ----------
#
# Rig geometry mirrors tests/test_zone_scope.py's Task 2 fixtures: a
# look-down camera at the origin, z=3, f=1000, c=(500, 500) — a world point
# (X, Y, z) projects to u = 1000*X/(3-z) + 500. A platform at 0.304 m is the
# live sortie_machine_1 height.

_PLAT_K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
_PLAT_R = np.diag([1.0, -1.0, -1.0])
_PLATFORM_Z = 0.304
_PLATFORM_POLY = np.array(
    [[2.9, -0.1], [3.1, -0.1], [3.1, 0.1], [2.9, 0.1]], dtype=np.float64)
# Foot pixel of an object standing at world (3.0, 0.0) ON the platform —
# plane-projects to exactly (3.0, 0.0); floor-projects to X≈3.34 (outside
# the platform's polygon even with the ±0.15 m tolerance cross).
_PLATFORM_FOOT = (1000.0 * 3.0 / (3.0 - _PLATFORM_Z) + 500.0, 500.0)


class _PlatformView:
    def __init__(self, image_size_wh=(2000, 1000)):
        self.K = _PLAT_K
        self.D = np.zeros(5)
        self.R = _PLAT_R
        self.t = np.array([0.0, 0.0, 3.0])
        self.H = floor_homography_from_K_R_t(self.K, self.R, self.t)
        self.image_size_wh = image_size_wh


class _PlatformViewH:
    """Mode-1 placeholder extrinsics — only ``H`` is real (K=I, R=I, t=0)."""

    def __init__(self, image_size_wh=(2000, 1000)):
        self.K = np.eye(3)
        self.D = np.zeros(5)
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.H = floor_homography_from_K_R_t(
            _PLAT_K, _PLAT_R, np.array([0.0, 0.0, 3.0]))
        self.image_size_wh = image_size_wh


class _PlatformRig:
    def __init__(self, views: dict):
        self._views = views

    @property
    def camera_ids(self):
        return tuple(self._views)

    def __getitem__(self, cam_id):
        return self._views[cam_id]

    def __contains__(self, cam_id):
        return cam_id in self._views


def _platform_registry(z_base_m: float, zone_id: str = "platform") -> ZoneRegistry:
    return ZoneRegistry([Zone(name="Platform", type="palette",
                              polygon=_PLATFORM_POLY, id=zone_id, z_base_m=z_base_m)])


def test_platform_zone_buckets_detection_floor_projection_would_miss():
    """A detection standing on the 0.304 m platform: with `rig` supplied, its
    evidence is bucketed on the ZONE'S OWN plane and lands inside; a
    same-shaped zone left at z_base_m=0 (floor) never sees it — proving
    bucketing now depends on each zone's own height, not one shared Z=0
    projection."""
    rig = _PlatformRig({"cam_a": _PlatformView()})
    pallet = Detection(camera_id="cam_a", capture_ts=0.0, cls="palette",
                       confidence=0.9,
                       bbox_xyxy=(_PLATFORM_FOOT[0] - 20, _PLATFORM_FOOT[1] - 20,
                                  _PLATFORM_FOOT[0] + 20, _PLATFORM_FOOT[1]),
                       foot_uv=_PLATFORM_FOOT)
    projector = _FakeProjector()
    raised = _platform_registry(_PLATFORM_Z)
    floor = _platform_registry(0.0)
    mgr_raised = PalletStateManager(raised, projector, PalletOccupancy(projector), rig=rig)
    mgr_floor = PalletStateManager(floor, projector, PalletOccupancy(projector), rig=rig)
    for _ in range(2):  # enter_after=2
        dec_raised = _zone_dec(mgr_raised.step({"cam_a": [pallet]}), "platform")
        dec_floor = _zone_dec(mgr_floor.step({"cam_a": [pallet]}), "platform")
    assert dec_raised.palette_state == "palette_empty"
    assert dec_raised.counts.get("palette") == 1
    assert dec_floor.palette_state == "no_palette"
    assert dec_floor.counts.get("palette", 0) == 0


def test_platform_occupancy_state_lands_in_platform_zone():
    """A pallet standing on the platform, loaded with a carton (the A
    image-overlap estimator decides "full" from bbox geometry alone,
    independent of the mock projector's metric accuracy): its occupancy
    verdict must bucket into the PLATFORM zone via plane-aware projection —
    the deliverable's explicit case, "a platform pallet's occupancy state
    must land in the platform zone." The plain `_FakeProjector` (scale=0.01)
    would map this foot to x≈16.1, nowhere near the platform's x∈[2.9,3.1]
    footprint, proving the bucketing no longer depends on it."""
    rig = _PlatformRig({"cam_a": _PlatformView()})
    fx, fy = _PLATFORM_FOOT
    pallet = Detection(camera_id="cam_a", capture_ts=0.0, cls="palette",
                       confidence=0.9, bbox_xyxy=(fx - 100, 440, fx + 100, 500),
                       foot_uv=(fx, fy))
    carton = Detection(camera_id="cam_a", capture_ts=0.0, cls="carton",
                       confidence=0.9, bbox_xyxy=(fx - 50, 400, fx + 50, 450),
                       foot_uv=(fx, 450.0))
    projector = _FakeProjector()
    raised = _platform_registry(_PLATFORM_Z)
    floor = _platform_registry(0.0)
    mgr_raised = PalletStateManager(raised, projector, PalletOccupancy(projector), rig=rig)
    mgr_floor = PalletStateManager(floor, projector, PalletOccupancy(projector), rig=rig)
    for _ in range(2):
        dec_raised = _zone_dec(mgr_raised.step({"cam_a": [pallet, carton]}), "platform")
        dec_floor = _zone_dec(mgr_floor.step({"cam_a": [pallet, carton]}), "platform")
    assert dec_raised.palette_state == "palette_loaded"
    assert dec_raised.content == ("carton",)
    assert dec_floor.palette_state == "no_palette"    # never entered presence


def test_floor_zone_bit_identical_with_or_without_rig():
    """z_base_m=0 zones must decide IDENTICALLY whether or not `rig` is
    supplied — `ZoneAwareProjector`'s z=0 path is pinned bit-identical to
    `FootProjector`'s plain floor projection (geometry's pixel_to_plane z=0
    parity test), so wiring the SAME rig into both `projector` and `rig`
    (exactly as the orchestrator does) must not change a single decision."""
    rig = _PlatformRig({"cam_a": _PlatformView()})
    zones = _zone_registry()  # default floor zone, z_base_m=0.0
    projector = FootProjector(rig)
    mgr_plain = PalletStateManager(zones, projector, PalletOccupancy(projector))
    mgr_aware = PalletStateManager(zones, projector, PalletOccupancy(projector), rig=rig)
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_a")
    carton = _det("carton", (150, 250, 250, 300), camera_id="cam_a")
    for _ in range(5):
        d_plain = _zone_dec(mgr_plain.step({"cam_a": [pallet, carton]}))
        d_aware = _zone_dec(mgr_aware.step({"cam_a": [pallet, carton]}))
        assert d_plain == d_aware


def test_mode1_h_only_rig_raised_zone_no_crash_keeps_floor_behavior():
    """Mode-1 (H-only placeholder K=I, R=I, t=0) rigs cannot lift a ray off
    the floor: a raised zone's decisions match a floor zone's exactly, no
    crash (Global Constraint: raised projection paths fall back to current
    behavior on H-only rigs)."""
    rig = _PlatformRig({"cam_a": _PlatformViewH()})
    projector = FootProjector(rig)
    raised = _platform_registry(_PLATFORM_Z)
    floor = _platform_registry(0.0)
    pallet = _det("palette", (700, 460, 900, 500), camera_id="cam_a")
    mgr_raised = PalletStateManager(raised, projector, PalletOccupancy(projector), rig=rig)
    mgr_floor = PalletStateManager(floor, projector, PalletOccupancy(projector), rig=rig)
    for _ in range(2):
        d_r = _zone_dec(mgr_raised.step({"cam_a": [pallet]}), "platform")
        d_f = _zone_dec(mgr_floor.step({"cam_a": [pallet]}), "platform")
        assert d_r == d_f


def test_unknown_camera_to_zone_aware_projector_contributes_no_evidence():
    """A detection whose camera isn't in `rig`: plane-aware bucketing skips
    it for every zone (fail-closed — same "a bad projection proves nothing"
    policy the single-projection path always had), no crash."""
    rig = _PlatformRig({"cam_a": _PlatformView()})
    zones = _zone_registry()
    projector = _FakeProjector()
    mgr = PalletStateManager(zones, projector, PalletOccupancy(projector), rig=rig)
    pallet = _det("palette", (100, 300, 300, 360), camera_id="cam_unknown")
    dec = _zone_dec(mgr.step({"cam_unknown": [pallet]}))
    assert dec.palette_state == "no_palette"
    assert dec.counts.get("palette", 0) == 0


def test_camera_loss_gate_unaffected_by_plane_aware_bucketing():
    """The camera-loss gate (partial-frame protection, Finding 2) works
    exactly as before when `rig` is supplied for plane-aware bucketing — the
    gate/hysteresis layer sits ABOVE bucketing and doesn't care how a
    detection's xy was derived."""
    rig = _PlatformRig({"cam_a": _PlatformView()})
    zones = _platform_registry(_PLATFORM_Z)
    projector = _FakeProjector()
    mgr = PalletStateManager(zones, projector, PalletOccupancy(projector),
                             rig=rig, camera_ids=("cam_a", "cam_b"))
    pallet = Detection(camera_id="cam_a", capture_ts=0.0, cls="palette",
                       confidence=0.9,
                       bbox_xyxy=(_PLATFORM_FOOT[0] - 20, _PLATFORM_FOOT[1] - 20,
                                  _PLATFORM_FOOT[0] + 20, _PLATFORM_FOOT[1]),
                       foot_uv=_PLATFORM_FOOT)
    for _ in range(2):  # enter_after=2, both cameras reporting
        mgr.step({"cam_a": [pallet], "cam_b": []}, reporting_cameras=("cam_a", "cam_b"))
    for _ in range(30):  # cam_a goes dark — must NOT read as evidence of absence
        dec = _zone_dec(mgr.step({"cam_b": []}, reporting_cameras=("cam_b",)), "platform")
        assert dec.palette_state == "palette_empty"
