"""``DisagreementGate`` — keep fused pairs that agree; demote those that don't."""

from __future__ import annotations

import pytest

from backbone.homography.cross_cam_fusion import FusedObservation
from backbone.homography.disagreement_gate import DisagreementGate


def _fused(
    cls: str = "person",
    *,
    xy_avg: tuple[float, float] = (1.0, 1.0),
    cam_a_xy: tuple[float, float] = (1.0, 1.0),
    cam_b_xy: tuple[float, float] = (1.0, 1.0),
    cam_a_conf: float = 0.9,
    cam_b_conf: float = 0.6,
) -> FusedObservation:
    return FusedObservation(
        cls=cls,
        xy_m=xy_avg,
        confidence=max(cam_a_conf, cam_b_conf),
        cameras_seeing=("cam_a", "cam_b"),
        capture_ts=0.0,
        per_camera_positions={"cam_a": cam_a_xy, "cam_b": cam_b_xy},
        per_camera_confidence={"cam_a": cam_a_conf, "cam_b": cam_b_conf},
    )


def _single(camera_id: str = "cam_a") -> FusedObservation:
    return FusedObservation(
        cls="person",
        xy_m=(0.0, 0.0),
        confidence=0.7,
        cameras_seeing=(camera_id,),
        capture_ts=0.0,
        per_camera_positions={camera_id: (0.0, 0.0)},
        per_camera_confidence={camera_id: 0.7},
    )


def test_agreement_within_threshold_passes_through() -> None:
    """Two cameras 30 cm apart for a person (threshold 40 cm) → keep fused."""
    gate = DisagreementGate()
    obs = _fused(
        cam_a_xy=(1.0, 1.0),
        cam_b_xy=(1.3, 1.0),
    )
    out = gate.check([obs])
    assert len(out) == 1
    assert out[0].cameras_seeing == ("cam_a", "cam_b")
    assert gate.rejected_count == 0


def test_disagreement_beyond_threshold_demotes_to_cleaner() -> None:
    """Two cameras 1 m apart for a person (threshold 40 cm) → reject + keep higher-conf."""
    gate = DisagreementGate()
    obs = _fused(
        cam_a_xy=(1.0, 1.0),
        cam_b_xy=(2.0, 1.0),
        cam_a_conf=0.9,
        cam_b_conf=0.6,
    )
    out = gate.check([obs])
    assert len(out) == 1
    assert out[0].cameras_seeing == ("cam_a",)
    assert out[0].xy_m == (1.0, 1.0)
    assert out[0].confidence == pytest.approx(0.9)
    assert gate.rejected_count == 1


def test_cleaner_observation_picks_higher_confidence_camera() -> None:
    """When cam_b is more confident, demote keeps cam_b's position."""
    gate = DisagreementGate()
    obs = _fused(
        cam_a_xy=(1.0, 1.0),
        cam_b_xy=(2.0, 1.0),
        cam_a_conf=0.4,
        cam_b_conf=0.95,
    )
    out = gate.check([obs])
    assert out[0].cameras_seeing == ("cam_b",)
    assert out[0].xy_m == (2.0, 1.0)


def test_single_cam_observations_pass_through_untouched() -> None:
    gate = DisagreementGate()
    obs = _single("cam_a")
    out = gate.check([obs])
    assert out[0] is obs
    assert gate.rejected_count == 0


def test_rejected_count_accumulates_across_calls() -> None:
    gate = DisagreementGate()
    bad = _fused(cam_a_xy=(0.0, 0.0), cam_b_xy=(5.0, 5.0))
    gate.check([bad])
    gate.check([bad])
    assert gate.rejected_count == 2


def test_per_class_threshold_respected() -> None:
    """Forklift threshold is 80 cm (vs 40 cm for person) — same disagreement keeps fused."""
    gate = DisagreementGate()
    obs = _fused(
        cls="forklift",
        cam_a_xy=(1.0, 1.0),
        cam_b_xy=(1.5, 1.0),   # 50 cm — passes 80 cm forklift gate, fails 40 cm person gate
    )
    out = gate.check([obs])
    assert len(out[0].cameras_seeing) == 2
    assert gate.rejected_count == 0


def test_custom_thresholds_can_tighten_gate() -> None:
    gate = DisagreementGate(agreement_distance_m={"person": 0.1})
    obs = _fused(
        cam_a_xy=(1.0, 1.0),
        cam_b_xy=(1.2, 1.0),  # 20 cm — beyond the tight 10 cm gate
    )
    out = gate.check([obs])
    assert out[0].cameras_seeing == ("cam_a",)
    assert gate.rejected_count == 1
