"""``CrossCamFusion`` — Hungarian matching across cameras within class thresholds."""

from __future__ import annotations

import pytest

from backbone.core.types import Detection
from backbone.homography.cross_cam_fusion import CrossCamFusion


def _det(camera_id: str, cls: str = "person", confidence: float = 0.9) -> Detection:
    return Detection(
        camera_id=camera_id,
        capture_ts=0.0,
        cls=cls,
        confidence=confidence,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        foot_uv=(5.0, 10.0),
    )


def test_matching_pair_within_threshold_is_fused() -> None:
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a"), (1.0, 2.0)),
        (_det("cam_b"), (1.2, 2.1)),
    ]
    out = fusion.fuse(items)
    assert len(out) == 1
    obs = out[0]
    assert obs.cls == "person"
    assert obs.cameras_seeing == ("cam_a", "cam_b")
    # Position is the averaged pair.
    assert obs.xy_m == pytest.approx((1.1, 2.05), abs=1e-9)
    assert obs.confidence == pytest.approx(0.9)


def test_pair_beyond_threshold_split_into_single_cam() -> None:
    """A 5 m separation is far beyond the 0.8 m person threshold."""
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a"), (0.0, 0.0)),
        (_det("cam_b"), (5.0, 5.0)),
    ]
    out = fusion.fuse(items)
    assert len(out) == 2
    for obs in out:
        assert len(obs.cameras_seeing) == 1


def test_classes_are_not_mixed() -> None:
    """A person and a forklift at the same position must not fuse."""
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a", cls="person"), (1.0, 1.0)),
        (_det("cam_b", cls="forklift"), (1.0, 1.0)),
    ]
    out = fusion.fuse(items)
    assert len(out) == 2
    classes = {obs.cls for obs in out}
    assert classes == {"person", "forklift"}
    for obs in out:
        assert len(obs.cameras_seeing) == 1


def test_single_camera_observations_pass_through() -> None:
    fusion = CrossCamFusion()
    items = [(_det("cam_a"), (0.0, 0.0))]
    out = fusion.fuse(items)
    assert len(out) == 1
    assert out[0].cameras_seeing == ("cam_a",)


def test_hungarian_picks_optimal_assignment() -> None:
    """Three people in cam_a + three in cam_b: optimal matching pairs nearest."""
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a"), (0.0, 0.0)),
        (_det("cam_a"), (2.0, 0.0)),
        (_det("cam_a"), (4.0, 0.0)),
        # cam_b sees the same three people, slightly shifted.
        (_det("cam_b"), (0.1, 0.05)),
        (_det("cam_b"), (2.1, 0.05)),
        (_det("cam_b"), (4.1, 0.05)),
    ]
    out = fusion.fuse(items)
    fused_pairs = [obs for obs in out if len(obs.cameras_seeing) == 2]
    assert len(fused_pairs) == 3
    xs = sorted(obs.xy_m[0] for obs in fused_pairs)
    assert xs == pytest.approx([0.05, 2.05, 4.05], abs=1e-9)


def test_unmatched_remainder_becomes_single_cam() -> None:
    """cam_a has 2 detections, cam_b has 1 → 1 fused pair + 1 single-cam."""
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a"), (0.0, 0.0)),
        (_det("cam_a"), (5.0, 5.0)),    # no partner in cam_b
        (_det("cam_b"), (0.1, 0.0)),
    ]
    out = fusion.fuse(items)
    fused_pairs = [obs for obs in out if len(obs.cameras_seeing) == 2]
    singles = [obs for obs in out if len(obs.cameras_seeing) == 1]
    assert len(fused_pairs) == 1
    assert len(singles) == 1
    assert singles[0].xy_m == pytest.approx((5.0, 5.0), abs=1e-9)


def test_fused_confidence_is_max_of_contributors() -> None:
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a", confidence=0.6), (0.0, 0.0)),
        (_det("cam_b", confidence=0.9), (0.05, 0.05)),
    ]
    out = fusion.fuse(items)
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(0.9)


def test_per_camera_positions_preserved_for_disagreement_gate() -> None:
    fusion = CrossCamFusion()
    items = [
        (_det("cam_a"), (1.0, 2.0)),
        (_det("cam_b"), (1.3, 2.1)),
    ]
    out = fusion.fuse(items)
    assert out[0].per_camera_positions["cam_a"] == (1.0, 2.0)
    assert out[0].per_camera_positions["cam_b"] == (1.3, 2.1)


def test_custom_thresholds_respected() -> None:
    """Override the person threshold to be very tight (5 cm)."""
    fusion = CrossCamFusion(match_distance_m={"person": 0.05})
    items = [
        (_det("cam_a"), (0.0, 0.0)),
        (_det("cam_b"), (0.10, 0.0)),  # 10 cm apart — beyond tight threshold
    ]
    out = fusion.fuse(items)
    assert len(out) == 2  # not fused
    for obs in out:
        assert len(obs.cameras_seeing) == 1
