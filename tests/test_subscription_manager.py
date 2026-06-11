"""``SubscriptionManager`` — predicate matching + rate limiting + zones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from backbone.core.types import Track2D
from backbone.shared.zones import Zone, ZoneRegistry
from backbone.triangulation.subscription_manager import (
    MatchRule,
    SubscriptionManager,
    SubscriptionRule,
)

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def _track(
    *,
    track_id: int = 1,
    cls: str = "person",
    xy_m: tuple[float, float] = (1.0, 1.0),
    cameras_seeing: tuple[str, ...] = ("cam_a", "cam_b"),
    capture_ts: float = 0.0,
) -> Track2D:
    return Track2D(
        track_id=track_id,
        cls=cls,
        capture_ts=capture_ts,
        xy_m=xy_m,
        vxy_m=(0.0, 0.0),
        confidence=0.9,
        cameras_seeing=cameras_seeing,
    )


# ---- match predicates ----


def test_class_filter() -> None:
    rule = SubscriptionRule("r1", "securite", MatchRule(cls="person"), "xyz")
    mgr = SubscriptionManager([rule])
    assert len(mgr.filter([_track(cls="person")], reference_ts=0.0)) == 1
    assert len(mgr.filter([_track(cls="forklift")], reference_ts=1.0)) == 0


def test_cameras_seeing_min_filter() -> None:
    rule = SubscriptionRule(
        "r1", "securite", MatchRule(cameras_seeing_min=2), "xyz",
    )
    mgr = SubscriptionManager([rule])
    assert len(mgr.filter([_track(cameras_seeing=("cam_a", "cam_b"))], reference_ts=0.0)) == 1
    assert len(mgr.filter([_track(cameras_seeing=("cam_a",))], reference_ts=1.0)) == 0


def test_in_zone_exact_match() -> None:
    zones = ZoneRegistry([Zone("rack_a", "storage", SQUARE)])
    rule = SubscriptionRule("r", "p", MatchRule(in_zone="rack_a"), "xyz")
    mgr = SubscriptionManager([rule], zones)
    assert len(mgr.filter([_track(xy_m=(1.0, 1.0))], reference_ts=0.0)) == 1
    assert len(mgr.filter([_track(xy_m=(10.0, 10.0))], reference_ts=1.0)) == 0


def test_in_zone_any_type_glob() -> None:
    zones = ZoneRegistry([
        Zone("rack_a", "storage", SQUARE),
        Zone("rack_b", "storage", SQUARE + np.array([5.0, 0.0])),
        Zone("press_1", "danger", SQUARE + np.array([0.0, 5.0])),
    ])
    rule = SubscriptionRule("r", "p", MatchRule(in_zone="any_storage"), "xyz")
    mgr = SubscriptionManager([rule], zones)
    # In rack_a → matches.
    assert len(mgr.filter([_track(xy_m=(1.0, 1.0))], reference_ts=0.0)) == 1
    # In rack_b → matches (same type).
    assert len(mgr.filter([_track(xy_m=(6.0, 1.0))], reference_ts=1.0)) == 1
    # In press_1 — danger, not storage → no match.
    assert len(mgr.filter([_track(xy_m=(1.0, 6.0))], reference_ts=2.0)) == 0


def test_multiple_predicates_must_all_hold() -> None:
    zones = ZoneRegistry([Zone("rack_a", "storage", SQUARE)])
    rule = SubscriptionRule(
        "r", "p",
        MatchRule(cls="pallet", cameras_seeing_min=2, in_zone="rack_a"),
        "xyz",
    )
    mgr = SubscriptionManager([rule], zones)
    # All predicates hold.
    assert len(mgr.filter(
        [_track(cls="pallet", cameras_seeing=("cam_a", "cam_b"), xy_m=(1.0, 1.0))],
        reference_ts=0.0,
    )) == 1
    # Wrong class.
    assert len(mgr.filter(
        [_track(cls="person", cameras_seeing=("cam_a", "cam_b"), xy_m=(1.0, 1.0))],
        reference_ts=1.0,
    )) == 0
    # Too few cameras.
    assert len(mgr.filter(
        [_track(cls="pallet", cameras_seeing=("cam_a",), xy_m=(1.0, 1.0))],
        reference_ts=2.0,
    )) == 0
    # Outside zone.
    assert len(mgr.filter(
        [_track(cls="pallet", cameras_seeing=("cam_a", "cam_b"), xy_m=(10.0, 1.0))],
        reference_ts=3.0,
    )) == 0


# ---- rate limiting ----


def test_rate_hz_throttles_repeated_emissions() -> None:
    rule = SubscriptionRule("r", "p", MatchRule(cls="person"), "xyz", rate_hz=10.0)
    mgr = SubscriptionManager([rule])
    # First call passes through.
    assert len(mgr.filter([_track()], reference_ts=0.000)) == 1
    # 50 ms later — under 100 ms throttle.
    assert len(mgr.filter([_track()], reference_ts=0.050)) == 0
    # 150 ms after first — past throttle, emits again.
    assert len(mgr.filter([_track()], reference_ts=0.150)) == 1


def test_rate_hz_is_per_track() -> None:
    rule = SubscriptionRule("r", "p", MatchRule(cls="person"), "xyz", rate_hz=10.0)
    mgr = SubscriptionManager([rule])
    # Two distinct tracks at the same instant — both should pass first time.
    pairs = mgr.filter([_track(track_id=1), _track(track_id=2)], reference_ts=0.0)
    assert len(pairs) == 2


def test_no_rate_means_every_frame() -> None:
    rule = SubscriptionRule("r", "p", MatchRule(cls="person"), "xyz")
    mgr = SubscriptionManager([rule])
    assert len(mgr.filter([_track()], reference_ts=0.0)) == 1
    assert len(mgr.filter([_track()], reference_ts=0.001)) == 1


# ---- validation / loading ----


def test_duplicate_rule_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SubscriptionManager([
            SubscriptionRule("dup", "p", MatchRule(), "xyz"),
            SubscriptionRule("dup", "p", MatchRule(), "xyz"),
        ])


def test_unknown_exact_zone_reference_rejected() -> None:
    with pytest.raises(ValueError, match="not a known zone"):
        SubscriptionManager(
            [SubscriptionRule("r", "p", MatchRule(in_zone="nope"), "xyz")],
            zones=ZoneRegistry([Zone("known", "storage", SQUARE)]),
        )


def test_any_type_glob_does_not_require_zones_to_exist() -> None:
    # Permissive — if no zones of that type are loaded, glob simply matches nothing.
    mgr = SubscriptionManager(
        [SubscriptionRule("r", "p", MatchRule(in_zone="any_phantom"), "xyz")],
        zones=ZoneRegistry.empty(),
    )
    assert len(mgr.filter([_track()], reference_ts=0.0)) == 0


def test_yaml_load(tmp_path: Path) -> None:
    path = tmp_path / "subs.yaml"
    path.write_text(yaml.safe_dump([
        {
            "name": "fall",
            "module": "securite",
            "match": {"cls": "person", "cameras_seeing_min": 2},
            "request": "xyz",
            "rate_hz": 10.0,
        },
    ]))
    mgr = SubscriptionManager.load(path)
    assert len(mgr.rules) == 1
    rule = mgr.rules[0]
    assert rule.name == "fall"
    assert rule.module == "securite"
    assert rule.match.cls == "person"
    assert rule.match.cameras_seeing_min == 2
    assert rule.request == "xyz"
    assert rule.rate_hz == 10.0


def test_yaml_load_rejects_missing_required(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump([{"name": "x", "module": "m"}]))   # missing request
    with pytest.raises(ValueError, match="required key 'request'"):
        SubscriptionManager.load(path)


def test_yaml_load_top_level_must_be_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"rules": []}))
    with pytest.raises(ValueError, match="must be a list"):
        SubscriptionManager.load(path)


# ---- single-view fallback (Mode 2 occlusion) ----


def test_allow_single_view_parsed_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "subs.yaml"
    p.write_text(yaml.safe_dump([{
        "name": "fall", "module": "securite", "request": "xyz",
        "match": {"cls": "person", "cameras_seeing_min": 1, "allow_single_view": True},
    }]))
    mgr = SubscriptionManager.load(p)
    assert mgr.rules[0].match.allow_single_view is True


def test_allow_single_view_defaults_false() -> None:
    rule = SubscriptionRule("r", "m", MatchRule(cls="person"), "xyz")
    assert rule.match.allow_single_view is False


def test_one_cam_track_matches_min1_rule_for_single_view() -> None:
    # cameras_seeing_min:1 lets a 1-camera (occluded-in-the-other) track reach the
    # triangulation stage, where allow_single_view drives the Z=0 fallback.
    rule = SubscriptionRule(
        "fall", "securite",
        MatchRule(cls="person", cameras_seeing_min=1, allow_single_view=True), "xyz")
    mgr = SubscriptionManager([rule])
    solo = _track(cameras_seeing=("cam_a",))
    assert [r.name for _t, r in mgr.filter([solo], reference_ts=0.0)] == ["fall"]
    # The strict ≥2-view rule still rejects the same 1-cam track.
    strict = SubscriptionManager(
        [SubscriptionRule("strict", "securite", MatchRule(cls="person", cameras_seeing_min=2), "xyz")])
    assert strict.filter([solo], reference_ts=0.0) == []
