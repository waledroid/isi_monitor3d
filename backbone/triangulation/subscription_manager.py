"""Subscription DSL for the triangulation layer.

A *subscription* declares when a module wants 3D for a given track. The
triangulation layer reads ``config/subscriptions.yaml``, filters the
``Track2D`` stream against each rule per frame, and triangulates only the
matched tracks — the architecture's "subscription, not polling" principle.

Per-rule rate limiting (`rate_hz`) is enforced per ``(rule_name, track_id)``
so a fast-firing rule on one track does not starve another track of the same
rule.

Example ``subscriptions.yaml``:

    - name: fall_detection
      module: securite
      match:
        cls: person
        cameras_seeing_min: 2
        in_zone: any_danger
      request: xyz
      rate_hz: 10.0

    - name: stacked_pallets
      module: palettes
      match: { cls: pallet, cameras_seeing_min: 2, in_zone: any_storage }
      request: xyz
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from backbone.core.types import Track2D
from backbone.shared.timestamps import now
from backbone.shared.zones import ZoneRegistry


@dataclass(slots=True, frozen=True)
class MatchRule:
    """Predicates a ``Track2D`` must satisfy to match a subscription."""

    cls: str | None = None
    cameras_seeing_min: int | None = None
    in_zone: str | None = None    # exact zone name or "any_<type>" glob
    # When true, a track this rule matches that's seen by only ONE camera still
    # gets a 3D position via the single-view floor fallback (Z=0), flagged
    # single_view. Pair with ``cameras_seeing_min: 1`` (or unset) so 1-cam tracks
    # reach the triangulation stage. Default keeps the strict ≥2-view behavior.
    allow_single_view: bool = False


@dataclass(slots=True, frozen=True)
class SubscriptionRule:
    """One row of ``subscriptions.yaml``."""

    name: str
    module: str
    match: MatchRule
    request: str           # "xyz" in v1 (S5.5: "keypoints_3d", "torso_xyz", ...)
    rate_hz: float | None = None


def _parse_match(raw: dict | None) -> MatchRule:
    if raw is None:
        return MatchRule()
    return MatchRule(
        cls=raw.get("cls"),
        cameras_seeing_min=raw.get("cameras_seeing_min"),
        in_zone=raw.get("in_zone"),
        allow_single_view=bool(raw.get("allow_single_view", False)),
    )


def _parse_rule(entry: dict) -> SubscriptionRule:
    for key in ("name", "module", "request"):
        if key not in entry:
            raise ValueError(f"subscription entry missing required key {key!r}: {entry}")
    return SubscriptionRule(
        name=entry["name"],
        module=entry["module"],
        match=_parse_match(entry.get("match")),
        request=entry["request"],
        rate_hz=entry.get("rate_hz"),
    )


class SubscriptionManager:
    """Filters ``Track2D`` per frame against a fixed set of rules."""

    def __init__(
        self,
        rules: Iterable[SubscriptionRule],
        zones: ZoneRegistry | None = None,
    ) -> None:
        self._rules = tuple(rules)
        names = [r.name for r in self._rules]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate subscription names: {duplicates}")
        self._zones = zones if zones is not None else ZoneRegistry.empty()
        self._last_emit: dict[tuple[str, int], float] = {}
        self._validate_zone_references()

    @classmethod
    def load(
        cls,
        path: str | Path,
        zones: ZoneRegistry | None = None,
    ) -> SubscriptionManager:
        data = yaml.safe_load(Path(path).read_text()) or []
        if not isinstance(data, list):
            raise ValueError("subscriptions.yaml: top-level must be a list")
        rules = [_parse_rule(entry) for entry in data]
        return cls(rules, zones)

    @property
    def rules(self) -> tuple[SubscriptionRule, ...]:
        return self._rules

    def filter(
        self,
        tracks: Iterable[Track2D],
        *,
        reference_ts: float | None = None,
    ) -> list[tuple[Track2D, SubscriptionRule]]:
        """Return all ``(track, rule)`` pairs where every match predicate holds."""
        ref_ts = now() if reference_ts is None else reference_ts
        results: list[tuple[Track2D, SubscriptionRule]] = []
        for track in tracks:
            for rule in self._rules:
                if not self._track_matches(track, rule.match):
                    continue
                if not self._rate_allows(rule, track.track_id, ref_ts):
                    continue
                results.append((track, rule))
                self._last_emit[(rule.name, track.track_id)] = ref_ts
        return results

    # ---- internals ----

    def _track_matches(self, track: Track2D, match: MatchRule) -> bool:
        if match.cls is not None and match.cls != track.cls:
            return False
        if (
            match.cameras_seeing_min is not None
            and len(track.cameras_seeing) < match.cameras_seeing_min
        ):
            return False
        if match.in_zone is not None and not self._in_zone(track.xy_m, match.in_zone):
            return False
        return True

    def _in_zone(self, xy_m: tuple[float, float], spec: str) -> bool:
        if spec.startswith("any_"):
            type_ = spec.removeprefix("any_")
            for name in self._zones.by_type(type_):
                if self._zones[name].contains(xy_m):
                    return True
            return False
        if spec not in self._zones:
            return False
        return self._zones[spec].contains(xy_m)

    def _rate_allows(self, rule: SubscriptionRule, track_id: int, ref_ts: float) -> bool:
        if rule.rate_hz is None or rule.rate_hz <= 0.0:
            return True
        min_dt = 1.0 / rule.rate_hz
        last = self._last_emit.get((rule.name, track_id))
        if last is None:
            return True
        return (ref_ts - last) >= min_dt

    def _validate_zone_references(self) -> None:
        """Refuse to start with subscriptions that reference unknown zones."""
        for rule in self._rules:
            spec = rule.match.in_zone
            if spec is None:
                continue
            if spec.startswith("any_"):
                continue   # type-glob is permissive — empty matches just return False
            if spec not in self._zones:
                raise ValueError(
                    f"subscription {rule.name!r}: in_zone={spec!r} is not a known zone "
                    f"(available: {list(self._zones.names)})"
                )


# Compatibility export for the orchestrator.
__all__ = [
    "MatchRule",
    "SubscriptionManager",
    "SubscriptionRule",
]
