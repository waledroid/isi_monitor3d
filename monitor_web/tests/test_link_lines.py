"""Unit tests for ``monitor_web.link_lines`` — schema, ``*`` glob, distance gating."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from monitor_web.link_lines import (
    LinkLineRule,
    load_link_lines,
    rules_to_dict,
    should_link,
)

# ---- schema ----


def test_rule_round_trip_via_alias() -> None:
    """Pydantic round-trips the YAML-friendly ``from`` alias."""
    r = LinkLineRule.model_validate({
        "from": "person",
        "to": ["palette", "forklift"],
        "max_distance_m": 5.0,
        "color": "#ffffff",
    })
    assert r.from_ == "person"
    assert r.to == ["palette", "forklift"]
    assert r.max_distance_m == 5.0
    assert r.color == "#ffffff"
    # rules_to_dict re-emits with the ``from`` key (not ``from_``).
    assert "from" in rules_to_dict([r])[0]


def test_rule_rejects_empty_to() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        LinkLineRule.model_validate({"from": "person", "to": []})


def test_rule_rejects_blank_to_entry() -> None:
    with pytest.raises(Exception):  # noqa: B017
        LinkLineRule.model_validate({"from": "person", "to": ["palette", "  "]})


def test_rule_rejects_zero_or_negative_distance() -> None:
    with pytest.raises(Exception):  # noqa: B017
        LinkLineRule.model_validate({"from": "person", "to": ["palette"], "max_distance_m": 0})
    with pytest.raises(Exception):  # noqa: B017
        LinkLineRule.model_validate({"from": "person", "to": ["palette"], "max_distance_m": -1})


def test_rule_accepts_missing_distance() -> None:
    r = LinkLineRule.model_validate({"from": "person", "to": ["palette"]})
    assert r.max_distance_m is None


# ---- should_link / matches ----


def test_should_link_undirected_pair() -> None:
    """A directional YAML rule is applied undirected at render time."""
    rules = [LinkLineRule.model_validate({"from": "person", "to": ["palette"]})]
    assert should_link(rules, "person", "palette") is True
    assert should_link(rules, "palette", "person") is True
    # Unmatched class.
    assert should_link(rules, "person", "forklift") is False


def test_star_glob_excludes_self() -> None:
    """``to: ['*']`` links the ``from`` class to every OTHER class, but not itself."""
    rules = [LinkLineRule.model_validate({"from": "person", "to": ["*"]})]
    assert should_link(rules, "person", "forklift") is True
    assert should_link(rules, "person", "palette") is True
    # Self-pairs never link, even under '*'.
    assert should_link(rules, "person", "person") is False
    # Pairs the rule doesn't ``from`` from also don't link.
    assert should_link(rules, "forklift", "palette") is False


def test_multiple_rules_compose() -> None:
    rules = [
        LinkLineRule.model_validate({"from": "person", "to": ["palette"]}),
        LinkLineRule.model_validate({"from": "forklift", "to": ["palette"]}),
    ]
    assert should_link(rules, "person", "palette") is True
    assert should_link(rules, "forklift", "palette") is True
    assert should_link(rules, "person", "forklift") is False


# ---- load_link_lines ----


def test_load_none_returns_empty() -> None:
    assert load_link_lines(None) == []


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_link_lines(tmp_path / "missing.yaml") == []


def test_load_valid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "link_lines.yaml"
    p.write_text(yaml.safe_dump({
        "rules": [
            {"from": "person", "to": ["palette", "forklift"], "max_distance_m": 5.0},
            {"from": "forklift", "to": ["*"]},
        ],
    }))
    rules = load_link_lines(p)
    assert len(rules) == 2
    assert rules[0].from_ == "person"
    assert rules[0].max_distance_m == 5.0
    assert rules[1].to == ["*"]


def test_load_rejects_bad_top_level(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just a list\n")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_link_lines(p)


def test_load_rejects_bad_rules_field(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"rules": "not-a-list"}))
    with pytest.raises(ValueError, match="'rules' must be a list"):
        load_link_lines(p)


def test_load_propagates_rule_validation_error(tmp_path: Path) -> None:
    """A malformed rule bubbles up as a ValueError (caller turns into HTTP 500)."""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"rules": [{"from": "", "to": ["palette"]}]}))
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        load_link_lines(p)
