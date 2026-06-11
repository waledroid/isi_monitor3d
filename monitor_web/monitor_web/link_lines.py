"""Server-side ``config/link_lines.yaml`` loader + matching helper (S16).

The dashboard's floor map draws a thin white line with a live metric distance
label between every pair of tracked objects whose classes satisfy at least one
of these rules. Rules are pure render preferences — the Backbone's `Track2D`
UDP envelope is untouched. Schema (`yaml`):

    rules:
      - from: person
        to: ["forklift", "palette", "robot"]   # or '*' for everything except `from`
        max_distance_m: 5.0                    # optional; omit/null = always draw
        color: "#ffffff"                       # optional override

The dashboard fetches the parsed rules over ``GET /api/link-lines`` and writes
through the existing ``POST /api/config`` flow (atomic + same pattern as zones).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LinkLineRule(BaseModel):
    """One distance-line rule. ``to`` accepts ``'*'`` (all classes except
    ``from``); ``max_distance_m`` is optional (None = always draw the line)."""

    from_: str = Field(..., min_length=1, alias="from")
    to: list[str] = Field(..., min_length=1)
    max_distance_m: float | None = None
    color: str | None = None

    model_config = {"populate_by_name": True}

    @field_validator("to")
    @classmethod
    def to_non_empty_strings(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for item in v:
            s = (item or "").strip()
            if not s:
                raise ValueError("'to' entries must be non-empty strings")
            out.append(s)
        return out

    @field_validator("max_distance_m")
    @classmethod
    def positive_finite_or_none(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not math.isfinite(v) or v <= 0:
            raise ValueError("max_distance_m must be a positive finite number")
        return v

    def matches(self, from_cls: str, to_cls: str) -> bool:
        """Does this rule link the (unordered) class pair (from_cls, to_cls)?

        Rules are directional in YAML but applied undirected at render time —
        we link a pair iff either ordering satisfies one rule.
        """
        if from_cls == to_cls:
            # '*' explicitly excludes the rule's own `from` class.
            return False
        if self.from_ != from_cls:
            return False
        if "*" in self.to:
            return True
        return to_cls in self.to


def parse_link_lines(raw: dict | None) -> list[LinkLineRule]:
    """Validate a link-lines doc (``{'rules': [...]}``) into rule objects.

    Shared by the legacy-file loader and the unified-config reader; raises
    ``ValueError`` on malformed rules so the API can 500 with detail.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("link_lines: top level must be a mapping")
    rules_raw = raw.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("link_lines: 'rules' must be a list")
    return [LinkLineRule.model_validate(r) for r in rules_raw]


def load_link_lines(path: Path | None) -> list[LinkLineRule]:
    """Parse a legacy standalone ``link_lines.yaml``. Missing/empty/unreadable → ``[]``.

    Kept for back-compat; the live path now reads the merged ``link_lines`` section
    of the unified dashboard config and validates it via :func:`parse_link_lines`.
    """
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"link_lines.yaml unreadable: {exc}") from exc
    return parse_link_lines(raw)


def should_link(rules: list[LinkLineRule], a_cls: str, b_cls: str) -> bool:
    """Return True if any rule links the (unordered) pair (a_cls, b_cls)."""
    for r in rules:
        if r.matches(a_cls, b_cls) or r.matches(b_cls, a_cls):
            return True
    return False


def rules_to_dict(rules: list[LinkLineRule]) -> list[dict[str, Any]]:
    """Serialize rules back to the JSON shape used by the GET endpoint + the
    YAML on-disk format (preserves ``from`` rather than the Python-safe
    ``from_`` alias)."""
    return [
        {
            "from": r.from_,
            "to": list(r.to),
            "max_distance_m": r.max_distance_m,
            "color": r.color,
        }
        for r in rules
    ]
