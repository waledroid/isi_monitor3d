"""Floor-frame zones — point-in-polygon over named regions.

A *zone* is a labelled polygon on the floor plane (Z = 0), described in metric
``(X, Y)`` coordinates. Authored once per node in ``config/zones.yaml`` and
consumed by:

* ``backbone.triangulation.subscription_manager`` — match rules can say
  ``in_zone: rack_a3`` or ``in_zone: any_danger``.
* Future module side (Pallets, Sécurité) over the UDP/JSON contract — modules
  receive raw ``xy_m`` and apply their own point-in-polygon, but reading the
  same zones.yaml keeps semantics consistent.

Zones are simple single-polygon regions in v1 — no holes, no multi-polygons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def slugify_zone_id(name: str) -> str:
    """Legacy fallback id for a zones.yaml entry that predates ``id:``."""
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return safe or "zone"


@dataclass(slots=True, frozen=True)
class Zone:
    """One polygonal floor region with a STABLE identity.

    ``id`` is immutable and never reused: external systems (AGVs, WMS, MQTT
    subscribers) key off it, so deleting or renaming a zone must not disturb
    the identity of any other. ``name`` is the operator-facing label and may
    be edited freely. Files written before ids existed load with an id derived
    from the name (``slugify_zone_id``) so nothing breaks on upgrade.
    """

    name: str
    type: str
    polygon: np.ndarray   # (N, 2) float64, ordered vertices in meters
    # Zone-category metadata (back-compat default — kind-less YAML still loads).
    # `kind` is the category (palette|etagere|danger); `severity` is meaningful
    # for kind="danger" (info|warning|critical) and drives the dashboard's pulse.
    # Sécurité (future module) reads the same fields.
    kind: str = "palette"
    severity: str = "info"
    id: str = ""          # set in __post_init__ when absent (legacy files)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", slugify_zone_id(self.name))
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2:
            raise ValueError(
                f"Zone {self.name!r}: polygon must be shape (N, 2), got {self.polygon.shape}"
            )
        if self.polygon.shape[0] < 3:
            raise ValueError(
                f"Zone {self.name!r}: polygon needs ≥3 vertices, got {self.polygon.shape[0]}"
            )

    def contains(self, xy_m: tuple[float, float]) -> bool:
        """Return True if ``xy_m`` lies inside (or on the boundary of) the polygon.

        Pure-numpy replacement for ``cv2.pointPolygonTest(..., measureDist=False)``
        which returns +1 inside / 0 on-edge / -1 outside; the old code returned
        ``signed >= 0`` ⇒ inside OR on the boundary. We preserve that exactly: an
        explicit on-edge/on-vertex check returns True, and a ray-casting (even-odd)
        interior test handles the strictly-inside case. cv2 reads its contour as
        float32, so we match that precision to agree on near-boundary float ties.
        """
        x, y = float(xy_m[0]), float(xy_m[1])
        poly = self.polygon.astype(np.float32)
        n = poly.shape[0]

        # On-boundary check (edge or vertex) → inside, matching cv2's 0 → >= 0.
        for i in range(n):
            ax, ay = float(poly[i, 0]), float(poly[i, 1])
            bx, by = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
            # Collinear (cross == 0) AND within the segment's bounding box.
            cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
            if cross == 0.0:
                if (
                    min(ax, bx) <= x <= max(ax, bx)
                    and min(ay, by) <= y <= max(ay, by)
                ):
                    return True

        # Ray casting (even-odd) for the strict interior.
        inside = False
        for i in range(n):
            ax, ay = float(poly[i, 0]), float(poly[i, 1])
            bx, by = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
            if (ay > y) != (by > y):
                x_cross = (bx - ax) * (y - ay) / (by - ay) + ax
                if x < x_cross:
                    inside = not inside
        return inside


class ZoneRegistry:
    """All zones loaded from ``zones.yaml``, indexed by name and by type."""

    def __init__(self, zones: list[Zone]) -> None:
        names = [z.name for z in zones]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate zone names: {duplicates}")
        ids = [z.id for z in zones]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate zone ids: {duplicates}")
        self._by_name: dict[str, Zone] = {z.name: z for z in zones}
        self._by_id: dict[str, Zone] = {z.id: z for z in zones}
        self._by_type: dict[str, list[str]] = {}
        for z in zones:
            self._by_type.setdefault(z.type, []).append(z.name)

    @classmethod
    def empty(cls) -> ZoneRegistry:
        return cls([])

    @classmethod
    def load(cls, path: str | Path) -> ZoneRegistry:
        """Load from a YAML file. See ``config/zones.yaml.example`` for the schema."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZoneRegistry:
        raw = data.get("zones", [])
        if not isinstance(raw, list):
            raise ValueError("zones.yaml: top-level 'zones' must be a list")
        zones: list[Zone] = []
        for entry in raw:
            if "name" not in entry or "type" not in entry or "polygon" not in entry:
                raise ValueError(
                    f"zone entry missing required keys (need name, type, polygon): {entry}"
                )
            polygon = np.asarray(entry["polygon"], dtype=np.float64)
            zones.append(
                Zone(
                    name=entry["name"],
                    type=entry["type"],
                    polygon=polygon,
                    kind=entry.get("kind", "palette"),
                    severity=entry.get("severity", "info"),
                    id=str(entry.get("id") or ""),   # "" ⇒ derived from name
                )
            )
        return cls(zones)

    # ---- lookup ----

    def by_id(self, zone_id: str) -> Zone | None:
        """The zone with this STABLE id, or None. External systems key here."""
        return self._by_id.get(zone_id)

    def id_of(self, name: str) -> str | None:
        z = self._by_name.get(name)
        return z.id if z is not None else None

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Zone:
        return self._by_name[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())

    @property
    def ids(self) -> tuple[str, ...]:
        """STABLE ids of every zone, in load order. Internal state keys off these."""
        return tuple(self._by_id.keys())

    def name_of(self, zone_id: str) -> str | None:
        """Operator-facing label for a stable id, or None if the id is unknown."""
        z = self._by_id.get(zone_id)
        return z.name if z is not None else None

    def by_type(self, type_: str) -> tuple[str, ...]:
        """All zone names with the given type. Empty tuple if no match."""
        return tuple(self._by_type.get(type_, ()))

    def which(self, xy_m: tuple[float, float]) -> tuple[str, ...]:
        """Names of zones containing ``xy_m`` — handles overlapping zones."""
        return tuple(name for name, z in self._by_name.items() if z.contains(xy_m))

    def which_ids(self, xy_m: tuple[float, float]) -> tuple[str, ...]:
        """STABLE ids of zones containing ``xy_m``.

        Prefer this over ``which`` for internal state that must survive a
        zone rename: ids are immutable, names are not.
        """
        return tuple(zid for zid, z in self._by_id.items() if z.contains(xy_m))


class ZoneMembershipHysteresis:
    """Debounce per-track zone membership at polygon boundaries.

    Raw point-in-polygon flips every few frames for an object sitting ON a
    zone edge (live 2026-08-06: a carton at the Sortie_1 boundary flapped
    the zone's object list and spammed passings although nothing moved). A
    track ENTERS a zone after ``enter_after`` consecutive inside frames
    (1 = immediately) and LEAVES only after ``exit_after`` consecutive
    outside frames (~0.6 s at 13 fps with the default 8) — so a genuine
    exit still registers fast while boundary jitter cannot flap the state.
    """

    def __init__(self, exit_after: int = 15, enter_after: int = 1) -> None:
        self._exit_after = int(exit_after)
        self._enter_after = int(enter_after)
        self._member: dict[int, set[str]] = {}
        self._in_streak: dict[tuple[int, str], int] = {}
        self._out_streak: dict[tuple[int, str], int] = {}

    def update(self, track_id: int, raw: tuple[str, ...]) -> tuple[str, ...]:
        """Fold this frame's raw membership into the debounced one."""
        raw_set = set(raw)
        member = self._member.setdefault(track_id, set())
        for zid in raw_set - member:
            key = (track_id, zid)
            self._in_streak[key] = self._in_streak.get(key, 0) + 1
            if self._in_streak[key] >= self._enter_after:
                member.add(zid)
                self._in_streak.pop(key, None)
        for key in [k for k in self._in_streak
                    if k[0] == track_id and k[1] not in raw_set]:
            self._in_streak.pop(key, None)
        for zid in list(member):
            key = (track_id, zid)
            if zid in raw_set:
                self._out_streak.pop(key, None)
            else:
                self._out_streak[key] = self._out_streak.get(key, 0) + 1
                if self._out_streak[key] >= self._exit_after:
                    member.discard(zid)
                    self._out_streak.pop(key, None)
        return tuple(sorted(member))

    def forget(self, live_ids: set[int]) -> None:
        """Drop state for tracks that no longer exist."""
        for tid in [t for t in self._member if t not in live_ids]:
            del self._member[tid]
        self._in_streak = {k: v for k, v in self._in_streak.items()
                           if k[0] in live_ids}
        self._out_streak = {k: v for k, v in self._out_streak.items()
                            if k[0] in live_ids}
