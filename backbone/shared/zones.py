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


@dataclass(slots=True, frozen=True)
class Zone:
    """One named polygonal floor region."""

    name: str
    type: str
    polygon: np.ndarray   # (N, 2) float64, ordered vertices in meters
    # Zone-category metadata (back-compat default — kind-less YAML still loads).
    # `kind` is the category (palette|etagere|danger); `severity` is meaningful
    # for kind="danger" (info|warning|critical) and drives the dashboard's pulse.
    # Sécurité (future module) reads the same fields.
    kind: str = "palette"
    severity: str = "info"

    def __post_init__(self) -> None:
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
        self._by_name: dict[str, Zone] = {z.name: z for z in zones}
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
                )
            )
        return cls(zones)

    # ---- lookup ----

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Zone:
        return self._by_name[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())

    def by_type(self, type_: str) -> tuple[str, ...]:
        """All zone names with the given type. Empty tuple if no match."""
        return tuple(self._by_type.get(type_, ()))

    def which(self, xy_m: tuple[float, float]) -> tuple[str, ...]:
        """Names of zones containing ``xy_m`` — handles overlapping zones."""
        return tuple(name for name, z in self._by_name.items() if z.contains(xy_m))
