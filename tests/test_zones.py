"""``Zone`` / ``ZoneRegistry`` — point-in-polygon + YAML round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from backbone.shared.zones import Zone, ZoneRegistry

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def test_zone_contains_inside_point() -> None:
    z = Zone("z", "danger", SQUARE)
    assert z.contains((1.0, 1.0))


def test_zone_does_not_contain_outside_point() -> None:
    z = Zone("z", "danger", SQUARE)
    assert not z.contains((3.0, 3.0))
    assert not z.contains((-0.1, 1.0))


def test_zone_includes_boundary() -> None:
    """cv2.pointPolygonTest returns 0 on edge — we treat that as inside."""
    z = Zone("z", "danger", SQUARE)
    assert z.contains((2.0, 1.0))
    assert z.contains((0.0, 0.0))


def test_zone_contains_clearly_inside() -> None:
    z = Zone("z", "danger", SQUARE)
    assert z.contains((0.5, 0.5))
    assert z.contains((1.9, 0.1))


def test_zone_contains_clearly_outside() -> None:
    z = Zone("z", "danger", SQUARE)
    assert not z.contains((5.0, 5.0))
    assert not z.contains((1.0, 2.5))


def test_zone_contains_point_on_edge() -> None:
    """A point lying on an edge (not a vertex) → True (cv2's 0 → >= 0)."""
    z = Zone("z", "danger", SQUARE)
    assert z.contains((1.0, 0.0))   # midpoint of bottom edge
    assert z.contains((2.0, 1.0))   # midpoint of right edge


def test_zone_contains_point_on_vertex() -> None:
    """A point exactly on a vertex → True."""
    z = Zone("z", "danger", SQUARE)
    assert z.contains((0.0, 0.0))
    assert z.contains((2.0, 2.0))


def test_zone_contains_just_outside_near_edge() -> None:
    """Just outside an edge → False."""
    z = Zone("z", "danger", SQUARE)
    assert not z.contains((-0.001, 1.0))
    assert not z.contains((2.001, 1.0))


def test_zone_contains_matches_cv2_random() -> None:
    """The pure-numpy contains() must agree with cv2.pointPolygonTest >= 0."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(20260626)
    mismatches = 0
    total = 0
    for _ in range(200):
        n_verts = int(rng.integers(3, 8))
        poly = rng.uniform(-5.0, 5.0, size=(n_verts, 2))
        z = Zone("z", "danger", poly)
        contour = poly.astype(np.float32).reshape(-1, 1, 2)
        for _ in range(40):
            pt = (float(rng.uniform(-6.0, 6.0)), float(rng.uniform(-6.0, 6.0)))
            ours = z.contains(pt)
            theirs = cv2.pointPolygonTest(contour, pt, False) >= 0
            total += 1
            if ours != theirs:
                mismatches += 1
    # Allow the rare exact-boundary float tie; demand near-perfect agreement.
    assert mismatches <= total * 0.001, f"{mismatches}/{total} mismatches vs cv2"


def test_zone_rejects_too_few_vertices() -> None:
    with pytest.raises(ValueError, match="3 vertices"):
        Zone("bad", "danger", np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_zone_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        Zone("bad", "danger", np.array([[0.0, 0.0, 0.0]]))


# ---- ZoneRegistry ----


def test_registry_empty() -> None:
    reg = ZoneRegistry.empty()
    assert len(reg) == 0
    assert reg.names == ()
    assert reg.by_type("anything") == ()
    assert reg.which((1.0, 1.0)) == ()


def test_registry_lookup_by_name() -> None:
    a = Zone("a", "storage", SQUARE)
    b = Zone("b", "danger", SQUARE + np.array([3.0, 0.0]))
    reg = ZoneRegistry([a, b])
    assert "a" in reg
    assert "missing" not in reg
    assert reg["a"] is a


def test_registry_by_type_groups_zones() -> None:
    reg = ZoneRegistry([
        Zone("s1", "storage", SQUARE),
        Zone("s2", "storage", SQUARE + np.array([5.0, 0.0])),
        Zone("d1", "danger", SQUARE + np.array([0.0, 5.0])),
    ])
    assert set(reg.by_type("storage")) == {"s1", "s2"}
    assert set(reg.by_type("danger")) == {"d1"}
    assert reg.by_type("nope") == ()


def test_registry_which_finds_containing_zones() -> None:
    reg = ZoneRegistry([
        Zone("z1", "storage", SQUARE),                              # contains (1, 1)
        Zone("z2", "storage", SQUARE + np.array([10.0, 10.0])),      # does not
    ])
    assert reg.which((1.0, 1.0)) == ("z1",)
    assert reg.which((20.0, 20.0)) == ()


def test_registry_which_handles_overlapping_zones() -> None:
    reg = ZoneRegistry([
        Zone("outer", "danger", np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])),
        Zone("inner", "danger", np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])),
    ])
    assert set(reg.which((2.0, 2.0))) == {"outer", "inner"}


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ZoneRegistry([Zone("dup", "a", SQUARE), Zone("dup", "b", SQUARE)])


def test_registry_load_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump({
        "zones": [
            {
                "name": "rack_a3",
                "type": "storage",
                "polygon": [[1.0, 0.0], [3.0, 0.0], [3.0, 1.5], [1.0, 1.5]],
            },
        ],
    }))
    reg = ZoneRegistry.load(path)
    assert reg.names == ("rack_a3",)
    assert reg["rack_a3"].type == "storage"
    assert reg["rack_a3"].contains((2.0, 0.5))


def test_registry_load_missing_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("zones:\n  - { name: x }\n")
    with pytest.raises(ValueError, match="required keys"):
        ZoneRegistry.load(path)


def test_registry_load_top_level_must_be_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("zones: not_a_list\n")
    with pytest.raises(ValueError, match="must be a list"):
        ZoneRegistry.load(path)


def test_registry_load_handles_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    reg = ZoneRegistry.load(path)
    assert len(reg) == 0


# ---- S10 extensions: kind + severity (back-compat defaults) ----


def test_zone_defaults_kind_and_severity() -> None:
    """Callers that don't pass kind/severity get the defaults."""
    z = Zone("z", "storage", SQUARE)
    assert z.kind == "palette"
    assert z.severity == "info"


def test_zone_accepts_kind_and_severity() -> None:
    z = Zone("z", "danger", SQUARE, kind="danger", severity="critical")
    assert z.kind == "danger"
    assert z.severity == "critical"


def test_registry_load_back_compat_without_kind(tmp_path: Path) -> None:
    """Pre-S10 zones.yaml (no kind/severity keys) still loads."""
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump({
        "zones": [{"name": "z", "type": "danger",
                   "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}],
    }))
    reg = ZoneRegistry.load(path)
    assert reg["z"].kind == "palette"
    assert reg["z"].severity == "info"


def test_registry_load_reads_kind_and_severity(tmp_path: Path) -> None:
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump({
        "zones": [{
            "name": "press",
            "type": "danger",
            "kind": "danger",
            "severity": "critical",
            "polygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        }],
    }))
    reg = ZoneRegistry.load(path)
    assert reg["press"].kind == "danger"
    assert reg["press"].severity == "critical"
