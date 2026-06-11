"""Registry contract tests.

These pin the behavior the orchestrator depends on: name-based lookup,
duplicate-name rejection, helpful errors on unknown names.
"""

from __future__ import annotations

import pytest

from backbone.core import Registry, RegistryError


class _Base:
    pass


def test_register_and_create() -> None:
    reg: Registry[_Base] = Registry("Test")

    @reg.register("alpha")
    class Alpha(_Base):
        def __init__(self, x: int = 1) -> None:
            self.x = x

    instance = reg.create("alpha", x=42)
    assert isinstance(instance, Alpha)
    assert instance.x == 42


def test_duplicate_name_rejected() -> None:
    reg: Registry[_Base] = Registry("Test")

    @reg.register("alpha")
    class _A(_Base):
        pass

    with pytest.raises(RegistryError):

        @reg.register("alpha")
        class _B(_Base):
            pass


def test_unknown_name_lists_available() -> None:
    reg: Registry[_Base] = Registry("Test")

    @reg.register("alpha")
    class _A(_Base):
        pass

    @reg.register("beta")
    class _B(_Base):
        pass

    with pytest.raises(RegistryError) as excinfo:
        reg.create("gamma")

    msg = str(excinfo.value)
    assert "gamma" in msg
    assert "alpha" in msg
    assert "beta" in msg


def test_names_and_contains() -> None:
    reg: Registry[_Base] = Registry("Test")

    @reg.register("zulu")
    class _Z(_Base):
        pass

    @reg.register("alpha")
    class _A(_Base):
        pass

    assert reg.names() == ["alpha", "zulu"]
    assert "alpha" in reg
    assert "missing" not in reg


def test_five_seams_present() -> None:
    """The architecture commits to exactly five plugin seams. Pin them here."""
    from backbone.core import (
        detector_registry,
        frame_source_registry,
        metadata_sink_registry,
        tracker_registry,
        triangulator_registry,
    )

    seams = {
        frame_source_registry.seam,
        detector_registry.seam,
        tracker_registry.seam,
        triangulator_registry.seam,
        metadata_sink_registry.seam,
    }
    assert seams == {
        "FrameSource",
        "Detector",
        "Tracker",
        "Triangulator",
        "MetadataSink",
    }
