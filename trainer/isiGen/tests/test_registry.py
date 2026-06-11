import pytest
from src.core.registry import Registry, RegistryError


def test_register_create_names_contains():
    reg = Registry("Widget")
    @reg.register("alpha")
    class Alpha:
        def __init__(self, x=1):
            self.x = x
    inst = reg.create("alpha", x=7)
    assert isinstance(inst, Alpha) and inst.x == 7
    assert reg.names() == ["alpha"]
    assert "alpha" in reg and "beta" not in reg


def test_duplicate_name_raises():
    reg = Registry("Widget")
    @reg.register("a")
    class A: ...
    with pytest.raises(RegistryError, match="already registered"):
        @reg.register("a")
        class B: ...


def test_unknown_create_lists_available():
    reg = Registry("Widget")
    @reg.register("only")
    class Only: ...
    with pytest.raises(RegistryError, match="available: only"):
        reg.create("nope")
