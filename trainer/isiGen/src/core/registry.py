"""Generic plugin registry (isiGen copy of the Backbone's canonical pattern).

Used at exactly eight seams: ControlMapExtractor, Masker, Captioner,
LoraTrainer, ScaffoldSource, ImageGenerator, QualityFilter, DatasetExporter.
Implementations register themselves with a name and are instantiated by the
phase runners from the project YAML — they never instantiate each other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a plugin name is unknown or already taken."""


class Registry(Generic[T]):
    """Name → class registry for one plugin seam."""

    def __init__(self, seam: str) -> None:
        self._seam = seam
        self._impls: dict[str, type[T]] = {}

    @property
    def seam(self) -> str:
        return self._seam

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: `@registry.register("canny")` on a concrete class."""

        def decorator(cls: type[T]) -> type[T]:
            if name in self._impls:
                raise RegistryError(
                    f"{self._seam}: name {name!r} already registered "
                    f"to {self._impls[name].__name__}"
                )
            self._impls[name] = cls
            return cls

        return decorator

    def create(self, name: str, **kwargs) -> T:
        """Instantiate the implementation registered under `name`."""
        try:
            cls = self._impls[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._impls)) or "<none>"
            raise RegistryError(
                f"{self._seam}: unknown implementation {name!r} (available: {available})"
            ) from exc
        return cls(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._impls)

    def __contains__(self, name: object) -> bool:
        return name in self._impls
