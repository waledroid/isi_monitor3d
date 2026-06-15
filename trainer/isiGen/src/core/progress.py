"""Lightweight progress hook.

Phase runners call ``report(done, total, label)`` inside their loops. A host
(the Studio JobRunner) installs a sink via ``set_sink`` to capture it onto the
running job; with no sink installed (CLI use) it is a no-op. Only one phase job
runs at a time, so a single module-level sink is safe.
"""

from __future__ import annotations

from collections.abc import Callable

_sink: Callable[[int, int, str], None] | None = None


def set_sink(fn: Callable[[int, int, str], None] | None) -> None:
    global _sink
    _sink = fn


def report(done: int, total: int, label: str = "") -> None:
    if _sink is not None:
        try:
            _sink(int(done), int(total), label)
        except Exception:
            pass
