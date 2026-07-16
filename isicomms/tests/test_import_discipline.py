"""Process-boundary discipline: backbone.runtime must never be imported.

The gateway is a consumer of the UDP/JSON schema and ZoneRegistry only.
If backbone.runtime appears in sys.modules after create_app(), the process
boundary has been violated.
"""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from isicomms.app import create_app
from isicomms.config import Settings


def test_backbone_runtime_not_imported_after_create_app():
    """create_app() must not pull in backbone.runtime (or its sub-modules)."""
    app = create_app(Settings(mqtt_port=1884))
    with TestClient(app):
        pass  # lifespan runs; still no backbone.runtime

    runtime_modules = [
        name for name in sys.modules
        if name == "backbone.runtime" or name.startswith("backbone.runtime.")
    ]
    assert runtime_modules == [], (
        f"backbone.runtime was imported — process boundary violated: {runtime_modules}"
    )


def test_only_allowed_backbone_modules_are_imported():
    """Only backbone.comms.schemas and backbone.shared.zones may be imported."""
    allowed_prefixes = (
        "backbone.comms",
        "backbone.shared",
        "backbone.core",   # schemas pulls in backbone.core.types
    )
    app = create_app(Settings(mqtt_port=1884))
    with TestClient(app):
        pass

    forbidden = [
        name for name in sys.modules
        if name.startswith("backbone.")
        and not any(name.startswith(p) for p in allowed_prefixes)
        and name != "backbone"
    ]
    assert forbidden == [], (
        f"Forbidden backbone modules imported: {forbidden}"
    )
