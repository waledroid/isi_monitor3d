"""ISI Monitor 3D operator dashboard.

Separate FastAPI process from the Backbone. Consumes UDP envelopes per the
public ``backbone.comms.schemas`` contract; never imports
``backbone.runtime``.
"""

from .app import create_app

__all__ = ["create_app"]
