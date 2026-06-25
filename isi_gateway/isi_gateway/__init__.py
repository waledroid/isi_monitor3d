"""ISI Gateway — central MQTT aggregator for distributed Backbone nodes."""

from .app import create_app

__all__ = ["create_app"]
