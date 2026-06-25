"""Metadata layer: UDP/JSON contract with the module ecosystem.

Importing this package auto-registers the ``udp`` and ``mqtt``
``MetadataSink`` plugins. After ``import backbone.metadata``,
``metadata_sink_registry`` includes both ``"udp"`` and ``"mqtt"``.

Public API:
    * ``Publisher`` — fan-out from the pipeline to any number of sinks.
    * ``UdpSink`` — concrete UDP/JSON emitter plugin.
    * ``MqttSink`` — concrete MQTT emitter plugin.
    * ``Track2DMessage`` / ``Track3DMessage`` / ``PassingEventMessage`` —
      the on-wire schema, also usable by module-side consumers for typed decode.
    * ``parse_envelope`` — discriminating parser for consumer code.
"""

from . import mqtt_sink as _mqtt_sink  # noqa: F401  — registers "mqtt"
from . import udp_sink as _udp_sink  # noqa: F401  — registers "udp"
from .mqtt_sink import MqttSink
from .publisher import Publisher
from .schemas import (
    SCHEMA_VERSION,
    MessageType,
    PassingEventMessage,
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    parse_envelope,
)
from .udp_sink import UdpSink

__all__ = [
    "SCHEMA_VERSION",
    "MessageType",
    "MqttSink",
    "PassingEventMessage",
    "Publisher",
    "SchemaVersionError",
    "Track2DMessage",
    "Track3DMessage",
    "UdpSink",
    "parse_envelope",
]
