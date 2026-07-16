"""Communications layer: UDP/JSON contract with the module ecosystem.

Importing this package auto-registers the ``udp`` and ``mqtt``
``MetadataSink`` plugins. After ``import backbone.comms``,
``metadata_sink_registry`` includes ``"udp"`` — and ``"mqtt"`` when paho-mqtt
is installed. The mqtt import is DEFENSIVE: the light consumer surface
(``schemas`` + ``zones``, e.g. a lean isicomms image or an exported module
using only the wire contract) must import without the ``mqtt`` extra; a
config that names the ``mqtt`` sink still fails loudly at registry lookup.

Public API:
    * ``Publisher`` — fan-out from the pipeline to any number of sinks.
    * ``UdpSink`` — concrete UDP/JSON emitter plugin.
    * ``MqttSink`` — concrete MQTT emitter plugin.
    * ``Track2DMessage`` / ``Track3DMessage`` / ``PassingEventMessage`` /
      ``ImageRefMessage`` / ``DiagnosticsMessage`` / ``ConfigMessage`` —
      the on-wire schema, also usable by module-side consumers for typed decode.
    * ``parse_envelope`` — discriminating parser for consumer code.
"""

from . import udp_sink as _udp_sink  # noqa: F401  — registers "udp"

try:  # optional extra: paho-mqtt (see docstring)
    from . import mqtt_sink as _mqtt_sink  # noqa: F401  — registers "mqtt"
    from .mqtt_sink import MqttSink
except ImportError:  # pragma: no cover — exercised by the wheel light-surface test
    MqttSink = None  # type: ignore[assignment,misc]

from .publisher import Publisher
from .schemas import (
    SCHEMA_VERSION,
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    ImageRefMessage,
    LatencyStats,
    MessageType,
    PassingEventMessage,
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    ZoneSpec,
    parse_envelope,
)
from .udp_sink import UdpSink

__all__ = [
    "SCHEMA_VERSION",
    "CalibrationFactCheck",
    "ConfigMessage",
    "DiagnosticsMessage",
    "ImageRefMessage",
    "LatencyStats",
    "MessageType",
    "MqttSink",
    "PassingEventMessage",
    "Publisher",
    "SchemaVersionError",
    "Track2DMessage",
    "Track3DMessage",
    "UdpSink",
    "ZoneSpec",
    "parse_envelope",
]
