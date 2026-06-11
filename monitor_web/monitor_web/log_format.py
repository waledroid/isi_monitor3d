"""Format Backbone log lines for the dashboard LOGS panel.

The supervisor's ring buffer holds raw subprocess lines like::

    15:57:48 | backbone.detection.rfdetr_onnx_seg | INFO | RfdetrOnnxSegDetector: loaded …

This module parses them into clean rows (short source tag, level, message) and
collapses consecutive identical lines into one row with a count, so the panel
reads as a short, meaningful event list instead of a raw scroll. Lines that don't
match the expected shape (multi-line tracebacks, stray prints) pass through as
``level="raw"`` so nothing is silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

# Map a backbone module path (leading "backbone." stripped) to a short source tag.
_SOURCE_TAGS = {
    "runtime": "core",
    "detection": "detect",
    "ingestion": "cam",
    "homography": "geo",
    "triangulation": "geo",
    "metadata": "bus",
    "calibration": "calib",
}

_KNOWN_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(slots=True)
class LogRow:
    time: str
    source: str
    level: str       # debug/info/warning/error/critical, or "raw" for unparsed
    message: str
    count: int = 1


def _short_source(module: str) -> str:
    """``backbone.detection.rfdetr_onnx_seg`` → ``detect``; unknown → last segment."""
    mod = module.removeprefix("backbone.")
    head = mod.split(".", 1)[0]
    return _SOURCE_TAGS.get(head, mod.rsplit(".", 1)[-1])


def parse_line(raw: str) -> LogRow:
    """Parse ``TIME | module | LEVEL | message``. Non-matching lines pass through
    as ``level="raw"`` with the original text preserved."""
    parts = raw.split(" | ", 3)
    if len(parts) == 4 and parts[2].strip().upper() in _KNOWN_LEVELS:
        time, module, level, message = parts
        return LogRow(
            time=time.strip(),
            source=_short_source(module.strip()),
            level=level.strip().lower(),
            message=message.strip(),
        )
    return LogRow(time="", source="", level="raw", message=raw)


def format_lines(raw_lines) -> list[LogRow]:
    """Parse each line and collapse consecutive identical ``(level, message)`` rows
    into one with a ``count``. Order preserved (oldest→newest, as stored). Raw lines
    are never collapsed."""
    rows: list[LogRow] = []
    for raw in raw_lines:
        row = parse_line(raw)
        last = rows[-1] if rows else None
        if (last is not None and row.level != "raw"
                and last.level == row.level and last.message == row.message):
            last.count += 1
            if row.time:
                last.time = row.time   # keep the most recent timestamp
        else:
            rows.append(row)
    return rows
