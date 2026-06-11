"""Tests for the dashboard LOGS-panel formatter (monitor_web/log_format.py)."""
from monitor_web.log_format import format_lines, parse_line


def test_parse_wellformed_line():
    row = parse_line(
        "15:57:48 | backbone.detection.rfdetr_onnx_seg | INFO | loaded model"
    )
    assert row.time == "15:57:48"
    assert row.source == "detect"        # detection.* → short tag
    assert row.level == "info"
    assert row.message == "loaded model"
    assert row.count == 1


def test_source_tags_map_known_modules():
    assert parse_line("00:00:00 | backbone.runtime.orchestrator | INFO | x").source == "core"
    assert parse_line("00:00:00 | backbone.ingestion._gst_source | ERROR | x").source == "cam"
    assert parse_line("00:00:00 | backbone.metadata.udp_sink | WARNING | x").source == "bus"
    # Unknown module → last path segment.
    assert parse_line("00:00:00 | backbone.weird.thing | INFO | x").source == "thing"


def test_nonmatching_line_is_raw():
    raw = '  File "foo.py", line 12, in bar'
    row = parse_line(raw)
    assert row.level == "raw"
    assert row.message == raw
    assert row.time == ""


def test_collapse_consecutive_duplicates_keeps_latest_time():
    lines = [
        "15:00:01 | backbone.ingestion._gst_source | ERROR | RTSP not-linked",
        "15:00:02 | backbone.ingestion._gst_source | ERROR | RTSP not-linked",
        "15:00:03 | backbone.ingestion._gst_source | ERROR | RTSP not-linked",
    ]
    rows = format_lines(lines)
    assert len(rows) == 1
    assert rows[0].count == 3
    assert rows[0].level == "error"
    assert rows[0].source == "cam"
    assert rows[0].time == "15:00:03"   # most recent timestamp retained


def test_non_consecutive_duplicates_not_collapsed():
    lines = [
        "15:00:01 | backbone.runtime.orchestrator | INFO | a",
        "15:00:02 | backbone.runtime.orchestrator | INFO | b",
        "15:00:03 | backbone.runtime.orchestrator | INFO | a",
    ]
    assert len(format_lines(lines)) == 3


def test_raw_lines_never_collapse():
    # Identical traceback lines stay separate (raw is never merged).
    assert len(format_lines(["trace x", "trace x"])) == 2
