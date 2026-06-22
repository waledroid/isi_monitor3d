"""isical — a guided camera-calibration Studio (Multical Mode-2).

A sibling of monitor_web (runs in the monitor3d conda env): a named calibration
"project" with three phase cards — Intrinsic → Extrinsic → Export — that capture
ChArUco/AprilGrid board images live from the cameras (auto-snap on a well-detected,
steady, sharp board), then drive the existing Multical backend (calibration/) and
assemble calibration.json. Modeled on the isiGen Studio (FastAPI + JobRunner +
phase board), but CPU-only and consuming live RTSP/USB frames.
"""
