# G3 — End-to-end latency artifact (capture → publish)

**Date:** 2026-07-20, 16:4x CEST · **Method:** passive capture of the production
system's MQTT diagnostics heartbeat (`isiMonitor3D/+/+/diagnostics/heartbeat`,
5 s interval) for 310 s — no probe process, no interference with the live
pipeline. Latency is `LatencyMeter` capture→publish (the single capture-time
clock, per the KPI definition); each heartbeat carries `{p50,p95,p99,n}` over
the meter's rolling window (n = 2048 at capture).

**System state:** deployed default (points mode; isistream producer + metric
engine, motion gate on, TRT EP, zone-scoped detection, 2 cameras alive,
25.8 fps aggregate, frame_count 18k+ at capture; dashboard + MQTT + gateway all
running — i.e. full production load).

**Raw:** `G3_mqtt_diagnostics_20260720.jsonl` (61 heartbeats, all with data).

## Result (median across 61 heartbeats; min–max)

| Percentile | Latency | Range over window |
|---|---|---|
| p50 | **40.3 ms** | 39.5 – 42.1 |
| p95 | **78.1 ms** | 76.2 – 79.9 |
| p99 | **94.0 ms** | 90.4 – 102.4 |

**KPI check: p95 = 78 ms « 200 ms target ✅** (margin ×2.6).

## Notes for the Results section

- These figures are BETTER than the July prose numbers in CLAUDE.md
  (p50 77 / p95 126 ms) — measured after the perf-lever work (720p GPU
  downscale, pose_imgsz 480, motion gate). Report THIS artifact as primary;
  cite the config state. The motion gate's cached re-emissions are included in
  the distribution (see §2.3 of the draft) — state this as a measurement
  condition.
- fps_by_camera and per-source alive status are in the raw JSONL for T4.
- Caveat for honesty: capture_ts is stamped at the appsink callback (lags the
  optical event by the ~100 ms rtspsrc latency buffer + decode; documented in
  backbone/ingestion/rtsp.py) — the KPI is defined against this clock.
