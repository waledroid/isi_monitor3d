# 3. Results

All figures in this section trace to on-disk artifacts: the measurement-campaign records under `thesis/measurements/`, training-run files under `trainer/isidet/`, and the deployed calibration under `isical/data/c1/`. Sample sizes and measurement conditions are stated with each table. This section reports observations only; interpretation follows in Section 4.

## 3.1 Detector accuracy (validation split)

**Data provenance.** A data-provenance audit of the training corpus, whose record is archived with the measurement artifacts, established two facts that frame every accuracy number below. First, the three-class corpus used for all reported detector training (`pallet3`: 5,540 training / 1,049 validation images; classes `palette`, `carton`, `polybag`) consists entirely of real photographs — filename, byte-level hash, timeline, and visual checks found zero synthetic images in it. Second, the corpus has no independent held-out test split: the COCO-format `test` folder is a byte-identical duplicate of the validation folder, created only because the RF-DETR trainer requires the directory to exist. **All detector accuracy reported in this article is therefore validation-split accuracy**, and no test-split figures are given; the implications are discussed with the limitations in Section 4.

**Aggregate accuracy.** Table T1 reports the three detector configurations evaluated, all trained on `pallet3` under the isidet trainer (Section 2.6). For the RF-DETR run, which logs per-epoch metrics for both regular and exponential-moving-average (EMA) weights, we adopt the following checkpoint-selection rule, mirroring the YOLO trainer's own `best.pt` convention: **the EMA checkpoint of the epoch maximizing validation box mAP@0.5:0.95** (here epoch 23 of 41; the non-EMA metrics at the same epoch are 0.971 / 0.930).

**Table T1 — Detector accuracy on the pallet3 validation split (1,049 real images, 1,436 instances; three classes).**

| Model (checkpoint) | Input | Box P | Box R | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| YOLO26l-seg (`best.pt`, epoch 154/172) | 640 | 0.950 | 0.951 | **0.977** | **0.948** | 0.972 | 0.921 |
| YOLO26n-seg (`best.pt`, epoch 89/100) | 320 | 0.915 | 0.899 | 0.962 | 0.895 | 0.953 | 0.846 |
| RF-DETR medium-seg (best-EMA, epoch 23/41) | 432 | 0.953 | 0.933 | 0.973 | 0.938 | 0.962 | 0.906 |

Sources: YOLO26l row — independent re-evaluation (command recorded in the measurement archive; `isi-train` env, ultralytics 8.4.22, torch 2.10.0+cu128, RTX 5070); YOLO26n row — the run's `results.csv` at its best epoch; RF-DETR row — the run's `metrics.csv` at the selected checkpoint.

**Reproduction.** The headline YOLO26l-seg numbers were regenerated from the stored `best.pt` by an independent validation pass on the same 1,049 images: box mAP@0.5 reproduced exactly (0.977) and box mAP@0.5:0.95 within 0.001 of the training-time report (0.948 vs 0.947); precision/recall differ slightly (0.950/0.951 vs 0.960/0.939) because the evaluator selects a different F1-optimal confidence point per run. The re-evaluation also recorded the mask mAPs (0.972 / 0.921), which the original training report did not headline. Evaluation speed in that pass was 1.2 ms preprocess / 10.3 ms inference / 1.2 ms postprocess per image at batch 8. Raw outputs (PR curves, confusion matrices) are archived with the measurement artifacts.

**Per-class accuracy.** Table T2 gives the per-class breakdown of the independently re-evaluated YOLO26l-seg model; class↔label mapping is taken directly from the dataset's `data.yaml` and is authoritative by construction.

**Table T2 — YOLO26l-seg per-class validation accuracy (independent re-evaluation).**

| Class | Images | Instances | Box P | Box R | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|---|
| palette | 808 | 1,047 | 0.973 | 0.945 | 0.989 | 0.933 | 0.974 | 0.895 |
| carton | 74 | 174 | 0.909 | 0.948 | 0.960 | 0.943 | 0.960 | 0.900 |
| polybag | 186 | 215 | 0.967 | 0.958 | 0.983 | 0.969 | 0.980 | 0.967 |

For RF-DETR medium-seg, the trainer's COCO evaluator reports per-class box AP at the selected checkpoint of 0.916 (palette), 0.905 (carton), and 0.971 (polybag).

**KPI observation.** All three configurations exceed the mAP@0.5 ≥ 0.90 acceptance target on the validation split (0.962–0.977); the smallest model does so at a 320 px input.

**KPI coverage note.** The pallet empty/full classification target (precision/recall ≥ 0.95/0.93) is **not validated in this work**: the two-estimator occupancy mechanism is implemented and deployed (Section 2.5), but no labeled empty/full evaluation was performed. This gap is recorded with the limitations in Section 4.

## 3.2 Calibration accuracy

The deployed two-camera rig (project `c1`) was calibrated with the two-stage workflow of Section 2.3: 25 accepted ChArUco shots per camera (intrinsics), 8 synchronized pairs of the 6-board AprilGrid target (extrinsics, intrinsics fixed), and 8 flat ChArUco floor placements per camera (world anchor, RANSAC consensus plane).

**Table T3 — Deployed-rig calibration reprojection RMS (`isical/data/c1/`).**

| Quantity | cam_a | cam_b | Applicable gate |
|---|---|---|---|
| Intrinsic-stage reprojection RMS (25 shots/cam) | 0.603 px | 0.451 px | — (feeds the fixed-K extrinsic solve) |
| Joint extrinsic consensus reprojection RMS | 1.621 px | 1.621 px | ≤ 2.0 px (assembly gate) |

The written calibration passed the 2.0 px assembly gate with a joint consensus RMS of 1.621 px on both cameras (`calibration_refined.json`). This bundle-adjustment residual is the quantity checked against the ≤ 2 px homography-error acceptance target at calibration time; its relation to runtime floor-projection error is discussed in Section 4. The intrinsic-stage values (`work/intrinsic_rms.json`) are the per-camera ChArUco solve residuals consumed, with intrinsics frozen, by the joint solve.

## 3.3 End-to-end runtime

**Live latency measurement.** End-to-end capture→publish latency was measured on the live production system by passively recording its MQTT diagnostics heartbeat for 310 s — 61 heartbeats, each carrying the `LatencyMeter` percentiles over a rolling window of n = 2,048 published messages; no probe process touched the pipeline. System state at capture: the deployed default configuration (points mode with the isistream producer and metric engine as separate processes, motion gate enabled, pose stride 1 — the pose model runs on every non-gated tick, `pose_every_n: 1` — TensorRT execution provider, zone-scoped detection, both cameras alive, 25.8 fps aggregate, frame count > 18,000), with the dashboard, MQTT broker, and gateway all running — i.e., full production load. Whole-GPU memory under this load was 7.6 of 12.2 GiB (a machine total across all processes, including the desktop and dashboard; WSL2 exposes no per-process VRAM attribution). Because the motion gate was active, its cached re-emissions (Section 2.4) are included in the measured distribution. Latency is defined against `capture_ts`, whose clock definition and ~100 ms pre-appsink lag are stated in Section 2.4; the KPI is defined against this clock.

**Table T4 — Capture→publish latency, live production system (median across 61 heartbeats; min–max over the 310 s window).**

| Percentile | Latency | Range |
|---|---|---|
| p50 | **40.3 ms** | 39.5 – 42.1 ms |
| p95 | **78.1 ms** | 76.2 – 79.9 ms |
| p99 | **94.0 ms** | 90.4 – 102.4 ms |

The p95 of 78.1 ms meets the < 200 ms acceptance target with a ×2.6 margin. Throughout the 310 s window every heartbeat reported the dual-camera mode with both sources alive and balanced per-camera rates (≈13.4–13.8 fps per camera, ≈26 fps aggregate). An earlier configuration of the same system — before the ingest downscale to 720p, the pose input-size reduction, and the motion gate were introduced — had been measured at p50 77 / p95 126 ms; that prior figure is retained here only to date the configuration change, and the live measurement above is the reported result.

[Figure F5: Capture→publish latency of the live production system over the 310 s measurement window — per-heartbeat p50/p95/p99 traces (or distribution box plot) from the 61 diagnostics heartbeats, with the 200 ms KPI line.]

[Figure F6: The operator dashboard during the measured production run — live camera view with instance-segmented pallets, configured zone polygons, a tracked person with pose skeleton and metric proximity distances (1.0 m / 0.5 m to a zone), per-zone occupancy cards driven by the MQTT zone-state stream, and the runtime status panel (dual-camera Live, Track2D count, UDP counters, GPU memory).]

**Execution-provider benchmark.** The production detector (`yolo26n-seg`, fp16 ONNX, three classes, mask decoding enabled as deployed) was benchmarked under the TensorRT and CUDA execution providers of ONNX Runtime 1.23.2 (TensorRT 10.16, CUDA 12.9, RTX 5070, WSL2), at the production input size (320) and at 640. Each (EP × size) configuration ran in its own process, strictly sequentially: N = 50 timed calls after 5 warm-ups, for both the isolated `session.run` (pure inference) and the full `detect()` call (preprocess + inference + NMS + mask decode). The TensorRT engine cache was warm (detector build 1.0–1.9 s; no engine compilation occurred), so the figures reflect steady-state production behavior. The operator dashboard held ≈5 GB VRAM and was streaming throughout; medians are therefore the robust statistic, and occasional p95 spikes (e.g., CUDA-640 inference p95 62.6 ms vs median 16.4 ms) coincide with this background load.

**Table T5 — TensorRT vs CUDA execution provider (N = 50 per cell; medians).**

| Measurement | EP | 320 px | 640 px |
|---|---|---|---|
| Isolated inference (median ms) | CUDA | 13.85 | 16.39 |
| | TensorRT | 4.55 | 5.83 |
| | *TRT speedup* | *3.0×* | *2.8×* |
| Full `detect()` (median ms) | CUDA | 22.35 | 64.35 |
| | TensorRT | 8.74 | 46.68 |
| | *TRT speedup* | *2.6×* | *1.4×* |
| Bench VRAM footprint (Δ MB) | CUDA | 207 | 432 |
| | TensorRT | 309 | 321 |

At 640 px, the full-`detect()` gain collapses to 1.4× although the isolated-inference gain remains 2.8×: of the 46.7 ms TensorRT median, ≈41 ms is CPU-side work (letterboxing, fp16 conversion, NMS, and full-frame mask assembly with `decode_masks=True`). VRAM deltas are the benchmark's own footprint over a ≈5 GB baseline held by the concurrently running dashboard.

## 3.4 Synthetic-data training ablation (polybag)

The contribution of the isiGen synthetic data is measured by a single-class (polybag) instance-segmentation training ablation: three training arms with identical model and hyperparameters (`yolo26n-seg.pt`, 640 px, 80 epochs, batch 8, fixed seeds), each evaluated on one common **real** test set never seen by any arm — the 186 polybag-containing images of the pallet3 validation split (215 instances). Arm S (synthetic-only) trains on the 238 training images among the 267 CLIP-filtered SDXL+ControlNet+LoRA generations of the isiGen export (the remaining 29 form its validation split), with the 53 real LoRA-source photos explicitly excluded; Arm R (real-only) trains on a count-matched random subsample (seed 42) of the 737 polybag-containing pallet3 *training* images; Arm R+S trains on their union. Each arm's checkpoint is its own `best.pt`, selected on that arm's validation split — Arm S therefore selects on synthetic validation images, Arms R and R+S on their own splits. Leakage checks: the test set draws only from pallet3 *val*, Arm R only from pallet3 *train* (split preserved from the source dataset), the synthetic images share zero bytes with pallet3 (the provenance audit's hash/size intersection: 0 collisions), and an md5 intersection of the 53 real LoRA-source photos against the 186 test images found 0 overlaps (recorded in the ablation record) — the generator never saw any test image.

**Table T6 — Ablation on the common real test set (186 images, 215 polybag instances).**

| Arm | Train images (real/syn) | Box P | Box R | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| S (synthetic-only) | 0 / 238 | 0.319 | 0.391 | 0.223 | 0.185 | 0.219 | 0.179 |
| R (real-only) | 238 / 0 | 0.922 | 0.885 | 0.941 | 0.927 | 0.946 | 0.930 |
| R+S (merged) | 238 / 238 | 0.906 | 0.935 | **0.962** | **0.950** | 0.965 | 0.950 |

All three trainings completed normally (exit code 0; 80 epochs each; 0.26 h for Arm R and 0.50 h for Arm R+S on the RTX 5070); raw per-run outputs and logs are archived with the measurement artifacts.

Three observations follow from the table. First, the synthetic-only model is a functioning detector on synthetic imagery — on its own synthetic validation split it reaches box mAP@0.5 ≈ 0.936 — but transfers to real frames at 0.223 box mAP@0.5: a real-transfer drop of ~0.71 mAP@0.5 for this class and pipeline configuration. Second, adding the 238 synthetic images to the 238 real ones raises every mAP metric on the real test set: box mAP@0.5 +0.021 (0.941 → 0.962), box mAP@0.5:0.95 +0.023 (0.927 → 0.950), mask mAP@0.5:0.95 +0.020 (0.930 → 0.950), and recall +0.050 (0.885 → 0.935), while precision moves from 0.922 to 0.906. Third, as a measurement condition, each arm ran with a single seed; differences of ~0.02 mAP are within single-seed training noise for datasets of this size. The test set contains only polybag-positive real images from one site's camera family, so false positives on object-free frames are not measured here. Interpretation of these observations — including how the R+S margins relate to the noise floor — is deferred to Section 4.

## 3.5 Geometric verification bounds (synthetic ground truth)

The geometric core's accuracy is continuously verified against synthetic ground truth by the repository's hermetic end-to-end tests (721 tests collected on 2026-07-20); these are **software verification bounds under a synthetic camera model, not field accuracy measurements** — no tape-measured ground-truth campaign has yet been run on the deployed rig (Section 4).

**Table T7 — Synthetic end-to-end accuracy bounds enforced by the test suite.**

| Path | Condition | Enforced bound |
|---|---|---|
| Homography chain (foot → `Track2D`) | zero pixel noise | ≤ 1 mm vs ground truth (asserted at 10⁻³ m) |
| Homography chain | 2 px Gaussian noise on every detection | < 10 cm vs ground truth |
| Triangulation (foot centroid → `Track3D`) | zero pixel noise, 2 cameras | ≤ 1 mm in X, Y, Z |
| Reprojection gate | all triangulations | max per-view error ≤ 5 px (deployed default; 5–8 px allowed) |

The bounds are assertions in `tests/test_e2e_homography_synthetic.py` and `tests/test_e2e_triangulation_synthetic.py`, exercised on every suite run; the pipeline under test is composed from the production classes (projector, fusion, gates, trackers, triangulator), not mocks.

<!--
Source traceability — §3 Results (drafted 2026-07-20):

3.1 Detector accuracy:
  /home/aatanda/isi_monitor3d/thesis/measurements/G0_data_provenance.md (all-real pallet3;
    no held-out test split — test folder byte-identical to valid; dataset_v2 never trained on)
  /home/aatanda/isi_monitor3d/thesis/measurements/G1_test_split_eval.md (yolo26l re-eval:
    aggregate + per-class table (now T2) verbatim; reproduction deltas; speed line; raw outputs path)
  /home/aatanda/isi_monitor3d/trainer/isidet/runs/segment/models/yolo/yolo26l-seg_e200_640px_09-06-2026_00-24-57/report.md
    (best epoch 154/172, 0.977/0.947, P 0.960 R 0.939)
  /home/aatanda/isi_monitor3d/trainer/isidet/runs/segment/models/yolo/yolo26n-seg_e100_320px_03-07-2026_15-09-28/results.csv
    (best epoch 89/100: box 0.96175/0.89495, mask 0.95327/0.84581, P 0.91521 R 0.89946; args.yaml imgsz 320)
  /home/aatanda/isi_monitor3d/trainer/isidet/models/rfdetr/rfdetr-medium-seg_e41_432px/metrics.csv
    (best-EMA epoch 23: ema mAP50 0.9731, ema mAP50-95 0.9380, ema segm 0.9622/0.9056;
     non-EMA same epoch 0.9713/0.9305; per-class AP palette 0.9156 / carton 0.9051 / polybag 0.9707;
     precision 0.9532 recall 0.9327)
  CHECKPOINT RULE ADOPTED (resolves the §1 traceability note): best-EMA by val box mAP@0.5:0.95,
    mirroring ultralytics best.pt; PLAN T1's "0.975/0.933" line should be updated to 0.973/0.938.

3.2 Calibration:
  /home/aatanda/isi_monitor3d/isical/data/c1/calibration_refined.json (reprojection_rms_px 1.621 both cams)
  /home/aatanda/isi_monitor3d/isical/data/c1/work/intrinsic_rms.json (cam_a 0.6034, cam_b 0.4506)
  /home/aatanda/isi_monitor3d/isical/data/c1/{intrinsic,extrinsic,floor}/ (25 shots/cam, 8 pairs, 8 floor placements/cam — counted)
  /home/aatanda/isi_monitor3d/calibration/calibrate.py (assembly gates 0.5 px single-stage / 2.0 px two-stage)

3.3 Runtime:
  /home/aatanda/isi_monitor3d/thesis/measurements/G3_summary.md + G3_mqtt_diagnostics_20260720.jsonl
    (61 heartbeats, n=2048 window, 310 s, p50 40.3 / p95 78.1 / p99 94.0 + ranges, 25.8 fps,
     conditions, capture-clock caveat, motion-gate inclusion; per-camera fps 13.4-13.8 and
     mode/sources-alive read from the raw JSONL payloads — node_id deliberately not printed)
  /home/aatanda/isi_monitor3d/CLAUDE.md (prior-configuration figure p50 77 / p95 126 — cited only as such)
  /home/aatanda/isi_monitor3d/thesis/measurements/G4_trt_vs_cuda.md (+ raw/g4_*.json): all EP-table (now T5) numbers,
    N=50, warm-cache note, dashboard-contention caveat, 640 px CPU-postprocess decomposition (~41 of 46.7 ms)

3.4 Ablation:
  /home/aatanda/isi_monitor3d/thesis/measurements/G6_synth_ablation.md (FINAL, all arms complete:
    design, leakage checks, all three table rows verbatim incl. mask columns; synthetic-val sanity
    0.936; deltas R->R+S +0.021/+0.023/+0.020 mAP and +0.050 recall / precision 0.922->0.906 from
    the artifact's Observation; single-seed ~0.02-noise caveat and polybag-positive-only test-set
    caveat carried into the prose per the artifact's Caveats; training times 0.262 h / 0.497 h,
    exit 0, raw outputs under G6_ablation/runs/; post-review leakage check at the artifact's
    bottom: md5 53 LoRA-source photos x 186 test images = 0 overlaps — cited in §3.4;
    267 = 238 train + 29 val synthetic images counted on disk in the isiGen export)

3.5 Verification bounds:
  /home/aatanda/isi_monitor3d/tests/test_e2e_homography_synthetic.py (asserts abs=1e-3 m zero-noise;
    2 px noise → <10 cm per module docstring/test)
  /home/aatanda/isi_monitor3d/tests/test_e2e_triangulation_synthetic.py (xyz approx abs=1e-3)
  /home/aatanda/isi_monitor3d/backbone/triangulation/reprojection_gate.py (default 5.0 px; 5-8 range)
  pytest --collect-only on 2026-07-20: 721 tests collected (supersedes PLAN T5's stale "644 tests";
    noted for PLAN maintenance)

Deviations / flags:
  - PLAN T1 RF-DETR discrepancy RESOLVED (PLAN now carries 0.973/0.938 + the best-EMA rule).
  - PLAN T5 "644 tests" superseded by today's count (721).
  - yolo26m-seg exists in runs/ but is not in PLAN T1's evidence list; omitted here.
  - G3 numbers supersede CLAUDE.md prose (77/126) — handled per coordinator instruction.

FULL.REVIEW-2 fixes (2026-07-20):
  - C-1: deployed pose stride (pose_every_n: 1, verified in config/backbone.yaml:58) added to
    §3.3's G3 condition sentence — §2.3's "declared in Section 3" promise now fulfilled.
  - C-2: tables renumbered consecutively by order of appearance: T1, T2 (was T1b), T3, T4,
    T5 (was T4b), T6 (ablation, unchanged), T7 (bounds, was T5). Cross-references updated in
    04_discussion.md; abstract/conclusions traceability references unaffected (T1/T3/T4/T6).

FULL.REVIEW-1 revision log (2026-07-20, applied after verification):
  - MC-1: empty/full KPI explicitly de-scoped in §3.1 ("KPI coverage note") — no silent omission.
  - MC-2: whole-GPU 7.6/12.2 GiB context added to §3.3 (WSL2 = no per-process VRAM; per
    coordinator, the -52%/tick numbers were REMOVED from contribution 1 and moved to the
    FOR-SECTION-4 note below as dev-log measurements).
  - MC-3: §3.4 interpretive clause ("should be read with that uncertainty...") removed;
    single-seed noise floor kept as a measurement condition; reading moved to FOR-SECTION-4.
  - MC-4: §2.3 now states 384 default / 320 deployed+measured (config/backbone.yaml:64
    zone_imgsz: 320 — verified).
  - Hardening 2: 53-source-vs-186-test md5 check (0 overlaps, recorded at the bottom of
    G6_synth_ablation.md) cited in §3.4's leakage sentence.
  - m3: 238-vs-267 reconciled (238 train + 29 val = 267 counted on disk in the export).
  - m6: arm-local checkpoint-selection asymmetry disclosed in §3.4.
  - §3.2: "assembly gate = KPI" mapping softened to observation; BA-residual-vs-runtime-error
    argument assigned to §4.
  - "sim-to-real gap" renamed "real-transfer drop" in §3.4 (naming moved to §4).

Author corrections 2026-07-21 (this session):
  - Item 7: internal campaign labels G0/G1/G3/G4/G6 removed from all prose, headings, and
    table captions — replaced by descriptive phrases (data-provenance audit, independent
    re-evaluation, live latency measurement, execution-provider benchmark, training
    ablation). Artifact FILENAMES retained where cited as archive paths
    (thesis/measurements/raw/g1_val_yolo26l/, thesis/measurements/G6_ablation/runs/).
    The artifact mapping in this comment block is unchanged and remains the traceability
    record (G0->provenance audit, G1->re-evaluation, G3->live latency, G4->EP benchmark,
    G6->ablation).
  - Item 2: capture-clock caveat reduced to a back-reference — its single full statement
    now lives in §2.4 (Capture).
  - Item 3: §2 cross-references shifted for the new §2.1 Background (2.5->2.6 trainer,
    2.2->2.3 calibration workflow, 2.4->2.5 occupancy, 2.3->2.4 perception).
  - No numbers changed.
-->

<!-- FOR-SECTION-4 (drafting notes — material REMOVED from §1/§3 that §4 must own):

1. DEV-LOG NUMBERS (removed from contribution 1 per FULL.REVIEW-1 MC-2): during the July 2026
   split-process migration ("Direction 1"), development-log measurements under the PRE-CAMPAIGN
   configuration recorded (a) total VRAM ~2.5 GB in points mode vs ~5.2 GB with in-process
   perception (~-52 %), and (b) a perception tick costing ~55 ms standalone vs ~2,200 ms
   in-process (GIL + ONNX-Runtime thread-pool contention). Provenance: CLAUDE.md dev log, not a
   campaign artifact; NOT re-measurable now — WSL2 exposes no per-process VRAM attribution.
   Present in §4 as design rationale with explicit "measured during development, prior
   configuration" provenance. Context from G3-era load: whole-GPU 7.6/12.2 GiB (machine total).
2. G6 reading (removed from §3.4): the R+S over R mAP margins (+0.021/+0.023/+0.020) are within
   the ~0.02 single-seed noise floor -> state as "consistent with a small benefit, not
   established"; the recall gain (+0.050) is the only movement plausibly outside the floor.
   "Real-transfer drop" may be interpreted as the sim-to-real gap here. Also own: data-budget
   confound (476 vs 238; an R-full 737-image arm was possible), single class/site, arm-local
   checkpoint selection.
3. Latency ×2.6 margin must be paired with the capture-clock caveat: the KPI clock starts at the
   appsink callback (~100 ms rtspsrc buffer + decode after the optical event); optical-to-publish
   is therefore roughly 100 ms larger than reported and the KPI is met as defined by the spec's
   clock.
4. Calibration: argue (or bound) the relation between the 1.621 px BA residual and the runtime
   floor-projection homography error (the KPI's operative quantity).
5. Checkpoint-selection bias: best.pt / best-EMA selected on the same val split that is reported
   -> optimistic selection bias on top of the no-test-split limitation.
6. Full mandatory-limitations list: FULL.REVIEW-1 §3b items 1-9 (val-only accuracy; G6 caveats;
   no field ground truth — G2/G5 not run, 5 px gate is a threshold not a measurement; 2-cam DLT
   exactly determined; capture clock; empty/full KPI unvalidated; dev-rig-only measurements —
   Jetson port argued not demonstrated; single site; motion-gate re-emissions inside the latency
   distribution).
-->

