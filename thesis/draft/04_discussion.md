# 4. Discussion

## 4.1 Interpretation

**Why the architecture meets the latency KPI.** The live latency result (p95 78.1 ms under full production load, Table T4) is attributable to the accumulated design choices of Section 2 rather than to any single optimization. The split-process design was motivated by development-log measurements made during the July 2026 migration, under a pre-campaign configuration: running perception in-process with the dashboard cost ~2,200 ms per perception tick versus ~55 ms standalone (interpreter-lock and ONNX Runtime thread-pool contention), and the points-mode engine reduced total VRAM from ~5.2 GB to ~2.5 GB. We report these figures as design rationale with their provenance stated: they are development measurements, not campaign artifacts, and the per-process VRAM split cannot be re-measured today because WSL2 exposes no per-process attribution — the current whole-GPU figure under production load is 7.6 of 12.2 GiB (Section 3.3). The improvement from the earlier configuration's p50 77 / p95 126 ms to the measured 40.3 / 78.1 ms reflects the combined introduction of the in-pipeline 720p downscale, the reduced pose input size, the motion gate, and the TensorRT provider; the per-lever decomposition was not isolated and we do not apportion it.

The ×2.6 margin against the 200 ms target must be read with the clock definition of Section 2.4: `capture_ts` starts ~100 ms (jitter buffer plus decode) after the optical event. Optical-event-to-publish is therefore roughly 100 ms larger than the reported figures; the KPI is met as defined against the specification's capture clock, and the remaining headroom is what the Jetson port will consume.

**Why 320 px zone crops are the enabling choice.** The execution-provider benchmark shows TensorRT's isolated inference gain (2.8–3.0×) collapsing to 1.4× end-to-end at 640 px, because ≈41 ms of the 46.7 ms median is CPU-side letterboxing, decoding, and mask assembly (Table T5); medians are used throughout this comparison because the benchmark ran beside the ≈5 GB dashboard load (Section 3.3). At the deployed 320 px the end-to-end gain remains 2.6×. Zone scoping is what makes the small input viable: each zone crop receives more model pixels than it would occupy in a full frame (Section 2.4), so accuracy-relevant resolution is preserved while the postprocessing volume that throttles the GPU gain stays small. Conversely, further model acceleration at 640 px would be unproductive without moving postprocessing off the CPU.

**What the ablation shows about synthetic data.** The training ablation supports reading the isiGen pipeline as an *augmenter*, not a replacement for real data: the synthetic-only arm transfers to real frames at 0.223 box mAP@0.5 against 0.941 for the count-matched real arm — a real-transfer drop consistent with an appearance domain gap (single site and camera family, CLIP filtering notwithstanding). The merged arm's mAP margins over real-only (+0.021/+0.023) lie within the stated ~0.02 single-seed noise floor and are therefore consistent with a small benefit rather than established; the recall gain (+0.050) is the only movement plausibly outside the floor and is the effect we consider robust. Two design facts bound this reading: the merged arm trains on twice the images of the real arm, confounding "synthetic data" with "more data" (737 real images existed; an R-full arm would separate the confound), and the count-matched design represents the scarce-real-data scenario rather than this site's actual data budget. The decision-relevant experiment remains unrun: an arm of ~53 real images with and without synthetic augmentation — the actual fifty-photo deployment scenario claimed in Section 1.

**Calibration residual versus the homography KPI.** The 1.176 px consensus figure (Table T3) is a bundle-adjustment residual over board corners, not a direct measurement of runtime floor-projection error. It is a meaningful proxy — the same $\mathbf{K}, \mathbf{R}, \mathbf{t}$ that generate the residual compose the runtime $\mathbf{H}$, so systematic calibration error would surface in both — but it does not bound projection error on floor regions the boards never covered. A field check against surveyed floor points is required before the ≤ 2 px KPI can be considered verified in the runtime sense. The August re-solve adds an operational observation: fixed cameras drift, and the calibration is a maintained artifact rather than a one-time step — the operator workflow of Section 2.3 absorbed a full extrinsic re-solve in one session, at the price of a new world origin and redrawn zones.

**Raised surfaces.** The 0.5–0.6 m cross-camera separation of platform pallets (Section 2.5) is the expected parallax of projecting an elevated contact point onto $z = 0$ from two viewpoints, and it exposed how much of the pipeline silently assumed one plane: displaced zone crops, rejected fusion pairs, wobbling zone decisions. A per-zone base height routed through one ray–plane primitive restored consistency without re-calibrating; the widened thresholds should be tightened again once the platform zones are confirmed stable.

## 4.2 Limitations

1. **Validation-split accuracy with validation-selected checkpoints.** No held-out test split exists (Section 3.1); `best.pt`/best-EMA checkpoints are selected on the same split that is reported, adding optimistic selection bias. Deployment-frame accuracy is unmeasured, and per-class support is imbalanced (carton: 74 images), widening per-class uncertainty.
2. **Ablation scope.** The training ablation uses one seed per arm with margins at its ~0.02 noise floor, one class, 238-image arms, a 186-image single-site positives-only test set (no false-positive measurement on object-free frames), a data-budget confound (476 vs 238 training images), and arm-local checkpoint selection.
3. **No archived field geometric ground truth.** `Track3D` has been exercised on the deployed rig (August 2026) with the reprojection gate widened to 60 px (Section 2.5), but neither a reprojection-error distribution nor a tape-measure campaign has been archived; geometric accuracy rests on synthetic verification bounds (Table T7), and the deployed gate — 12× the code default — is a configured threshold that currently rejects little.
4. **Two-camera gate blindness.** With exactly two cameras the DLT is exactly determined, so the reprojection gate cannot detect cross-camera disagreement at the 3D stage; mitigation is the upstream 2D disagreement gate, and the 3D gate becomes informative only with ≥3 cameras.
5. **Clock definition.** Reported latency excludes ~100 ms of pre-appsink pipeline (jitter buffer + decode); see Section 4.1.
6. **Pallet empty/full KPI unvalidated.** The occupancy mechanism is deployed but no labeled precision/recall measurement exists (Section 3.1).
7. **Platform.** All measurements were made on the development workstation (RTX 5070, WSL2); the Jetson Orin NX port is argued portable via the ONNX artifact, not demonstrated.
8. **Single deployment.** One site, one rig, one camera family underlie every finding, including the calibration and ablation test data.
9. **Motion-gate re-emissions in the latency distribution.** Gated (cached) ticks share the measured latency distribution with inferred ticks; worst-case always-inferring latency was not isolated.
10. **Raised-surface height; un-remeasured latency.** For boxy objects on a platform the two-view `Track3D` height absorbs foot-point correspondence error and can fall below the surface (wire value raw; displays anchor to the zone plane). The latency campaign (Table T4) and EP benchmark (Table T5) predate the August changes (crop-trained detector, plane-aware zones, widened gates) and were not re-run.

<!--
Source traceability — §4 Discussion (drafted 2026-07-20):

Inputs (mandatory per coordinator):
  /home/aatanda/isi_monitor3d/thesis/draft/FULL.REVIEW-1.md §3b (drafting brief: every
    "Interpret" bullet covered in 4.1; limitations 1-9 mapped one-to-one to 4.2 items 1-9)
  /home/aatanda/isi_monitor3d/thesis/draft/03_results.md FOR-SECTION-4 comment (notes 1-6:
    dev-log VRAM/tick numbers + provenance caveats -> 4.1 first paragraph; G6 reading ->
    4.1 third paragraph; capture-clock/x2.6 pairing -> 4.1 second paragraph; BA-residual
    argument -> 4.1 fourth paragraph; checkpoint-selection bias -> 4.2 item 1;
    limitations list -> 4.2)

Numbers re-verified against artifacts:
  G3_summary.md (78.1 / 40.3 ms; prior config 77/126; conditions), G4_trt_vs_cuda.md
  (2.8-3.0x isolated, 1.4x/2.6x e2e, ~41 of 46.7 ms CPU-side), G6_synth_ablation.md
  (0.223 / 0.941, +0.021/+0.023, +0.050 recall, ~0.02 noise floor, 476 vs 238, 737 real
  available, 53 reals), isical/data/c1 (1.621 px), CLAUDE.md dev log (2,200 vs 55 ms tick,
  5.2 vs 2.5 GB — presented WITH "development measurement, prior configuration, not
  re-measurable under WSL2" provenance per FULL.REVIEW-1 MC-2 option b), §3.3 (7.6/12.2 GiB),
  backbone/ingestion/rtsp.py docstring + CLAUDE.md (~100 ms rtspsrc latency + decode),
  tests/test_e2e docs + CLAUDE.md S5 gotcha (exactly-determined 2-cam DLT).

Discipline notes:
  - Interpretations anchor to §3 tables (T3, T4, T5, T6 — post-FULL.REVIEW-2 C-2
    renumbering: T1b->T2, T4b->T5, bounds T5->T7) or to explicitly-provenanced
    dev-log figures; no new measurements introduced.
  - "Consistent with a small benefit, not established" phrasing per FULL.REVIEW-1 MC-3.
  - The 77/126 -> 40/78 change is attributed to the combined levers WITHOUT per-lever
    apportioning (not isolated — stated).
  - The ~53-real +/- synthetic arm is named as the decision-relevant unrun experiment
    (FULL.REVIEW-1 §8.6), tying back to §1's reworded economics claim.

Author corrections 2026-07-21 (this session):
  - Item 7: campaign labels G2/G3/G4/G5/G6 removed from reader-facing prose (mapping:
    G3->live latency result/measured latency distribution, G4->execution-provider
    benchmark, G6->training ablation, G2/G5->named descriptively in limitation 3).
  - Item 2: the ×2.6-margin paragraph now back-references §2.4's full capture-clock
    statement instead of restating it.
  - Item 3: Section 2.3 references shifted to 2.4 (perception) for the new Background.
  - No numbers changed.
-->
