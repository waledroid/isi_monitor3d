# 5. Conclusions

This article presented isiMonitor3d, an integrated industrial vision pipeline that turns one or two fixed cameras into metric, identity-stable warehouse monitoring, from operator-guided calibration to machine-consumable delivery. Against the customer acceptance targets, the measured outcomes are: end-to-end capture-to-publish latency of p95 78.1 ms under full production load, within the < 200 ms target as defined by the specification's capture clock; a calibration bundle-adjustment residual of 1.621 px against the ≤ 2 px homography-error target, noting that this residual is a calibration-time proxy for the runtime projection error (Section 4.1); and detection mAP@0.5 of 0.962–0.977, above the ≥ 0.90 target, on the validation split. Two targets remain unverified: the pallet empty/full precision/recall KPI (mechanism deployed, no labeled measurement) and field geometric accuracy (the triangulation gate is a configured threshold, not a measured error).

The contributions stand as follows. A modular five-module architecture with frozen communication contracts delivers single-decode, multi-consumer processing with the measured real-time performance above. A dual-method localization framework serves always-on ground-plane homography and on-demand stereo triangulation from a shared calibration and identity space. An operator-guided calibration workflow reaches the 1.62 px reprojection consensus on the deployed rig using printed calibration boards only. A synthetic-data generation pipeline based on SDXL, ControlNet, and LoRA is evaluated quantitatively against real training data, its measured role being to augment, not replace, it.

Future work follows directly from the limitations. First, the field measurements not yet run: live triangulation reprojection-error logging and a tape-measured ground-truth campaign on the deployed rig. Second, the decision-relevant training experiments: the ~53-real ± synthetic arm that tests the actual fifty-photo scenario, together with seed replication of the ablation. Third, the Jetson Orin NX port, argued portable through the shared ONNX artifact but not yet demonstrated. Fourth, extension to ≥3 cameras via aniposelib triangulation, which makes the reprojection gate informative. Fifth, the deferred pose-mode extension, lifting per-keypoint 3D rather than the foot centroid alone.

<!--
Source traceability — §5 Conclusions (drafted 2026-07-20):
  KPI outcomes: 03_results.md (§3.3 T4 p95 78.1; §3.2 T3 1.621 px; §3.1 T1 0.962-0.977) with
    the §4.1 proxy caveat and §4.2 items 3/6 (unverified: occupancy KPI, field accuracy) —
    restated per coordinator, no new numbers.
  Contributions: one sentence each, mirroring 01_introduction.md's four-item list
    (post 5-to-1 merge; augments-not-replaces wording per FULL.REVIEW-1 A1/MC-3).
  Future work: G2/G5 (PLAN §5 + 04_discussion item 3), ~53-real+/-synthetic + seeds
    (FULL.REVIEW-1 §8.1/§8.6), Jetson port (CLAUDE.md portability rationale; §4.2 item 7),
    >=3-cam aniposelib + pose-mode S5.5 (CLAUDE.md deferred-scope statements; PLAN §3 row 5).
  Author corrections 2026-07-21: "ISI Monitor 3D" -> "isiMonitor3d" (item 1); no G-labels
    were present in this section's prose; numbers unchanged.
  §1-restructure sync 2026-07-21 (author constraint e): contributions paragraph reworded
    to mirror the NEW 4-item list (architecture / dual-method localization framework /
    calibration 1.62 px / synthetic-data generation + quantitative evaluation).
    "with explicit failure gating" dropped here too, matching contribution 2; the
    augment-not-replace reading (FULL.REVIEW-1 A1/MC-3) is retained.
-->
