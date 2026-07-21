# 1. Introduction

Warehouse operations are automating rapidly, and with automation comes a new monitoring problem: automated guided vehicles (AGVs) and autonomous mobile robots increasingly share floor space with people, forklifts, and moving goods, under safety requirements codified for driverless industrial trucks in ISO 3691-4 [1]. Vehicle-mounted safety sensors protect the immediate surroundings of each vehicle, but site-level questions — who is inside a danger zone, which storage zone holds which pallets, whether a pallet left a zone loaded or empty — require infrastructure-side perception: fixed cameras observing the shared floor and reporting its state to the fleet controllers and warehouse management systems that act on it. Deep-learning computer vision has been identified as a key enabler for exactly these warehousing tasks, including entity re-identification, multi-view localization on the shop floor, and category-agnostic segmentation of bin items for robotic grasping [2].

For such a system's output to be actionable, pixel-space perception is insufficient. A fleet controller cannot consume bounding boxes in one camera's image plane; it needs object positions in metric world coordinates, identities that persist over time so that zone entries, exits, and dwell times are attributable to individual objects, and the zone-level events derived from both. Conventional CCTV analytics — detections, counts, heatmaps — live in image space, cannot express an inter-object distance in meters, and therefore cannot directly support safety monitoring or vehicle coordination.

Deploying such a system in an industrial warehouse introduces several practical challenges beyond the perception task itself. The system must operate in real time on a single edge GPU without cloud connectivity, support operator-friendly installation and maintenance, tolerate camera failures mid-shift, and adapt to site-specific layouts and object appearances for which no public datasets exist.

The research literature addresses these needs in four largely separate bodies of work. **Multi-camera localization.** Multi-camera people tracking is a mature field with dedicated surveys [3]; the probabilistic occupancy map of Fleuret et al. demonstrated metrically accurate ground-plane tracking from few synchronized cameras [4], more recent bird's-eye-view methods such as MVDet project learned features onto the ground plane before detection [5], and tracking-by-detection has converged on Kalman-filter motion models with Hungarian association [6], refined by ByteTrack's two-pass association over high- and low-confidence detections [7]. However, these works focus on benchmark datasets and do not address industrial deployment, calibration workflows, or fault tolerance.

**Calibration and geometry.** The geometric machinery itself is textbook material [8]: planar-target camera calibration [9], its fiducial infrastructure (AprilTag [10], ArUco/ChArUco [11]), plane-induced homographies, and direct linear transformation (DLT) triangulation. Existing work, however, typically treats ground-plane homography and multi-view triangulation as alternative localization methods — the former cheap but confined to floor contact, the latter fully 3D but requiring synchronized calibrated views. To our knowledge, no deployed system described in the literature we reviewed integrates both from a single calibration while maintaining a shared identity space.

**Detection.** Real-time detection evolves along two lines — the single-stage YOLO family from its origin [12] through its surveyed evolution [13] to the NMS-free, edge-oriented YOLO26 generation [14], and the detection-transformer line from DETR [15] through real-time variants [16] to the fine-tuning-oriented RF-DETR [17] — while portable inference runtimes (ONNX Runtime [22], TensorRT [23]) make these models deployable on heterogeneous edge hardware through a single exchanged model artifact. Existing detector research, however, emphasizes benchmark accuracy and throughput rather than deployment-aware inference that exploits static scene priors — in a fixed-camera installation, most of every frame never changes and activity concentrates in known zones.

**Synthetic training data.** On the data side, synthetic imagery has progressed from domain randomization in rendered scenes [21] to diffusion-based generation: SDXL provides photorealistic synthesis [18], ControlNet constrains it with geometric control maps so that labels can be derived by construction [19], and low-rank adaptation (LoRA), introduced for language models [20] and since applied to diffusion backbones, adapts the generator to a specific object class from a few dozen examples. Although these techniques have shown promise on public datasets, their effectiveness within a complete industrial monitoring pipeline remains largely unvalidated.

Across the reviewed literature, we did not find a complete industrial vision system integrating operator-guided calibration, synthetic-data-assisted detector training, metric multi-camera localization, and machine-consumable delivery into a single measured pipeline. Bridging this integration gap is the objective of this work.

Closing the gap matters for three practical reasons. **Safety:** the safety functions ISO 3691-4 requires of driverless trucks are vehicle-centric [1]; infrastructure-side metric tracking complements them with continuous site-level supervision — person positions in meters, evaluated against danger-zone polygons and delivered to the fleet controller before any vehicle's own sensors are in range. **Inventory and flow:** the same tracks give warehouse management systems a live, per-zone view of goods, zone entry and exit events, and pallet empty/full state — signals otherwise obtained by manual scanning. **Deployment economics:** lightweight calibration from printed boards and one mid-range GPU keep the hardware side modest, while synthetic imagery generated from on the order of fifty real photographs of a class can augment — though, on our own ablation evidence, not replace — the per-site collection and annotation campaign (Section 3.4).

This article describes isiMonitor3d, a complete industrial vision pipeline built to a warehouse-logistics customer specification (cahier des charges) whose acceptance targets are sub-200 ms capture-to-publish latency (95th percentile), homography reprojection error at most 2 px, per-view triangulation reprojection error gated at 5–8 px, detection mAP@0.5 of at least 0.90, and pallet empty/full classification precision/recall of at least 0.95/0.93. The system spans five modules — operator-guided calibration, a perception producer, a metric-geometry engine, a synthetic-data/training pair, and a communication gateway — and offers two localization modes: continuous 2D floor tracking by ground-plane homography and on-demand 3D tracking by stereo triangulation, delivered as versioned JSON over UDP and MQTT with a polling REST gateway for AGV fleet controllers.

The main contributions of this work are:

1. A **modular five-module architecture** with frozen communication contracts, enabling single-decode multi-consumer processing and measured real-time performance (capture-to-publish latency of 40 ms p50 / 78 ms p95 under full production load on one 12 GB edge GPU).
2. A **dual-method localization framework** combining always-on ground-plane homography and on-demand stereo triangulation from a shared calibration and identity space.
3. An **operator-guided calibration workflow** (isical) achieving a 1.62 px reprojection consensus on the deployed rig using printed calibration boards only.
4. A **synthetic-data generation pipeline** (isiGen: SDXL + ControlNet + LoRA), together with a quantitative evaluation of synthetic versus real training data for industrial object detection (Section 3.4).

The remainder of the article is organized as follows. Section 2 presents the materials and methods: background on the underlying architectures and tools (2.1), system architecture and wire contracts (2.2), calibration (2.3), perception and detection models (2.4), the geometric core (2.5), synthetic data generation (2.6), and communication and deployment (2.7). Section 3 reports experimental results against the acceptance targets — calibration accuracy, detector accuracy, runtime performance, and synthetic geometric-accuracy bounds. Section 4 discusses these results, including the design decisions the measurements justify, and states the limitations of the current validation. Section 5 concludes and outlines future work.

<!--
Source traceability — §1 Introduction (RESTRUCTURED 2026-07-21 per the author's guideline):

Structure follows the author's organization EXACTLY:
  problem (2 paras: application+safety context; pixel-space insufficiency) →
  engineering challenges (1 merged para, author's sample text as base) →
  literature in FOUR topic blocks, each closing with its limitation sentence
    (A multi-camera localization; B calibration & geometry incl. the HEDGED
     single-calibration/shared-identity novelty claim — "To our knowledge, no
     deployed system described in the literature we reviewed..."; C detection;
     D synthetic data — limitation sentences use the author's wording) →
  research gap (1 para, author's wording) →
  why it matters (1 shortened para: safety / inventory / economics) →
  system overview (isiMonitor3d FIRST introduced here; five-module list — one of
    the two permitted homes per the de-repetition rule; KPI list RETAINED here
    because §2.2 back-references "the five acceptance criteria listed in Section 1") →
  4 tightened contributions (author's wording, verified numbers) → roadmap.

Citation mapping (author's illustrative numbers replaced by references.md reality):
  [1] ISO 3691-4 (para 1 + why-it-matters); [2] Rutinowski (para 1);
  [3] survey, [4] Fleuret POM, [5] MVDet, [6] SORT, [7] ByteTrack (block A);
  [8] Hartley–Zisserman, [9] Zhang, [10] AprilTag, [11] ArUco/ChArUco (block B);
  [12] YOLO, [13] YOLO survey, [14] YOLO26, [15] DETR, [16] RT-DETR,
  [17] RF-DETR, [22] ONNX Runtime, [23] TensorRT (block C);
  [21] domain randomization, [18] SDXL, [19] ControlNet, [20] LoRA (block D).
  ORPHAN CHECK: every reference previously cited only in §1 ([1]–[5], [21])
  remains cited in the new §1 — NO reference became orphaned.

Numbers (unchanged, all previously verified):
  40 ms p50 / 78 ms p95 — live latency measurement (G3_summary.md: 40.3/78.1,
    stated rounded as in the prior contribution list); 1.62 px — deployed-rig
    calibration_refined.json (1.621, rounded as before); KPI targets — cahier des
    charges via CLAUDE.md KPI table.
  DROPPED FROM THE CONTRIBUTION LIST per the author's tightened wording (values
    remain in §3/abstract): ~26 fps aggregate (contribution 1); intrinsics
    0.60/0.45 px (contribution 3); 53→500 counts and the 0.96–0.98 mAP@0.5 range
    (contribution 4 — reviewer M4's KPI-checkability note is superseded by the
    author's directive; the range still appears in §3.1, §5, and the abstract).
  "with explicit failure gating" dropped from contribution 2 per the author's
    tightened wording (the gates remain fully described in §2.5).

Sync applied elsewhere this session (author constraints e/f):
  - 05_conclusions.md contributions paragraph reworded to mirror the NEW list.
  - thesis/PLAN.md §4 contribution list replaced with the NEW list (dated note).
  - isimonitor3d.tex §1 replaced wholesale; conclusions paragraph synced; rebuilt.

Carried from the superseded §1 (facts re-checked, hedges preserved):
  - All 23 original references verified 2026-07-20; [24]–[28] verified 2026-07-21
    (those are cited in §2.1 Background, not §1).
  - Integration-gap claim scoped to "the reviewed literature"; LoRA diffusion
    transfer hedged; no claim that integrated systems "do not exist".
  - Factual corrections from prior review rounds remain in force: all-real corpus
    attribution (provenance audit), live-measurement latency figures replacing the
    pre-lever 77/126 prose, validation-split qualifiers, 4-contribution merge.
-->
