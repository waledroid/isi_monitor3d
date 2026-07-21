# Review — §2 Materials and Methods (draft v1)

Reviewer: cv-reviewer agent. Date: 2026-07-20.
Scope: `thesis/draft/02_methods.md` against `thesis/PLAN.md` and the repository at HEAD (branch `deepstream`).

---

## Summary

The section describes the five-module ISI Monitor 3D system: the split-process Direction-1 architecture (§2.1), the two-stage ChArUco/AprilGrid calibration with floor anchoring and the K,D,R,t→H,P derivation (§2.2), RTSP capture and zone-scoped detection in the isistream producer (§2.3), the dual-method geometric core — undistort→H homography chain with fusion, disagreement gating, ByteTrack-in-meters, temporal stabilization, plus subscription-driven two-view DLT triangulation with a reprojection gate and 3D Kalman (§2.4) — the isiGen synthetic-data pipeline (§2.5), and the comms/deployment stack (§2.6). Depth allocation follows the plan (2.4 deepest); measured KPI values are correctly withheld for Section 3.

Verification performed: board/capture specs vs `isical/data/c1/calib.yaml`; RMS hard limits and consensus plane fit vs `calibration/calibrate.py` (lines 76–83, 760–879); H/P derivation vs `backbone/shared/geometry.py` (`projection_from_K_R_t`, `floor_homography_from_K_R_t`); NVDEC chain vs `backbone/ingestion/rtsp.py`; fusion/gate semantics vs `backbone/homography/{cross_cam_fusion,disagreement_gate}.py`; DLT vs `backbone/triangulation/opencv_dlt.py`; gate default vs `reprojection_gate.py`; associator undistortion vs `keypoint_associator.py`; zone scoping vs `backbone/detection/zone_scope.py`; tick/emission vs `isistream/core.py`; schema/MQTT vs `backbone/comms/{schemas,mqtt_sink}.py` and `docs/REUSE.md`. The overwhelming majority of claims verify exactly, including several that contradict stale documentation elsewhere (e.g., the NVDEC decode chain, which `CLAUDE.md` still describes as software-only — the draft correctly follows the code).

## Strengths

1. **Traceability discipline.** The HTML source-comment maps every subsection to repo files, and its three recorded deviations (floor placements 8 vs code target 4; 500 scaffolds vs 2001 files; KPI-targets-only) are all correct calls. This is exactly the grounding standard the plan demands.
2. **Correct geometry.** The pose convention ($\mathbf{R}$ world←camera, projection via $\mathbf{R}^\top, -\mathbf{R}^\top\mathbf{t}$), the $\mathbf{H} = (\mathbf{K}[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}'])^{-1}$ derivation, and the undistort-then-H ordering all match `geometry.py` exactly — dimensionally consistent and notation-consistent between §2.2 and §2.4.
3. **Methods/Results separation is clean.** The only performance numbers present are the five KPI design targets, correctly attributed to the customer specification. No leaked measurements (no 77 ms, no 1.62 px, no mAP values).
4. **Honest-failure mechanisms are given first-class methodological treatment** (disagreement-gate demotion semantics, explicit-empty heartbeat vs silence, degraded solo emission, reprojection gate), matching the code's behavior including the exactly-determined 2-view DLT caveat.
5. **Length and depth allocation are on budget**: ~3,190 words ≈ 4.7 pp before equations/figures, within the ≈5-page (~3,400-word) budget; §2.4 is the deepest subsection as planned.

## Major Concerns

**M1 — The motion gate is omitted, and it contradicts two stated claims (§2.1(a), §2.3 "Perception tick").**
`config/backbone.yaml` sets `motion_gate: true` and `isistream/core.py::tick` (lines ~285–330) re-emits **cached** wire detections for motion-gated cameras "under the NEW frame's capture_ts" (its own comment). Likewise the person cache `self._wire_person` is re-sent on *every* tick, not only pose ticks. This contradicts: (a) §2.1's claim that each message carries "the `capture_ts` of the exact source frame the detections were computed on" — for cached re-emissions the stamped `capture_ts` is *newer* than the frame the detections came from; and (b) §2.3's claim that "between pose ticks, person tracks coast on the metric engine's Kalman prediction" — in fact the engine receives repeated identical measurements, which is a different estimator behavior than coasting (it re-anchors the Kalman at the stale position). The draft inherited the stale `core.py` docstring here. **Fix:** describe the motion gate (it is a real, enabled inference-economy mechanism a reader must know to reproduce the latency/GPU profile) and qualify the capture_ts statement, or verify and state the deployed configuration explicitly.

**M2 — ByteTrack first-pass pool misstated (§2.4).**
The draft: "two Hungarian passes match tracked tracks to high-confidence observations, then remaining (including lost) tracks to low-confidence observations." The implementation (`backbone/homography/bytetrack.py`, comment at line 101: "First pass: TRACKED + NEW + LOST tracks ↔ high-conf observations") includes LOST tracks in the **first** pass. The draft's wording describes the original ByteTrack paper's association, not this variant — a reviewer comparing against Zhang et al. would call the deviation unreported. One clause fixes it.

**M3 — Dangling cross-reference: "Sections 4 and 9 (Limitations)" (§2.4, last sentence).**
The article has five sections (PLAN §3); Limitations is a subsection of §4 Discussion. "Section 9" does not exist. Replace with "Section 4".

**M4 — Mode-1 minimum correspondences overstated (§2.2, floor-anchor paragraph).**
The draft: "at least five operator-measured pixel-to-floor point correspondences." The tool accepts **four** (`calibration/calibrate_single_cam.py`, "4+ point floor-plane fit"); five is the *recommendation* because with exactly four the fit is exactly determined and the residual gate cannot fire (`CLAUDE.md` calibration commands). State: minimum four, five or more recommended so the residual check is informative — this also reinforces the section's fail-honestly theme.

**M5 — isiGen stage enumeration matches neither repo source (§2.5).**
The draft's ten named stages include "pipeline initialization" and omit "detection". The `src/stages/` packages (the ten-count source) are: captioning, control_maps, curate, **detection** (auto-box prompt detectors feeding SAM2 masking — `src/stages/detection/base.py`), exporting, filtering, generation, lora, masking, scaffolds — no "pipeline initialization" package. The README's workflow lists **eight** phases (with "Pipeline init" as P5). Pick one authoritative enumeration; as written, neither count nor names can be checked against the repo.

## Minor Concerns

1. **DLT row count (§2.4, Eq. for the homogeneous system).** $[\mathbf{p}_i]_\times \mathbf{P}_i \mathbf{X} = \mathbf{0}$ yields three rows of which **two are independent**; say "two independent rows". (Optionally note the standard construction actually used by `cv2.triangulatePoints`: rows $u\,\mathbf{p}^{3\top}-\mathbf{p}^{1\top}$, $v\,\mathbf{p}^{3\top}-\mathbf{p}^{2\top}$.)
2. **SAHI appears once, unintroduced (§2.3, TRT bucket sentence).** "batch sizes under tiled (SAHI) inference" — tiled inference is never defined or motivated. Either add one sentence introducing it (it exists: `backbone/detection/tiling.py`, `zone_scope.py` SAHI params) or drop the parenthetical and say "batch sizes are padded to a fixed bucket set".
3. **"loopback MTU" (§2.6).** The fragmentation envelope exists because WSL2 mirrored-mode loopback silently drops UDP datagrams above ~1.5 KB — not a classical MTU limit. "datagrams exceeding the transport's safe payload size" is more accurate and portable.
4. **Duplication risk with Results tables.** The deployed-rig counts (25 shots/cam, 8 pairs, 8 floor placements) and dataset counts (53 reals; 500/553/500; 5,540/1,049) also appear in PLAN's T2/T3. Acceptable as materials description here, but decide the single home for each number before assembling §3 to avoid repeating them.
5. **"1 × 2 tags per board" (§2.2)** is correct per `calib.yaml` (`april_tags_x: 1, april_tags_y: 2`) but reads ambiguously; "2 tags per board (1 × 2 grid)" is clearer.
6. **§2.5 dataset provenance gap.** "500 generated images" (polybag) sits beside "5,540 training images" (three classes) with no bridge; one clause on how per-class synthetic output composes with real/other-class data into pallet3 would prevent a reviewer asking whether 5,540 are all synthetic.
7. **Figure F2 caption** lists the chain ending "temporal stabilizer → `Track2D`" — consistent with §2.4; ensure the eventual figure also shows the pallet-occupancy side module mentioned in the text, or drop that sentence from the caption's scope.

### Reproducibility — must add (small, all have repo-verified values)

- **Frame-pairing skew tolerance:** default `max_skew_ms = 33` (`backbone/ingestion/frame_sync.py` line 50). §2.4 says "a skew tolerance" with no value; the 100 ms degraded grace *is* given, so the asymmetry is conspicuous.
- **Fusion / disagreement default distances:** match 0.8 m (person/pallet) / 1.6 m (forklift); agreement 0.4 / 0.8 m (`cross_cam_fusion.py` line 27, `disagreement_gate.py` line 25). The draft says only "permissive" vs "tighter" — a reviewer will demand the numbers; state them as per-class configurable defaults.
- **ByteTrack confidence split:** `conf_high = 0.5`, `conf_low = 0.1`, below-low dropped (`bytetrack.py` lines 57–58).
- **RANSAC floor-plane inlier threshold:** 0.03 m (`calibrate.py` line 838).

### Reproducibility — configurable, acceptable to state as such (no change required)

Tick pacing (`isistream.fps`), pose stride default (`pose_every_n = 1`), stabilizer window / `min_frames_confirmed`, detection confidence thresholds, zone `crop_height_m` (2.0 m default is already given), `zone_imgsz` 384 (given).

## Recommended Experiments

None required for this section (Methods). Two items feed forward:
1. The G3 latency-probe artifact should record whether the **motion gate** and pose stride were active — after M1, these are declared methods parameters and the Results must state their settings.
2. If G2 (live reprojection-error logging) runs, log the **disagreement-gate rejection count** (`DisagreementGate.rejected_count`) alongside — §2.4 presents the gate as the 2-camera consistency mechanism, so §3 should show it operating.

## Final Recommendation

**Minor revision.** The draft is technically sound, well-grounded, on budget, and correctly separates methods from results. The five major concerns are all localized (one mechanism to add + four wording/consistency fixes); none require restructuring. Address M1–M5 and the four "must add" parameter values, then the section is ready for assembly.
