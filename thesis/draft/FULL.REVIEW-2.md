# Full peer review — pass 2: abstract, discussion, conclusions (2026-07-20)

Reviewer role: senior CV researcher / journal editor. Scope: `00_abstract.md`,
`04_discussion.md`, `05_conclusions.md`, checked against FULL.REVIEW-1 (its §3b
drafting brief and MC-1…MC-5), §1–§3 as revised, PLAN.md, and the measurement
artifacts (G3/G4/G6 numbers re-verified against the files this pass).

---

## Verdicts

| File | Verdict |
|---|---|
| `04_discussion.md` | **Ready-for-mentor** (one optional half-sentence, D-1) |
| `00_abstract.md` | **Minor revision** (one sentence, A-1; line-count slack, A-2) |
| `05_conclusions.md` | **Ready-for-mentor** |
| **Whole package** | **Minor revision → Ready-for-mentor** once A-1 is fixed and the assembly checklist below is executed. No major concerns remain. |

---

## 1. §4 Discussion vs the FULL.REVIEW-1 §3b drafting brief

**Compliance: complete on every mandatory item.**

Mandatory interpretations, all present and correctly anchored:

- **G3 latency** — split-process rationale delivered with the exact MC-2(b)
  treatment: 55 vs ~2,200 ms tick and 5.2→2.5 GB VRAM presented as
  development-log measurements, prior configuration, explicitly not
  re-measurable (WSL2, no per-process attribution), with the campaign-era
  whole-GPU 7.6/12.2 GiB anchored to §3.3. The 77/126→40/78 change is
  attributed to the combined levers **without** per-lever apportioning, stated
  as not isolated. Correct and disciplined.
- **Capture-clock definition** — the ×2.6 margin is paired with the ~100 ms
  pre-appsink caveat in the same paragraph, and "met as defined against the
  specification's capture clock" is stated plainly. Exactly what the brief
  required; m5 discharged.
- **G4** — the 2.8–3.0× → 1.4× collapse is explained via the ≈41 of 46.7 ms
  CPU-side postprocess (matches the artifact), and the productive conclusion
  (320 px zone crops keep the EP gain realizable; further model speedup at
  640 px is unproductive without moving postprocess off CPU) is drawn. Verified
  against T4b: 22.35/8.74 = 2.6× at 320 ✓.
- **T1/T1b** — checkpoint-selection bias, per-class support (carton: 74),
  deployment-frame accuracy unmeasured: present (in 4.2 item 1 rather than 4.1
  — acceptable placement).
- **G6** — all four required readings present: augmenter-not-replacement with
  domain-gap causes; mAP margins within the ~0.02 floor stated as "consistent
  with a small benefit rather than established" (the exact MC-3 downgrade
  language) with recall +0.050 as the only robust effect; the data-budget
  confound named with the 737-image R-full admission; count-matched design
  defended as the scarce-real-data scenario. The unrun ~53-real ± synthetic arm
  is named as decision-relevant, closing the loop to §1's reworded economics
  claim.
- **Calibration proxy argument** — §4.1 owns the BA-residual-vs-runtime-error
  mapping that §3.2 deferred, including the honest "does not bound projection
  error on floor regions the boards never covered" and the field-check
  requirement. This resolves the §3a interpretive-leak assignment.

**All nine mandatory limitations present, one-to-one, correctly stated** (4.2
items 1–9 map exactly to the brief's list 1–9; item 2 additionally folds in the
no-FP-measurement and arm-local-selection disclosures — good consolidation).

Interpretive-claim anchoring: every number in §4 traces to a §3 table (T3, T4,
T4b, T6), to §3.3 prose (7.6/12.2 GiB, 77/126), or to explicitly-provenanced
dev-log figures. **No unanchored interpretive claim found.**

- **D-1 (minor, optional):** the brief asked §4's G4 paragraph to also restate
  the dashboard-contention caveat ("medians robust, p95s contaminated"). It
  lives in §3.3 as a measurement condition but is not echoed in 4.1. A
  half-sentence ("medians are used throughout because the benchmark ran beside
  the ≈5 GB dashboard, Section 3.3") would perfect brief compliance. Not
  blocking.
- **D-2 (note):** PLAN §3's discussion row lists "degraded-mode behavior" as an
  interpretation topic; §4 does not discuss it. The binding brief
  (FULL.REVIEW-1 §3b) did not require it and §2.4 covers the mechanism; leave
  as-is unless the mentor asks.

---

## 2. Abstract

**Line count: exactly 15 physical lines** (body lines 3–17 of the file),
counted this pass — at the UGA cap with zero slack (see A-2). ~195 words.

**UGA element completeness: all six present, in order** — problem
(infrastructure-side metric monitoring), limitation (pixel-space analytics;
per-site calibration + detector cost), approach (five modules, one calibration
→ two geometric methods, one identity space), validation (deployed two-camera
rig, full production load), quantitative results, contribution (final
sentence).

**Number identity vs §3: all identical.** 40.3 / 78.1 ms = T4 ✓ (G3 artifact
re-checked); 1.621 px = T3 ✓; 0.977 box mAP@0.5 = T1 with "on its validation
split" qualifier ✓; 0.962 vs 0.941 = T6 R+S vs R ✓. No acronym violations
(mAP spelled out, "graphics processor").

**Over-claim scan:**

- **A-1 (the one substantive concern):** *"A controlled ablation showed that
  synthetic images generated from 53 real photographs augment rather than
  replace real training data (0.962 versus 0.941 box mAP@0.5, merged versus
  real-only)."* The "not replace" half is robustly established (0.223 transfer
  — not cited here). The "augment" half is evidenced by a +0.021 margin that §4
  itself rules **within the single-seed noise floor** and "not established."
  "Showed that … augment" is establishment language resting on a within-noise
  number — the exact claim structure MC-3 downgraded elsewhere; a reviewer who
  reads §4 will notice the abstract kept the stronger verb. Fix (choose one):
  - lead with the robust halves: "…cannot replace real training data (0.223
    versus 0.941 box mAP@0.5 on real frames, synthetic-only versus real-only)
    and are consistent with a small augmentation benefit (recall +0.050)"; or
  - keep the current numbers but soften the verb: "supports their role as an
    augmentation of, not a replacement for, real training data".
  Same softening should be mirrored in PLAN §1's thesis statement
  ("measurably augments" → "is consistent with a small augmentation benefit /
  augments recall"), since the abstract inherited its tone from there.
- Latency: "capture-to-publish … within the 200 ms target" — honest as written;
  the clock is named by the metric itself and §3/§4 carry the caveat. OK.
- "one edge graphics processor" — consistent with the article's framing;
  limitation 7 (dev workstation, not Jetson) covers the exposure. OK.
- **A-2 (formatting risk):** 15/15 lines at the current markdown wrap. The
  journal template will rewrap; re-verify the count after typesetting and keep
  one candidate cut in reserve (e.g., fold "within the 200 ms target" into the
  latency clause).

---

## 3. §5 Conclusions

**No new claims or numbers.** Every figure (78.1 ms, 200 ms, 1.621 px, 2 px,
0.962–0.977, 0.90, ~50 photographs) appears in §1/§3; qualifiers travel with
them (capture-clock definition on latency; "calibration-time proxy" on the
residual, with a Section 4.1 pointer; "on the validation split" on mAP).

**KPI restatement honesty: complete and correctly split.** All five acceptance
targets are accounted for: three met-as-qualified (latency, homography-residual
proxy, detection mAP) and two explicitly unverified (empty/full P/R; field
geometric accuracy with the "configured threshold, not a measured error"
formulation). This is the honest 3-of-5 accounting FULL.REVIEW-1 A2 demanded.

**Contributions restated** — four items, one sentence each, mirroring §1's
post-merge list; the synthetic-data item carries the augments-not-replaces
wording ("whose measured role is to augment, not replace" — defensible here
because §5 restates §3.4+§4.1 jointly, where the replace-failure is
established; if A-1's rewording lands, consider harmonizing to "measured role
is augmentation, with the mAP margin within single-seed noise" only if the
mentor wants belt-and-braces — not required).

**Future work: concrete, five items**, each traceable to a limitation or a
deferred repo capability (G2/G5 field measurements; ~53-real ± synthetic arm +
seed replication; Jetson port; ≥3-camera aniposelib; pose-mode extension). No
vague "we will explore" items. Ready.

---

## 4. Whole-package coherence

The article now tells one consistent story: thesis statement → integration gap
(§1) → five modules + two contracts (§2) → KPI-structured observations (§3) →
interpretation + nine limitations (§4) → honest 3-of-5 KPI accounting + future
work (§5). The augments-not-replaces arc is consistent across §1 economics,
§3.4 observations, §4.1 reading, and §5 — with the abstract as the one
remaining stronger-toned outlier (A-1).

**Forward-reference audit — no orphans of substance.** All delivered: §2.1
("process split analyzed in Section 4") → 4.1 ¶1; §2.4 ("examined in Section
4") → 4.2 item 4; §3.1 (implications) → 4.2 items 1/6; §3.2 (residual-vs-
runtime relation) → 4.1 ¶4; §3.4 (noise-floor reading) → 4.1 ¶3.

- **C-1 (micro-orphan):** §2.3 promises "Gate and stride settings are declared
  alongside the measurements in Section 3"; §3.3 declares the motion gate
  enabled and its re-emissions included, but not the pose-stride value. Either
  add the deployed stride to §3.3's condition sentence or trim §2.3's promise
  to "gate settings".
- **C-2 (table numbering):** PLAN's T2 (datasets) was folded into §2.5/§3.1
  prose; the article's tables run T1, T1b, T3, T4, T4b, T5, T6 — a visible gap
  at T2. Renumber sequentially at assembly (or resurrect T2 as a small dataset
  table if page budget allows).
- **C-3 (tone):** no violations found in the three new files — no marketing
  adjectives, claims verb-disciplined ("attributable", "consistent with",
  "supports reading … as"). §4.1's "the enabling choice" is earned by the T4b
  decomposition. Clean.
- Residual FULL.REVIEW-1 items: MC-1 ✓ (de-scoped in §3.1 + 4.2 item 6 + §5),
  MC-2 ✓ (option b executed verbatim), MC-3 ✓ (downgrade path taken in §3.4/§4;
  abstract lags — A-1), MC-4 ✓ (§2.3 384/320), MC-5 ✓ (PLAN abstract row now
  carries G3 numbers), Hardening 2 ✓ (md5 0-overlap check in G6 artifact and
  cited in §3.4), m1/m3/m6 ✓. m7 (ref [22]/[23] versions + access dates)
  still open by design — carried to the checklist.

---

## 5. Submission checklist (ordered; before this goes to the mentor)

1. **Fix A-1** — the abstract's ablation sentence (only blocking text edit);
   mirror the softening in PLAN §1's thesis statement.
2. **Produce figures.** `thesis/figures/` is **empty**. F1–F5 have placeholders
   in the drafts (F1 §2.1 from `illu.png`/cheatsheet topology; F2 §2.4 pipeline
   diagram — remember pass-1 Minor 7: depict or exclude the occupancy module;
   F3 §2.2 boards + Multical viewer export; F4 §2.5 confidential-safe isiGen
   samples; F5 §3.3 from `G3_mqtt_diagnostics_20260720.jsonl`). **F6
   (dashboard/deployment) exists only in PLAN §6 — no placeholder in any
   section**: either add it (§2.6 or §3.3, RTSP URLs/IPs redacted) or strike it
   from PLAN.
3. **Title page (does not exist yet):** working title (PLAN §1), author line
   (student name placeholder, Isitec mentor, academic supervisor — TBD),
   affiliations (Univ. Grenoble Alpes + ISITEC), **journal name** (mentor must
   validate the shortlist; Computers in Industry recommended), and the **red
   confidentiality notice** — all UGA-required on p. 1.
4. **Assemble** 00→05 + references into one document; renumber tables (C-2);
   fix C-1; verify every figure/table cross-reference; strip the traceability
   HTML comments from the mentor-facing PDF (keep them in the .md masters).
5. **References formatting** (m7): add versions + access dates to [22]/[23],
   apply the journal's citation style once chosen.
6. **Format in the journal's LaTeX template**, then enforce the two hard caps:
   **≤ 15 pages** total and **abstract ≤ 15 lines after rewrap** (A-2 —
   currently at exactly 15 in markdown).
7. **Resolve the MENTOR DECISION marker** in §1/PLAN (contributions merged
   5→4) explicitly with the mentor.
8. Optional, non-blocking (now framed as future work, but cheap if rig/GPU
   time appears before Aug 31): G6 seed replication (~2 h), ~53-real ±
   synthetic arm, negative-frame FP eval, idle-GPU G4 rerun.

---

## 6. Final recommendation

**Minor revision — one sentence (A-1) plus assembly mechanics.** §4 and §5 are
ready for the mentor as written; the discussion in particular executes the
pass-1 drafting brief completely, with all nine limitations present and no
unanchored interpretation. After A-1 and checklist items 2–4, the package is
**Ready-for-mentor**. The remaining risk to submission is entirely logistical
(figures, template, page cap), not scientific.
