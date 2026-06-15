# isiGen — User Manual

A step-by-step guide to turning a handful of real photos into a large,
perfectly-labeled **synthetic** training dataset, using the isiGen Studio web app.

If you just want the commands, see `README.md`. This document explains **what
each phase does and what you need to do** at every step.

---

## The big idea (read this first)

You have maybe 50–100 real photos per object class (palette, carton, polybag).
That's far too few to train a good detector, and hand-labeling masks is slow.

isiGen solves both problems at once:

1. You give it a small set of **real photos** — that's the only manual data prep.
2. It learns what your objects *look like* (a small **LoRA** fine-tune).
3. It then **generates unlimited new images** where the object's shape is forced
   by a depth map and the background is randomized by text prompts.
4. Because the shape is forced, isiGen **already knows the exact mask** of every
   generated object — so every synthetic image comes **pre-labeled**, with zero
   hand-annotation.

The result is exported as a YOLO-segmentation dataset you train with `isidet`.

> **You do NOT import annotations.** No LabelMe/COCO JSON. You import *photos
> only*; isiGen produces the labels itself.

---

## Before you start

- **Launch Studio:** open a terminal and type `gen`. It activates the
  `isi-train` environment and starts the app. Then open
  **http://localhost:8200** in your browser.
- **One GPU job at a time.** isiGen runs heavy phases (masks, LoRA, minting) one
  at a time so it never exhausts the 12 GB card. If a phase seems to "wait,"
  another job is probably still running — watch the **Job log** at the bottom of
  the pipeline page.
- **Get your photos onto this machine.** Studio reads images from a **folder
  path on the server**, not a browser upload. Put them somewhere like
  `/home/aatanda/photos/palette/`. (Windows files under `/mnt/c/...` work too.)

---

## The pipeline at a glance

The project page shows 8 phase cards. They run **in order** — each one needs the
previous one's output. A card turns **green ✓** when it's fully done, **amber ◐**
when partly done, and its button reads **Re-run** once complete.

| # | Phase | What it produces | Needs a GPU? |
|---|-------|------------------|:---:|
| 1 | Curate real images | Cleaned, class-tagged photos | no |
| 2 | Control maps | Depth + edge maps (generation guides) | yes (fast) |
| 3 | Ground-truth masks | Per-object masks (your future labels) | yes |
| 4 | Anti-bleed captions | One text prompt per image | no |
| 5 | LoRA training | A small model of *your* objects' look | yes (hours) |
| 6 | Synthetic scaffolds | Procedural layouts → control map + mask pairs | no |
| 7 | Mint synthetics | The generated, auto-labeled images | yes |
| 8 | Filter + export | Quality-filtered YOLO dataset | yes (fast) |

**Buttons on each card:** `open ›` jumps to that phase's detail page; `Run` /
`Re-run` starts the job. **⟳ Refresh** (top right) updates the counts instantly.

---

## Step 0 — Create a project

On the **Projects** page, use the **New project** form:

- **Name** — letters, numbers, `_`, `-` (e.g. `pallets_v1`).
- **Classes** — comma-separated, as `name:TRIGGER`, e.g.
  `palette:ISI_PLT, carton:ISI_CRTN, polybag:ISI_PLYBG`.

> **Your classes are entirely up to you.** `palette / carton / polybag` is just
> the example. Nothing is hardcoded — create a project with whatever classes you
> need (e.g. `forklift, worker_vest, shelf_label`) and the rest of the pipeline
> works the same. The trigger is optional; leave it off and isiGen fills in
> `ISI_<NAME>` automatically. **Remember the exact class names you choose** — your
> photo folders must match them (see Phase 1).

### What is the `TRIGGER` (`ISI_PLT`)?

The **class name** (`palette`) is your real label — it's what the final dataset
uses. The **trigger** (`ISI_PLT`) is a throwaway code word used **only** to teach
the image generator.

When the LoRA trains, it ties a *word* to the *look* of your specific objects. If
you used the real word "pallet," you'd fight the generator's pre-existing idea of
pallets and get generic ones. A made-up token the model has never seen
(`ISI_PLT`) gives it a clean slot to store *your* pallet's exact appearance. You
**never see the trigger again** after generation — exported labels say `palette`.

You can leave the trigger blank and isiGen auto-fills `ISI_PALETTE` etc. To delete
a project (and all its files + trained LoRA), click the **✕** at the right of its
row and confirm.

---

## Phase 1 — Curate real images

**What it does:** imports your photos, removes duplicates (by content), strips
EXIF/orientation data, and tags each image with its class. Idempotent — re-running
the same folder adds nothing new.

### Preferred folder structure (recommended)

Sort your photos into **one subfolder per class, named exactly like your project
classes**, then point isiGen at the parent folder once:

```
~/photos/                  ← this is the "Server folder" you enter
├── palette/               ← folder name = class name (must match EXACTLY)
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
├── carton/
│   ├── ...
└── polybag/
    └── ...
```

The class name is taken **from the folder name** — so for a different project
the tree is just your own class names:

```
~/photos/
├── forklift/
├── worker_vest/
└── shelf_label/
```

> **Exact match matters.** A subfolder whose name isn't one of your project's
> classes is **skipped** (you'll see it noted in the job log). `Palette` ≠
> `palette`. Files directly in the parent (not in a class subfolder) are skipped
> in folder-name mode. Sub-subfolders are fine — ingest scans recursively and
> uses each image's *immediate* parent folder as its class, so
> `palette/aisle3/img.jpg` → class `palette`.

**What you do:**

1. Click **open ›** on the Curate card (or the Curate page).
2. In the **Ingest** form, set **Server folder** to the **parent** folder
   (`~/photos`) and leave **Class** on the default **"— from subfolder names —"**.
   Click **Ingest** — every class is imported and tagged in one pass.
   - *Alternative:* if your images aren't sorted into class folders, pick a
     specific **Class** from the dropdown instead, and every image in that folder
     is tagged with it. Then repeat per class folder.
3. **Review the gallery:** use the class chips to filter, click a thumbnail to
   **retag** (wrong class) or **exclude** it (blurry, off-topic, near-duplicate).
   Excluded images stay but are skipped by every later phase.

**Tips for good photos:**
- **Keep the environment — don't crop tight.** Whole scenes (object in a real
  warehouse) caption better and reduce class-bleed.
- Aim for **one class prominent per photo**. Multi-class photos can be tagged with
  their dominant class; you'll fix the masks in Phase 3.
- 50–100 per class is a good target.

**Done when:** you have active (non-excluded) images.

---

## Phase 2 — Control maps

**What it does:** for each photo, computes two **generation guides**:
- a **depth map** (DepthAnythingV2) — the 3D shape the generator will follow,
- a **Canny edge map** — an alternate structural guide.

These are *inputs to generation*, not labels.

**What you do:** just click **Run** on the Control maps card. It's automatic and
fast. Check results on the **Maps viewer** (the maps page shows image / depth /
canny / mask side by side).

**Done when:** depth and canny exist for every active image.

---

## Phase 3 — Ground-truth masks

**What it does:** runs **SAM2** to segment each object and paints a
**color-coded mask** (each class gets its color). **These masks become the labels
for everything downstream**, so this is the one phase worth your attention.

**What you do:**

1. Click **Run** on the masks card for an automatic first pass.
2. Open the **Maps viewer** and check each mask. Auto-masks that the model wasn't
   confident about are flagged **needs review**.
3. To fix a mask, use the **SAM2 prompt canvas** on that image:
   - **click** = add a *positive* point (this pixel is the object),
   - **shift-click** = add a *negative* point (this pixel is NOT the object),
   - **drag** = draw a box around the object.
   Then re-run — only the images you re-prompted are recomputed.

**Done when:** every active image has a mask **and** none are left "needs review."
(The card stays **amber** while masks still need review.)

> Spend your time here. A wrong mask now becomes a wrong label in thousands of
> synthetic images later.

---

## Phase 4 — Anti-bleed captions

**What it does:** writes one text prompt per image, combining the **trigger word**
(your object's identity) + a class phrase + a randomized **background**
description. The varied backgrounds are what stop the model from memorizing one
scene ("anti-bleed").

**What you do:** click **Run** — captions are generated automatically. Open the
**Caption editor** to tweak any wording. **Your manual edits are never overwritten**
by re-running the phase.

**Done when:** every active image has a caption.

---

## Phase 5 — LoRA training

**What it does:** fine-tunes a small **LoRA** adapter on your curated images +
captions, teaching the SDXL generator what `ISI_PLT` (etc.) actually look like.
This is the longest step.

**What you do:**

1. Make sure the GPU is free (stop the dashboard/Backbone if running).
2. Click **Run**. Expect roughly **~4 hours** for the default 2000 steps on the
   RTX 5070. Watch progress in the **Job log**.
3. When it finishes, the trained weights are saved automatically and the project
   is pointed at them — no manual path-copying needed. The card turns **green**.

**Tips:**
- Fewer steps train faster but learn your objects less well. The default is a good
  balance; you can lower it in the project's `lora` config for a quick test.
- This phase only needs your **real** images — it runs before any synthetic ones
  exist.

**Done when:** trained LoRA weights exist for the project.

---

## Phase 6 — Synthetic scaffolds

**What it does:** procedurally generates **layout pairs** — a **control map**
(depth) plus its **perfect mask** — describing where objects will sit in each new
synthetic scene (e.g. stacks of cartons on a pallet). No AI yet; this is geometry.

**What you do:** click **Run**. To control how many to create, set the scaffold
**count** in the project config (default 500). Each pair is one future synthetic
image.

**Done when:** at least one scaffold pair exists.

---

## Phase 7 — Mint synthetics

**What it does:** the payoff. For each scaffold, the **SDXL + depth-ControlNet**
pipeline (with your LoRA) generates a photorealistic image: the ControlNet forces
the scaffold's geometry, the prompt randomizes the background, and the **scaffold's
mask is attached as the label** — aligned by construction. About **~30 s per
image**.

**What you do:** click **Run**. Generated images appear under the project's
`generated/` folder and as new synthetic records. You can mint in batches and
re-run to continue where it left off.

**Done when:** images are minted and the scaffold queue is empty.

---

## Phase 8 — Filter + export

**What it does:** two things:
1. **Quality filter** — scores each synthetic with CLIP and drops images that
   don't match their prompt (the hallucination guard; default threshold 0.25).
2. **Export** — writes a ready-to-train **YOLO-segmentation** dataset (and
   optionally LabelMe JSONs).

**What you do:** click **Run**. The dataset lands at:

```
data/<project>/export/yolo_seg/
├── images/{train,val}/<id>.jpg
├── labels/{train,val}/<id>.txt     # class + normalized polygon points
└── data.yaml                       # class names + paths
```

**Done when:** `data.yaml` exists.

---

## Handoff to training (isidet)

Point the isidet trainer's dataset path at the exported folder:

```
data/<project>/export/yolo_seg
```

Then train as usual (see `trainer/isidet`). The `.onnx` that training produces is
what the Backbone's detector consumes.

---

## Quick reference

| You want to… | Do this |
|---|---|
| Start the app | `gen` → http://localhost:8200 |
| See if a phase is done | Card is **green ✓**; **amber ◐** = partly done |
| Update counts now | **⟳ Refresh** (top right of the pipeline) |
| Re-do a phase | Click **Re-run** on its card |
| Fix a mask | Maps viewer → click / shift-click / drag, then Re-run masks |
| Edit a caption | Caption editor (edits survive re-runs) |
| Delete a project | **✕** on its row → confirm (removes data **and** its LoRA) |
| Find the final dataset | `data/<project>/export/yolo_seg/` |

## Troubleshooting

- **A phase won't start / seems stuck** — another GPU job is running (only one at
  a time). Watch the Job log; wait for it to finish.
- **Card stays amber** — some items aren't done. For masks, that usually means
  images still flagged *needs review*; open the Maps viewer and prompt them.
- **Minting/LoRA errors about model weights** — the SDXL weights aren't cached
  yet. See `README.md` → Setup for the one-time `hf download` commands (SDXL is
  ungated, no login needed).
- **Out-of-memory during LoRA or minting** — make sure nothing else is using the
  GPU (stop the dashboard/Backbone), then re-run.
