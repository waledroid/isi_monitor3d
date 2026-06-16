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

> **You don't *need* to import annotations** — isiGen produces the labels itself,
> so importing *photos only* is the normal path. But if you **already have masks**
> (**LabelMe**, **YOLO**, or **COCO**), curate imports them automatically and
> skips the masking phase for those images. See Phase 1.

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

The project page shows 8 phase cards that you run **in sequence, 1 → 8**. The
board enforces this as a **chain**: a phase's **Run** button stays **locked 🔒**
until the phase before it is finished, then unlocks. A card turns **green ✓** when
it's fully done, **amber ◐** when partly done, **🔒** when still locked, and its
button reads **Re-run** once complete (re-running a finished phase is always
allowed).

| # | Phase | What it produces | GPU? |
|---|-------|------------------|:---:|
| 1 | Curate real images | Cleaned, class-tagged photos | no |
| 2 | Control maps | Depth + edge maps (generation guides) | yes (fast) |
| 3 | Ground-truth masks | Per-object masks (your future labels) | yes |
| 4 | Anti-bleed captions | One text prompt per image | no |
| 5 | LoRA training | A small model of *your* objects' look | yes (hours) |
| 6 | Synthetic scaffolds | Layout → control map + mask pairs | no |
| 7 | Mint synthetics | The generated, auto-labeled images | yes |
| 8 | Filter + export | Quality-filtered YOLO dataset | yes (fast) |

Just work top to bottom: finish a phase, the next one lights up. If a card is
**🔒**, complete the one above it first (hover its Run button and it tells you
which).

> **Advanced (CLI only).** Under the hood the engine is more permissive than the
> board — the phases form a dependency graph, not a strict line: masks don't
> actually need the control maps, and **export works on real masks alone**. Power
> users can take shortcuts from the command line (e.g. `run_curate` →
> `run_masks` → `run_export` for a dataset from just hand-checked real images,
> skipping captions/LoRA/minting). The Studio board deliberately hides this and
> guides you straight through the full 1 → 8 chain.

**Buttons on each card:** `open ›` jumps to that phase's detail page (always
available, even when the phase is locked); `Run` / `Re-run` starts the job.
**⟳ Refresh** (top right) updates the counts instantly.

**Progress:** while a phase runs, the **top bar** shows a live `… 60% (6/10)`
for the active job (on every page), and the pipeline page shows a **progress
bar** above the Job log. Maps/masks/captions/scaffolds/mint count per item; LoRA
counts per training step.

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

### Background images for paste-then-harmonize (optional but recommended)

If you'll use the **copy_paste** scaffold path (Phase 6), add a folder named
**`bg`** or **`background`** with **empty-scene photos that contain no objects**
(e.g. an empty conveyor, a bare aisle). isiGen pastes your real objects onto these
clean backgrounds, so the pasted object can never overlap an existing one — you
get exact labels and clean, deliberate "doubles". Two equivalent layouts:

```
~/photos/                         OR        ~/photos/
└── polybag/                                ├── polybag/          ← class images
    ├── img/      ← class images            │   ├── img_0001.jpg
    │   ├── img_0001.jpg                     │   └── ...
    │   └── ...                              └── bg/               ← empty backgrounds
    └── bg/       ← empty backgrounds            ├── empty_01.jpg
        ├── empty_01.jpg                         └── ...
        └── ...
```

A folder named `bg`/`background` is detected automatically (even if you picked a
specific **Class**); an `img` folder is a pass-through to its parent class. Background
images carry no class, get no mask/caption, and are never training samples — they
are paste targets only. **~10–30 backgrounds is plenty for hundreds of minted
images** (each background is reused evenly, but every reuse differs in object,
size, placement, count, and harmonization seed). The Curate card shows a
*"N backgrounds"* count.

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

### Already have masks? Import them (skip SAM2)

If you already labeled some images, curate **auto-detects** the annotations and
rasterizes them into the project-colored mask — those records arrive **already
masked** (phase 3 turns green for them, no SAM2 needed). The ingest result
reports `masks_imported: N`. Three formats are supported:

| Format | Where it goes | What's read |
|---|---|---|
| **LabelMe** | `image.json` next to `image.jpg` (same name) | `shapes` — polygon / linestrip / rectangle |
| **YOLO** | `image.txt` next to `image.jpg` (same name) | each line `cls x1 y1 …` (polygon-seg) or `cls cx cy w h` (box) — normalized |
| **COCO** | **one** `*.json` at the ingest-folder root (has `images`/`annotations`/`categories`) | polygon `segmentation` (or `bbox`; RLE → box) |

Class mapping per format:
- **LabelMe / COCO** — the shape `label` / category `name` must match one of your
  project's class names (others are skipped + logged). One file may carry
  multiple classes.
- **YOLO** — class **indices** are mapped to names via a `data.yaml` (`names:`)
  or `classes.txt` at the ingest root if present; otherwise by your **project's
  class order** (index 0 = first project class). Keep them aligned.

Notes:
- **Mix freely:** images with annotations get the imported mask; images without
  any still go through SAM2 in Phase 3. SAM2 never overwrites an imported mask.
  Precedence when several are present: **COCO → LabelMe → YOLO**.
- With masks imported, the quickest real-only dataset is just **curate → export**
  (Phase 8) — no captions/LoRA/scaffolds/minting.
- *Limitation:* coordinates are scaled if the annotation's declared image size
  differs from the stored image, but EXIF **rotation** isn't corrected — annotate
  on already-oriented images.

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
fast. Its **open ›** page (Phase 2 — Control maps) shows **image / depth / canny**
side by side. Masks are a separate page (Phase 3).

**Done when:** depth and canny exist for every active image.

---

## Phase 3 — Ground-truth masks

**What it does:** runs **SAM2** to segment each object and paints a
**color-coded mask** (each class gets its color). **These masks become the labels
for everything downstream**, so this is the one phase worth your attention.

> Records whose masks you **imported** in Phase 1 (LabelMe/YOLO/COCO) already have a mask
> and are **skipped** here — SAM2 only fills the un-masked ones, so import and
> SAM2 coexist. To redo an imported mask, prompt + **Save** on it in the Maps
> viewer (that clears the mask), then **Run masks**.

**What you do:**

1. Click **Run** on the masks card for an automatic first pass. With no prompts,
   SAM2 **auto-guesses the single main object** (the largest, most central thing
   in the frame) and masks just that — it does *not* mask the whole scene. The
   guess can still be wrong, so review it.
2. Open the **Ground-truth masks** page (the masks card's **open ›**) and check
   each mask. Multi-class auto-guesses are flagged **needs review** (which class
   is ambiguous).
3. To fix or set a mask precisely, use the **SAM2 prompt canvas** on that image:
   - **click** = add a *positive* point (this pixel is the object),
   - **shift-click** = add a *negative* point (this pixel is NOT the object),
   - **drag** = draw a box around the object.
   **Save**, then **Run masks** — a prompted object always overrides the guess,
   and only the records you changed are recomputed.
   - **Clear** wipes your points **and** the shown mask (it removes the saved
     mask too, so nothing stale lingers). A later **Run masks** will auto-guess
     again unless you've drawn new prompts.

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
4. **See it:** the card's **open ›** opens the LoRA page — a **training loss
   curve** + the run report (base, rank, steps, final loss) for each run. (Runs
   trained before this feature show the report only.) To judge the LoRA's actual
   effect, look at the minted images on the Mint page.

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
Two **sources** (set `scaffolds.sources` in the project config):
- **`depth_remix`** — jittered copies of your **real** depth+mask pairs; faithful
  to your data and your classes. Best for single-class or realistic layouts.
- **`box3d_procedural`** — invented stacked-box geometry for **all** project
  classes; looks abstract/blocky and will include classes your data doesn't have
  (e.g. carton/palette when you only shot polybags). Drop it if that's not what
  you want. `count` (default 500) sets how many pairs — lower it for quick tests.

Each scaffold's `status` is **`pending`** until it's minted in phase 7 — that's a
queue marker, **not** a quality flag.

**Third source — `copy_paste` (paste-then-harmonize):** instead of a new scene,
it cuts a real object (via its mask) and pastes it onto a **real background** at a
depth-aware position/scale, then phase 7's **`sdxl_inpaint`** generator
regenerates *only* the pasted region (depth-ControlNet inpaint) so the background
stays pixel-exact and the object blends in. Set `scaffolds.sources: [copy_paste]`
+ `generation.generator: sdxl_inpaint`. Good for "same scene, varied object";
`generation.strength` tunes blend (low) vs regenerate (high).

> **Use background images (avoids overlap).** Pasting onto your **object** photos
> lands the object on top of an existing one → merged blobs. Instead ingest
> empty-scene photos in a **`bg`/`background`** folder (Phase 1) — copy_paste then
> pastes onto those clean backgrounds, so there's nothing to overlap. With no
> backgrounds it falls back to pasting onto object images (placement still tries to
> dodge the existing object).
>
> **Objects per background** — when `copy_paste` is a source, the Phase 6 page
> shows an **Objects per background** toggle: **Exactly 1** or **1–2 (random)**.
> "1–2" gives a mix of clean singles and deliberate doubles, all exactly labeled.
> The choice is saved as the project default (`scaffolds.copy_paste.paste_count`).

**What you do:** click **Generate scaffolds** (or **Run** on the board). To control
how many to create, set the scaffold **count** in the project config (default 500);
for copy_paste set the **Objects per background** toggle first. Each pair is one
future synthetic image. **See it:** the card's **open ›** opens a gallery of every
scaffold's **control map + mask** side by side.

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
re-run to continue where it left off. **See it:** the card's **open ›** opens a
gallery of the minted images, with a **show masks** toggle to view each one's
label.

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
| Re-do a phase **cleanly** | Click **Reset** on its card → confirm (deletes that phase's outputs), then Run. Mask-reset keeps your prompts; caption-reset keeps hand-edited captions. |
| Fix a mask | Ground-truth masks page → click / shift-click / drag, then Re-run masks |
| Edit a caption | Caption editor (edits survive re-runs) |
| Delete a project | **✕** on its row → confirm (removes data **and** its LoRA) |
| Find the final dataset | `data/<project>/export/yolo_seg/` |

## Troubleshooting

- **A phase won't start / seems stuck** — another GPU job is running (only one at
  a time). Watch the Job log; wait for it to finish.
- **Card stays amber** — some items aren't done. For masks, that usually means
  images still flagged *needs review*; open the Ground-truth masks page and
  prompt them.
- **Minting/LoRA errors about model weights** — the SDXL weights aren't cached
  yet. See `README.md` → Setup for the one-time `hf download` commands (SDXL is
  ungated, no login needed).
- **Out-of-memory during LoRA or minting** — make sure nothing else is using the
  GPU (stop the dashboard/Backbone), then re-run.
