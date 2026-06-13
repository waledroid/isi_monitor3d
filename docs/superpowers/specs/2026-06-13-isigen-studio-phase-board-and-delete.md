# isiGen Studio: phase done-state, refresh/rerun, deletable projects

**Date:** 2026-06-13 · **Status:** implemented

## Why

Two operator-UX gaps on the Studio phase board: (1) no visual signal that a
phase is finished — a completed phase looked identical to a pending one; (2) no
way to delete a project from the UI (only create). Rerun already existed (the
per-card Run button) but wasn't legible.

## Changes (Studio web layer only — no runner/generation changes)

1. **Per-phase done-state** (`static/js/phases.js`): each phase gains a
   `state(s)` → `done` / `partial` / `todo`, computed from the existing
   `/status` counts (`active = records − excluded`). done = all active records
   covered (masks also require `needs_review = 0`; mint requires empty queue);
   partial = some covered. Render applies `.phase-card.done` (light green) /
   `.phase-card.partial` (amber). The board's existing 5 s poll repaints
   automatically.

2. **New status field** `lora_trained` (`api/routes_projects.py`): the only
   phase with no existing completion signal. True when the project's
   `generation.lora_weights` resolves to a file, or any
   `runs/lora/<name>_*/pytorch_lora_weights.safetensors` exists.

3. **Controls**: a **⟳ Refresh** button on the phase page (`phases.html`,
   `#refresh-board`) re-pulls status immediately; each card's button relabels
   **Run → Re-run** when that phase is `done`.

4. **Deletable projects**: `core/project.delete_project(data_dir, name,
   runs_dir)` — guards against traversal/symlinks (target must sit directly
   under `data_dir` with a `project.yaml`), `rmtree`s the data dir, and (per the
   chosen scope) also removes `runs/lora/<name>_*` dirs and
   `runs/jobs/*_<name>_*.log`. Exposed as `DELETE /api/projects/{name}`
   (404 if missing). UI: each project row gets a `✕` button →
   native `confirm(...)` → DELETE → re-render.

## Tests (hermetic)

`tests/test_studio_routes.py`: `lora_trained` reported false→true when a
weights file appears; `DELETE` removes the data dir **and** matching
`runs/lora` dir, status 404s after, double-delete 404s. Suite 35 → 37 green.
The `state()` JS is pure and was validated by node + the live server.
