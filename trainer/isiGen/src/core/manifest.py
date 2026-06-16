"""Per-project manifest — the single source of truth every phase reads/extends.

One JSON object per line (`manifest.jsonl` in the project directory), one record
per curated source image. Phases fill in their fields (depth_map, mask, caption…)
and the Studio visualizers render straight from it. Unknown keys are tolerated
(`extra="allow"`) so later phases can add fields without a migration.

Writes are atomic: serialize to `manifest.jsonl.tmp` then `os.replace`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_NAME = "manifest.jsonl"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MaskPrompt(BaseModel):
    """One SAM2 prompt, authored in the Studio maps page.

    ``point`` → ``xy`` (+ ``label`` 1=foreground / 0=background);
    ``box``   → ``xyxy``. ``class_name`` decides which class's mask the
    resulting segment is painted into.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["point", "box"]
    class_name: str
    xy: list[float] | None = None
    label: int = 1
    xyxy: list[float] | None = None


class ManifestRecord(BaseModel):
    """One curated source image and everything the phases derived from it.

    All paths are RELATIVE to the project directory so a project folder can be
    moved/copied wholesale.
    """

    model_config = ConfigDict(extra="allow")

    id: str                                   # sha256[:12] of image content — stable, dedupe key
    sha256: str
    source_path: str = ""                     # provenance (original import path)
    image: str = ""                           # raw/<class>/<id>.jpg
    class_name: str = ""
    width: int = 0
    height: int = 0
    split: Literal["train", "val"] = "train"
    excluded: bool = False                    # Studio reject — all phases skip when true
    background: bool = False                  # empty-scene image (no object) — a paste
                                              # target for copy_paste; skips mask/caption/
                                              # LoRA/export, never a training sample

    # Phase 2 — dual-layer maps
    depth_map: str | None = None              # maps/depth/<id>.png
    canny_map: str | None = None              # maps/canny/<id>.png
    mask: str | None = None                   # maps/mask/<id>.png (color-coded ground truth)
    mask_prompts: list[MaskPrompt] = Field(default_factory=list)
    mask_source: Literal["prompted", "auto", "imported"] | None = None
    needs_review: bool = False                # auto-mask in a multi-class project

    # Phase 3 — captions
    caption_path: str | None = None           # captions/<id>.txt
    caption_edited: bool = False              # true ⇒ re-runs never overwrite

    updated: str = Field(default_factory=utcnow)
    notes: str = ""


class Manifest:
    """Load/modify/save the project manifest. Small (≤ a few hundred records),
    so a full rewrite on save is trivially cheap — no DB."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.path = self.project_dir / MANIFEST_NAME
        self.records: dict[str, ManifestRecord] = {}

    @classmethod
    def load(cls, project_dir: Path) -> Manifest:
        m = cls(project_dir)
        if m.path.exists():
            for line in m.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = ManifestRecord.model_validate_json(line)
                m.records[rec.id] = rec
        return m

    def upsert(self, record: ManifestRecord) -> None:
        record.updated = utcnow()
        self.records[record.id] = record

    def get(self, record_id: str) -> ManifestRecord | None:
        return self.records.get(record_id)

    def active(self) -> list[ManifestRecord]:
        """Records phases should process (not excluded), in stable id order."""
        return [r for _, r in sorted(self.records.items()) if not r.excluded]

    def save(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for _, rec in sorted(self.records.items()):
                fh.write(json.dumps(rec.model_dump(mode="json")) + "\n")
        os.replace(tmp, self.path)
