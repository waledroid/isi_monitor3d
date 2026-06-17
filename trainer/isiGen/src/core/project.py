"""Project config — `data/<project>/project.yaml`, validated by pydantic.

A *project* is one dataset type (e.g. warehouse pallets) with its own classes,
trigger words, colors, and per-phase parameters. The phase runners and the
Studio both load it through here; `configs/project_template.yaml` is the
commented schema copied on project creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_YAML = "project.yaml"

_ISIGEN_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = _ISIGEN_ROOT / "configs" / "project_template.yaml"


class ClassSpec(BaseModel):
    """One object class: dataset name + LoRA trigger word + ground-truth color."""

    name: str
    trigger: str                          # unique token the LoRA binds to (e.g. ISI_PLT)
    color: list[int]                      # RGB 0-255, unique per class (mask painting)

    @field_validator("color")
    @classmethod
    def _color_rgb(cls, v: list[int]) -> list[int]:
        if len(v) != 3 or not all(0 <= c <= 255 for c in v):
            raise ValueError("color must be [R, G, B] with 0-255 components")
        return v


class ProjectConfig(BaseModel):
    """The whole project.yaml. Phase params stay an open dict per phase so each
    plugin reads its own sub-dict (`phases.<phase>.<plugin_name>`) — adding a
    plugin never needs a schema change here."""

    model_config = ConfigDict(extra="allow")

    name: str
    classes: list[ClassSpec] = Field(min_length=1)
    # "generate" = full synthetic pipeline; "label" = Studio used as an annotation
    # tool (curate → SAM2 masks → LabelMe export, no captions/LoRA/scaffolds/mint).
    mode: Literal["generate", "label"] = "generate"
    phases: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_class_fields(self) -> ProjectConfig:
        names = [c.name for c in self.classes]
        triggers = [c.trigger for c in self.classes]
        colors = [tuple(c.color) for c in self.classes]
        for label, vals in (("name", names), ("trigger", triggers), ("color", colors)):
            if len(vals) != len(set(vals)):
                raise ValueError(f"class {label}s must be unique: {vals}")
        return self

    # ---- helpers ----

    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    def class_by_name(self, name: str) -> ClassSpec:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"unknown class {name!r} (project has: {self.class_names()})")

    def phase(self, key: str) -> dict:
        v = self.phases.get(key)
        return v if isinstance(v, dict) else {}


def load_project(project_dir: Path) -> ProjectConfig:
    path = Path(project_dir) / PROJECT_YAML
    data = yaml.safe_load(path.read_text()) or {}
    return ProjectConfig.model_validate(data)


def save_project(project_dir: Path, cfg: ProjectConfig) -> None:
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / PROJECT_YAML).write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )


def create_project(data_dir: Path, name: str, classes: list[ClassSpec], *,
                   mode: Literal["generate", "label"] = "generate") -> Path:
    """New project dir from the template: template phases + the given classes.

    ``mode="label"`` makes the project a LabelMe annotation dataset (no synthetic
    generation) — the export defaults to the labelme exporter. Refuses to
    overwrite an existing project."""
    project_dir = Path(data_dir) / name
    if (project_dir / PROJECT_YAML).exists():
        raise FileExistsError(f"project {name!r} already exists at {project_dir}")
    template = yaml.safe_load(TEMPLATE_PATH.read_text()) or {}
    cfg = ProjectConfig.model_validate(
        {**template, "name": name, "mode": mode,
         "classes": [c.model_dump() for c in classes]}
    )
    if mode == "label":
        cfg.phases.setdefault("export", {})["exporters"] = ["labelme"]
    save_project(project_dir, cfg)
    for sub in ("raw", "maps/depth", "maps/canny", "maps/mask",
                "captions", "thumbs", "scaffolds", "generated", "export"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)
    return project_dir


def list_projects(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(p.parent.name for p in data_dir.glob(f"*/{PROJECT_YAML}"))


def delete_project(data_dir: Path, name: str, runs_dir: Path | None = None) -> None:
    """Remove a project's data dir and (if given) its trained LoRA + job logs.

    Guards against path traversal/symlinks: the target must sit directly under
    ``data_dir`` and contain a ``project.yaml``. Raises FileNotFoundError if no
    such project exists."""
    import shutil

    data_dir = Path(data_dir).resolve()
    project_dir = (data_dir / name).resolve()
    if project_dir.parent != data_dir or not (project_dir / PROJECT_YAML).is_file():
        raise FileNotFoundError(f"project {name!r} not found under {data_dir}")
    shutil.rmtree(project_dir)
    if runs_dir is not None:
        runs_dir = Path(runs_dir)
        for d in (runs_dir / "lora").glob(f"{name}_*"):       # LoRA run dirs
            shutil.rmtree(d, ignore_errors=True)
        for f in (runs_dir / "jobs").glob(f"*_{name}_*.log"):  # job log files
            f.unlink(missing_ok=True)
