"""Template captioner — deterministic per record id (re-runs are stable), picks
a pattern + background from the project's editable bank and fills the slots:
{trigger} {class_phrase} {background}.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base import CAPTIONERS, Captioner

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig

_DEFAULT_PATTERNS = ["a photo of {trigger} {class_phrase}, {background}"]
_DEFAULT_BACKGROUNDS = ["a plain industrial environment with a concrete floor"]


@CAPTIONERS.register("template")
class TemplateCaptioner(Captioner):
    def __init__(self, patterns: list[str] | None = None,
                 class_phrases: dict[str, str] | None = None,
                 backgrounds: list[str] | None = None, **cfg) -> None:
        super().__init__(**cfg)
        self.patterns = list(patterns or _DEFAULT_PATTERNS)
        self.class_phrases = dict(class_phrases or {})
        self.backgrounds = list(backgrounds or _DEFAULT_BACKGROUNDS)

    def caption(self, record: ManifestRecord, project: ProjectConfig) -> str:
        spec = project.class_by_name(record.class_name)
        rng = random.Random(record.id)          # deterministic per image
        pattern = rng.choice(self.patterns)
        background = rng.choice(self.backgrounds)
        phrase = self.class_phrases.get(record.class_name, record.class_name)
        return pattern.format(trigger=spec.trigger, class_phrase=phrase,
                              background=background).strip()
