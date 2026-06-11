"""Tiny i18n loader — load string bundles from ``static/i18n/{lang}.json``.

Server-side use: templates pre-render the default language's strings into
the HTML. Client-side use: ``static/js/lang.js`` fetches the JSON bundle
for the chosen language and swaps elements tagged with ``data-i18n``.

Validation: ``available_langs()`` returns only languages whose bundles share
the same key set as the reference language (``en``) — prevents silent missing
translations.
"""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE_LANG = "en"
_I18N_DIR = Path(__file__).resolve().parent / "static" / "i18n"


def load_bundle(lang: str) -> dict[str, str]:
    """Read ``{lang}.json``; raises FileNotFoundError if missing."""
    path = _I18N_DIR / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def available_langs() -> list[str]:
    """Languages whose bundles parse and have the same keys as the reference."""
    if not _I18N_DIR.exists():
        return []
    try:
        reference_keys = set(load_bundle(REFERENCE_LANG).keys())
    except FileNotFoundError:
        return []
    out: list[str] = []
    for path in sorted(_I18N_DIR.glob("*.json")):
        lang = path.stem
        try:
            keys = set(load_bundle(lang).keys())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if keys == reference_keys:
            out.append(lang)
    return out
