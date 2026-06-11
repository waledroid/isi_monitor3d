"""i18n — bundles parse and have identical keys across languages."""

from __future__ import annotations

import pytest

from monitor_web.i18n import REFERENCE_LANG, available_langs, load_bundle


def test_reference_bundle_loads() -> None:
    strings = load_bundle(REFERENCE_LANG)
    assert isinstance(strings, dict)
    assert "title" in strings


def test_french_bundle_loads_and_matches_reference_keys() -> None:
    en = load_bundle("en")
    fr = load_bundle("fr")
    assert set(en.keys()) == set(fr.keys()), (
        f"language bundles drift: en-only={set(en) - set(fr)}, "
        f"fr-only={set(fr) - set(en)}"
    )


def test_available_langs_lists_both() -> None:
    langs = available_langs()
    assert "en" in langs
    assert "fr" in langs


def test_load_missing_bundle_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle("zz")
