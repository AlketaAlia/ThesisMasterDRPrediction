"""Internationalization: load translations from JSON files in `translations/`.

Each language is one JSON file (`en.json`, `sq.json`). The display name shown
in the language selector is mapped from the file's stem in `LANGUAGES`.
"""
import json
import os
from functools import lru_cache


# Display name → JSON filename stem
LANGUAGES = {
    "English": "en",
    "Shqip": "sq",
}


@lru_cache(maxsize=8)
def _load(language_code):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    path = os.path.join(project_root, "translations", f"{language_code}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_translator(display_name):
    """Return a `tr(key)` callable bound to the chosen UI language.

    Falls back to the key itself if the lookup misses, so missing translations
    are visible but non-fatal.
    """
    code = LANGUAGES.get(display_name, "en")
    table = _load(code)

    def tr(key):
        return table.get(key, key)

    return tr
