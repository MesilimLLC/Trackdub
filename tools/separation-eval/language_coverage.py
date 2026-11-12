"""Language rules for the corpus coverage check: which recipes apply, which dialogue each may use, and A13's reach."""

from __future__ import annotations

import ingest
from recipe_data import RECIPES, Recipe, item_language, language_allowed

SPLITS = ("dev", "test")
MIN_LANGUAGES = 5
LANGUAGE_MIN_SPEAKERS = 8


def has_multilingual_dialogue(items: list[dict]) -> bool:
    return any(i["role"] == "dialogue" and language_allowed(i, ()) for i in items)


def active_recipes(items: list[dict]) -> list[Recipe]:
    """Optional recipes (A13) are checked only when the items include non-English dialogue."""
    multilingual = has_multilingual_dialogue(items)
    return [r for r in RECIPES.values() if multilingual or not r.optional]


def items_for(recipe: Recipe, items: list[dict]) -> list[dict]:
    """The items a recipe may draw on: dialogue is limited to the recipe's languages."""
    return [i for i in items if i["role"] != "dialogue" or language_allowed(i, recipe.languages)]


def recipe_pools(items: list[dict]) -> list[tuple[Recipe, list[dict]]]:
    """Each recipe that applies to these items, with the items it may draw on."""
    return [(recipe, items_for(recipe, items)) for recipe in active_recipes(items)]


def language_speakers(items: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Speaker groups of non-English dialogue, by (language, split)."""
    groups: dict[tuple[str, str], set[str]] = {}
    for i in items:
        if i["role"] == "dialogue" and language_allowed(i, ()):
            groups.setdefault((item_language(i), ingest.assign_split(i["group"])), set()).add(i["group"])
    return groups


def enough_languages(groups: dict[tuple[str, str], set[str]], split: str, need: int) -> list[str]:
    return [lang for (lang, s), speakers in groups.items() if s == split and len(speakers) >= need]


def language_gaps(items: list[dict], min_groups: int) -> list[str]:
    """A13 needs MIN_LANGUAGES non-English languages, each with enough speakers, in both splits."""
    groups = language_speakers(items)
    need = max(min_groups, LANGUAGE_MIN_SPEAKERS)
    gaps = []
    for split in SPLITS if groups else ():
        enough = enough_languages(groups, split, need)
        if len(enough) < MIN_LANGUAGES:
            gaps.append(f"A13 [{split}]: {len(enough)} non-English language(s) with {need}+ speakers, need {MIN_LANGUAGES}")
    return gaps
