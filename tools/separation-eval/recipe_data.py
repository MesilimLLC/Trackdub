"""Dependency-free definitions of synthetic separation evaluation recipes."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LANGUAGE = "en"


def item_language(item: dict) -> str:
    """Items from corpus v1.0 carry no language and are English."""
    return item.get("language", DEFAULT_LANGUAGE)


def language_allowed(item: dict, languages: tuple[str, ...]) -> bool:
    """True when the item's language is listed; an empty list means every language except the default."""
    language = item_language(item)
    return language in languages if languages else language != DEFAULT_LANGUAGE


@dataclass(frozen=True)
class Layer:
    """One background layer: items of `role` carrying every tag in `tags`."""

    role: str
    tags: tuple[str, ...] = ()
    probability: float = 1.0


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    summary: str
    count: int
    duration_s: float = 30.0
    dialogue: bool = True
    dialogue_tags: tuple[str, ...] = ()
    languages: tuple[str, ...] = (DEFAULT_LANGUAGE,)
    optional: bool = False
    overlap: bool = False
    layers: tuple[Layer, ...] = (Layer("music", ("instrumental",)),)
    snr_db: tuple[float, float] = (0.0, 15.0)
    reverb: bool = False
    rt60_s: tuple[float, float] = (0.4, 2.0)
    codec_p: float = 0.25
    variants: tuple[str, ...] = ()
    gap_s: tuple[float, float] = (0.2, 1.0)


RECIPES: dict[str, Recipe] = {recipe.recipe_id: recipe for recipe in (
    Recipe("A1", "clean dialogue over instrumental score", 30),
    Recipe("A2", "dialogue over loud SFX", 30,
           layers=(Layer("sfx", ("loud",)), Layer("music", ("instrumental",), 0.5)), snr_db=(-3.0, 6.0)),
    Recipe("A3", "reverberant dialogue", 30, reverb=True),
    Recipe("A4", "dialogue over crowd and walla", 20,
           layers=(Layer("ambience", ("crowd",)),), snr_db=(0.0, 12.0)),
    Recipe("A5", "score vocals present", 30,
           layers=(Layer("music", ("vocals",)),), snr_db=(0.0, 12.0)),
    Recipe("A6", "overlapping dialogue", 20, overlap=True),
    Recipe("A7", "low SNR dialogue", 20, snr_db=(-10.0, 0.0)),
    Recipe("A8", "whisper and soft speech", 15, dialogue_tags=("whisper",),
           layers=(Layer("ambience"), Layer("music", ("instrumental",), 0.5)), snr_db=(0.0, 10.0)),
    Recipe("A9", "dialogue only, no background", 15, layers=()),
    Recipe("A10", "background only, no dialogue", 15, dialogue=False),
    Recipe("A11", "input format robustness", 15, codec_p=0.0,
           variants=("mono", "sr8000", "sr16000", "5.1")),
    Recipe("A12", "long form", 3, duration_s=1200.0, gap_s=(0.5, 4.0),
           layers=(Layer("music", ("instrumental",)), Layer("ambience", (), 0.5))),
    # Corpus v1.1 only: needs non-English dialogue items, so it is left out of default runs and coverage checks.
    Recipe("A13", "multilingual dialogue", 30, languages=(), optional=True),
)}

DEFAULT_RECIPE_IDS = tuple(rid for rid, recipe in RECIPES.items() if not recipe.optional)
