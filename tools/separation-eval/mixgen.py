#!/usr/bin/env python3
"""Generate Tier A synthetic mixtures with exact ground truth.

Consumes the item manifest from ingest.py and renders, per clip, a mixture plus
its dialogue and bed (music and effects) references at 48 kHz float32. Every clip
is derived from (seed, clip id) only, so any subset regenerates identically.

Design: docs/audits/separation-eval-corpus-and-rubric.md (section 2).

    python mixgen.py generate --manifest items.manifest.json --cache ./cache --out ./corpus \
        --split dev --recipes A1,A2 --seed 1234
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

import audiomath as am
import ingest
from recipe_data import DEFAULT_LANGUAGE, DEFAULT_RECIPE_IDS, Recipe, RECIPES, item_language, language_allowed

GENERATOR_VERSION = "0.1"
TARGET_LUFS = -23.0
PEAK_LIMIT = 0.98
ALIGN_SEARCH_SECONDS = 10
AUDIO_CACHE_CAPACITY = 16


class GenerationError(Exception):
    pass


def rng_for(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def split_counts(count: int) -> dict[str, int]:
    dev = round(count * ingest.DEV_PERCENT / 100)
    return {"dev": dev, "test": count - dev}


class ItemPool:
    def __init__(self, items: list[dict], cache: Path, split: str,
                 audio_cache_capacity: int = AUDIO_CACHE_CAPACITY):
        if audio_cache_capacity < 1:
            raise ValueError("audio_cache_capacity must be positive")
        self.cache = cache
        self.items = sorted((i for i in items if i["split"] == split), key=lambda i: i["id"])
        self.audio_cache_capacity = audio_cache_capacity
        self._audio: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def with_languages(self, languages: tuple[str, ...]) -> ItemPool:
        """A view whose dialogue items are limited to `languages`; it shares this pool's audio cache."""
        view = copy.copy(self)
        view.items = [i for i in self.items if i["role"] != "dialogue" or language_allowed(i, languages)]
        return view

    def candidates(self, role: str, tags: tuple[str, ...] = (), exclude_groups: frozenset[str] = frozenset()) -> list[dict]:
        return [i for i in self.items
                if i["role"] == role and set(tags) <= set(i.get("tags", [])) and i["group"] not in exclude_groups]

    def pick(self, rng: np.random.Generator, role: str, tags: tuple[str, ...] = (),
             exclude_groups: frozenset[str] = frozenset(), pool: list[dict] | None = None) -> dict:
        cands = pool if pool is not None else self.candidates(role, tags, exclude_groups)
        if not cands:
            raise GenerationError(f"no '{role}' items with tags {list(tags)} in this split")
        return cands[int(rng.integers(len(cands)))]

    def load(self, item: dict, channels: int) -> np.ndarray:
        key = (item["id"], channels)
        if key in self._audio:
            self._audio.move_to_end(key)
        else:
            self._audio[key] = am.load_audio(ingest.cache_path(self.cache, item), channels=channels)
            if len(self._audio) > self.audio_cache_capacity:
                self._audio.popitem(last=False)
        return self._audio[key]


def scoped_dialogue_pool(recipe: Recipe, pool: ItemPool, index: int, meta: dict) -> ItemPool:
    """The pool limited to the languages this clip may use. A multilingual recipe cycles through the
    non-English languages present, one per clip, and records which one it used."""
    languages = recipe.languages
    if not languages:
        present = sorted({item_language(i) for i in pool.items if i["role"] == "dialogue"} - {DEFAULT_LANGUAGE})
        if not present:
            raise GenerationError(f"{recipe.recipe_id} needs non-English dialogue items in this split")
        languages = (present[index % len(present)],)
        meta["dialogue_language"] = languages[0]
    return pool.with_languages(languages)


def loop_to_length(a: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Random excerpt of length n; loops with a short crossfade when the source is shorter."""
    m = a.shape[0]
    if m >= n:
        off = int(rng.integers(0, m - n + 1))
        return a[off:off + n], off
    if m == 0:
        raise ValueError("cannot loop an empty audio source")
    if m == 1:
        return np.repeat(a, n, axis=0), 0

    xf = max(1, min(2400, m // 4))
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32).reshape((xf,) + (1,) * (a.ndim - 1))
    out = np.empty((n,) + a.shape[1:], dtype=np.result_type(a.dtype, np.float32))
    out[:m] = a
    written = m
    while written < n:
        overlap_start = written - xf
        out[overlap_start:written] = out[overlap_start:written] * (1.0 - ramp) + a[:xf] * ramp
        appended = min(n - written, m - xf)
        out[written:written + appended] = a[xf:xf + appended]
        written += appended
    return out, 0


def assemble_dialogue(pool: ItemPool, rng: np.random.Generator, n: int, tags: tuple[str, ...],
                      exclude_groups: frozenset[str], gap_s: tuple[float, float]) -> tuple[np.ndarray, list[dict], str]:
    """One speaker group laid out on a timeline of n samples with pauses. Returns mono audio."""
    first = pool.pick(rng, "dialogue", tags, exclude_groups)
    group_items = [i for i in pool.candidates("dialogue", tags) if i["group"] == first["group"]]
    out = np.zeros(n, dtype=np.float32)
    used: list[dict] = []
    pos = int(rng.uniform(0.0, 0.5) * am.SR)
    item = first
    while pos < n:
        a = pool.load(item, 1)[:, 0]
        take = min(a.shape[0], n - pos)
        off = int(rng.integers(0, a.shape[0] - take + 1))
        out[pos:pos + take] = a[off:off + take]
        used.append({"id": item["id"], "sha256": item["sha256"], "offset_s": round(off / am.SR, 3)})
        pos += take + int(rng.uniform(*gap_s) * am.SR)
        item = group_items[int(rng.integers(len(group_items)))]
    return out, used, first["group"]


def pan_and_delay(mono: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    pan = float(rng.uniform(-0.5, 0.5))
    theta = (pan + 1.0) * math.pi / 4.0
    stereo = np.stack([mono * math.cos(theta) * math.sqrt(2), mono * math.sin(theta) * math.sqrt(2)], axis=1)
    haas_ms = 0.0
    if rng.random() < 0.3:
        haas_ms = float(rng.uniform(1.0, 20.0))
        d = int(haas_ms / 1000.0 * am.SR)
        side = int(rng.integers(2))
        shifted = np.zeros_like(stereo[:, side])
        shifted[d:] = stereo[:-d, side]
        stereo[:, side] = shifted
    return stereo.astype(np.float32), {"pan": round(pan, 3), "haas_ms": round(haas_ms, 2)}


def widen(stereo: np.ndarray, w: float) -> np.ndarray:
    mid = (stereo[:, 0] + stereo[:, 1]) / 2.0
    side = (stereo[:, 0] - stereo[:, 1]) / 2.0 * w
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def reverberate(mono: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a RIR aligned so its direct path sits at t=0, keeping dialogue timing."""
    h = rir.astype(np.float64).reshape(-1)
    h = h[int(np.argmax(np.abs(h))):]
    h = h / math.sqrt(float((h ** 2).sum()))
    return signal.fftconvolve(mono.astype(np.float64), h)[: mono.shape[0]].astype(np.float32)


def apply_variant(arrays: dict[str, np.ndarray], variant: str) -> tuple[dict[str, np.ndarray], int]:
    """Re-render refs and mixture into an alternate input format. Returns (arrays, sample_rate)."""
    if variant == "mono":
        return {k: v.mean(axis=1, keepdims=True) for k, v in arrays.items()}, am.SR
    if variant in ("sr8000", "sr16000"):
        sr = int(variant[2:])
        return {k: am.resample(v, am.SR, sr) for k, v in arrays.items()}, sr
    if variant == "5.1":
        out = {}
        for k in ("dialogue", "bed"):
            v = arrays[k]
            z = np.zeros(v.shape[0], dtype=np.float32)
            if k == "dialogue":
                out[k] = np.stack([z, z, v.mean(axis=1), z, z, z], axis=1)
            else:
                out[k] = np.stack([v[:, 0], v[:, 1], z, z, 0.5 * v[:, 0], 0.5 * v[:, 1]], axis=1)
        out["mixture"] = out["dialogue"] + out["bed"]
        return out, am.SR
    raise GenerationError(f"unknown variant '{variant}'")


def codec_roundtrip(x: np.ndarray, codec: str) -> tuple[np.ndarray, int]:
    """Encode and decode through ffmpeg, then undo codec delay by cross-correlation."""
    if not am.have_ffmpeg():
        raise GenerationError("codec round trip requested but ffmpeg is not installed (use --no-codec)")
    args = {"aac": (".m4a", ["-c:a", "aac", "-b:a", "128k"]), "ac3": (".ac3", ["-c:a", "ac3", "-b:a", "192k"])}[codec]
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / f"out{args[0]}"
        am.write_wav(src, x, am.SR)
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *args[1], str(dst)],
                              capture_output=True, check=False)
        if proc.returncode != 0:
            raise GenerationError(f"ffmpeg {codec} encode failed: {proc.stderr.decode(errors='replace')[:200]}")
        y = am.load_audio(dst, channels=x.shape[1])
    seg = min(x.shape[0], y.shape[0], ALIGN_SEARCH_SECONDS * am.SR)
    xs, ys = x[:seg].mean(axis=1), y[:seg].mean(axis=1)
    corr = signal.correlate(ys, xs, mode="full", method="fft")
    lag = int(np.argmax(corr)) - (seg - 1)
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.concatenate([np.zeros((-lag, y.shape[1]), dtype=np.float32), y], axis=0)
    n = x.shape[0]
    y = y[:n] if y.shape[0] >= n else np.concatenate([y, np.zeros((n - y.shape[0], y.shape[1]), dtype=np.float32)], axis=0)
    return y.astype(np.float32), lag


@dataclass
class Context:
    items: list[dict]
    cache: Path
    seed: int
    no_codec: bool = False
    duration_override: float | None = None
    pools: dict[str, ItemPool] = field(default_factory=dict)

    def pool(self, split: str) -> ItemPool:
        if split not in self.pools:
            self.pools[split] = ItemPool(self.items, self.cache, split)
        return self.pools[split]


def generate_clip(recipe: Recipe, index: int, split: str, ctx: Context) -> tuple[dict[str, np.ndarray], int, dict]:
    clip_id = f"{recipe.recipe_id.lower()}-{split}-{index:03d}"
    rng = rng_for(ctx.seed, clip_id)
    pool = ctx.pool(split)
    duration = recipe.duration_s if ctx.duration_override is None else ctx.duration_override
    if not math.isfinite(duration) or duration <= 0:
        raise GenerationError("duration override must be finite and positive")
    n = int(duration * am.SR)
    meta: dict = {"clip_id": clip_id, "recipe": recipe.recipe_id, "split": split, "seed": ctx.seed, "sources": []}

    d = np.zeros((n, 2), dtype=np.float32)
    if recipe.dialogue:
        rir_item, rt60 = None, None
        if recipe.reverb:
            ok = []
            for cand in pool.candidates("rir"):
                t = am.estimate_rt60(pool.load(cand, 1))
                if not math.isnan(t) and recipe.rt60_s[0] <= t <= recipe.rt60_s[1]:
                    ok.append((cand, t))
            if not ok:
                raise GenerationError(f"no RIR with RT60 in {recipe.rt60_s} in this split")
            rir_item, rt60 = ok[int(rng.integers(len(ok)))]
            meta.update({"rir_id": rir_item["id"], "rt60_s": round(rt60, 3)})
            meta["sources"].append({"id": rir_item["id"], "sha256": rir_item["sha256"], "role": "rir"})
        used_groups: frozenset[str] = frozenset()
        speaker_meta = []
        dialogue_pool = scoped_dialogue_pool(recipe, pool, index, meta)
        for s in range(2 if recipe.overlap else 1):
            mono, used, group = assemble_dialogue(dialogue_pool, rng, n, recipe.dialogue_tags, used_groups, recipe.gap_s)
            used_groups = used_groups | {group}
            if rir_item is not None:
                mono = reverberate(mono, pool.load(rir_item, 1))
            stereo, pd = pan_and_delay(mono, rng)
            if s == 1:
                stereo *= 10.0 ** (float(rng.uniform(-6.0, 0.0)) / 20.0)
            d += stereo
            speaker_meta.append({"group": group, **pd})
            meta["sources"].extend({**u, "role": "dialogue"} for u in used)
        meta["speakers"] = speaker_meta

    b = np.zeros((n, 2), dtype=np.float32)
    layer_meta = []
    for layer in recipe.layers:
        if layer.probability < 1.0 and rng.random() >= layer.probability:
            continue
        item = pool.pick(rng, layer.role, layer.tags)
        seg, off = loop_to_length(pool.load(item, 2), n, rng)
        rel_db = float(rng.uniform(-6.0, 0.0)) if layer_meta else 0.0
        b += seg * 10.0 ** (rel_db / 20.0)
        layer_meta.append({"id": item["id"], "offset_s": round(off / am.SR, 3), "rel_db": round(rel_db, 2)})
        meta["sources"].append({"id": item["id"], "sha256": item["sha256"], "role": layer.role})
    if layer_meta:
        w = float(rng.uniform(1.0, 1.6)) if rng.random() < 0.3 else 1.0
        b = widen(b, w)
        meta["bed_width"] = round(w, 3)
    meta["bed_layers"] = layer_meta

    if recipe.dialogue and layer_meta:
        snr = float(rng.uniform(*recipe.snr_db))
        mask = am.frame_activity(d, am.SR)
        dr, br = am.active_rms(d, mask, am.SR), am.active_rms(b, mask, am.SR)
        if dr == 0.0 or br == 0.0:
            raise GenerationError(f"{clip_id}: cannot calibrate SNR (silent dialogue or bed)")
        b = (b * (dr / (br * 10.0 ** (snr / 20.0)))).astype(np.float32)
        meta["target_snr_db"] = round(snr, 2)

    mix = d + b
    lufs = am.integrated_lufs(mix)
    if math.isinf(lufs):
        raise GenerationError(f"{clip_id}: mixture is silent or too short to measure loudness")
    gain = 10.0 ** ((TARGET_LUFS - lufs) / 20.0)
    peak = max(float(np.abs(a).max()) for a in (mix, d, b)) * gain
    if peak > PEAK_LIMIT:
        gain *= PEAK_LIMIT / peak
        meta["peak_limited"] = True
    d, b, mix = d * gain, b * gain, mix * gain
    meta.update({"gain_db": round(20.0 * math.log10(gain), 3), "mixture_lufs": round(am.integrated_lufs(mix), 2)})
    if recipe.dialogue and layer_meta:
        mask = am.frame_activity(d, am.SR)
        meta["measured_snr_db"] = round(20.0 * math.log10(am.active_rms(d, mask, am.SR) / am.active_rms(b, mask, am.SR)), 2)

    arrays = {"mixture": mix.astype(np.float32), "dialogue": d.astype(np.float32), "bed": b.astype(np.float32)}
    sr = am.SR
    if recipe.variants:
        variant = recipe.variants[index % len(recipe.variants)]
        arrays, sr = apply_variant(arrays, variant)
        meta["variant"] = variant
    meta.update({"sample_rate": sr, "channels": arrays["mixture"].shape[1], "codec": None})

    if recipe.codec_p > 0.0 and not ctx.no_codec and rng.random() < recipe.codec_p:
        codec = ["aac", "ac3"][int(rng.integers(2))]
        arrays["mixture"], lag = codec_roundtrip(arrays["mixture"], codec)
        meta.update({"codec": codec, "codec_lag_samples": lag})
    return arrays, sr, meta


def write_clip(out_dir: Path, arrays: dict[str, np.ndarray], sr: int, meta: dict) -> dict:
    clip_dir = out_dir / meta["clip_id"]
    clip_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arr in arrays.items():
        path = clip_dir / f"{name}.wav"
        am.write_wav(path, arr, sr)
        files[name] = ingest.sha256_file(path)
    meta = {**meta, "files": files}
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def generate_corpus(items: list[dict], cache: Path, out_dir: Path, *, split: str, recipes: list[str],
                    seed: int, items_sha256: str, count_override: int | None = None,
                    duration_override: float | None = None, no_codec: bool = False) -> dict:
    ctx = Context(items, cache, seed, no_codec=no_codec, duration_override=duration_override)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for rid in recipes:
        if rid not in RECIPES:
            raise GenerationError(f"unknown recipe '{rid}'")
        recipe = RECIPES[rid]
        count = count_override if count_override is not None else split_counts(recipe.count)[split]
        if count <= 0:
            raise GenerationError("clip count must be positive")
        for index in range(count):
            arrays, sr, meta = generate_clip(recipe, index, split, ctx)
            clips.append(write_clip(out_dir, arrays, sr, meta))
            print(f"  {meta['clip_id']}", file=sys.stderr)
    manifest = {
        "corpus_schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "items_sha256": items_sha256,
        "seed": seed,
        "split": split,
        "target_lufs": TARGET_LUFS,
        "clips": clips,
    }
    (out_dir / "corpus.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="render clips for one split")
    g.add_argument("--manifest", type=Path, required=True)
    g.add_argument("--cache", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--split", choices=("dev", "test"), required=True)
    g.add_argument("--recipes", default=",".join(DEFAULT_RECIPE_IDS),
                   help="comma-separated recipe ids (default: every recipe except the optional ones, such as A13)")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--count", type=int, help="clips per recipe (default: recipe count split dev/test)")
    g.add_argument("--duration", type=float, help="override clip duration in seconds (development only)")
    g.add_argument("--no-codec", action="store_true", help="never apply the codec round trip")
    sub.add_parser("recipes", help="list recipes")
    args = parser.parse_args(argv)

    if args.command == "recipes":
        for r in RECIPES.values():
            print(f"{r.recipe_id:4} {r.count:3} clips  {r.summary}{'  (optional)' if r.optional else ''}")
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems = ingest.verify_manifest(manifest, args.cache)
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        print("item manifest does not verify against the cache; refusing to generate", file=sys.stderr)
        return 1
    try:
        result = generate_corpus(
            manifest["items"], args.cache, args.out, split=args.split,
            recipes=[r.strip().upper() for r in args.recipes.split(",") if r.strip()],
            seed=args.seed, items_sha256=manifest["items_sha256"], count_override=args.count,
            duration_override=args.duration, no_codec=args.no_codec)
    except (GenerationError, am.AudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(result['clips'])} clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
