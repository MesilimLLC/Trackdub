from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .corpus import CHUNKS_DIR, MANIFEST_PATH, Chunk, find_repo_root, load_chunks, load_sources


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._/-]{1,}", re.IGNORECASE)
_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_STOPWORDS = frozenset(
    "a an and are as by do does for from how i in is it of on or our the to we what which with".split()
)

_K1 = 1.2
_B = 0.75
_PART_WEIGHT = 0.4
_TITLE_WEIGHT = 1.0
_SYMBOL_BONUS = 3.0
_MAX_PER_SOURCE = 3
_SNIPPET_CHARS = 800


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _parts(token: str) -> list[str]:
    """Split an identifier like RegisterExecutionProviderLibrary or ep_refit_table into word parts."""
    pieces: list[str] = []
    for chunk in re.split(r"[_./+-]+", token):
        pieces.extend(m.lower() for m in _PART_RE.findall(chunk))
    return [p for p in pieces if len(p) >= 2] if len(pieces) > 1 else []


def _terms(text: str) -> list[str]:
    """Whole tokens plus identifier word parts, so prose queries reach code symbols."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        out.append(raw.lower())
        out.extend(_parts(raw))
    return out


@dataclass(frozen=True)
class _Index:
    chunks: tuple[Chunk, ...]
    counts: tuple[Counter, ...]
    lengths: tuple[int, ...]
    avg_length: float
    df: Counter
    titles: tuple[frozenset[str], ...]


def _signature() -> tuple:
    stamps: list[tuple[str, int]] = []
    if MANIFEST_PATH.is_file():
        stamps.append((MANIFEST_PATH.name, MANIFEST_PATH.stat().st_mtime_ns))
    if CHUNKS_DIR.is_dir():
        stamps.extend((p.name, p.stat().st_mtime_ns) for p in sorted(CHUNKS_DIR.glob("*.txt")))
    repo_root = find_repo_root()
    for source in load_sources():
        if source.kind == "repo-file" and source.path:
            path = repo_root / source.path
            if path.is_file():
                stamps.append((source.path, path.stat().st_mtime_ns))
    return tuple(stamps)


@lru_cache(maxsize=2)
def _build(signature: tuple) -> _Index:
    chunks = tuple(load_chunks())
    counts = tuple(Counter(_terms(c.text)) for c in chunks)
    lengths = tuple(sum(c.values()) for c in counts)
    df: Counter = Counter()
    for c in counts:
        df.update(c.keys())
    avg = (sum(lengths) / len(lengths)) if lengths else 1.0
    titles = tuple(frozenset(_terms(c.title)) for c in chunks)
    return _Index(chunks, counts, lengths, avg, df, titles)


@dataclass(frozen=True)
class _Query:
    weights: dict[str, float]
    idf: dict[str, float]
    symbols: list[str]


def _is_symbol(raw: str) -> bool:
    return bool(re.search(r"[_.]|[a-z][A-Z]", raw))


def _add_parts(weights: dict[str, float], raw: str) -> None:
    for part in _parts(raw):
        if part not in _STOPWORDS:
            weights.setdefault(part, _PART_WEIGHT)


def _query_weights(query: str) -> tuple[dict[str, float], list[str]]:
    weights: dict[str, float] = {}
    symbols: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        whole = raw.lower()
        if whole in _STOPWORDS:
            continue
        weights[whole] = 1.0
        if _is_symbol(raw):
            symbols.append(whole)
        _add_parts(weights, raw)
    return weights, symbols


def _match_positions(lowered: str, terms: dict[str, float], idf: dict[str, float]) -> list[tuple[int, float]]:
    hits: list[tuple[int, float]] = []
    for term, weight in terms.items():
        value = weight * idf.get(term, 0.0)
        if value > 0:
            hits.extend((m.start(), value) for m in re.finditer(re.escape(term), lowered))
    return sorted(hits)


def _best_window_start(hits: list[tuple[int, float]]) -> int:
    def window_score(anchor: int) -> float:
        start = max(0, anchor - 120)
        return sum(value for pos, value in hits if start <= pos < start + _SNIPPET_CHARS)

    return max(0, max((pos for pos, _ in hits), key=window_score) - 120)


def _line_aligned(text: str, start: int) -> int:
    line_start = text.rfind("\n", max(0, start - 80), start)
    return line_start + 1 if line_start != -1 else start


def _snippet(text: str, terms: dict[str, float], idf: dict[str, float]) -> str:
    """Window of the chunk with the densest weighted term matches, trimmed to line starts."""
    if len(text) <= _SNIPPET_CHARS:
        return text
    hits = _match_positions(text.lower(), terms, idf)
    if not hits:
        return text[:_SNIPPET_CHARS]
    start = _line_aligned(text, _best_window_start(hits))
    end = min(len(text), start + _SNIPPET_CHARS)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _term_score(tf: int, in_title: bool, weight_idf: float, norm: float) -> float:
    score = weight_idf * (tf * (_K1 + 1.0)) / (tf + norm) if tf else 0.0
    return score + _TITLE_WEIGHT * weight_idf if in_title else score


def _score_chunk(index: _Index, i: int, query: _Query) -> float:
    norm = _K1 * (1.0 - _B + _B * index.lengths[i] / index.avg_length)
    score = sum(
        _term_score(index.counts[i].get(term, 0), term in index.titles[i], weight * query.idf[term], norm)
        for term, weight in query.weights.items()
    )
    if score <= 0:
        return 0.0
    text_l = index.chunks[i].text.lower()
    score += sum(_SYMBOL_BONUS for symbol in query.symbols if symbol in text_l)
    return score * (1.0 + index.chunks[i].priority / 1000.0)


def _has_tag(chunk: Chunk, tag_l: str | None) -> bool:
    return tag_l is None or any(tag_l == t.lower() for t in chunk.tags)


def _rank(index: _Index, query: _Query, tag_l: str | None) -> list[tuple[float, int]]:
    scored = [(_score_chunk(index, i, query), i) for i, chunk in enumerate(index.chunks) if _has_tag(chunk, tag_l)]
    ranked = [item for item in scored if item[0] > 0]
    ranked.sort(key=lambda item: (-item[0], -index.chunks[item[1]].priority, index.chunks[item[1]].source_id, index.chunks[item[1]].chunk_index))
    return ranked


def _cap_per_source(index: _Index, ranked: list[tuple[float, int]], cap: int) -> list[tuple[float, int]]:
    per_source: Counter = Counter()
    kept: list[tuple[float, int]] = []
    for score, i in ranked:
        source_id = index.chunks[i].source_id
        if per_source[source_id] < _MAX_PER_SOURCE:
            per_source[source_id] += 1
            kept.append((score, i))
    return kept[:cap]


def _result(chunk: Chunk, score: float, query: _Query) -> dict:
    text_l = chunk.text.lower()
    return {
        "score": round(score, 3),
        "sourceId": chunk.source_id,
        "title": chunk.title,
        "origin": chunk.origin,
        "tags": list(chunk.tags),
        "chunkIndex": chunk.chunk_index,
        "matched": sorted(t for t in query.weights if t in text_l),
        "excerpt": _snippet(chunk.text, query.weights, query.idf),
    }


def search_chunks(query: str, *, limit: int = 8, tag: str | None = None) -> list[dict]:
    weights, symbols = _query_weights(query)
    index = _build(_signature())
    total = len(index.chunks)
    if not weights or total == 0:
        return []
    idf = {t: math.log(1.0 + (total - index.df.get(t, 0) + 0.5) / (index.df.get(t, 0) + 0.5)) for t in weights}
    query_data = _Query(weights, idf, symbols)
    ranked = _rank(index, query_data, tag.lower() if tag else None)
    kept = _cap_per_source(index, ranked, max(1, min(limit, 20)))
    return [_result(index.chunks[i], score, query_data) for score, i in kept]


def get_source_text(source_id: str, *, max_chars: int = 12000) -> dict:
    chunks = [c for c in load_chunks() if c.source_id == source_id]
    if not chunks:
        remote_hint = (
            "Source not loaded. For remote URLs, run: uv run trackdub-gpu-docs-ingest"
        )
        return {"found": False, "sourceId": source_id, "message": remote_hint}

    text = "\n\n".join(c.text for c in sorted(chunks, key=lambda c: c.chunk_index))
    truncated = len(text) > max_chars
    return {
        "found": True,
        "sourceId": source_id,
        "title": chunks[0].title,
        "origin": chunks[0].origin,
        "tags": list(chunks[0].tags),
        "truncated": truncated,
        "text": text[:max_chars],
    }
