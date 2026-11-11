from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PACKAGE_ROOT / "corpus" / "manifest.v0.json"
DATA_DIR = PACKAGE_ROOT / ".data"
CHUNKS_DIR = DATA_DIR / "chunks"


@dataclass(frozen=True)
class CorpusSource:
    id: str
    kind: str
    title: str
    tags: tuple[str, ...]
    priority: int
    path: str | None = None
    url: str | None = None
    pin_warning: str | None = None


@dataclass(frozen=True)
class Chunk:
    source_id: str
    title: str
    tags: tuple[str, ...]
    priority: int
    origin: str
    text: str
    chunk_index: int


def find_repo_root(start: Path | None = None) -> Path:
    cursor = (start or Path(__file__).resolve()).parent
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "runtime" / "trt-rtx-ep.manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate Trackdub repo root (missing runtime/trt-rtx-ep.manifest.json)."
    )


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _docs_rag_corpus() -> dict[str, Any]:
    path = find_repo_root() / "tools" / "docs-rag" / "corpus.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources() -> list[CorpusSource]:
    manifest = load_manifest()
    sources: list[CorpusSource] = []
    for raw in manifest["sources"]:
        sources.append(
            CorpusSource(
                id=raw["id"],
                kind=raw["kind"],
                title=raw["title"],
                tags=tuple(raw.get("tags", [])),
                priority=int(raw.get("priority", 0)),
                path=raw.get("path"),
                url=raw.get("url"),
                pin_warning=raw.get("pinWarning"),
            )
        )
    # Remote pages live once, in the docs-rag corpus; a vendor source opts in with a "gpu" block.
    for vendor in _docs_rag_corpus()["vendors"]:
        for entry in vendor["sources"]:
            gpu = entry.get("gpu") if isinstance(entry, dict) else None
            if gpu is None:
                continue
            sources.append(
                CorpusSource(
                    id=gpu["id"],
                    kind="url",
                    title=gpu["title"],
                    tags=tuple(gpu.get("tags", [])),
                    priority=int(gpu.get("priority", 0)),
                    url=entry["url"],
                    pin_warning=gpu.get("pinWarning"),
                )
            )
    return sources


def _chunk_text(text: str, *, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    parts: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        if end < len(cleaned):
            split_at = cleaned.rfind("\n\n", start, end)
            if split_at <= start + max_chars // 3:
                split_at = cleaned.rfind("\n", start, end)
            if split_at > start + max_chars // 3:
                end = split_at
        parts.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]


def _read_repo_file(source: CorpusSource, repo_root: Path) -> str:
    if not source.path:
        raise ValueError(f"repo-file source {source.id} missing path")
    path = repo_root / source.path
    if not path.is_file():
        raise FileNotFoundError(f"Missing repo file for {source.id}: {path}")
    return path.read_text(encoding="utf-8")


def _read_ingested_url(source: CorpusSource) -> str | None:
    cached = CHUNKS_DIR / f"{source.id}.txt"
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    return None


def load_chunks(*, include_remote_if_cached: bool = True) -> list[Chunk]:
    repo_root = find_repo_root()
    chunks: list[Chunk] = []
    for source in load_sources():
        text: str | None = None
        origin: str
        if source.kind == "repo-file":
            text = _read_repo_file(source, repo_root)
            origin = source.path or source.id
        elif source.kind == "url":
            if not include_remote_if_cached:
                continue
            text = _read_ingested_url(source)
            if text is None:
                continue
            origin = source.url or source.id
        else:
            continue

        for index, piece in enumerate(_chunk_text(text)):
            chunks.append(
                Chunk(
                    source_id=source.id,
                    title=source.title,
                    tags=source.tags,
                    priority=source.priority,
                    origin=origin,
                    text=piece,
                    chunk_index=index,
                )
            )
    return chunks


def pin_summary() -> dict[str, Any]:
    manifest = load_manifest()
    pin = dict(manifest["pin"])
    ep_manifest_path = find_repo_root() / "runtime" / "trt-rtx-ep.manifest.json"
    ep_manifest = json.loads(ep_manifest_path.read_text(encoding="utf-8"))
    pin["liveEpManifest"] = {
        "version": ep_manifest.get("version"),
        "cudaVariant": ep_manifest.get("cudaVariant"),
        "matchesCorpusPin": (
            ep_manifest.get("version") == pin.get("epAbiVersion")
            and ep_manifest.get("cudaVariant") == pin.get("cudaVariant")
        ),
    }
    return pin
