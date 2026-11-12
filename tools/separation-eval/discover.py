#!/usr/bin/env python3
"""Discover candidate corpus items with machine-readable licenses, without downloading any audio.

Reads a source spec, queries Openverse (Freesound, Jamendo, Wikimedia Commons) and archive.org's LibriVox
collection, keeps only items whose license is on ingest.py's allow-list, and writes the candidate list that
`ingest.py ingest` consumes. API responses are cached so reruns do not spend Openverse's anonymous quota
(20 requests/min, 200/day).

Tags on music, SFX and ambience items come from the search query that found them (`tag_basis` records
this). They are a starting point, not a verification: spot-check by listening before trusting a stratum.

    python discover.py --spec corpus-sources.v1.json --out items.v1.jsonl --api-cache D:/api-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import ingest

DISCOVER_VERSION = "1"
OPENVERSE_URL = "https://api.openverse.org/v1/audio/"
ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA_URL = "https://archive.org/metadata/"
ARCHIVE_DOWNLOAD_URL = "https://archive.org/download/"
OPENVERSE_PAGE_SIZE = 20
OPENVERSE_PAUSE_S = 3.1
ARCHIVE_PAUSE_S = 0.5
AUDIO_EXTENSIONS = {"mp3", "ogg", "oga", "wav", "flac"}
FILETYPE_ALIASES = {"mp32": "mp3"}  # Jamendo labels its MP3 streams "mp32"
PUBLIC_DOMAIN_URLS = (
    "creativecommons.org/publicdomain/mark/1.0",
    "creativecommons.org/licenses/publicdomain",
    "creativecommons.org/publicdomain/zero/1.0",
)
NON_SOLO_TITLE = re.compile(r"collection|anthology|dramatic|poetry|poems|\bvol(ume|\.)?\b|short stor", re.IGNORECASE)
# LibriVox collaborative (multi-reader) readings carry "collab" or "multivoice" in the archive identifier
# even when the title gives no hint (e.g. "The Wind in the Willows"), so the identifier must be checked too.
NON_SOLO_IDENTIFIER = re.compile(r"collab|multivoice", re.IGNORECASE)

FetchJson = Callable[[str], dict]


class DiscoverError(Exception):
    pass


class QuotaError(DiscoverError):
    pass


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            raise DiscoverError("HTTPS request redirected to a non-HTTPS URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _is_loopback_http(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower() == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def creator_group_key(text: str) -> str:
    """Encode a normalized creator name without discarding non-ASCII identity."""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"[^\w]+", "-", normalized, flags=re.UNICODE).strip("-") or "unknown"


def _json_request(url: str, auth_header: str | None) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "trackdub-separation-eval/1", "Accept": "application/json"})
    if auth_header:
        # urllib does not copy unredirected headers onto a redirected request.
        req.add_unredirected_header("Authorization", auth_header)
    return req


def cached_fetch_json(cache_dir: Path | None, pause_s: float, *, retries: int = 3,
                      auth_header: str | None = None) -> FetchJson:
    """GET JSON with an on-disk response cache, a polite pause, and retries on transient errors.
    `auth_header` is sent as `Authorization` to the requested URL only (never to a redirect target) and
    is not part of the cache key, so it must not change what a URL returns."""
    last_call = [0.0]

    def fetch(url: str) -> dict:
        path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(retries):
            wait = pause_s - (time.monotonic() - last_call[0])
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.monotonic()
            if urllib.parse.urlsplit(url).scheme.lower() != "https" and not _is_loopback_http(url):
                raise DiscoverError("url must be https")
            req = _json_request(url, auth_header)
            try:
                opener = urllib.request.build_opener(_HttpsRedirectHandler())
                with opener.open(req, timeout=60) as resp:
                    if urllib.parse.urlsplit(resp.geturl()).scheme.lower() != "https" and not _is_loopback_http(url):
                        raise DiscoverError("HTTPS request redirected to a non-HTTPS URL")
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    raise QuotaError(f"rate limited by {urllib.parse.urlsplit(url).netloc} "
                                     f"(Retry-After {exc.headers.get('Retry-After', 'unknown')}s); rerun later, "
                                     "cached responses are kept") from exc
                if exc.code >= 500 and attempt + 1 < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise DiscoverError(f"GET {url} failed with HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise DiscoverError(f"GET {url} failed: {exc}") from exc
        if path is not None:
            path.write_text(json.dumps(data), encoding="utf-8")
        return data

    return fetch


def spdx_for_openverse(license_code: str, version: str | None) -> str | None:
    if license_code == "cc0":
        return "CC0-1.0"
    if license_code == "pdm":
        return "LicenseRef-PublicDomain"
    if license_code == "by" and version in ("3.0", "4.0"):
        return f"CC-BY-{version}"
    return None


def openverse_candidate(result: dict, spec: dict, retrieved_at: str) -> tuple[dict | None, str | None]:
    spdx = spdx_for_openverse(str(result.get("license", "")).lower(), str(result.get("license_version") or ""))
    if spdx is None:
        return None, f"license {result.get('license')} {result.get('license_version')} not accepted"
    url = result.get("url") or ""
    if not url.startswith("https://"):
        return None, "no https audio url"
    filetype = FILETYPE_ALIASES.get(str(result.get("filetype", "")).lower(), str(result.get("filetype", "")).lower())
    if filetype not in AUDIO_EXTENSIONS:
        return None, f"filetype {result.get('filetype')} not supported"
    creator = (result.get("creator") or "").strip()
    if not creator:
        return None, "no creator"
    duration_s = (result.get("duration") or 0) / 1000.0
    if not spec.get("min_duration_s", 0) <= duration_s <= spec.get("max_duration_s", 1e9):
        return None, f"duration {duration_s:.1f}s outside bounds"
    title = (result.get("title") or "untitled").strip()
    landing = result.get("foreign_landing_url") or ""
    source = result.get("source") or result.get("provider") or "openverse"
    item = {
        "id": f"{slug(source)}-{spec['role'][:3]}-{str(result['id']).replace('-', '')[:12]}",
        "source": f"openverse:{source}",
        "url": url,
        "role": spec["role"],
        "group": f"{source}:{creator_group_key(creator)}",
        "license_spdx": spdx,
        "attribution_text": f'"{title}" by {creator}, {spdx}, {landing}'.strip(", "),
        "license_evidence_url": landing,
        "tags": sorted({str(t).lower() for t in spec.get("tags", [])}),
        "tag_basis": "search-query",
        "ext": filetype,
        "title": title,
        "creator": creator,
        "query": spec["q"],
        "retrieved_at": retrieved_at,
    }
    range_bytes = spec.get("range_bytes")
    filesize = result.get("filesize")
    if range_bytes and (not filesize or int(filesize) > range_bytes):
        item["range_bytes"] = int(range_bytes)
    return item, None


def discover_openverse(specs: list[dict], fetch_json: FetchJson, retrieved_at: str,
                       log: Callable[[str], None] = lambda _: None) -> list[dict]:
    items: dict[str, dict] = {}
    seen_urls: set[str] = set()
    for spec in specs:
        creator_counts: dict[str, int] = {}
        kept = 0
        for page in range(1, int(spec.get("pages", 1)) + 1):
            if kept >= int(spec.get("limit", 40)):
                break
            params = {"q": spec["q"], "license": "cc0,pdm,by", "page_size": OPENVERSE_PAGE_SIZE, "page": page}
            for key in ("source", "category"):
                if spec.get(key):
                    params[key] = spec[key]
            data = fetch_json(OPENVERSE_URL + "?" + urllib.parse.urlencode(params))
            for result in data.get("results", []):
                if kept >= int(spec.get("limit", 40)):
                    break
                item, _ = openverse_candidate(result, spec, retrieved_at)
                if item is None or item["url"] in seen_urls or item["id"] in items:
                    continue
                if creator_counts.get(item["group"], 0) >= int(spec.get("per_creator_cap", 2)):
                    continue
                creator_counts[item["group"]] = creator_counts.get(item["group"], 0) + 1
                seen_urls.add(item["url"])
                items[item["id"]] = item
                kept += 1
            if page >= int(data.get("page_count", page)):
                break
        log(f"openverse '{spec['name']}': {kept} candidates")
    return list(items.values())


def discover_librivox(config: dict, fetch_json: FetchJson, retrieved_at: str,
                      log: Callable[[str], None] = lambda _: None) -> list[dict]:
    query = ("collection:librivoxaudio AND language:(eng OR English) AND mediatype:audio")
    params = [("q", query), ("fl[]", "identifier"), ("fl[]", "title"), ("fl[]", "creator"),
              ("fl[]", "licenseurl"), ("fl[]", "runtime"), ("rows", str(config.get("search_rows", 300))),
              ("sort[]", "downloads desc"), ("output", "json")]
    docs = fetch_json(ARCHIVE_SEARCH_URL + "?" + urllib.parse.urlencode(params)).get("response", {}).get("docs", [])

    def long_enough(doc: dict) -> bool:
        parts = [int(p) for p in str(doc.get("runtime", "0")).split(":") if p.isdigit()]
        seconds = sum(p * m for p, m in zip(reversed(parts), (1, 60, 3600)))
        return seconds >= int(config.get("min_runtime_minutes", 30)) * 60

    eligible = [d for d in docs
                if any(u in str(d.get("licenseurl", "")) for u in PUBLIC_DOMAIN_URLS)
                and not NON_SOLO_TITLE.search(str(d.get("title", "")))
                and not NON_SOLO_IDENTIFIER.search(str(d.get("identifier", "")))
                and long_enough(d)]
    random.Random(int(config.get("seed", 1))).shuffle(eligible)

    items = []
    range_bytes = int(config.get("range_bytes", 3_000_000))
    for doc in eligible:
        if len(items) >= int(config.get("books", 50)):
            break
        identifier = doc["identifier"]
        files = fetch_json(ARCHIVE_METADATA_URL + identifier).get("files", [])
        mp3s = sorted((f for f in files if f.get("format") == "64Kbps MP3" and int(f.get("size", 0)) >= range_bytes),
                      key=lambda f: f["name"])
        if not mp3s:
            continue
        chosen = mp3s[min(int(config.get("section_index", 3)), len(mp3s) - 1)]
        title = str(doc.get("title", identifier)).strip()
        items.append({
            "id": f"librivox-dlg-{slug(identifier)}"[:64],
            "source": "librivox",
            "url": ARCHIVE_DOWNLOAD_URL + identifier + "/" + urllib.parse.quote(chosen["name"]),
            "role": "dialogue",
            "group": f"librivox:{identifier}",
            "license_spdx": "LicenseRef-PublicDomain",
            "attribution_text": f'LibriVox recording of "{title}" by {doc.get("creator", "unknown")} (public domain), '
                                f"https://archive.org/details/{identifier}",
            "license_evidence_url": f"https://archive.org/details/{identifier}",
            "tags": [],
            "tag_basis": "none",
            "ext": "mp3",
            "title": title,
            "creator": str(doc.get("creator", "")),
            "range_bytes": range_bytes,
            "retrieved_at": retrieved_at,
        })
    log(f"librivox: {len(items)} candidates from {len(eligible)} eligible books")
    return items


def discover(spec: dict, openverse_fetch: FetchJson, archive_fetch: FetchJson, retrieved_at: str,
             only: str | None = None, log: Callable[[str], None] = lambda _: None) -> list[dict]:
    items: list[dict] = []
    if only in (None, "openverse"):
        items += discover_openverse(spec.get("openverse", []), openverse_fetch, retrieved_at, log)
    if only in (None, "librivox") and spec.get("librivox"):
        items += discover_librivox(spec["librivox"], archive_fetch, retrieved_at, log)
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise DiscoverError("duplicate item ids across sources")
    for item in items:
        problem = ingest.validate_record(item, allow_sharealike=False)
        if problem:
            raise DiscoverError(f"{item['id']}: {problem}")
    return sorted(items, key=lambda i: i["id"])


def write_items(path: Path, items: list[dict], spec_name: str, retrieved_at: str) -> None:
    lines = [f"# Candidate items from {spec_name}, discovered {retrieved_at} by discover.py v{DISCOVER_VERSION}.",
             "# Metadata only: no audio has been downloaded. Licenses come from the source APIs; tags marked "
             "tag_basis=search-query are unverified."]
    lines += [json.dumps(i, sort_keys=True) for i in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--api-cache", type=Path, help="directory for cached API responses (recommended)")
    p.add_argument("--only", choices=("openverse", "librivox"))
    args = p.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = args.api_cache
    try:
        items = discover(spec,
                         cached_fetch_json(cache / "openverse" if cache else None, OPENVERSE_PAUSE_S),
                         cached_fetch_json(cache / "archive" if cache else None, ARCHIVE_PAUSE_S),
                         retrieved_at, args.only, log=lambda m: print(m, file=sys.stderr))
    except DiscoverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_items(args.out, items, args.spec.name, retrieved_at)
    print(f"wrote {len(items)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
