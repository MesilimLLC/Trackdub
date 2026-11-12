#!/usr/bin/env python3
"""Ingest source items for the separation evaluation corpus.

Reads a JSONL list of candidate items, rejects anything whose license is not
clearly commercial-permissive, downloads and hashes each file into a local
cache (never committed), probes audio properties, assigns a deterministic
dev/test split by source group, and writes an item manifest.

Design: docs/audits/separation-eval-corpus-and-rubric.md (section 3).

    python ingest.py ingest --items items.jsonl --cache ./cache --out items.manifest.json
    python ingest.py verify --manifest items.manifest.json --cache ./cache
"""

from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

SCHEMA_VERSION = 1
SPLIT_SALT = "trackdub-sep-eval-v1"
DEV_PERCENT = 30
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024

ROLES = ("dialogue", "music", "sfx", "ambience", "rir")
ITEM_PASSTHROUGH_KEYS = ("range_bytes", "ext", "tag_basis", "title", "creator", "query", "retrieved_at", "freesound_id",
                         "language", "gender_class", "provenance", "local_only", "band_limit_hz")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
EXT_PATTERN = re.compile(r"^[a-z0-9]{1,5}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# SPDX id (or LicenseRef) -> share-alike flag. Anything not listed is rejected.
ALLOWED_LICENSES: dict[str, bool] = {
    "CC0-1.0": False,
    "LicenseRef-PublicDomain": False,
    "CC-BY-3.0": False,
    "CC-BY-4.0": False,
    "Apache-2.0": False,
    "MIT": False,
    "CC-BY-SA-3.0": True,
    "CC-BY-SA-4.0": True,
}
NONCOMMERCIAL_OR_NODERIVS = re.compile(r"(^|[-_])(NC|ND)([-_]|$)", re.IGNORECASE)

Fetcher = Callable[[str, Path, int | None], None]


class IngestError(Exception):
    pass


@dataclass(frozen=True)
class Rejection:
    item_id: str
    reason: str


def check_license(spdx: str, allow_sharealike: bool) -> str | None:
    """Return a rejection reason, or None when the license is acceptable."""
    if NONCOMMERCIAL_OR_NODERIVS.search(spdx):
        return f"license '{spdx}' is noncommercial or no-derivatives"
    if spdx not in ALLOWED_LICENSES:
        return f"license '{spdx}' is not in the allow-list"
    if ALLOWED_LICENSES[spdx] and not allow_sharealike:
        return f"license '{spdx}' is share-alike; pass --allow-sharealike to accept"
    return None


def validate_record(rec: dict, allow_sharealike: bool) -> str | None:
    item_id = rec.get("id", "")
    if not isinstance(item_id, str) or not ID_PATTERN.match(item_id):
        return f"id '{item_id}' must match {ID_PATTERN.pattern}"
    for key in ("source", "url", "license_spdx", "license_evidence_url", "attribution_text", "group"):
        value = rec.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"missing or empty '{key}'"
    if rec.get("role") not in ROLES:
        return f"role must be one of {ROLES}"
    try:
        parsed_url = urllib.parse.urlsplit(rec["url"])
    except ValueError:
        return "url must be a valid https URL"
    if parsed_url.scheme.lower() != "https" or not parsed_url.hostname:
        return "url must be https"
    try:
        evidence_url = urllib.parse.urlsplit(rec["license_evidence_url"])
    except ValueError:
        return "license_evidence_url must be a valid https URL"
    if evidence_url.scheme.lower() != "https" or not evidence_url.hostname:
        return "license_evidence_url must be https"
    ext = rec.get("ext")
    if ext is not None and (not isinstance(ext, str) or not EXT_PATTERN.fullmatch(ext)):
        return "ext, when given, must be 1 to 5 lowercase ASCII letters or digits"
    expected = rec.get("sha256")
    if expected is not None and not SHA256_PATTERN.match(str(expected)):
        return "sha256, when given, must be 64 lowercase hex characters"
    range_bytes = rec.get("range_bytes")
    if range_bytes is not None and (not isinstance(range_bytes, int) or isinstance(range_bytes, bool)
                                    or not 1024 <= range_bytes <= MAX_DOWNLOAD_BYTES):
        return f"range_bytes, when given, must be an integer between 1024 and {MAX_DOWNLOAD_BYTES}"
    tags = rec.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        return "tags, when given, must be a list of strings"
    return check_license(rec["license_spdx"], allow_sharealike)


def assign_split(group: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{group}".encode("utf-8")).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 100 < DEV_PERCENT else "test"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_size(resp, limit: int) -> int | None:
    """Bytes the server says it will send, capped at `limit`, or None when it does not say."""
    expected = None
    content_range = resp.headers.get("Content-Range")
    if content_range:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip(), re.IGNORECASE)
        if match and int(match.group(1)) == 0 and match.group(3) != "*":
            expected = min(limit, int(match.group(3)))
        elif resp.status == 206:
            raise IngestError("download has an invalid Content-Range header")
    if expected is None and resp.headers.get("Content-Length") is not None:
        expected = min(limit, int(resp.headers["Content-Length"]))
    return expected


def _copy_limited(resp, out, limit: int, range_bytes: int | None) -> int:
    """Write at most `limit` bytes and return the count. Going over the limit is an error unless
    `range_bytes` asked for a truncated download."""
    total = 0
    while chunk := resp.read(min(1024 * 1024, limit - total + 1)):
        total += len(chunk)
        if total > limit:
            if range_bytes is None:
                raise IngestError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
            chunk = chunk[: len(chunk) - (total - limit)]
            total = limit
        out.write(chunk)
        if total == limit:
            break
    return total


def http_fetch(url: str, dest: Path, range_bytes: int | None = None, *, bearer_token: str | None = None) -> None:
    """Download `url` to `dest`. With `range_bytes`, keep only the first that many bytes, which is
    deterministic and still decodable for MP3 and similar streams; servers that ignore Range are cut off.
    `bearer_token` is sent only to `url` itself, never to a redirect target."""
    try:
        parsed_url = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise IngestError(f"invalid download URL: {exc}") from exc
    if parsed_url.scheme.lower() != "https" and not _is_loopback_http(url):
        raise IngestError("url must be https")
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "trackdub-separation-eval/1"}
    limit = MAX_DOWNLOAD_BYTES if range_bytes is None else range_bytes
    if range_bytes is not None:
        headers["Range"] = f"bytes=0-{range_bytes - 1}"
    req = urllib.request.Request(url, headers=headers)
    if bearer_token:
        # urllib does not copy unredirected headers onto a redirected request.
        req.add_unredirected_header("Authorization", f"Bearer {bearer_token}")
    opener = urllib.request.build_opener(_HttpsRedirectHandler())
    try:
        with opener.open(req, timeout=60) as resp:
            if urllib.parse.urlsplit(resp.geturl()).scheme.lower() != "https" and not _is_loopback_http(url):
                raise IngestError("HTTPS download redirected to a non-HTTPS URL")

            expected = _expected_size(resp, limit)
            with tmp.open("wb") as out:
                total = _copy_limited(resp, out, limit, range_bytes)
            if expected is not None and total < expected:
                raise IngestError(f"incomplete download: received {total} of {expected} bytes")
        tmp.replace(dest)
    except (OSError, ValueError, http.client.HTTPException, urllib.error.URLError, IngestError) as exc:
        tmp.unlink(missing_ok=True)
        if isinstance(exc, IngestError):
            raise
        raise IngestError(f"download failed: {exc}") from exc


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        loopback_fixture = req is not None and _is_loopback_http(req.full_url) and _is_loopback_http(newurl)
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https" and not loopback_fixture:
            raise IngestError("HTTPS download redirected to a non-HTTPS URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _is_loopback_http(url: str) -> bool:
    """Allow plain HTTP only for local fixture servers used by unit tests."""
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower() == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}


def probe_audio(path: Path) -> dict:
    """Duration, sample rate, channels and codec via ffprobe, falling back to wave."""
    if shutil.which("ffprobe"):
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            try:
                info = json.loads(proc.stdout)
                streams = info["streams"]
                s = streams[0]
                duration = float(info["format"]["duration"])
                sample_rate, channels = int(s["sample_rate"]), int(s["channels"])
                if not math.isfinite(duration) or duration <= 0 or sample_rate <= 0 or channels <= 0:
                    raise ValueError("duration, sample_rate and channels must be positive")
                return {"duration_s": round(duration, 3), "sample_rate": sample_rate,
                        "channels": channels, "codec": s.get("codec_name", "unknown")}
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
                raise IngestError(f"invalid ffprobe output for {path.name}: {exc}") from exc
    try:
        with wave.open(str(path), "rb") as w:
            return {
                "duration_s": round(w.getnframes() / w.getframerate(), 3),
                "sample_rate": w.getframerate(),
                "channels": w.getnchannels(),
                "codec": "pcm",
            }
    except (wave.Error, EOFError) as exc:
        raise IngestError(f"cannot probe audio for {path.name}: {exc}") from exc


def cache_path(cache: Path, rec: dict) -> Path:
    record_ext = rec.get("ext")
    if record_ext is not None and (not isinstance(record_ext, str) or not EXT_PATTERN.fullmatch(record_ext)):
        raise IngestError("ext, when given, must be 1 to 5 lowercase ASCII letters or digits")
    ext = f".{record_ext}" if record_ext else (Path(rec["url"].split("?", 1)[0]).suffix.lower() or ".bin")
    root = cache.resolve()
    destination = (root / rec["role"] / f"{rec['id']}{ext}").resolve()
    if not destination.is_relative_to(root):
        raise IngestError("cache destination escapes the cache directory")
    return destination


def _local_problem(rec: dict, cache: Path) -> str | None:
    """Why a local-only item cannot be used from the cache, or None when its file is there and intact."""
    try:
        path = cache_path(cache, rec)
    except (IngestError, KeyError, TypeError, AttributeError):
        return "has no usable cache location"
    if not path.is_file():
        return "is missing from the cache"
    expected = rec.get("sha256")
    if expected is not None and sha256_file(path) != expected:
        return "differs from its recorded sha256"
    return None


def split_unusable_local(records: Iterable[dict], cache: Path) -> tuple[list[dict], list[Rejection]]:
    """Set aside local-only items (built on this machine from archives) that are missing or corrupted in the
    cache. Their `url` is only provenance, so there is nothing to download."""
    kept: list[dict] = []
    rejected: list[Rejection] = []
    for rec in records:
        problem = _local_problem(rec, cache) if isinstance(rec, dict) and rec.get("local_only") is True else None
        if problem:
            rejected.append(Rejection(str(rec.get("id", "<missing>")),
                                      f"local-only item {problem}; rebuild it with build_local_items.py"))
        else:
            kept.append(rec)
    return kept, rejected


def _cache_matches(rec: dict, dest: Path) -> bool:
    expected = rec.get("sha256")
    if not dest.is_file():
        return False
    return expected is None or sha256_file(dest) == expected


def split_pending(valid: list[tuple[dict, Path]]) -> tuple[list[tuple[dict, Path]], dict[str, Exception]]:
    """Items that need downloading, and local-only items whose file changed after the first check. The latter
    are never downloaded: their `url` is only provenance."""
    pending: list[tuple[dict, Path]] = []
    changed_local: dict[str, Exception] = {}
    for rec, dest in valid:
        if _cache_matches(rec, dest):
            continue
        if rec.get("local_only") is True:
            changed_local[rec["id"]] = IngestError("local-only file changed or went missing; rebuild it with build_local_items.py")
        else:
            pending.append((rec, dest))
    return pending, changed_local


def fetch_all(
    pending: list[tuple[dict, Path]],
    fetch: Fetcher,
    workers: int,
    per_host: int,
    progress: Callable[[int, int, str], None] | None,
) -> dict[str, Exception]:
    """Download every pending item, at most `workers` at once and `per_host` per host.
    Returns the failures by item id; a failed download never stops the others."""
    if per_host < 1:
        raise IngestError("per_host must be a positive integer")
    failures: dict[str, Exception] = {}
    gates: dict[str, threading.Semaphore] = {}
    lock = threading.Lock()
    done = [0]

    def gate_for(url: str) -> threading.Semaphore:
        host = urllib.parse.urlsplit(url).netloc
        with lock:
            return gates.setdefault(host, threading.Semaphore(per_host))

    def work(job: tuple[dict, Path]) -> None:
        rec, dest = job
        try:
            with gate_for(rec["url"]):
                fetch(rec["url"], dest, rec.get("range_bytes"))
        except Exception as exc:  # network errors are per-item, not fatal
            with lock:
                failures[rec["id"]] = exc
        finally:
            with lock:
                done[0] += 1
                if progress:
                    progress(done[0], len(pending), rec["id"])

    if workers <= 1:
        for job in pending:
            work(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, pending))
    return failures


def ingest_items(
    records: Iterable[dict],
    cache: Path,
    *,
    allow_sharealike: bool = False,
    fetch: Fetcher = http_fetch,
    probe: Callable[[Path], dict] = probe_audio,
    workers: int = 1,
    per_host: int = 2,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict], list[Rejection]]:
    if per_host < 1:
        raise IngestError("per_host must be a positive integer")
    accepted: list[dict] = []
    records, rejected = split_unusable_local(records, cache)
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}

    valid: list[tuple[dict, Path]] = []
    for rec in records:
        if not isinstance(rec, dict):
            rejected.append(Rejection("<invalid>", "record must be a JSON object"))
            continue
        item_id = str(rec.get("id", "<missing>"))
        reason = validate_record(rec, allow_sharealike)
        if reason is None and item_id in seen_ids:
            reason = "duplicate id"
        if reason:
            rejected.append(Rejection(item_id, reason))
            continue
        seen_ids.add(item_id)
        dest = cache_path(cache, rec)
        dest.parent.mkdir(parents=True, exist_ok=True)
        valid.append((rec, dest))

    pending, changed_local = split_pending(valid)
    failures = fetch_all(pending, fetch, workers, per_host, progress)
    failures.update(changed_local)

    for rec, dest in valid:
        item_id = rec["id"]
        expected = rec.get("sha256")
        if item_id in failures:
            rejected.append(Rejection(item_id, f"download failed: {failures[item_id]}"))
            continue

        digest = sha256_file(dest)
        if expected is not None and digest != expected:
            dest.unlink(missing_ok=True)
            rejected.append(Rejection(item_id, f"sha256 mismatch: expected {expected}, got {digest}"))
            continue
        if digest in seen_hashes:
            rejected.append(Rejection(item_id, f"same bytes as '{seen_hashes[digest]}'"))
            continue
        seen_hashes[digest] = item_id

        try:
            audio = probe(dest)
        except (IngestError, KeyError, TypeError, ValueError, OverflowError) as exc:
            rejected.append(Rejection(item_id, str(exc)))
            continue

        accepted.append({
            "id": item_id,
            "source": rec["source"],
            "url": rec["url"],
            "role": rec["role"],
            "group": rec["group"],
            "license_spdx": rec["license_spdx"],
            "share_alike": ALLOWED_LICENSES[rec["license_spdx"]],
            "license_evidence_url": rec.get("license_evidence_url", ""),
            "attribution_text": rec["attribution_text"],
            "tags": sorted({t.strip().lower() for t in rec.get("tags", [])}),
            **{k: rec[k] for k in ITEM_PASSTHROUGH_KEYS if k in rec},
            "sha256": digest,
            "size_bytes": dest.stat().st_size,
            **audio,
            "split": assign_split(rec["group"]),
            "excluded_from_training": True,
        })

    accepted.sort(key=lambda r: r["id"])
    return accepted, rejected


def items_hash(items: list[dict]) -> str:
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(items: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "split_salt": SPLIT_SALT,
        "dev_percent": DEV_PERCENT,
        "items_sha256": items_hash(items),
        "items": items,
    }


def verify_manifest(manifest: dict, cache: Path) -> list[str]:
    problems: list[str] = []
    items = manifest.get("items", [])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"unsupported schema_version {manifest.get('schema_version')}")
    if manifest.get("items_sha256") != items_hash(items):
        problems.append("items_sha256 does not match items (manifest was edited)")
    groups: dict[str, set[str]] = {}
    for item in items:
        groups.setdefault(item["group"], set()).add(item["split"])
        if item["split"] != assign_split(item["group"]):
            problems.append(f"{item['id']}: split does not match deterministic assignment")
        reason = check_license(item["license_spdx"], allow_sharealike=True)
        if reason:
            problems.append(f"{item['id']}: {reason}")
        path = cache_path(cache, item)
        if not path.is_file():
            problems.append(f"{item['id']}: missing from cache ({path})")
        elif sha256_file(path) != item["sha256"]:
            problems.append(f"{item['id']}: cached file hash differs from manifest")
    for group, splits in groups.items():
        if len(splits) > 1:
            problems.append(f"group '{group}' spans both dev and test")
    return problems


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path}:{n}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise IngestError(f"{path}:{n}: expected a JSON object")
        records.append(record)
    return records


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _select_fetcher(records: list[dict], args: argparse.Namespace) -> Fetcher:
    """Plain downloads, or authorized and rate-limited ones when any item is a Freesound original."""
    import freesound  # freesound imports this module
    budget = freesound.DownloadBudget(args.freesound_interval, args.freesound_max_downloads)
    return freesound.fetcher_for_records(records, args.freesound_token_file, budget)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="validate, download, hash and write a manifest")
    p_ingest.add_argument("--items", type=Path, required=True)
    p_ingest.add_argument("--cache", type=Path, required=True)
    p_ingest.add_argument("--out", type=Path, required=True)
    p_ingest.add_argument("--allow-sharealike", action="store_true")
    p_ingest.add_argument("--workers", type=positive_int, default=4, help="concurrent downloads (default 4)")
    p_ingest.add_argument("--per-host", type=positive_int, default=2, help="concurrent downloads per host (default 2)")
    p_ingest.add_argument("--skip-rejected", action="store_true",
                          help="write the manifest from accepted items even if some were rejected")
    p_ingest.add_argument("--freesound-token-file", type=Path,
                          help="OAuth2 token file from `freesound.py exchange` (default: the per-user location)")
    p_ingest.add_argument("--freesound-interval", type=float, default=2.2,
                          help="seconds between Freesound original downloads; Freesound allows 30 a minute (default 2.2)")
    p_ingest.add_argument("--freesound-max-downloads", type=positive_int, default=450,
                          help="most Freesound original downloads to make in this run; the daily limit is 500 (default 450)")

    p_verify = sub.add_parser("verify", help="re-check a manifest against the cache")
    p_verify.add_argument("--manifest", type=Path, required=True)
    p_verify.add_argument("--cache", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "verify":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        problems = verify_manifest(manifest, args.cache)
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        print(f"{len(manifest.get('items', []))} items, {len(problems)} problems")
        return 1 if problems else 0

    try:
        records = read_jsonl(args.items)
        fetch = _select_fetcher(records, args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    accepted, rejected = ingest_items(
        records, args.cache, allow_sharealike=args.allow_sharealike, fetch=fetch, workers=args.workers,
        per_host=args.per_host, progress=lambda done, total, item_id: print(f"[{done}/{total}] {item_id}", file=sys.stderr, flush=True))
    for r in rejected:
        print(f"REJECTED {r.item_id}: {r.reason}", file=sys.stderr)
    if rejected and not args.skip_rejected:
        print(f"{len(rejected)} rejected; manifest not written (use --skip-rejected to override)", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(build_manifest(accepted), indent=2) + "\n", encoding="utf-8")
    by_split = {s: sum(1 for i in accepted if i["split"] == s) for s in ("dev", "test")}
    print(f"wrote {args.out}: {len(accepted)} accepted ({by_split['dev']} dev, {by_split['test']} test), {len(rejected)} rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
