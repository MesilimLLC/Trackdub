#!/usr/bin/env python3
"""Freesound original files for the separation evaluation corpus.

Discovers sounds with the Freesound APIv2 text search (API-key auth) and lets `ingest.py` download the
original files, which Freesound only serves to OAuth2 clients. Credentials come from FREESOUND_CLIENT_ID and
FREESOUND_CLIENT_SECRET, read from the process environment or, on Windows, the per-user environment. The
OAuth2 token is stored per user, outside the repository, and is never printed.

Check the Freesound API terms of use for the intended use before running any command that calls the API.

    python freesound.py authorize
    python freesound.py exchange --code <code shown by Freesound after approval>
    python freesound.py status
    python freesound.py discover --spec corpus-sources.freesound-originals.json --out items.freesound.jsonl \\
        --api-cache D:/api-cache --exclude-ids <FSD50K dev.csv> --exclude-ids <FSD50K eval.csv>
    python ingest.py ingest --items items.freesound.jsonl --cache D:/corpus-cache --out items.freesound.manifest.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple

import discover
import ingest

API_ROOT = "https://freesound.org/apiv2/"
SEARCH_URL = API_ROOT + "search/text/"
AUTHORIZE_URL = API_ROOT + "oauth2/authorize/"
TOKEN_URL = API_ROOT + "oauth2/access_token/"
DOWNLOAD_URL = API_ROOT + "sounds/{sound_id}/download/"
DOWNLOAD_PATH = re.compile(r"/apiv2/sounds/\d+/download/")
SEARCH_FIELDS = "id,name,username,license,type,channels,samplerate,duration,filesize,url"
SEARCH_PAGE_SIZE = 50
SEARCH_PAUSE_S = 1.1
CLIENT_ID_VAR = "FREESOUND_CLIENT_ID"
CLIENT_SECRET_VAR = "FREESOUND_CLIENT_SECRET"
# Refresh well before expiry so a long ingest does not run out of token part way.
REFRESH_MARGIN_S = 3600

LICENSE_URLS = {
    "creativecommons.org/publicdomain/zero/1.0": "CC0-1.0",
    "creativecommons.org/licenses/by/4.0": "CC-BY-4.0",
    "creativecommons.org/licenses/by/3.0": "CC-BY-3.0",
}
LICENSE_FILTER_NAMES = {"CC0-1.0": "Creative Commons 0", "CC-BY-4.0": "Attribution", "CC-BY-3.0": "Attribution"}
TYPE_ALIASES = {"aif": "aiff"}
DEFAULT_TYPES = ("wav", "flac", "aiff")
DEFAULT_MIN_SAMPLERATE = 44100
DEFAULT_MAX_CHANNELS = 2
DEFAULT_MAX_FILESIZE = 300 * 1024 * 1024
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": int, "name": str, "username": str, "license": str, "type": str, "channels": int,
    "samplerate": (int, float), "duration": (int, float), "filesize": int, "url": str,
}

Post = Callable[[str, dict], dict]


class FreesoundError(ingest.IngestError):
    pass


class Credentials(NamedTuple):
    client_id: str
    secret: str


class SearchContext(NamedTuple):
    retrieved_at: str
    excluded_ids: frozenset[int]


# ---- Credentials and token storage --------------------------------------------------------------------------

def _windows_user_variable(name: str) -> str | None:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value) or None


def read_credential(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    return _windows_user_variable(name) if sys.platform == "win32" else None


def credentials() -> Credentials:
    client_id, secret = read_credential(CLIENT_ID_VAR), read_credential(CLIENT_SECRET_VAR)
    if not (client_id and secret):
        raise FreesoundError(f"set {CLIENT_ID_VAR} and {CLIENT_SECRET_VAR} to the Freesound API credentials")
    return Credentials(client_id, secret)


def default_token_file() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "trackdub" / "separation-eval" / "freesound-token.json"


def save_token(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(path)


def _is_token_record(record: object) -> bool:
    if not isinstance(record, dict) or not isinstance(record.get("access_token"), str):
        return False
    expires = record.get("expires_at")
    return isinstance(expires, (int, float)) and not isinstance(expires, bool)


def read_token(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FreesoundError(f"no Freesound token at {path}; run `freesound.py authorize`, then `exchange`") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise FreesoundError(f"cannot read the Freesound token at {path}: {exc}") from exc
    if not _is_token_record(record):
        raise FreesoundError(f"the Freesound token at {path} is malformed; run `freesound.py authorize`")
    return record


# ---- OAuth2 -------------------------------------------------------------------------------------------------

def authorize_url(client_id: str, state: str) -> str:
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode({"client_id": client_id, "response_type": "code", "state": state})


def http_post_form(url: str, fields: dict) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode("utf-8"),
                                 headers={"User-Agent": "trackdub-separation-eval/1", "Accept": "application/json"})
    opener = urllib.request.build_opener(discover._HttpsRedirectHandler())
    try:
        with opener.open(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode("utf-8", errors="replace")
        raise FreesoundError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, discover.DiscoverError) as exc:
        raise FreesoundError(f"POST {url} failed: {exc}") from exc


def _token_record(response: dict, now: float) -> dict:
    try:
        access, expires_in = response["access_token"], int(response["expires_in"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FreesoundError(f"unexpected token response shape: {exc!r}") from exc
    if not isinstance(access, str) or not access:
        raise FreesoundError("unexpected token response shape: empty access_token")
    refresh = response.get("refresh_token")
    return {"access_token": access, "refresh_token": refresh if isinstance(refresh, str) else None,
            "expires_at": now + expires_in, "scope": response.get("scope")}


def exchange_code(code: str, creds: Credentials, *, post: Post = http_post_form,
                  now: Callable[[], float] = time.time) -> dict:
    fields = {"client_id": creds.client_id, "client_secret": creds.secret,
              "grant_type": "authorization_code", "code": code}
    return _token_record(post(TOKEN_URL, fields), now())


def refresh_access(record: dict, creds: Credentials, *, post: Post = http_post_form,
                   now: Callable[[], float] = time.time) -> dict:
    if not record.get("refresh_token"):
        raise FreesoundError("the Freesound token has expired and has no refresh token; run `freesound.py authorize`")
    fields = {"client_id": creds.client_id, "client_secret": creds.secret,
              "grant_type": "refresh_token", "refresh_token": record["refresh_token"]}
    return _token_record(post(TOKEN_URL, fields), now())


def load_access_token(path: Path | None = None, *, post: Post = http_post_form,
                      now: Callable[[], float] = time.time,
                      creds: Callable[[], Credentials] = credentials) -> str:
    """Return a usable access token, refreshing and saving it first when it expires within the margin."""
    path = path or default_token_file()
    record = read_token(path)
    if record["expires_at"] - REFRESH_MARGIN_S <= now():
        record = refresh_access(record, creds(), post=post, now=now)
        save_token(path, record)
    return record["access_token"]


# ---- Authorized downloads -----------------------------------------------------------------------------------

def is_download_url(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    on_freesound = (parts.scheme, parts.hostname) == ("https", "freesound.org")
    return on_freesound and DOWNLOAD_PATH.fullmatch(parts.path) is not None


def authorized_fetcher(access_token: str, fetch: Callable[..., None] = ingest.http_fetch) -> ingest.Fetcher:
    """An ingest fetcher that sends the token to Freesound download URLs only."""
    def fetch_one(url: str, dest: Path, range_bytes: int | None = None) -> None:
        fetch(url, dest, range_bytes, bearer_token=access_token if is_download_url(url) else None)
    return fetch_one


def fetcher_for_records(records: Iterable[object], token_file: Path | None = None) -> ingest.Fetcher:
    """Plain downloads, unless some item is a Freesound original, which needs the stored token."""
    if not any(isinstance(r, dict) and is_download_url(str(r.get("url", ""))) for r in records):
        return ingest.http_fetch
    return authorized_fetcher(load_access_token(token_file))


# ---- Search specs -------------------------------------------------------------------------------------------

def license_spdx(url: str) -> str | None:
    parts = urllib.parse.urlsplit(url.strip())
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return LICENSE_URLS.get((host + parts.path).rstrip("/"))


def spec_licenses(spec: dict) -> list[str]:
    licenses = spec.get("licenses", ["CC0-1.0"])
    known = isinstance(licenses, list) and set(licenses) <= set(LICENSE_FILTER_NAMES)
    if not (known and licenses):
        raise FreesoundError(f"spec '{spec.get('name')}': licenses must be a non-empty subset of "
                             f"{sorted(LICENSE_FILTER_NAMES)}")
    return licenses


def spec_types(spec: dict) -> tuple[str, ...]:
    return tuple(TYPE_ALIASES.get(t.lower(), t.lower()) for t in spec.get("types", DEFAULT_TYPES))


def _license_clause(spec: dict) -> str:
    names = sorted({LICENSE_FILTER_NAMES[l] for l in spec_licenses(spec)})
    if len(names) == 1:
        return f'license:"{names[0]}"'
    return "license:(" + " OR ".join(f'"{n}"' for n in names) + ")"


def search_filter(spec: dict) -> str:
    return " ".join([
        _license_clause(spec),
        "type:(" + " OR ".join(spec_types(spec)) + ")",
        f"duration:[{spec.get('min_duration_s', 0)} TO {spec.get('max_duration_s', 3600)}]",
        f"channels:[1 TO {int(spec.get('max_channels', DEFAULT_MAX_CHANNELS))}]",
        f"samplerate:[{int(spec.get('min_samplerate', DEFAULT_MIN_SAMPLERATE))} TO *]",
    ])


# ---- Results ------------------------------------------------------------------------------------------------

def _check_shape(result: object) -> dict:
    if not isinstance(result, dict):
        raise FreesoundError("unexpected search result shape: a result is not an object")
    for key, kind in REQUIRED_FIELDS.items():
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, kind):
            raise FreesoundError(f"unexpected search result shape: field '{key}' is {type(value).__name__}")
    return result


def _excluded_reason(r: dict, spec: dict, excluded_ids: Iterable[int]) -> str | None:
    return "on the exclusion list" if r["id"] in excluded_ids else None


def _license_reason(r: dict, spec: dict, excluded_ids: Iterable[int]) -> str | None:
    spdx = license_spdx(r["license"])
    if spdx is None or spdx not in spec_licenses(spec):
        return f"license {r['license']} not accepted"
    return None


def _format_reason(r: dict, spec: dict, excluded_ids: Iterable[int]) -> str | None:
    if TYPE_ALIASES.get(r["type"].lower(), r["type"].lower()) not in spec_types(spec):
        return f"type {r['type']} not accepted"
    if not 1 <= r["channels"] <= int(spec.get("max_channels", DEFAULT_MAX_CHANNELS)):
        return f"{r['channels']} channels not accepted"
    if r["samplerate"] < int(spec.get("min_samplerate", DEFAULT_MIN_SAMPLERATE)):
        return f"sample rate {r['samplerate']} too low"
    return None


def _size_reason(r: dict, spec: dict, excluded_ids: Iterable[int]) -> str | None:
    if not spec.get("min_duration_s", 0) <= r["duration"] <= spec.get("max_duration_s", 3600):
        return f"duration {r['duration']:.1f}s outside bounds"
    if not 0 < r["filesize"] <= int(spec.get("max_filesize_bytes", DEFAULT_MAX_FILESIZE)):
        return f"file size {r['filesize']} outside bounds"
    return None


def _source_reason(r: dict, spec: dict, excluded_ids: Iterable[int]) -> str | None:
    if not r["url"].startswith("https://freesound.org/"):
        return "unexpected landing url"
    return None if r["username"].strip() else "no username"


REASON_CHECKS = (_excluded_reason, _license_reason, _format_reason, _size_reason, _source_reason)


def _rejection_reason(r: dict, spec: dict, excluded_ids: Iterable[int]) -> str | None:
    for check in REASON_CHECKS:
        reason = check(r, spec, excluded_ids)
        if reason:
            return reason
    return None


def _build_item(r: dict, spec: dict, retrieved_at: str) -> dict:
    sound_id, username, landing = r["id"], r["username"].strip(), r["url"]
    spdx = license_spdx(r["license"])
    title = r["name"].strip() or "untitled"
    item = {
        "id": f"fsorig-{spec['role'][:3]}-{sound_id}",
        "source": "freesound",
        "url": DOWNLOAD_URL.format(sound_id=sound_id),
        "role": spec["role"],
        # Same group format as the v1 Openverse items, so a creator keeps its split across corpus versions.
        "group": f"freesound:{discover.creator_group_key(username)}",
        "license_spdx": spdx,
        "attribution_text": f'"{title}" by {username}, {spdx}, {landing}',
        "license_evidence_url": landing,
        "tags": sorted({str(t).lower() for t in spec.get("tags", [])}),
        "tag_basis": "search-query",
        "ext": TYPE_ALIASES.get(r["type"].lower(), r["type"].lower()),
        "title": title,
        "creator": username,
        "query": spec["q"],
        "retrieved_at": retrieved_at,
        "freesound_id": sound_id,
    }
    problem = ingest.validate_record(item, allow_sharealike=False)
    if problem:
        raise FreesoundError(f"{item['id']}: {problem}")
    return item


def freesound_candidate(result: object, spec: dict, retrieved_at: str,
                        excluded_ids: Iterable[int]) -> tuple[dict | None, str | None]:
    """Return (item, None) for an acceptable original, or (None, reason). Raises on an unexpected shape."""
    r = _check_shape(result)
    reason = _rejection_reason(r, spec, excluded_ids)
    if reason:
        return None, reason
    return _build_item(r, spec, retrieved_at), None


# ---- Discovery ----------------------------------------------------------------------------------------------

def _fetch_results(fetch_json: discover.FetchJson, spec: dict, page: int) -> tuple[list, bool]:
    query = {"query": spec["q"], "filter": search_filter(spec), "fields": SEARCH_FIELDS,
             "page_size": SEARCH_PAGE_SIZE, "page": page}
    data = fetch_json(SEARCH_URL + "?" + urllib.parse.urlencode(query))
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise FreesoundError("unexpected search response shape: no 'results' list")
    return results, bool(data.get("next"))


def _spec_candidates(spec: dict, fetch_json: discover.FetchJson, context: SearchContext) -> Iterator[dict]:
    """Acceptable results of one search spec, fetching the next page only when the caller asks for more."""
    for page in range(1, int(spec.get("pages", 1)) + 1):
        results, has_next = _fetch_results(fetch_json, spec, page)
        for result in results:
            item, _ = freesound_candidate(result, spec, context.retrieved_at, context.excluded_ids)
            if item is not None:
                yield item
        if not has_next:
            return


def _select(candidates: Iterable[dict], spec: dict, seen_sounds: set[int]) -> list[dict]:
    """Apply the per-creator cap, the overall limit and cross-spec de-duplication."""
    limit, cap = int(spec.get("limit", 40)), int(spec.get("per_creator_cap", 2))
    per_creator: collections.Counter[str] = collections.Counter()
    kept: list[dict] = []
    for item in candidates if limit > 0 else ():
        if item["freesound_id"] in seen_sounds or per_creator[item["group"]] >= cap:
            continue
        per_creator[item["group"]] += 1
        seen_sounds.add(item["freesound_id"])
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def discover_freesound(specs: list[dict], fetch_json: discover.FetchJson, context: SearchContext,
                       log: Callable[[str], None] = lambda _: None) -> list[dict]:
    items: list[dict] = []
    seen_sounds: set[int] = set()
    for spec in specs:
        if spec.get("role") not in ingest.ROLES:
            raise FreesoundError(f"spec '{spec.get('name')}': role must be one of {ingest.ROLES}")
        kept = _select(_spec_candidates(spec, fetch_json, context), spec, seen_sounds)
        log(f"freesound '{spec.get('name', spec['q'])}': {len(kept)} candidates")
        items += kept
    return sorted(items, key=lambda i: i["id"])


def read_excluded_ids(paths: Iterable[Path]) -> set[int]:
    """Freesound ids from plain lists or CSVs whose first column is the id (FSD50K ground-truth `fname`)."""
    ids: set[int] = set()
    for path in paths:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        fields = (line.split(",", 1)[0].strip().strip('"') for line in lines)
        found = {int(f) for f in fields if f.isdigit()}
        if not found:
            raise FreesoundError(f"{path}: no Freesound ids found")
        ids |= found
    return ids


# ---- CLI ----------------------------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("authorize", help="print the URL that grants this tool access to original files")
    p_exchange = sub.add_parser("exchange", help="trade the code Freesound showed for a stored token")
    p_exchange.add_argument("--code", required=True)
    p_exchange.add_argument("--token-file", type=Path)
    p_status = sub.add_parser("status", help="show whether a stored token exists and when it expires")
    p_status.add_argument("--token-file", type=Path)
    p_discover = sub.add_parser("discover", help="search Freesound originals and write candidate items")
    p_discover.add_argument("--spec", type=Path, required=True)
    p_discover.add_argument("--out", type=Path, required=True)
    p_discover.add_argument("--api-cache", type=Path, help="directory for cached API responses (recommended)")
    exclusion = p_discover.add_mutually_exclusive_group(required=True)
    exclusion.add_argument("--exclude-ids", type=Path, action="append",
                           help="file of Freesound ids to skip, such as the FSD50K ground-truth CSVs (repeatable)")
    exclusion.add_argument("--no-exclusion-list", action="store_true",
                           help="skip no ids; sounds used to train DnR-derived models may then enter the corpus")
    return parser


def _cmd_authorize(args: argparse.Namespace) -> None:
    print(authorize_url(credentials().client_id, secrets.token_urlsafe(16)))
    print("Open the URL, sign in, approve access, then run "
          "`freesound.py exchange --code <code>` within 10 minutes.", file=sys.stderr)


def _cmd_exchange(args: argparse.Namespace) -> None:
    path = args.token_file or default_token_file()
    record = exchange_code(args.code, credentials())
    save_token(path, record)
    expires = datetime.fromtimestamp(record["expires_at"], timezone.utc)
    print(f"saved token to {path}; expires {expires:%Y-%m-%d %H:%M} UTC")


def _cmd_status(args: argparse.Namespace) -> None:
    path = args.token_file or default_token_file()
    record = read_token(path)
    left = record["expires_at"] - time.time()
    state = "expired" if left <= 0 else f"valid for {left / 3600:.1f} h"
    print(f"{path}: {state}; refresh token {'present' if record.get('refresh_token') else 'absent'}")


def _cmd_discover(args: argparse.Namespace) -> None:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    specs = spec.get("freesound")
    if not isinstance(specs, list) or not specs:
        raise FreesoundError(f"{args.spec}: no 'freesound' search list")
    excluded = read_excluded_ids(args.exclude_ids or [])
    cache = args.api_cache / "freesound" if args.api_cache else None
    fetch_json = discover.cached_fetch_json(cache, SEARCH_PAUSE_S, auth_header=f"Token {credentials().secret}")
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    items = discover_freesound(specs, fetch_json, SearchContext(retrieved_at, frozenset(excluded)),
                               log=lambda m: print(m, file=sys.stderr))
    discover.write_items(args.out, items, args.spec.name, retrieved_at)
    print(f"wrote {len(items)} candidates to {args.out} ({len(excluded)} ids on the exclusion list)")


COMMANDS = {"authorize": _cmd_authorize, "exchange": _cmd_exchange, "status": _cmd_status,
            "discover": _cmd_discover}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (ingest.IngestError, discover.DiscoverError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
