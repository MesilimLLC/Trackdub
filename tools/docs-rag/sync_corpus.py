"""Stage first-party and vendor docs, then upload the tree to R2 for AI Search."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin


TOOL_ROOT = Path(__file__).resolve().parent
TRACKDUB_ROOT = TOOL_ROOT.parents[1]
STAGING = TOOL_ROOT / ".staging"
MANIFEST_PATH = TOOL_ROOT / "corpus.v1.json"
RUNTIME_MANIFEST = TRACKDUB_ROOT / "runtime" / "trt-rtx-ep.manifest.json"
SYNC_MANIFEST_KEY = "first-party/_sync.json"
USER_AGENT = "TrackdubDocsRag/0.1 (corpus sync)"


# Page chrome that repeats on every page of a doc site. Left in, it dominates chunk
# embeddings: a Sphinx sidebar table of contents is longer than many pages' actual content.
_CHROME_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form", "button"}
_CHROME_ROLES = {"navigation", "banner", "contentinfo", "search", "complementary"}
_CHROME_CLASS_WORDS = ("sidebar", "toctree", "breadcrumb", "navbar", "headerlink", "prev-next", "skip-link", "bd-toc")
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_BLOCK_TAGS = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "pre", "table", "dt", "dd", "blockquote"}


def _is_chrome(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    if tag in _CHROME_TAGS:
        return True
    values = dict(attrs)
    if (values.get("role") or "").lower() in _CHROME_ROLES:
        return True
    classes = (values.get("class") or "").lower()
    return any(word in classes for word in _CHROME_CLASS_WORDS)


def _is_content_root(tag: str, attrs: list[tuple[str, str | None]], root: str) -> bool:
    if root == "article":
        return tag == "article"
    return tag == "main" or (dict(attrs).get("role") or "").lower() == "main"


class _RootProbe(HTMLParser):
    """Finds which main-content container the page uses, if any."""

    def __init__(self) -> None:
        super().__init__()
        self.has_article = False
        self.has_main = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self.has_article = True
        elif tag == "main" or (dict(attrs).get("role") or "").lower() == "main":
            self.has_main = True


class _TextExtractor(HTMLParser):
    """Extracts readable text, keeping only the page's main content region when it has one.

    Root preference: the first <article> (GitHub README body, Sphinx page body), else <main> /
    role="main" (Just-the-Docs), else the whole page. Chrome inside the root (nav, header,
    footer, sidebars, Sphinx "#" header links) is always dropped. Non-chrome links inside the
    content are preserved as Markdown ``[text](url)`` so a reader can follow them (fragment-only
    anchors are left as plain text). Labels and destinations are escaped and destinations are
    wrapped in angle brackets so they survive ingestion verbatim. With ``emit_links=False``
    link text is emitted as plain text instead (used to size candidate roots without letting
    rendered URLs inflate the comparison).
    """

    def __init__(self, root: str | None = None, base_url: str | None = None, emit_links: bool = True) -> None:
        super().__init__()
        self._root = root
        self._base_url = base_url
        self._emit_links = emit_links
        self._root_tag: str | None = None  # actual tag of the matched root (role="main" can sit on a div)
        self._root_depth = 0  # open elements named _root_tag, counting the root itself
        self._root_done = False
        self._skip_stack: list[str] = []
        self._parts: list[str] = []
        self._link_stack: list[tuple[str, list[str]]] = []  # (href, inner text parts) for open anchors

    def _capturing(self) -> bool:
        if self._skip_stack:
            return False
        return self._root is None or self._root_depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            if tag == "br" and self._capturing():
                if self._link_stack:
                    self._link_stack[-1][1].append(" ")
                else:
                    self._parts.append("\n")
            return
        if self._skip_stack:
            self._skip_stack.append(tag)
            return
        if self._root_depth:
            if _is_chrome(tag, attrs):
                self._skip_stack.append(tag)
            elif tag == self._root_tag:
                self._root_depth += 1
        elif self._root is not None and not self._root_done and _is_content_root(tag, attrs, self._root):
            self._root_tag = tag
            self._root_depth = 1
        elif self._root is None and _is_chrome(tag, attrs):
            self._skip_stack.append(tag)
        if tag == "a" and self._capturing():
            href = (dict(attrs).get("href") or "").strip()
            # Fragment-only anchors describe in-page navigation, not other documents; keep
            # their label as plain text. Resolve relative targets against the source page.
            if href and not href.startswith("#"):
                if self._base_url:
                    href = urljoin(self._base_url, href)
                self._link_stack.append((href, []))

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self._skip_stack:
            # Pop to the matching open tag so unclosed children cannot leave us skipping forever.
            if tag in self._skip_stack:
                while self._skip_stack and self._skip_stack.pop() != tag:
                    pass
            return
        if tag == "a" and self._link_stack:
            href, parts = self._link_stack.pop()
            inner = " ".join("".join(parts).split())
            if inner:
                if self._emit_links:
                    label = inner.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
                    destination = href.replace("\\", "\\\\").replace("<", "\\<").replace(">", "\\>")
                    self._parts.append(f"[{label}](<{destination}>) ")
                else:
                    self._parts.append(inner + " ")
            return
        if self._root_depth and tag == self._root_tag:
            self._root_depth -= 1
            if self._root_depth == 0:
                # Only the first root: a later <article> is usually a related-content card.
                self._root_done = True
                return
        if self._capturing() and tag in _BLOCK_TAGS:
            if self._link_stack:
                self._link_stack[-1][1].append("\n")
            else:
                self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._capturing():
            return
        if self._link_stack:
            # Keep raw data (including whitespace) so handle_endtag can normalize the complete
            # link label without fusing boundaries between nested text nodes.
            self._link_stack[-1][1].append(data)
        else:
            text = data.strip()
            if text:
                self._parts.append(text + " ")

    def text(self) -> str:
        # html.parser does not synthesize a missing </a>, and a closing </a> that arrives while a
        # chrome element opened inside the anchor is still open is swallowed by the skip branch;
        # both orphan the _link_stack entry and would silently drop everything after it. Flush any
        # stragglers as plain text instead of losing the content.
        while self._link_stack:
            _, parts = self._link_stack.pop(0)
            inner = " ".join("".join(parts).split())
            if inner:
                self._parts.append(inner + " ")
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def _extract(html: str, root: str | None, base_url: str | None = None, emit_links: bool = True) -> str:
    extractor = _TextExtractor(root, base_url, emit_links)
    extractor.feed(html)
    return extractor.text()


# A first <article> this much smaller than <main> is a card inside the page (Hugging Face puts
# dataset/paper cards in <article> and the model card in <main>), not the page body.
_ARTICLE_MIN_SHARE_OF_MAIN = 0.25


def html_to_text(html: str, base_url: str | None = None) -> str:
    probe = _RootProbe()
    probe.feed(html)
    # Choose the content root by visible-text size, not rendered size: markdown destinations would
    # inflate a small <article> card past the share threshold and discard the real page body. Render
    # links only for the selected root.
    root: str | None = None
    if probe.has_article:
        if probe.has_main and len(_extract(html, "article", emit_links=False)) < len(_extract(html, "main", emit_links=False)) * _ARTICLE_MIN_SHARE_OF_MAIN:
            root = "main"
        else:
            root = "article"
    elif probe.has_main:
        root = "main"
    text = _extract(html, root, base_url) if root is not None else ""
    if not text:
        # No content region, or it was empty (client-rendered page); use the whole document.
        text = _extract(html, None, base_url)
    return text


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def repo_root(entry: dict) -> Path | None:
    override = os.environ.get(entry["rootEnv"])
    if override:
        path = Path(override)
    else:
        path = (TRACKDUB_ROOT / entry["defaultFromTrackdub"]).resolve()
    return path if path.is_dir() else None


def glob_matches(rel: str, pattern: str) -> bool:
    """Match a repo-relative path against a manifest glob on any supported Python.

    Pathlib's pattern matcher needs 3.13 and fnmatch's star crosses ``/``, so the
    corpus manifests are matched segment by segment instead.
    """
    parts = rel.split("/")
    segments = [segment for segment in pattern.split("/") if segment]

    def walk(index: int, rule: int) -> bool:
        if rule == len(segments):
            return index == len(parts)
        if segments[rule] == "**":
            return any(walk(depth, rule + 1) for depth in range(index, len(parts) + 1))
        if index >= len(parts) or not fnmatch.fnmatchcase(parts[index], segments[rule]):
            return False
        return walk(index + 1, rule + 1)

    return walk(0, 0)


def stage_repos(manifest: dict, staging: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0
    max_bytes = int(manifest["maxBytes"])
    for repo in manifest["repos"]:
        root = repo_root(repo)
        if root is None:
            raise RuntimeError(f"repo {repo['id']}: root missing")
        prefix = repo["prefix"]
        exclude = repo.get("exclude", [])
        matched_excludes: set[str] = set()
        repo_copied = copied
        seen: set[Path] = set()
        for pattern in repo["include"]:
            for path in root.glob(pattern):
                if not path.is_file() or path in seen:
                    continue
                if any(part in {".git", "node_modules", ".venv"} for part in path.parts):
                    continue
                seen.add(path)
                rel = path.relative_to(root).as_posix()
                excluded = [glob for glob in exclude if glob_matches(rel, glob)]
                if excluded:
                    matched_excludes.update(excluded)
                    continue
                if path.stat().st_size > max_bytes:
                    print(f"skip large {path}")
                    skipped += 1
                    continue
                dest = staging / prefix / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
                copied += 1
        if copied == repo_copied:
            raise RuntimeError(f"repo {repo['id']}: no eligible documents")
        unmatched = sorted(set(exclude) - matched_excludes)
        if unmatched:
            raise RuntimeError(f"repo {repo['id']}: exclude patterns matched no document: {unmatched}")
    return copied, skipped


def fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain"})
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    text = body.decode("utf-8", errors="replace")
    if "html" in content_type or text.lstrip().lower().startswith("<!doctype html") or text.lstrip().lower().startswith("<html"):
        text = html_to_text(text, base_url=url)
    return text.strip()


def vendor_sources(vendor: dict) -> list[tuple[str, str]]:
    """Vendor sources are [id, url] pairs or {"id", "url", ...} objects (e.g. carrying a "gpu" block)."""
    pairs: list[tuple[str, str]] = []
    for entry in vendor["sources"]:
        if isinstance(entry, dict):
            pairs.append((entry["id"], entry["url"]))
        else:
            source_id, url = entry
            pairs.append((source_id, url))
    return pairs


def stage_vendors(manifest: dict, staging: Path) -> tuple[int, int]:
    ok = 0
    fail = 0
    max_bytes = int(manifest["maxBytes"])
    for vendor in manifest["vendors"]:
        for source_id, url in vendor_sources(vendor):
            dest = staging / vendor["prefix"] / f"{source_id}.md"
            try:
                text = fetch_url(url)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"FAIL {vendor['id']}/{source_id}: {exc}")
                fail += 1
                continue
            if not text:
                print(f"FAIL {vendor['id']}/{source_id}: empty")
                fail += 1
                continue
            payload = f"Source: {url}\n\n{text}\n"
            if len(payload.encode("utf-8")) > max_bytes:
                payload = payload.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(payload, encoding="utf-8")
            print(f"ok {vendor['id']}/{source_id}")
            ok += 1
    return ok, fail


def read_pin(runtime_manifest: Path = RUNTIME_MANIFEST) -> dict:
    """Pin facts come from runtime/trt-rtx-ep.manifest.json so docs text cannot drift from the shipped bundle."""
    data = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    packages = data.get("packages", {})

    def package_version(platform: str) -> str:
        match = re.search(r"/v(\d+\.\d+\.\d+)/", packages.get(platform, {}).get("archiveUrl", ""))
        return match.group(1) if match else data["version"]

    trt = data["trtRtxRuntimeVersion"]
    major, minor = trt.split(".")[:2]
    return {
        "windows": package_version("win-x64"),
        "linux": package_version("linux-x64"),
        "cuda": data["cudaVariant"],
        "trt": trt,
        "dll": f"tensorrt_rtx_{major}_{minor}.dll",
    }


def pin_text(pin: dict) -> str:
    if pin["windows"] == pin["linux"]:
        version = f"version {pin['windows']} on Windows and Linux"
    else:
        version = f"version {pin['windows']} on Windows and {pin['linux']} on Linux"
    return "\n".join(
        [
            "# Docs RAG pin",
            "",
            f"Trackdub ships TensorRT-RTX as the standalone ONNX Runtime EP ABI plugin, {version} (TensorRT-RTX {pin['trt']}; Linux installs through LinML.Catalog), CUDA {pin['cuda']}.",
            f"Windows bundle includes {pin['dll']} (CUDA runtime statically linked). Do not treat Windows ML catalog NvTensorRtRtxExecutionProvider as the primary route.",
            "Vendor /latest/ pages may describe a newer TensorRT-RTX than this pin. First-party docs win on disagreement.",
            "",
        ]
    )


def stage_pin(staging: Path) -> None:
    pin = staging / "first-party" / "trackdub" / "docs" / "reference" / "docs-rag-pin.md"
    pin.parent.mkdir(parents=True, exist_ok=True)
    if pin.exists():
        return
    pin.write_text(pin_text(read_pin()), encoding="utf-8")


def git_state(root: Path) -> dict:
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "--short=12", "HEAD")
    status = run("status", "--porcelain", "--", "docs", "AGENTS.md", "runtime")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def stage_sync_manifest(manifest: dict, staging: Path) -> None:
    """Record which commit of each repo this corpus came from; the Worker stamps it onto first-party hits."""
    repos = {}
    for repo in manifest["repos"]:
        root = repo_root(repo)
        repos[repo["id"]] = git_state(root) if root is not None else {"commit": None, "dirty": None}
    payload = {"syncedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repos": repos}
    dest = staging / SYNC_MANIFEST_KEY
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def wrangler_prefix(api_root: Path) -> list[str]:
    """Resolve a Win32-safe wrangler invocation (`.bin/wrangler` is a POSIX shim)."""
    js = api_root / "node_modules" / "wrangler" / "bin" / "wrangler.js"
    if js.is_file():
        return ["node", str(js)]
    cmd = api_root / "node_modules" / ".bin" / "wrangler.cmd"
    if cmd.is_file():
        return [str(cmd)]
    return ["npx", "wrangler"]


def upload(staging: Path, bucket: str, api_root: Path, workers: int = 8, prune: bool = False) -> int:
    # Wrangler's object CLI cannot set custom metadata; use its remote R2 binding.
    command = ["node", str(TOOL_ROOT / "upload_corpus.mjs"), str(api_root), str(staging), bucket, str(max(1, workers))]
    if prune:
        command.append("--prune")
    result = subprocess.run(command, cwd=api_root, check=False)
    return result.returncode


def reindex(api_root: Path, instance: str) -> int:
    result = subprocess.run(
        [*wrangler_prefix(api_root), "ai-search", "jobs", "create", instance, "--json"],
        cwd=api_root,
        check=False,
    )
    if result.returncode:
        print("Upload succeeded but reindex request failed; check ai-search jobs list before retrying because the job may have started.", flush=True)
    else:
        print("Reindex requested; check ai-search jobs list for completion.", flush=True)
    return result.returncode


def wait_and_verify(api_root: Path, timeout_minutes: int, poll_seconds: int = 30) -> int:
    """Poll the read-only verifier until the reindex job is done, then return its verdict."""
    command = ["node", str(TOOL_ROOT / "verify_index.mjs"), "--api-root", str(api_root)]
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        if "Indexing job active" not in output:
            print(output, flush=True)
            return result.returncode
        if time.monotonic() >= deadline:
            print(f"Indexing job still active after {timeout_minutes} minutes; re-run verify_index.mjs later.", flush=True)
            return 1
        time.sleep(poll_seconds)


def refresh_staging(manifest: dict, staging: Path, skip_fetch: bool) -> None:
    previous = staging.with_name(staging.name + ".previous")
    if previous.exists():
        raise RuntimeError(f"recover previous staging at {previous} before refreshing")
    with tempfile.TemporaryDirectory(prefix=".corpus-refresh-", dir=staging.parent) as temporary:
        candidate = Path(temporary) / "candidate"
        candidate.mkdir()
        copied, skipped = stage_repos(manifest, candidate)
        stage_pin(candidate)
        stage_sync_manifest(manifest, candidate)
        vendor_ok, vendor_fail = (0, 0) if skip_fetch else stage_vendors(manifest, candidate)
        print(f"staged repo files={copied} skipped={skipped} vendor_ok={vendor_ok} vendor_fail={vendor_fail}")
        if vendor_fail:
            raise RuntimeError(f"{vendor_fail} vendor documents failed")
        if staging.exists():
            staging.rename(previous)
        try:
            candidate.rename(staging)
        except OSError:
            if previous.exists():
                previous.rename(staging)
            raise
        if previous.exists():
            shutil.rmtree(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and optionally upload and reindex the Trackdub docs corpus")
    parser.add_argument("--upload", action="store_true", help="Put staged objects with metadata into R2, then request reindex")
    parser.add_argument("--skip-fetch", action="store_true", help="Stage repo files only")
    parser.add_argument(
        "--reuse-staging",
        action="store_true",
        help="Upload existing .staging without re-fetching or clearing",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel R2 upload workers")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="After a successful upload, delete bucket objects missing from staging; requires a full refresh",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="After the reindex request, poll until the job finishes, then run verify_index and use its exit code",
    )
    parser.add_argument("--wait-timeout", type=int, default=40, help="Minutes --wait polls before giving up")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.wait and not args.upload:
        print("--wait needs --upload: there is no reindex job to wait for", flush=True)
        raise SystemExit(2)
    blocking = [
        name
        for name, refused in (
            ("--skip-fetch", args.skip_fetch),
            ("--reuse-staging", args.reuse_staging),
            ("missing --upload", not args.upload),
        )
        if refused
    ]
    if args.prune and blocking:
        print(f"--prune deletes bucket objects missing from staging; refusing because of {', '.join(blocking)}", flush=True)
        raise SystemExit(2)


def prepare_staging(manifest: dict, args: argparse.Namespace) -> None:
    if args.reuse_staging:
        if not STAGING.is_dir() or not any(path.is_file() for path in STAGING.rglob("*")):
            raise SystemExit(f"no staging tree at {STAGING}; run without --reuse-staging first")
        print(f"reusing staging at {STAGING}")
        return
    try:
        refresh_staging(manifest, STAGING, args.skip_fetch)
    except (OSError, RuntimeError) as exc:
        print(f"Refresh failed; previous staging preserved; nothing uploaded: {exc}", flush=True)
        raise SystemExit(1) from exc


def publish(manifest: dict, args: argparse.Namespace) -> int:
    api_root = Path(os.environ.get("API_TRACKDUB_ROOT", TRACKDUB_ROOT.parent / "api.trackdub")).resolve()
    uploaded = upload(STAGING, manifest["bucket"], api_root, workers=args.workers, prune=args.prune)
    if uploaded:
        return uploaded
    reindexed = reindex(api_root, manifest["aiSearchInstance"])
    if reindexed or not args.wait:
        return reindexed
    return wait_and_verify(api_root, args.wait_timeout)


def main() -> None:
    args = parse_args()
    validate_args(args)
    manifest = load_manifest()
    prepare_staging(manifest, args)
    if not args.upload:
        print(f"dry-run staging at {STAGING}")
        return
    raise SystemExit(publish(manifest, args))


if __name__ == "__main__":
    main()
