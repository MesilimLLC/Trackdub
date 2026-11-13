#!/usr/bin/env python3
"""Scan the public-core legal/policy surface for stale monorepo-era claims.

Guards against regressions like the GPLv3/dual-license CLA wording and the
false "Implemented in Application layer" IExportTierGate claim found during
the open-core split: license text that contradicts the Apache-2.0 LICENSE,
and desktop/cloud project names bleeding into files a public consumer of
this core would read as authoritative.

This intentionally does not scan docs/architecture, docs/specs,
docs/operations, docs/development, docs/decisions, docs/audits, docs/plans,
docs/archive,
or tools/**: those contain legitimate historical/internal-engineering
references to pre-split monorepo projects (Trackdub.App.Avalonia,
Trackdub.Api, Trackdub.Worker, Trackdub.WebhookDelivery, the activation
service) that are a separate, larger documentation cleanup — not part of
this check.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# License-claim patterns: must never appear anywhere in the tracked tree
# (excluding historical planning/audit records and third-party notices,
# which may legitimately discuss or quote other licenses).
LICENSE_CLAIM_PATTERNS = [
    # Matches GPL/GPLv3/GPL-3.0-only/"GPL 3.0" etc., case-insensitively,
    # while excluding LGPL/AGPL (a letter immediately before "GPL").
    re.compile(r"(?<![A-Za-z])GPL", re.IGNORECASE),
    # Same exclusion for the prose form: "GNU Lesser/Affero General Public
    # License" are distinct, permissive-enough-to-be-legitimate licenses.
    re.compile(r"(?<!Lesser )(?<!Affero )General Public License", re.IGNORECASE),
    re.compile(r"dual[- ]licens", re.IGNORECASE),
]
LICENSE_CLAIM_EXCLUDE_DIRS = [
    "docs/archive/",
    "docs/plans/",
    "docs/decisions/",
    "docs/audits/",
    "docs/architecture/",
    "docs/specs/",
    "docs/operations/",
    "docs/development/",
    "tools/",
]
LICENSE_CLAIM_EXCLUDE_FILES = [
    "docs/legal/THIRD_PARTY_NOTICES.md",
    "docs/legal/LICENSE-HISTORY.md",
    "scripts/ci/check-repository-boundary.py",
    # Per docs/repository-policy.md's "Native Dependencies" section, downloaded
    # native binaries aren't tracked here — the acquisition scripts, URLs, and
    # hashes are. This file's whole job is describing which third-party (GPL,
    # in this case) ffmpeg binary variant gets fetched and why — that's the
    # same legitimate "discuss another license" case as the notices file
    # above, just in the acquisition script itself rather than in docs/legal/.
    "src/Trackdub.Media/Process/FfmpegAutoDownloader.cs",
    "tests/Trackdub.Media.Tests/FfmpegAutoDownloaderTests.cs",
]

# Repository-boundary patterns: desktop/cloud project names that must not
# appear as if they were part of this public core's legal/policy/contract
# surface.
BOUNDARY_PATTERNS = [
    re.compile(r"Trackdub\.App\.Avalonia", re.IGNORECASE),
    re.compile(r"Trackdub\.Api\b", re.IGNORECASE),
    re.compile(r"Trackdub\.Worker\b", re.IGNORECASE),
    re.compile(r"Trackdub\.WebhookDelivery", re.IGNORECASE),
    re.compile(r"activation-service", re.IGNORECASE),
]
BOUNDARY_SCAN_PATHS = [
    "LICENSE",
    "NOTICE",
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docs/repository-policy.md",
    "docs/legal/",
    "src/Trackdub.Contracts/",
]


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [REPO_ROOT / line for line in output.splitlines() if line]


def matches_any(rel_posix: str, prefixes: list[str]) -> bool:
    return any(rel_posix == p.rstrip("/") or rel_posix.startswith(p) for p in prefixes)


def scan(files: list[Path], patterns: list[re.Pattern[str]]) -> list[str]:
    violations: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{lineno}: matched /{pattern.pattern}/: {line.strip()}")
    return violations


def is_machine_generated_lock_file(path: Path) -> bool:
    name = path.name
    return name == "packages.lock.json" or name.endswith(".lock") or name.endswith(".lock.json")


def main() -> int:
    all_files = tracked_files()
    violations: list[str] = []

    license_files = [
        f
        for f in all_files
        if not matches_any(f.relative_to(REPO_ROOT).as_posix(), LICENSE_CLAIM_EXCLUDE_DIRS)
        and f.relative_to(REPO_ROOT).as_posix() not in LICENSE_CLAIM_EXCLUDE_FILES
        and not is_machine_generated_lock_file(f)
    ]
    violations += scan(license_files, LICENSE_CLAIM_PATTERNS)

    boundary_files = [
        f
        for f in all_files
        if matches_any(f.relative_to(REPO_ROOT).as_posix(), BOUNDARY_SCAN_PATHS)
    ]
    violations += scan(boundary_files, BOUNDARY_PATTERNS)

    if violations:
        print("Repository-boundary scan failed:")
        for v in violations:
            print(f"  {v}")
        print()
        print(
            "Stale license or desktop/cloud-boundary wording found in the public-core "
            "legal/policy surface. See docs/legal/LICENSE-HISTORY.md and "
            "docs/repository-policy.md."
        )
        return 1

    print("Repository-boundary scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
