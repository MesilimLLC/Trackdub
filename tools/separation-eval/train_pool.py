#!/usr/bin/env python3
"""Check candidate training items against the frozen eval corpus and write the training-pool records.

Rubric 2.3 (as amended 2026-11-16) allows training on corpus sources only when the training items are disjoint from
every eval item, dev and test alike. A candidate is rejected when it shares any identity with an eval item or an
explicitly excluded record:
  - item id;
  - audio content (sha256);
  - group: speaker (Common Voice, VCTK), uploader (Freesound), artist (Jamendo), book (LibriVox) or room (SLR26),
    compared case-insensitively, the same key the eval split uses;
  - upstream id: Freesound sound id, Jamendo track id.
It is also rejected when it fails ingest's record validation (licence allow-list, share-alike always refused), has
no sha256, is itself marked excluded_from_training, or repeats audio or an id already accepted into the pool.

Known limit: eval LibriVox items record the book, not the reader, so a reader who also read an eval book is not
caught; LibriVox candidates are accepted with a warning until readers are recorded (next-scope section 4).

Usage:
  python train_pool.py check --eval-manifest items.v1.2.manifest.json --candidates cand.jsonl [--candidates ...]
      [--exclude-records items.v1.1.records.jsonl] --out items.train.jsonl --report train-pool-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import ingest

CHECKER_VERSION = 1
UNCHECKED_READER_SOURCES = ("librivox",)


@dataclass
class EvalIdentity:
    """Every identity key held by the eval set and the excluded records, by kind."""
    ids: set[str] = field(default_factory=set)
    sha256: set[str] = field(default_factory=set)
    groups: set[str] = field(default_factory=set)
    upstream: set[str] = field(default_factory=set)

    def add(self, item: dict) -> None:
        self.ids.add(item["id"])
        if item.get("sha256"):
            self.sha256.add(item["sha256"])
        if item.get("group"):
            self.groups.add(normalize_group(item["group"]))
        self.upstream.update(upstream_keys(item))


@dataclass
class Verdict:
    record: dict
    reasons: list[str]
    warnings: list[str]


def normalize_group(group: str) -> str:
    return group.strip().casefold()


def upstream_keys(item: dict) -> set[str]:
    """Upstream catalogue ids that identify the same recording across re-encodes."""
    keys = set()
    if item.get("freesound_id") is not None:
        keys.add(f"freesound:{int(item['freesound_id'])}")
    track = jamendo_track_id(item.get("url", ""))
    if track:
        keys.add(f"jamendo:{track}")
    return keys


def jamendo_track_id(url: str) -> str | None:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if not (parts.hostname or "").endswith("jamendo.com"):
        return None
    track = urllib.parse.parse_qs(parts.query).get("trackid", [None])[0]
    if track:
        return track
    segments = [s for s in parts.path.split("/") if s]
    after = [segments[i + 1] for i, s in enumerate(segments[:-1]) if s == "track" and segments[i + 1].isdigit()]
    return after[0] if after else None


def load_eval_identity(manifest: dict, excluded: list[dict]) -> EvalIdentity:
    identity = EvalIdentity()
    for item in [*manifest["items"], *excluded]:
        identity.add(item)
    return identity


def eval_overlap(rec: dict, identity: EvalIdentity) -> list[str]:
    reasons = []
    if rec["id"] in identity.ids:
        reasons.append("eval-id: id is an eval or excluded item id")
    if rec["sha256"] in identity.sha256:
        reasons.append("eval-audio: audio is identical to an eval or excluded item")
    if normalize_group(rec["group"]) in identity.groups:
        reasons.append(f"eval-group: group '{rec['group']}' is an eval or excluded group")
    reasons.extend(f"eval-upstream: upstream id {key} is used by an eval or excluded item"
                   for key in sorted(upstream_keys(rec) & identity.upstream))
    return reasons


def record_problems(rec: dict) -> list[str]:
    reason = ingest.validate_record(rec, allow_sharealike=False)
    if reason:
        return [f"invalid: {reason}"]
    problems = []
    if not ingest.SHA256_PATTERN.match(str(rec.get("sha256", ""))):
        problems.append("no-sha256: ingest or hash the audio before checking")
    if ":" not in rec["group"].strip(":"):
        problems.append(f"bad-group: group '{rec['group']}' must be '<scheme>:<value>'")
    if rec.get("excluded_from_training") is True:
        problems.append("marked-excluded: record is marked excluded_from_training")
    return problems


def check_record(rec: dict, identity: EvalIdentity, seen: EvalIdentity) -> Verdict:
    reasons = record_problems(rec)
    if not reasons:
        reasons = eval_overlap(rec, identity)
    if not reasons:
        if rec["id"] in seen.ids:
            reasons.append("pool-duplicate-id: id repeats an item already in the pool")
        if rec["sha256"] in seen.sha256:
            reasons.append("pool-duplicate-audio: audio repeats an item already in the pool")
    warnings = []
    if rec.get("source") in UNCHECKED_READER_SOURCES:
        warnings.append("reader not checked: eval LibriVox items record the book, not the reader")
    return Verdict(rec, reasons, warnings)


def check_candidates(candidates: list[dict], identity: EvalIdentity) -> list[Verdict]:
    seen = EvalIdentity()
    verdicts = []
    for rec in candidates:
        verdict = check_record(rec, identity, seen)
        if not verdict.reasons:
            seen.ids.add(rec["id"])
            seen.sha256.add(rec["sha256"])
        verdicts.append(verdict)
    return verdicts


def train_record(rec: dict, eval_items_sha256: str) -> dict:
    return {**rec, "split": "train", "excluded_from_training": False,
            "disjointness": {"checker_version": CHECKER_VERSION, "eval_items_sha256": eval_items_sha256}}


def reason_codes(verdicts: list[Verdict]) -> dict[str, int]:
    return dict(Counter(r.split(":", 1)[0] for v in verdicts for r in v.reasons))


def warning_counts(verdicts: list[Verdict]) -> dict[str, int]:
    return dict(Counter(w for v in verdicts for w in v.warnings))


def summarize(verdicts: list[Verdict], provenance: dict) -> dict:
    accepted = [v for v in verdicts if not v.reasons]
    rejected = [v for v in verdicts if v.reasons]
    return {
        **provenance,
        "checker_version": CHECKER_VERSION,
        "candidates": len(verdicts), "accepted": len(accepted), "rejected": len(rejected),
        "accepted_by_source": dict(Counter(v.record.get("source", "?") for v in accepted)),
        "rejected_by_reason": reason_codes(rejected),
        "warnings": warning_counts(accepted),
        "rejections": [{"id": v.record.get("id"), "reasons": v.reasons} for v in rejected],
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(paths: list[Path]) -> list[dict]:
    return [r for p in paths for r in ingest.read_jsonl(p)]


def load_eval_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("items_sha256") != ingest.items_hash(manifest.get("items", [])):
        raise SystemExit(f"{path}: items_sha256 does not match its items")
    return manifest


def write_pool(path: Path, records: list[dict], eval_items_sha256: str) -> None:
    header = (f"# Training pool checked against eval items_sha256 {eval_items_sha256} "
              f"(train_pool.py checker version {CHECKER_VERSION}); never mix into the eval manifest.")
    lines = [header, *(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def input_provenance(args: argparse.Namespace, eval_items_sha256: str) -> dict:
    return {"eval_manifest_sha256": file_sha256(args.eval_manifest), "eval_items_sha256": eval_items_sha256,
            "exclude_records": {str(p): file_sha256(p) for p in args.exclude_records},
            "candidate_files": {str(p): file_sha256(p) for p in args.candidates}}


def run_check(args: argparse.Namespace) -> dict:
    manifest = load_eval_manifest(args.eval_manifest)
    items_sha = manifest["items_sha256"]
    identity = load_eval_identity(manifest, read_records(args.exclude_records))
    verdicts = check_candidates(read_records(args.candidates), identity)
    write_pool(args.out, [train_record(v.record, items_sha) for v in verdicts if not v.reasons], items_sha)
    report = summarize(verdicts, input_provenance(args, items_sha))
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="check candidates and write the training-pool records")
    p.add_argument("--eval-manifest", type=Path, required=True, help="frozen eval items manifest (items.v1.2.manifest.json)")
    p.add_argument("--candidates", type=Path, action="append", required=True, help="candidate records (JSONL); repeatable")
    p.add_argument("--exclude-records", type=Path, action="append", default=[],
                   help="extra records that must never be trained on, e.g. items dropped at a freeze; repeatable")
    p.add_argument("--out", type=Path, required=True, help="accepted training records (JSONL)")
    p.add_argument("--report", type=Path, required=True, help="summary and every rejection with its reasons (JSON)")
    args = parser.parse_args(argv)
    report = run_check(args)
    print(f"{report['accepted']} accepted, {report['rejected']} rejected of {report['candidates']}; "
          f"report in {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
