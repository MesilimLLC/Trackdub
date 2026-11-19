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
import re
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
    urls = [str(item.get(key) or "") for key in ("url", "license_evidence_url")]
    sounds = {freesound_id_field(item), *(freesound_id_in_url(u) for u in urls)}
    tracks = {jamendo_track_id(u) for u in urls}
    return {f"freesound:{s}" for s in sounds if s} | {f"jamendo:{t}" for t in tracks if t}


def freesound_id_field(item: dict) -> str | None:
    """The record's freesound_id as a canonical decimal string, or None when absent or not a whole number."""
    value = item.get("freesound_id")
    text = "" if value is None or isinstance(value, bool) else str(value).strip()
    return str(int(text)) if text.isdigit() else None


FREESOUND_SOUND_PATH = re.compile(r"/(?:sounds|s)/(\d+)(?:/|$)")
FREESOUND_PREVIEW_PATH = re.compile(r"/previews/\d+/(\d+)_")


def freesound_id_in_url(url: str) -> str | None:
    """Sound id in a freesound.org sound page, API or preview URL."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host != "freesound.org" and not host.endswith(".freesound.org"):
        return None
    match = FREESOUND_SOUND_PATH.search(parts.path) or FREESOUND_PREVIEW_PATH.search(parts.path)
    return str(int(match.group(1))) if match else None


def is_jamendo_host(host: str | None) -> bool:
    host = (host or "").lower()
    return host == "jamendo.com" or host.endswith(".jamendo.com")


def track_in_path(path: str) -> str | None:
    """The numeric segment after 'track', as in /track/339322/name or /download/track/685596/mp32/."""
    segments = [s for s in path.split("/") if s]
    after = [nxt for cur, nxt in zip(segments, segments[1:]) if cur == "track" and nxt.isdigit()]
    return after[0] if after else None


def jamendo_track_id(url: str) -> str | None:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if not is_jamendo_host(parts.hostname):
        return None
    return urllib.parse.parse_qs(parts.query).get("trackid", [None])[0] or track_in_path(parts.path)


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


def field_problems(rec: dict) -> list[str]:
    checks = [
        (not ingest.SHA256_PATTERN.match(str(rec.get("sha256", ""))),
         "no-sha256: ingest or hash the audio before checking"),
        (":" not in rec["group"].strip(":"), f"bad-group: group '{rec['group']}' must be '<scheme>:<value>'"),
        (rec.get("excluded_from_training") is True, "marked-excluded: record is marked excluded_from_training"),
        ("freesound_id" in rec and freesound_id_field(rec) is None,
         f"bad-freesound-id: freesound_id {rec.get('freesound_id')!r} is not a whole number"),
    ]
    return [message for failed, message in checks if failed]


def record_problems(rec: dict) -> list[str]:
    reason = ingest.validate_record(rec, allow_sharealike=False)
    return [f"invalid: {reason}"] if reason else field_problems(rec)


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


@dataclass(frozen=True)
class InputFile:
    """An input read once: its hash and its parsed content both come from these bytes."""
    path: Path
    sha256: str
    text: str


def read_input(path: Path) -> InputFile:
    data = path.read_bytes()
    return InputFile(path, hashlib.sha256(data).hexdigest(), data.decode("utf-8-sig"))


def parse_records(inputs: list[InputFile]) -> list[dict]:
    return [r for inp in inputs for r in ingest.parse_jsonl(inp.text, str(inp.path))]


def load_eval_manifest(inp: InputFile) -> dict:
    manifest = json.loads(inp.text)
    if manifest.get("items_sha256") != ingest.items_hash(manifest.get("items", [])):
        raise SystemExit(f"{inp.path}: items_sha256 does not match its items")
    return manifest


def same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False


def refuse_aliased_outputs(args: argparse.Namespace) -> None:
    """--out and --report must be two new or distinct files that are not any input."""
    if same_file(args.out, args.report):
        raise SystemExit("--out and --report must be different files")
    inputs = [args.eval_manifest, *args.exclude_records, *args.candidates]
    clashes = sorted({str(out) for out in (args.out, args.report) for inp in inputs if same_file(out, inp)})
    if clashes:
        raise SystemExit(f"output would overwrite an input: {', '.join(clashes)}")


def write_pool(path: Path, records: list[dict], eval_items_sha256: str) -> None:
    header = (f"# Training pool checked against eval items_sha256 {eval_items_sha256} "
              f"(train_pool.py checker version {CHECKER_VERSION}); never mix into the eval manifest.")
    lines = [header, *(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def input_provenance(manifest: InputFile, excluded: list[InputFile], candidates: list[InputFile],
                     eval_items_sha256: str) -> dict:
    return {"eval_manifest_sha256": manifest.sha256, "eval_items_sha256": eval_items_sha256,
            "exclude_records": {str(i.path): i.sha256 for i in excluded},
            "candidate_files": {str(i.path): i.sha256 for i in candidates}}


def run_check(args: argparse.Namespace) -> dict:
    refuse_aliased_outputs(args)
    manifest_in = read_input(args.eval_manifest)
    excluded_in = [read_input(p) for p in args.exclude_records]
    candidates_in = [read_input(p) for p in args.candidates]
    manifest = load_eval_manifest(manifest_in)
    items_sha = manifest["items_sha256"]
    identity = load_eval_identity(manifest, parse_records(excluded_in))
    verdicts = check_candidates(parse_records(candidates_in), identity)
    write_pool(args.out, [train_record(v.record, items_sha) for v in verdicts if not v.reasons], items_sha)
    report = summarize(verdicts, input_provenance(manifest_in, excluded_in, candidates_in, items_sha))
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
