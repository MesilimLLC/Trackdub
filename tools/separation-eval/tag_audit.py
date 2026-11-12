#!/usr/bin/env python3
"""Tag audit for corpus items whose tags came from the search query (`tag_basis: search-query`).

`sample` writes a review pack: a seeded sample of items (every whisper item, 10 of each other tag) as short
numbered WAV excerpts plus an audit.csv to fill in, with the claimed tag and the question to answer for each.
`score` reads the filled-in CSV and reports the failure rate per tag. A tag over 10% failures, counting
"unsure" as a failure, makes that stratum provisional in the rubric. Nothing here listens to the audio:
a person has to, and the only automatic check is for clips that are silent.

    python tag_audit.py sample --manifest items.v1.1.manifest.json --cache D:/cache/corpus-cache --out-dir D:/cache/tag-audit
    python tag_audit.py score --csv D:/cache/tag-audit/audit.csv --out D:/cache/tag-audit/tag_audit.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.io import wavfile

import audiomath as am
import ingest

QUESTIONS = {
    "whisper": "Is the speech whispered or very soft, not a normal speaking voice?",
    "loud": "Is it a loud sound effect (impact, explosion, gunshot, engine or thunder)?",
    "crowd": "Is it crowd or background-talker ambience, with many people?",
    "instrumental": "Is it music with no singing or speech?",
    "vocals": "Does the music include a singing voice?",
}
ALL_ITEMS_TAGS = ("whisper",)
SAMPLE_PER_TAG = 10
FAIL_LIMIT = 0.10
EXCERPT_S = 20.0
SILENT_DBFS = -60.0
SEED = "tag-audit-v1"
VERDICTS = ("pass", "fail", "unsure")
PACK_LIST = "pack.json"
FIELDS = ["file", "item_id", "claimed_tag", "question", "rms_dbfs", "auto_flag", "verdict", "notes"]


class AuditError(Exception):
    pass


def order_key(*parts: str) -> str:
    return hashlib.sha256(":".join([SEED, *parts]).encode("utf-8")).hexdigest()


def sample_items(items: list[dict]) -> list[tuple[str, dict]]:
    """(tag, item) pairs: all items for tags in ALL_ITEMS_TAGS, SAMPLE_PER_TAG seeded picks for the rest."""
    picked: list[tuple[str, dict]] = []
    for tag in QUESTIONS:
        pool = sorted((i for i in items if tag in i.get("tags", []) and i.get("tag_basis") == "search-query"),
                      key=lambda i: order_key(tag, i["id"]))
        picked += [(tag, i) for i in (pool if tag in ALL_ITEMS_TAGS else pool[:SAMPLE_PER_TAG])]
    return sorted(picked, key=lambda pair: order_key("order", pair[1]["id"]))


def loudest_excerpt(audio: np.ndarray, seconds: float = EXCERPT_S, sr: int = am.SR) -> np.ndarray:
    """The loudest window of `seconds` (found on a half-second grid), or the whole clip when it is shorter."""
    window, hop = int(seconds * sr), sr // 2
    if audio.shape[0] <= window:
        return audio
    starts = np.union1d(np.arange(0, audio.shape[0] - window + 1, hop), [audio.shape[0] - window])
    energy = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    totals = energy[starts + window] - energy[starts]
    start = int(starts[int(np.argmax(totals))])
    return audio[start:start + window]


def rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    return -math.inf if rms == 0.0 else 20.0 * math.log10(rms)


def write_excerpt(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    wavfile.write(path, am.SR, pcm)


class Pack(NamedTuple):
    cache: Path
    out_dir: Path


def verified_path(item: dict, cache: Path) -> Path:
    """The cached file for the item, checked against the manifest hash so a replaced file is not reviewed."""
    path = ingest.cache_path(cache, item)
    expected = item.get("sha256")
    if expected is not None and ingest.sha256_file(path) != expected:
        raise AuditError(f"{item['id']}: cached file does not match the manifest sha256")
    return path


def review_row(number: int, tag: str, item: dict, pack: Pack) -> dict:
    try:
        audio = loudest_excerpt(am.load_audio(verified_path(item, pack.cache), channels=1)[:, 0])
    except (OSError, ValueError, am.AudioError, ingest.IngestError) as exc:
        raise AuditError(f"{item['id']}: cannot load audio: {exc}") from exc
    name = f"{number:03d}.wav"
    write_excerpt(pack.out_dir / name, audio)
    level = rms_dbfs(audio)
    return {"file": name, "item_id": item["id"], "claimed_tag": tag, "question": QUESTIONS[tag],
            "rms_dbfs": "-inf" if math.isinf(level) else f"{level:.1f}",
            "auto_flag": "silent" if level < SILENT_DBFS else "", "verdict": "", "notes": ""}


def build_pack(manifest: dict, cache: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "audit.csv").exists():
        raise AuditError(f"{out_dir / 'audit.csv'} exists; it may hold verdicts, so choose a new directory")
    pack = Pack(cache, out_dir)
    rows = [review_row(n, tag, item, pack) for n, (tag, item) in enumerate(sample_items(manifest["items"]), 1)]
    with (out_dir / "audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / PACK_LIST).write_text(json.dumps({"files": [r["file"] for r in rows]}) + "\n", encoding="utf-8")
    return rows


def read_verdicts(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    check_complete_rows(rows, path)
    bad = [r["file"] for r in rows if r["verdict"].strip().lower() not in ("", *VERDICTS)]
    if bad:
        raise AuditError(f"verdict must be pass, fail or unsure (or blank); see {', '.join(bad[:5])}")
    return rows


def check_complete_rows(rows: list[dict], path: Path) -> None:
    """Every row needs every column, and the rows must be the ones the pack was written with."""
    if not rows:
        raise AuditError(f"{path} has no rows")
    for number, row in enumerate(rows, 2):
        if any(row.get(field) is None for field in FIELDS):
            raise AuditError(f"{path} line {number} is missing a column; expected {', '.join(FIELDS)}")
    check_against_pack_list(rows, path.with_name(PACK_LIST))


def check_against_pack_list(rows: list[dict], pack_list: Path) -> None:
    if not pack_list.is_file():
        return
    expected = set(json.loads(pack_list.read_text(encoding="utf-8"))["files"])
    found = {r["file"] for r in rows}
    if found != expected:
        raise AuditError(f"rows differ from the pack: missing {sorted(expected - found)[:5]}, "
                         f"unexpected {sorted(found - expected)[:5]}")


def tag_result(rows: list[dict]) -> dict:
    verdicts = collections.Counter(r["verdict"].strip().lower() for r in rows)
    answered = sum(verdicts[v] for v in VERDICTS)
    rate = (verdicts["fail"] + verdicts["unsure"]) / answered if answered else 0.0
    status = "incomplete" if verdicts[""] else ("provisional" if rate > FAIL_LIMIT else "ok")
    return {"items": len(rows), **{v: verdicts[v] for v in VERDICTS}, "blank": verdicts[""],
            "fail_rate": round(rate, 3), "status": status,
            "failed_ids": [r["item_id"] for r in rows if r["verdict"].strip().lower() in ("fail", "unsure")],
            "silent_ids": [r["item_id"] for r in rows if r["auto_flag"] == "silent"]}


def score(rows: list[dict]) -> dict:
    by_tag: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_tag[row["claimed_tag"]].append(row)
    return {"fail_limit": FAIL_LIMIT, "tags": {tag: tag_result(r) for tag, r in sorted(by_tag.items())}}


def cmd_sample(args: argparse.Namespace) -> str:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = build_pack(manifest, args.cache, args.out_dir)
    per_tag = dict(collections.Counter(r["claimed_tag"] for r in rows))
    silent = sum(1 for r in rows if r["auto_flag"])
    return f"wrote {len(rows)} excerpts {per_tag} and audit.csv to {args.out_dir}; {silent} silent"


def cmd_score(args: argparse.Namespace) -> str:
    if args.out.resolve() == args.csv.resolve():
        raise AuditError("--out must not be the verdict CSV; it would overwrite the verdicts")
    report = score(read_verdicts(args.csv))
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return "; ".join(f"{t}: {r['status']} ({r['fail_rate']:.0%} of {r['items']})" for t, r in report["tags"].items())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_sample = sub.add_parser("sample", help="write the review pack")
    p_sample.add_argument("--manifest", type=Path, required=True)
    p_sample.add_argument("--cache", type=Path, required=True)
    p_sample.add_argument("--out-dir", type=Path, required=True)
    p_score = sub.add_parser("score", help="score a filled-in audit.csv")
    p_score.add_argument("--csv", type=Path, required=True)
    p_score.add_argument("--out", type=Path, required=True)
    return parser


COMMANDS = {"sample": cmd_sample, "score": cmd_score}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(COMMANDS[args.command](args))
    except (AuditError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
