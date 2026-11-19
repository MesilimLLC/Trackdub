#!/usr/bin/env python3
"""Select Common Voice speakers for separator training and extract their clips from the downloaded archives.

Speakers in the frozen eval corpus are never chosen. Every other speaker with at least --min-clips good clips
(validated, up_votes >= 2, down_votes == 0, 2 to 15 s; the eval subset's filter) contributes up to
--clips-per-speaker of them, chosen by a seeded hash so a rerun picks the same clips. Speaker groups are built
exactly as for the eval subset, `commonvoice:<lang>:<first 16 hex of sha256(client_id)>`, so train_pool.py can
compare them; raw client ids are never written out.

The output directory gets `<lang>/clips/*.mp3` and `commonvoice-subset.manifest.json` in the schema
build_local_items.py reads, with `split: train`. Then:

    python select_cv_train.py --cv-root D:/cache --eval-manifest D:/cache/items.v1.2.manifest.json \\
        --out-dir D:/train-cache/commonvoice
    python build_local_items.py --cv-dir D:/train-cache/commonvoice --cache D:/train-cache/corpus-cache \\
        --out D:/train-cache/cv-train.candidates.jsonl
    python train_pool.py check --eval-manifest D:/cache/items.v1.2.manifest.json \\
        --candidates D:/train-cache/cv-train.candidates.jsonl --out ... --report ...

--cv-root holds the `*cv-corpus-<version>-<date>-<lang>.tar.gz` archives and a `cv-meta/<lang>/` folder with
each archive's `validated.tsv` and `clip_durations.tsv`.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterator, NamedTuple

csv.field_size_limit(sys.maxsize)

DEFAULT_SEED = "cv-train-v1"
FILTER = "up_votes>=2, down_votes==0, 2-15 s"
ARCHIVE_NAME = re.compile(r"cv-corpus-([\d.]+)-(\d{4}-\d{2}-\d{2})-(.+)\.tar\.gz$")
GENDER_CLASS = {"female_feminine": "F", "male_masculine": "M"}


class Clip(NamedTuple):
    path: str
    duration_s: float
    up_votes: int
    gender: str
    age: str
    accent: str


class Speaker(NamedTuple):
    group: str
    gender_class: str
    clips: list[Clip]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def speaker_group(lang: str, client_id: str) -> str:
    return f"commonvoice:{lang}:{sha256_text(client_id)[:16]}"


def tsv_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def read_durations(path: Path) -> dict[str, float]:
    durations = {}
    for row in tsv_rows(path):
        try:
            durations[row["clip"]] = float(row["duration[ms]"]) / 1000.0
        except (KeyError, ValueError):
            continue
    return durations


def votes(row: dict) -> tuple[int, int] | None:
    try:
        return int(row["up_votes"]), int(row["down_votes"])
    except (KeyError, ValueError):
        return None


def well_rated(up_down: tuple[int, int] | None) -> bool:
    return up_down is not None and up_down[0] >= 2 and up_down[1] == 0


def usable_length(duration: float | None) -> bool:
    return duration is not None and 2.0 <= duration <= 15.0


def good_clip(row: dict, durations: dict[str, float]) -> Clip | None:
    up_down, duration = votes(row), durations.get(row.get("path", ""))
    if not (well_rated(up_down) and usable_length(duration)):
        return None
    return Clip(row["path"], duration, up_down[0], row.get("gender") or "", row.get("age") or "",
                row.get("accent") or "")


def good_clips_by_client(meta: Path) -> dict[str, list[Clip]]:
    durations = read_durations(meta / "clip_durations.tsv")
    clips: dict[str, list[Clip]] = collections.defaultdict(list)
    for row in tsv_rows(meta / "validated.tsv"):
        clip = good_clip(row, durations)
        if clip:
            clips[row["client_id"]].append(clip)
    return clips


class Rules(NamedTuple):
    min_clips: int
    clips_per_speaker: int
    seed: str


def select_speakers(lang: str, clips_by_client: dict[str, list[Clip]], excluded_groups: set[str],
                    rules: Rules) -> list[Speaker]:
    """Every non-excluded speaker with enough good clips, each capped to a seeded choice of clips."""
    speakers = []
    for client, clips in clips_by_client.items():
        group = speaker_group(lang, client)
        if len(clips) < rules.min_clips or group.casefold() in excluded_groups:
            continue
        picked = sorted(clips, key=lambda c: sha256_text(f"{rules.seed}:{lang}:{c.path}"))[:rules.clips_per_speaker]
        speakers.append(Speaker(group, GENDER_CLASS.get(clips[0].gender, "U"), picked))
    return sorted(speakers, key=lambda s: s.group)


def eval_groups(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {item["group"].casefold() for item in manifest["items"] if item.get("group")}


class Archive(NamedTuple):
    path: Path
    lang: str


def manifest_entry(archive: Archive, member: str, raw: bytes, pick: tuple[Speaker, Clip]) -> dict:
    speaker, clip = pick
    version, date, _ = ARCHIVE_NAME.search(archive.path.name).groups()
    lang, base = archive.lang, Path(member).name
    return {"language": lang, "path": f"commonvoice/{lang}/clips/{base}", "group": speaker.group, "split": "train",
            "gender_class": speaker.gender_class, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "duration_s": clip.duration_s, "age": clip.age, "accent": clip.accent, "up_votes": clip.up_votes,
            "archive": archive.path.name, "member": member, "corpus": f"cv-corpus-{version}-{date}"}


def wanted_clip(member: tarfile.TarInfo, wanted: dict) -> str | None:
    """The clip's file name when this archive member is one of the wanted clips, else None."""
    base = Path(member.name).name
    is_clip = member.isfile() and "/clips/" in member.name
    return base if is_clip and base in wanted else None


def extract(archive: Archive, speakers: list[Speaker], out_dir: Path) -> list[dict]:
    """Stream the archive once and write each wanted clip; returns manifest entries for the clips found."""
    wanted = {c.path: (s, c) for s in speakers for c in s.clips}
    dest = out_dir / archive.lang / "clips"
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(archive.path, mode="r|gz") as tf:
        for member in tf:
            base = wanted_clip(member, wanted)
            if base is None:
                continue
            raw = tf.extractfile(member).read()
            part = dest / (base + ".part")
            part.write_bytes(raw)
            part.replace(dest / base)
            entries.append(manifest_entry(archive, member.name, raw, wanted[base]))
    return entries


def archives_in(cv_root: Path, languages: set[str] | None) -> list[Archive]:
    found = [Archive(p, ARCHIVE_NAME.search(p.name).group(3)) for p in sorted(cv_root.glob("*cv-corpus-*.tar.gz"))]
    return [a for a in found if languages is None or a.lang in languages]


def select_and_extract(args: argparse.Namespace) -> dict:
    excluded = eval_groups(args.eval_manifest)
    languages = set(args.languages.split(",")) if args.languages else None
    files, summary = [], {}
    rules = Rules(args.min_clips, args.clips_per_speaker, args.seed)
    for archive in archives_in(args.cv_root, languages):
        lang = archive.lang
        clips = good_clips_by_client(args.cv_root / "cv-meta" / lang)
        speakers = select_speakers(lang, clips, excluded, rules)
        entries = extract(archive, speakers, args.out_dir)
        files.extend(entries)
        summary[lang] = {"speakers": len(speakers), "clips_wanted": sum(len(s.clips) for s in speakers),
                         "clips_extracted": len(entries), "hours": round(sum(e["duration_s"] for e in entries) / 3600, 2)}
        print(f"{lang}: {summary[lang]}", file=sys.stderr, flush=True)
    return {"purpose": "separator training (never eval)", "seed": args.seed, "min_good_clips": args.min_clips,
            "clips_per_speaker": args.clips_per_speaker, "filter": FILTER,
            "excluded_eval_items_sha256": json.loads(args.eval_manifest.read_text(encoding="utf-8"))["items_sha256"],
            "summary": summary, "files": files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv-root", type=Path, required=True, help="folder with the archives and cv-meta/<lang>/")
    parser.add_argument("--eval-manifest", type=Path, required=True, help="frozen eval items manifest; its speakers are skipped")
    parser.add_argument("--out-dir", type=Path, required=True, help="where clips and the subset manifest go")
    parser.add_argument("--languages", help="comma-separated locale codes (default: every archive found)")
    parser.add_argument("--min-clips", type=int, default=10, help="good clips a speaker needs to be chosen (default 10)")
    parser.add_argument("--clips-per-speaker", type=int, default=60, help="cap per speaker (default 60)")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.out_dir.resolve() == (args.cv_root / "commonvoice").resolve():
        parser.error("--out-dir must not be the eval subset folder")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = select_and_extract(args)
    path = args.out_dir / "commonvoice-subset.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"{len(manifest['files'])} clips in {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
