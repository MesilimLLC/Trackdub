#!/usr/bin/env python3
"""Turn locally extracted speech subsets (VCTK, Common Voice) into per-speaker dialogue items for the corpus.

mixgen lays one speaker's items out on a timeline, so each speaker becomes one item: their selected clips,
decoded to 48 kHz mono and joined with short silences, written as 16-bit WAV straight into the corpus cache.
Items are marked `local_only`: their `url` is provenance, and ingest.py accepts them by hash instead of
downloading anything. Every clip is checked against the sha256 in its subset manifest before it is used.

    python build_local_items.py --vctk-dir D:/cache/vctk --cv-dir D:/cache/commonvoice \\
        --cache D:/cache/corpus-cache --out D:/cache/items.local-dialogue.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
from scipy.io import wavfile

import audiomath as am
import ingest

GAP_S = 0.25
VCTK_DOI = "https://doi.org/10.7488/ds/2645"
VCTK_PAGE = "https://datashare.ed.ac.uk/handle/10283/3443"
CV_LISTING_URL = "https://commonvoice.mozilla.org/en/datasets"
CV_TERMS_URL = "https://commonvoice.mozilla.org/en/terms"
CV_DATASET_URLS = {
    "ar": "https://commonvoice.mozilla.org/datasets/cmu5wom3z00ceo107guxtg4xa",
    "pt": "https://commonvoice.mozilla.org/datasets/cmu5wui2w00e8nq07idithpm6",
    "ru": "https://commonvoice.mozilla.org/datasets/cmu5x45pn00dao107j4o9w2yv",
    "ja": "https://commonvoice.mozilla.org/datasets/cmu5xbhnf00fmnq07z9p3aa70",
    "zh-TW": "https://commonvoice.mozilla.org/datasets/cmu5wkw5i00c6o107jnm3t51u",
    "tr": "https://commonvoice.mozilla.org/datasets/cmu5wkah500c2o10719lem8m8",
    "nl": "https://commonvoice.mozilla.org/datasets/cmu62gjbl00nso107n7ct8dyl",
}
TRANSFORM = ("decode with ffmpeg to 48 kHz mono float32, join the clips in the listed order with "
             f"{GAP_S} s of silence between them, round to 16-bit PCM WAV")


class BuildError(Exception):
    pass


class SpeakerSource(NamedTuple):
    """One speaker's clips and the metadata their item will carry."""

    item_id: str
    group: str
    language: str
    gender_class: str
    source: str
    license_spdx: str
    url: str
    evidence_url: str
    attribution: str
    clips: list[dict]
    provenance: dict


def ffmpeg_version() -> str:
    if not am.have_ffmpeg():
        raise BuildError("ffmpeg is required to decode the clips")
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False).stdout
    return out.splitlines()[0] if out else "unknown"


def read_manifest(directory: Path, name: str) -> dict:
    try:
        return json.loads((directory / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read {directory / name}: {exc}") from exc


def parse_speaker_info(path: Path) -> dict[str, str]:
    """VCTK speaker-info.txt: `ID AGE GENDER ACCENTS REGION`, with a header row."""
    genders: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3:
            genders[fields[0]] = fields[2].upper() if fields[2].upper() in ("F", "M") else "U"
    return genders


def clip_entry(path: Path, member: str, sha256: str, size: int) -> dict:
    return {"path": path, "member": member, "sha256": sha256, "bytes": size}


def vctk_sources(vctk_dir: Path) -> list[SpeakerSource]:
    manifest = read_manifest(vctk_dir, "vctk-subset.manifest.json")
    genders = parse_speaker_info(vctk_dir / "speaker-info.txt")
    by_speaker: dict[str, list[dict]] = collections.defaultdict(list)
    for f in manifest["files"]:
        by_speaker[f["speaker"]].append(clip_entry(vctk_dir / f["member"], f["member"], f["sha256"], f["bytes"]))
    return [SpeakerSource(
        item_id=f"vctk-dlg-{speaker}", group=f"vctk:{speaker}", language="en",
        gender_class=genders.get(speaker, "U"), source="vctk", license_spdx="CC-BY-4.0", url=VCTK_DOI,
        evidence_url=VCTK_PAGE,
        attribution=(f"CSTR VCTK Corpus (version 0.92), speaker {speaker}, by Veaux, Yamagishi and MacDonald, "
                     f"University of Edinburgh, CC-BY-4.0, {VCTK_DOI}"),
        clips=clips,
        provenance={"corpus": "CSTR VCTK Corpus 0.92", "archive_url": manifest["archive_url"],
                    "archive_size": manifest["archive_size"], "selection": manifest["selection"]},
    ) for speaker, clips in sorted(by_speaker.items())]


def commonvoice_sources(cv_dir: Path) -> list[SpeakerSource]:
    manifest = read_manifest(cv_dir, "commonvoice-subset.manifest.json")
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for f in manifest["files"]:
        by_group[f["group"]].append(f)
    sources = []
    for group, files in sorted(by_group.items()):
        lang, speaker = files[0]["language"], group.rsplit(":", 1)[-1]
        clips = [clip_entry(cv_dir / f["language"] / "clips" / Path(f["path"]).name, f["member"], f["sha256"], f["bytes"])
                 for f in files]
        sources.append(SpeakerSource(
            item_id=f"cv-dlg-{lang.lower()}-{speaker}", group=group, language=lang,
            gender_class=files[0]["gender_class"], source="commonvoice", license_spdx="CC0-1.0",
            url=CV_DATASET_URLS.get(lang, CV_LISTING_URL), evidence_url=CV_TERMS_URL,
            attribution=f"Mozilla Common Voice Scripted Speech 27.0 ({lang}), CC0-1.0, speaker {speaker}, {CV_LISTING_URL}",
            clips=clips,
            provenance={"corpus": files[0]["corpus"], "archive": files[0]["archive"],
                        "selection": f"seed {manifest['seed']}; {manifest['filter']}"},
        ))
    return sources


def load_clip(clip: dict) -> np.ndarray:
    data = clip["path"].read_bytes()
    if hashlib.sha256(data).hexdigest() != clip["sha256"]:
        raise BuildError(f"{clip['path']}: sha256 differs from the subset manifest")
    return am.load_audio(clip["path"], channels=1)[:, 0]


def concat_clips(clips: list[dict], gap_s: float = GAP_S) -> np.ndarray:
    gap = np.zeros(int(gap_s * am.SR), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for clip in clips:
        if pieces:
            pieces.append(gap)
        pieces.append(load_clip(clip))
    return np.concatenate(pieces)


def write_pcm16_wav(path: Path, mono: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype("<i2")
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as f:
        wavfile.write(f, am.SR, pcm)
    tmp.replace(path)


def build_item(src: SpeakerSource, cache: Path, retrieved_at: str, ffmpeg: str) -> dict:
    item = {
        "id": src.item_id, "source": src.source, "url": src.url, "role": "dialogue", "group": src.group,
        "license_spdx": src.license_spdx, "license_evidence_url": src.evidence_url,
        "attribution_text": src.attribution, "tags": [], "tag_basis": "none", "ext": "wav",
        "retrieved_at": retrieved_at, "language": src.language, "gender_class": src.gender_class,
        "local_only": True,
    }
    mono = concat_clips(src.clips)
    dest = ingest.cache_path(cache, item)
    write_pcm16_wav(dest, mono)
    item["sha256"] = ingest.sha256_file(dest)
    item["provenance"] = {**src.provenance, "transform": TRANSFORM, "ffmpeg": ffmpeg, "duration_s": round(mono.size / am.SR, 3),
                          "members": [{k: c[k] for k in ("member", "sha256", "bytes")} for c in src.clips]}
    problem = ingest.validate_record(item, allow_sharealike=False)
    if problem:
        raise BuildError(f"{src.item_id}: {problem}")
    return item


class BuildSettings(NamedTuple):
    cache: Path
    retrieved_at: str
    workers: int = 4
    log: Callable[[str], None] = lambda _: None


def build_all(sources: list[SpeakerSource], settings: BuildSettings) -> list[dict]:
    ffmpeg = ffmpeg_version()
    # Create the directory before the workers start: resolving paths while another thread is creating the
    # cache directories can make cache_path report an escape on Windows.
    (settings.cache / "dialogue").mkdir(parents=True, exist_ok=True)

    def one(src: SpeakerSource) -> dict:
        item = build_item(src, settings.cache, settings.retrieved_at, ffmpeg)
        settings.log(f"{src.item_id}: {item['provenance']['duration_s']:.0f} s from {len(src.clips)} clips")
        return item

    with ThreadPoolExecutor(max_workers=max(1, settings.workers)) as pool:
        return sorted(pool.map(one, sources), key=lambda i: i["id"])


def write_items(path: Path, items: list[dict]) -> None:
    header = ["# Per-speaker dialogue items built by build_local_items.py from local VCTK and Common Voice subsets.",
              "# local_only: the audio is in the corpus cache already; `url` is provenance, not a download."]
    path.write_text("\n".join(header + [json.dumps(i, sort_keys=True) for i in items]) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vctk-dir", type=Path)
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not (args.vctk_dir or args.cv_dir):
        parser.error("give --vctk-dir, --cv-dir or both")
    try:
        sources = (vctk_sources(args.vctk_dir) if args.vctk_dir else []) + (
            commonvoice_sources(args.cv_dir) if args.cv_dir else [])
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        items = build_all(sources, BuildSettings(args.cache, today, args.workers, lambda m: print(m, file=sys.stderr)))
    except (BuildError, ingest.IngestError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_items(args.out, items)
    print(f"wrote {len(items)} items to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
