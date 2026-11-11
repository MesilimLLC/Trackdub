#!/usr/bin/env python3
"""Pick room impulse responses from the OpenSLR SLR26 simulated RIR archive and write them as corpus items.

The archive has 3 room-size sets of 200 rooms with 100 RIRs each, all 16 kHz. Rooms are drawn in a seeded
order; a room is kept only if every probed RIR has a measured RT60 inside the A3 range, and the first RIRs of
each kept room become items. A room is one group, so it never spans dev and test. Items are marked
`local_only` and carry `band_limit_hz` because a 16 kHz RIR has nothing above 8 kHz. Run with the archive
you downloaded yourself; its sha256 is recorded in every item.

    python build_rir_items.py --zip D:/cache/slr26/sim_rir_16k.zip --cache D:/cache/corpus-cache \\
        --out D:/cache/items.slr26-rir.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import audiomath as am
import ingest

ARCHIVE_URL = "https://openslr.trmal.net/resources/26/sim_rir_16k.zip"
PAGE_URL = "https://www.openslr.org/26/"
CITATION = ("Simulated Room Impulse Response Database (OpenSLR SLR26), Ko, Peddinti, Povey, Seltzer and "
            "Khudanpur, ICASSP 2017, Apache-2.0")
SET_NAMES = {"smallroom": "small", "mediumroom": "medium", "largeroom": "large"}
MEMBER = re.compile(r"^simulated_rirs_16k/(smallroom|mediumroom|largeroom)/(Room\d+)/([^/]+\.wav)$")
# Inside the 0.4 to 2.0 s range that A3 draws on, so a small estimator difference cannot push an item out.
RT60_RANGE = (0.45, 1.9)
PROBES_PER_ROOM = 2
ROOMS_PER_SET = 20
SEED = "slr26-v1.1-seed1"
BAND_LIMIT_HZ = 8000


class BuildError(Exception):
    pass


class Choice(NamedTuple):
    room_set: str
    room: str
    members: list[str]
    rt60_s: list[float]


def rooms_in(zf: zipfile.ZipFile) -> dict[tuple[str, str], list[str]]:
    """RIR members of the archive by (room set, room), in file-name order."""
    rooms: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for name in zf.namelist():
        match = MEMBER.match(name)
        if match:
            rooms[(match.group(1), match.group(2))].append(name)
    return {key: sorted(members) for key, members in rooms.items()}


def measure_rt60(zf: zipfile.ZipFile, member: str) -> float:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rir.wav"
        path.write_bytes(zf.read(member))
        return float(am.estimate_rt60(am.load_audio(path, channels=1)))


def in_range(rt60: float) -> bool:
    return not math.isnan(rt60) and RT60_RANGE[0] <= rt60 <= RT60_RANGE[1]


def room_order(rooms: dict[tuple[str, str], list[str]], room_set: str) -> list[str]:
    names = sorted(room for (s, room) in rooms if s == room_set)
    return sorted(names, key=lambda room: hashlib.sha256(f"{SEED}:{room_set}:{room}".encode()).hexdigest())


def choose_set(zf: zipfile.ZipFile, rooms: dict[tuple[str, str], list[str]], room_set: str, quota: int) -> list[Choice]:
    chosen: list[Choice] = []
    for room in room_order(rooms, room_set):
        if len(chosen) >= quota:
            break
        members = rooms[(room_set, room)][:PROBES_PER_ROOM]
        times = [measure_rt60(zf, m) for m in members]
        if len(members) == PROBES_PER_ROOM and all(in_range(t) for t in times):
            chosen.append(Choice(room_set, room, members, times))
    return chosen


def choose_rooms(zf: zipfile.ZipFile, rooms_per_set: int = ROOMS_PER_SET) -> list[Choice]:
    rooms = rooms_in(zf)
    if not rooms:
        raise BuildError("no simulated_rirs_16k/<set>/RoomNNN/*.wav members found in the archive")
    return [c for room_set in SET_NAMES for c in choose_set(zf, rooms, room_set, rooms_per_set)]


class BuildContext(NamedTuple):
    zf: zipfile.ZipFile
    cache: Path
    archive: dict
    retrieved_at: str


def item_for(choice: Choice, index: int, ctx: BuildContext) -> dict:
    member = choice.members[index]
    item = {
        "id": f"slr26-{SET_NAMES[choice.room_set]}-{choice.room.lower()}-{Path(member).stem[-5:]}",
        "source": "openslr-slr26", "url": PAGE_URL, "role": "rir", "group": f"slr26:{choice.room_set}:{choice.room.lower()}",
        "license_spdx": "Apache-2.0", "license_evidence_url": PAGE_URL, "attribution_text": CITATION,
        "tags": [], "tag_basis": "none", "ext": "wav", "retrieved_at": ctx.retrieved_at, "local_only": True,
        "band_limit_hz": BAND_LIMIT_HZ,
    }
    dest = ingest.cache_path(ctx.cache, item)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(ctx.zf.read(member))
    tmp.replace(dest)
    item["sha256"] = ingest.sha256_file(dest)
    item["provenance"] = {**ctx.archive, "member": member, "room_set": choice.room_set, "room": choice.room,
                          "rt60_s": round(choice.rt60_s[index], 3), "sample_rate": 16000,
                          "simulator": "image method, per the archive README"}
    problem = ingest.validate_record(item, allow_sharealike=False)
    if problem:
        raise BuildError(f"{item['id']}: {problem}")
    return item


def build_items(zip_path: Path, cache: Path, rooms_per_set: int, retrieved_at: str) -> list[dict]:
    archive = {"archive_url": ARCHIVE_URL, "archive_sha256": ingest.sha256_file(zip_path),
               "archive_size": zip_path.stat().st_size}
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BuildError(f"cannot open {zip_path}: {exc}") from exc
    with zf:
        ctx = BuildContext(zf, cache, archive, retrieved_at)
        items = [item_for(choice, i, ctx) for choice in choose_rooms(zf, rooms_per_set) for i in range(PROBES_PER_ROOM)]
    return sorted(items, key=lambda item: item["id"])


def write_items(path: Path, items: list[dict]) -> None:
    header = ["# Room impulse responses from OpenSLR SLR26 (simulated, 16 kHz), built by build_rir_items.py.",
              "# local_only: the audio is in the corpus cache already; `url` is provenance, not a download."]
    path.write_text("\n".join(header + [json.dumps(i, sort_keys=True) for i in items]) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rooms-per-set", type=int, default=ROOMS_PER_SET)
    args = parser.parse_args(argv)
    try:
        items = build_items(args.zip, args.cache, args.rooms_per_set, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    except (BuildError, ingest.IngestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_items(args.out, items)
    per_set = collections.Counter(i["provenance"]["room_set"] for i in items)
    print(f"wrote {len(items)} items from {len({i['group'] for i in items})} rooms ({dict(per_set)}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
