#!/usr/bin/env python3
"""Flag corpus items whose Freesound sound is in FSD50K, which Divide-and-Remaster v3 drew its effects from.

Reads the ids from FSD50K.ground_truth.zip (dev.csv and eval.csv, whose `fname` column is the Freesound id)
and matches them against the `freesound_id` of each manifest item. The result is an exposure upper bound for
DnR-trained models, kept apart from the manifest so its hash does not change. For discovery, pass the same
CSVs to `freesound.py discover --exclude-ids`.

    python fsd50k_overlap.py --manifest items.v1.1.manifest.json \\
        --fsd50k-zip D:/cache/fsd50k/FSD50K.ground_truth.zip --out D:/cache/contamination/fsd50k_overlap.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

CSV_MEMBERS = {"FSD50K.ground_truth/dev.csv": "dev", "FSD50K.ground_truth/eval.csv": "eval"}


class OverlapError(Exception):
    pass


def ids_in(zf: zipfile.ZipFile, member: str) -> list[int]:
    """The numeric `fname` values of one CSV in the archive."""
    try:
        text = io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="")
    except KeyError as exc:
        raise OverlapError(f"{zf.filename} has no {member}") from exc
    return [int(row["fname"]) for row in csv.DictReader(text) if str(row.get("fname", "")).isdigit()]


def read_fsd50k_ids(zip_path: Path) -> dict[int, str]:
    """Freesound id -> FSD50K set ('dev' or 'eval')."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise OverlapError(f"cannot open {zip_path}: {exc}") from exc
    with zf:
        ids = {i: fsd_set for member, fsd_set in CSV_MEMBERS.items() for i in ids_in(zf, member)}
    if not ids:
        raise OverlapError(f"no Freesound ids found in {zip_path}")
    return ids


def flag_items(items: list[dict], ids: dict[int, str]) -> list[dict]:
    """The items whose `freesound_id` is in FSD50K, with the fields a report needs."""
    return [{"id": i["id"], "freesound_id": i["freesound_id"], "fsd50k_set": ids[i["freesound_id"]], "role": i["role"],
             "tags": i.get("tags", []), "split": i.get("split")}
            for i in items if i.get("freesound_id") in ids]


def count_by(rows: list[dict], key) -> dict[str, int]:
    return dict(sorted(collections.Counter(key(r) for r in rows).items()))


def summarize(items: list[dict], flagged: list[dict]) -> dict:
    sounds = [i for i in items if "freesound_id" in i]
    return {
        "freesound_items": len(sounds), "in_fsd50k": len(flagged),
        "share_in_fsd50k": round(len(flagged) / len(sounds), 4) if sounds else 0.0,
        "by_role": count_by(flagged, lambda r: r["role"]),
        "by_tag": count_by([dict(r, tag=t) for r in flagged for t in (r["tags"] or ["(none)"])], lambda r: r["tag"]),
        "by_split": count_by(flagged, lambda r: str(r["split"])),
        "by_fsd50k_set": count_by(flagged, lambda r: r["fsd50k_set"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fsd50k-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        ids = read_fsd50k_ids(args.fsd50k_zip)
    except (OSError, json.JSONDecodeError, OverlapError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    flagged = flag_items(manifest["items"], ids)
    report = {"items_sha256": manifest.get("items_sha256"), "fsd50k_ids": len(ids),
              "summary": summarize(manifest["items"], flagged), "items": flagged}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    s = report["summary"]
    print(f"{s['in_fsd50k']} of {s['freesound_items']} Freesound items are in FSD50K ({s['share_in_fsd50k']:.1%}); report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
