import contextlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import build_rir_items as bri
import ingest

RATE = 16000
ROOT = "simulated_rirs_16k"
# (room set, room, RT60 in seconds): only the ones inside 0.45 to 1.9 s should be chosen.
ROOMS = [
    ("smallroom", "Room001", 0.1), ("smallroom", "Room002", 0.6), ("smallroom", "Room003", 0.8),
    ("mediumroom", "Room001", 1.0), ("mediumroom", "Room002", 0.3), ("mediumroom", "Room003", 1.2),
    ("largeroom", "Room001", 2.5), ("largeroom", "Room002", 1.5),
]
EXPECTED_ROOMS = {("smallroom", "Room002"), ("smallroom", "Room003"), ("mediumroom", "Room001"),
                  ("mediumroom", "Room003"), ("largeroom", "Room002")}


def rir_wav(rt60, seed):
    rng = np.random.default_rng(seed)
    n = int(rt60 * 1.2 * RATE)
    t = np.arange(n) / RATE
    h = (rng.standard_normal(n) * np.exp(-6.9 * t / rt60)).astype(np.float32)
    h[0] = 8.0
    buffer = io.BytesIO()
    wavfile.write(buffer, RATE, h / 8.0)
    return buffer.getvalue()


def make_zip(path: Path, rirs_per_room=3):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{ROOT}/README", "simulated RIRs")
        for k, (room_set, room, rt60) in enumerate(ROOMS):
            for n in range(1, rirs_per_room + 1):
                zf.writestr(f"{ROOT}/{room_set}/{room}/{room}-{n:05d}.wav", rir_wav(rt60, 100 * k + n))


class RirItemTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.zip = self.root / "sim_rir_16k.zip"
        self.cache = self.root / "cache"
        make_zip(self.zip)

    def tearDown(self):
        self.tmp.cleanup()


class SelectionTests(RirItemTestBase):
    def choices(self, per_set=5):
        with zipfile.ZipFile(self.zip) as zf:
            return bri.choose_rooms(zf, per_set)

    def test_rooms_are_grouped_by_set_and_the_readme_is_ignored(self):
        with zipfile.ZipFile(self.zip) as zf:
            rooms = bri.rooms_in(zf)
        self.assertEqual(len(rooms), len(ROOMS))
        self.assertEqual(rooms[("smallroom", "Room002")][0], f"{ROOT}/smallroom/Room002/Room002-00001.wav")
        self.assertTrue(all(len(members) == 3 for members in rooms.values()))

    def test_only_rooms_with_both_probes_in_the_rt60_range_are_chosen(self):
        chosen = {(c.room_set, c.room) for c in self.choices()}
        self.assertEqual(chosen, EXPECTED_ROOMS)
        for c in self.choices():
            self.assertTrue(all(bri.RT60_RANGE[0] <= t <= bri.RT60_RANGE[1] for t in c.rt60_s))
            self.assertEqual(len(c.members), bri.PROBES_PER_ROOM)

    def test_the_quota_applies_to_each_set(self):
        per_set = [c.room_set for c in self.choices(per_set=1)]
        self.assertEqual(sorted(per_set), ["largeroom", "mediumroom", "smallroom"])

    def test_the_choice_is_deterministic(self):
        self.assertEqual(self.choices(per_set=1), self.choices(per_set=1))

    def test_an_archive_without_rirs_is_an_error(self):
        empty = self.root / "empty.zip"
        with zipfile.ZipFile(empty, "w") as zf:
            zf.writestr("notes.txt", "x")
        with zipfile.ZipFile(empty) as zf, self.assertRaisesRegex(bri.BuildError, "no simulated_rirs_16k"):
            bri.choose_rooms(zf)


class ItemTests(RirItemTestBase):
    def items(self):
        return bri.build_items(self.zip, self.cache, 5, "2026-11-11")

    def test_each_chosen_room_gives_its_first_rirs_as_items(self):
        items = self.items()
        self.assertEqual(len(items), len(EXPECTED_ROOMS) * bri.PROBES_PER_ROOM)
        self.assertEqual([i["id"] for i in items], sorted(i["id"] for i in items))
        first = next(i for i in items if i["group"] == "slr26:smallroom:room002")
        self.assertEqual(first["id"], "slr26-small-room002-00001")
        self.assertEqual(first["provenance"]["member"], f"{ROOT}/smallroom/Room002/Room002-00001.wav")

    def test_items_record_license_band_limit_and_archive(self):
        item = self.items()[0]
        self.assertEqual((item["role"], item["license_spdx"], item["band_limit_hz"], item["local_only"]),
                         ("rir", "Apache-2.0", 8000, True))
        self.assertEqual(item["provenance"]["archive_sha256"], ingest.sha256_file(self.zip))
        self.assertEqual(item["provenance"]["sample_rate"], 16000)
        self.assertTrue(bri.RT60_RANGE[0] <= item["provenance"]["rt60_s"] <= bri.RT60_RANGE[1])

    def test_a_room_is_one_group_so_it_cannot_span_both_splits(self):
        by_group = {}
        for item in self.items():
            by_group.setdefault(item["group"], set()).add(ingest.assign_split(item["group"]))
        self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))
        self.assertEqual(len(by_group), len(EXPECTED_ROOMS))

    def test_the_cached_files_match_their_hashes_and_ingest_without_downloading(self):
        items = self.items()
        for item in items:
            self.assertEqual(ingest.sha256_file(ingest.cache_path(self.cache, item)), item["sha256"])

        def refuse(url, dest, range_bytes=None):
            raise AssertionError(f"unexpected download of {url}")

        accepted, rejected = ingest.ingest_items(items, self.cache, fetch=refuse)
        self.assertEqual((len(accepted), rejected), (len(items), []))
        self.assertEqual({a["band_limit_hz"] for a in accepted}, {8000})
        self.assertEqual(ingest.verify_manifest(ingest.build_manifest(accepted), self.cache), [])

    def test_a_file_that_is_not_a_zip_is_an_error(self):
        bad = self.root / "bad.zip"
        bad.write_bytes(b"not a zip")
        with self.assertRaisesRegex(bri.BuildError, "cannot open"):
            bri.build_items(bad, self.cache, 5, "2026-11-11")


class MainTests(RirItemTestBase):
    def test_main_writes_the_items_file(self):
        out = self.root / "items.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            code = bri.main(["--zip", str(self.zip), "--cache", str(self.cache), "--out", str(out), "--rooms-per-set", "5"])
        self.assertEqual(code, 0)
        self.assertEqual(len(ingest.read_jsonl(out)), len(EXPECTED_ROOMS) * bri.PROBES_PER_ROOM)

    def test_main_reports_a_missing_archive(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = bri.main(["--zip", str(self.root / "missing.zip"), "--cache", str(self.cache), "--out", str(self.root / "x.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("error:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
