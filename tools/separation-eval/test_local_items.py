import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import wavfile

import audiomath as am
import build_local_items as bli
import ingest


def refuse_download(url, dest, range_bytes=None):
    raise AssertionError(f"unexpected download of {url}")


class LocalItemTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vctk, self.cv, self.cache = self.root / "vctk", self.root / "commonvoice", self.root / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def clip_file(self, path, freq, seconds=1.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(seconds * am.SR)) / am.SR
        am.write_wav(path, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(-1, 1), am.SR)
        return ingest.sha256_file(path), path.stat().st_size

    def make_vctk(self, speakers=("p225", "p226"), clips=3):
        files = []
        for k, speaker in enumerate(speakers):
            for n in range(clips):
                member = f"wav48_silence_trimmed/{speaker}/{speaker}_{n:03d}_mic1.wav"
                sha, size = self.clip_file(self.vctk / member, 300 + 50 * k + 7 * n)
                files.append({"member": member, "speaker": speaker, "bytes": size, "crc32": 0, "sha256": sha, "zip_offset": 0})
        manifest = {"archive_url": "https://example.org/vctk.zip", "archive_size": 1, "selection": "first mic1", "files": files}
        (self.vctk / "vctk-subset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.vctk / "speaker-info.txt").write_text(
            "ID  AGE  GENDER  ACCENTS  REGION\np225  23  F  English  Southern England\np226  22  M  English  Surrey\n",
            encoding="utf-8")

    def make_cv(self, languages=("ar", "pt"), speakers=2, clips=3):
        files = []
        for li, lang in enumerate(languages):
            for s in range(speakers):
                group = f"commonvoice:{lang}:{s:016x}"
                for n in range(clips):
                    name = f"common_voice_{lang}_{s}{n}.wav"
                    sha, size = self.clip_file(self.cv / lang / "clips" / name, 200 + 40 * li + 9 * s + n)
                    files.append({"language": lang, "path": f"commonvoice/{lang}/clips/{name}", "group": group, "split": "dev",
                                  "gender_class": "F", "bytes": size, "sha256": sha, "duration_s": 1.0,
                                  "archive": f"cv-{lang}.tar.gz", "member": f"cv/{lang}/clips/{name}",
                                  "corpus": "cv-corpus-27.0-2026-09-11"})
        manifest = {"seed": "s1", "filter": "up>=2", "files": files}
        (self.cv / "commonvoice-subset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class SourceTests(LocalItemTestBase):
    def test_vctk_gives_one_source_per_speaker_with_gender_and_group(self):
        self.make_vctk()
        sources = bli.vctk_sources(self.vctk)
        self.assertEqual([s.item_id for s in sources], ["vctk-dlg-p225", "vctk-dlg-p226"])
        self.assertEqual([s.group for s in sources], ["vctk:p225", "vctk:p226"])
        self.assertEqual([s.gender_class for s in sources], ["F", "M"])
        self.assertEqual({s.language for s in sources}, {"en"})
        self.assertEqual([len(s.clips) for s in sources], [3, 3])

    def test_common_voice_gives_one_source_per_speaker_group(self):
        self.make_cv()
        sources = bli.commonvoice_sources(self.cv)
        self.assertEqual(len(sources), 4)
        self.assertTrue(all(s.item_id.startswith("cv-dlg-") and s.license_spdx == "CC0-1.0" for s in sources))
        self.assertEqual(sorted({s.language for s in sources}), ["ar", "pt"])
        self.assertEqual(sources[0].group, "commonvoice:ar:0000000000000000")

    def test_unknown_or_missing_gender_is_u(self):
        path = self.root / "info.txt"
        path.write_text("ID AGE GENDER\np1 20 F\np2 30 X\np3 40\n", encoding="utf-8")
        self.assertEqual(bli.parse_speaker_info(path), {"p1": "F", "p2": "U"})

    def test_a_missing_manifest_is_a_build_error(self):
        with self.assertRaisesRegex(bli.BuildError, "cannot read"):
            bli.vctk_sources(self.vctk)


class BuildTests(LocalItemTestBase):
    def source(self):
        self.make_vctk()
        return bli.vctk_sources(self.vctk)[0]

    def test_clips_are_joined_with_the_gap_between_them(self):
        audio = bli.concat_clips(self.source().clips)
        self.assertEqual(audio.shape[0], 3 * am.SR + 2 * int(bli.GAP_S * am.SR))

    def test_a_changed_clip_is_refused(self):
        src = self.source()
        self.clip_file(src.clips[1]["path"], 999)
        with self.assertRaisesRegex(bli.BuildError, "sha256 differs"):
            bli.concat_clips(src.clips)

    def test_item_is_a_pcm16_wav_in_the_cache_with_provenance(self):
        src = self.source()
        item = bli.build_item(src, self.cache, "2026-11-11", "ffmpeg test")
        path = ingest.cache_path(self.cache, item)
        rate, pcm = wavfile.read(path)
        self.assertEqual((rate, pcm.dtype, pcm.ndim), (am.SR, np.dtype("<i2"), 1))
        self.assertEqual(item["sha256"], ingest.sha256_file(path))
        self.assertTrue(item["local_only"] and item["role"] == "dialogue" and item["language"] == "en")
        self.assertEqual([m["sha256"] for m in item["provenance"]["members"]], [c["sha256"] for c in src.clips])
        self.assertEqual(item["provenance"]["ffmpeg"], "ffmpeg test")
        self.assertAlmostEqual(item["provenance"]["duration_s"], 3.5, places=3)

    def test_building_twice_gives_identical_bytes(self):
        src = self.source()
        first = bli.build_item(src, self.cache, "2026-11-11", "ffmpeg test")["sha256"]
        second = bli.build_item(src, self.cache, "2026-11-11", "ffmpeg test")["sha256"]
        self.assertEqual(first, second)

    def test_build_all_is_sorted_whatever_the_worker_count(self):
        self.make_vctk()
        self.make_cv()
        sources = bli.commonvoice_sources(self.cv) + bli.vctk_sources(self.vctk)
        with mock.patch.object(bli, "ffmpeg_version", return_value="ffmpeg test"):
            items = bli.build_all(sources, bli.BuildSettings(self.cache, "2026-11-11", workers=3))
        self.assertEqual([i["id"] for i in items], sorted(i["id"] for i in items))
        self.assertEqual(len(items), 6)

    def test_ffmpeg_is_required(self):
        with mock.patch.object(am, "have_ffmpeg", return_value=False), self.assertRaisesRegex(bli.BuildError, "ffmpeg"):
            bli.ffmpeg_version()


class IngestTests(LocalItemTestBase):
    def built_items(self):
        self.make_vctk()
        self.make_cv(languages=("ar",))
        sources = bli.vctk_sources(self.vctk) + bli.commonvoice_sources(self.cv)
        with mock.patch.object(bli, "ffmpeg_version", return_value="ffmpeg test"):
            return bli.build_all(sources, bli.BuildSettings(self.cache, "2026-11-11", workers=2))

    def test_cached_local_items_are_accepted_without_any_download(self):
        accepted, rejected = ingest.ingest_items(self.built_items(), self.cache, fetch=refuse_download)
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 4)
        for item in accepted:
            self.assertEqual(item["split"], ingest.assign_split(item["group"]))
            self.assertTrue(item["local_only"])
            self.assertIn("language", item)
            self.assertIn("members", item["provenance"])
        self.assertEqual(ingest.verify_manifest(ingest.build_manifest(accepted), self.cache), [])

    def test_a_local_item_missing_from_the_cache_is_rejected_not_downloaded(self):
        items = self.built_items()
        ingest.cache_path(self.cache, items[0]).unlink()
        accepted, rejected = ingest.ingest_items(items, self.cache, fetch=refuse_download)
        self.assertEqual(len(accepted), 3)
        self.assertEqual([r.item_id for r in rejected], [items[0]["id"]])
        self.assertIn("local-only item is missing from the cache", rejected[0].reason)

    def test_a_corrupted_cached_local_item_is_rejected_by_its_hash(self):
        items = self.built_items()
        ingest.cache_path(self.cache, items[1]).write_bytes(b"not the audio")
        accepted, rejected = ingest.ingest_items(items, self.cache, fetch=refuse_download)
        self.assertEqual(len(accepted), 3)
        self.assertEqual([r.item_id for r in rejected], [items[1]["id"]])
        self.assertIn("local-only item differs from its recorded sha256", rejected[0].reason)

    def test_records_that_are_not_objects_do_not_crash_the_local_check(self):
        kept, rejected = ingest.split_unusable_local(["junk", {"local_only": True}], self.cache)
        self.assertEqual(kept, ["junk"])
        self.assertEqual([r.item_id for r in rejected], ["<missing>"])
        self.assertIn("no usable cache location", rejected[0].reason)

    def test_items_that_are_not_local_only_are_left_alone(self):
        records = [{"id": "x-dlg-000000000001", "role": "dialogue", "url": "https://x/a.wav"}]
        kept, rejected = ingest.split_unusable_local(records, self.cache)
        self.assertEqual((kept, rejected), (records, []))


class MainTests(LocalItemTestBase):
    def run_main(self, *argv):
        err = io.StringIO()
        with mock.patch.object(bli, "ffmpeg_version", return_value="ffmpeg test"), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            return bli.main(list(argv)), err.getvalue()

    def test_main_writes_sorted_items(self):
        self.make_vctk()
        self.make_cv()
        out = self.root / "items.jsonl"
        code, _ = self.run_main("--vctk-dir", str(self.vctk), "--cv-dir", str(self.cv), "--cache", str(self.cache), "--out", str(out))
        self.assertEqual(code, 0)
        items = ingest.read_jsonl(out)
        self.assertEqual(len(items), 6)
        self.assertEqual([i["id"] for i in items], sorted(i["id"] for i in items))

    def test_main_reports_a_missing_manifest(self):
        code, err = self.run_main("--vctk-dir", str(self.vctk), "--cache", str(self.cache), "--out", str(self.root / "x.jsonl"))
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)

    def test_main_needs_at_least_one_source(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            bli.main(["--cache", str(self.cache), "--out", str(self.root / "x.jsonl")])


if __name__ == "__main__":
    unittest.main()
