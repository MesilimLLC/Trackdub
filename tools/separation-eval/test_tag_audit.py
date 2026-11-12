import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import audiomath as am
import ingest
import tag_audit as ta

COUNTS = {"whisper": 3, "loud": 12, "crowd": 12, "instrumental": 12, "vocals": 12}


def tone(seconds, amplitude=0.3, freq=440.0):
    t = np.arange(int(seconds * am.SR)) / am.SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class AuditTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.items = self.make_items()
        self.manifest = {"items": self.items}

    def tearDown(self):
        self.tmp.cleanup()

    def make_items(self):
        items = []
        for tag, n in COUNTS.items():
            for k in range(n):
                item = {"id": f"{tag}-{k:02d}", "role": "sfx", "tags": [tag], "tag_basis": "search-query", "ext": "wav",
                        "url": "https://example.org/x.wav"}
                path = ingest.cache_path(self.cache, item)
                path.parent.mkdir(parents=True, exist_ok=True)
                am.write_wav(path, tone(2.0, freq=200 + 10 * k).reshape(-1, 1), am.SR)
                items.append(item)
        untagged = {"id": "plain-00", "role": "sfx", "tags": ["loud"], "tag_basis": "none", "ext": "wav", "url": "https://example.org/y.wav"}
        items.append(untagged)
        return items


class SampleTests(AuditTestBase):
    def test_every_whisper_item_and_ten_of_each_other_tag(self):
        counts = {}
        for tag, _ in ta.sample_items(self.items):
            counts[tag] = counts.get(tag, 0) + 1
        self.assertEqual(counts, {"whisper": 3, "loud": 10, "crowd": 10, "instrumental": 10, "vocals": 10})

    def test_items_whose_tags_were_not_from_the_search_query_are_left_out(self):
        self.assertNotIn("plain-00", [item["id"] for _, item in ta.sample_items(self.items)])

    def test_the_sample_is_deterministic_and_not_grouped_by_tag(self):
        first = [item["id"] for _, item in ta.sample_items(self.items)]
        self.assertEqual(first, [item["id"] for _, item in ta.sample_items(self.items)])
        tags = [tag for tag, _ in ta.sample_items(self.items)]
        self.assertNotEqual(tags, sorted(tags))


class SignalTests(unittest.TestCase):
    def test_the_loudest_window_is_chosen(self):
        audio = np.zeros(60 * am.SR, dtype=np.float32)
        audio[40 * am.SR:45 * am.SR] = 0.5
        excerpt = ta.loudest_excerpt(audio, 20.0)
        self.assertEqual(excerpt.shape[0], 20 * am.SR)
        self.assertEqual(int((excerpt > 0.1).sum()), 5 * am.SR)

    def test_a_clip_shorter_than_the_window_is_returned_whole(self):
        audio = tone(3.0)
        self.assertEqual(ta.loudest_excerpt(audio, 20.0).shape, audio.shape)

    def test_rms_in_dbfs(self):
        self.assertAlmostEqual(ta.rms_dbfs(tone(1.0, 0.5)), -9.03, places=1)
        self.assertEqual(ta.rms_dbfs(np.zeros(100, dtype=np.float32)), -np.inf)
        self.assertEqual(ta.rms_dbfs(np.zeros(0, dtype=np.float32)), -np.inf)


class PackTests(AuditTestBase):
    def test_the_pack_has_numbered_pcm16_excerpts_and_a_csv(self):
        out = self.root / "pack"
        rows = ta.build_pack(self.manifest, self.cache, out)
        self.assertEqual(len(rows), 3 + 4 * 10)
        self.assertEqual(rows[0]["file"], "001.wav")
        rate, pcm = wavfile.read(out / "001.wav")
        self.assertEqual((rate, pcm.dtype, pcm.ndim), (am.SR, np.dtype("<i2"), 1))
        with (out / "audit.csv").open(encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))
        self.assertEqual(list(written[0]), ta.FIELDS)
        self.assertTrue(all(r["verdict"] == "" and r["question"] for r in written))

    def test_a_silent_clip_is_flagged(self):
        item = next(i for i in self.items if i["id"] == "whisper-00")
        am.write_wav(ingest.cache_path(self.cache, item), np.zeros((am.SR, 1), dtype=np.float32), am.SR)
        rows = ta.build_pack(self.manifest, self.cache, self.root / "pack")
        flagged = [r for r in rows if r["auto_flag"] == "silent"]
        self.assertEqual([r["item_id"] for r in flagged], ["whisper-00"])
        self.assertEqual(flagged[0]["rms_dbfs"], "-inf")

    def test_an_existing_audit_csv_is_never_overwritten(self):
        out = self.root / "pack"
        out.mkdir()
        (out / "audit.csv").write_text("file,verdict\n001.wav,pass\n", encoding="utf-8")
        with self.assertRaisesRegex(ta.AuditError, "may hold verdicts"):
            ta.build_pack(self.manifest, self.cache, out)
        self.assertIn("pass", (out / "audit.csv").read_text(encoding="utf-8"))

    def test_missing_audio_is_an_error_naming_the_item(self):
        ingest.cache_path(self.cache, self.items[0]).unlink()
        with self.assertRaisesRegex(ta.AuditError, "cannot load audio"):
            ta.build_pack(self.manifest, self.cache, self.root / "pack")


def rows_for(tag, verdicts):
    return [{"file": f"{i:03d}.wav", "item_id": f"{tag}-{i}", "claimed_tag": tag, "question": "q", "rms_dbfs": "-20.0",
             "auto_flag": "", "verdict": v, "notes": ""} for i, v in enumerate(verdicts)]


class ScoreTests(unittest.TestCase):
    def result(self, verdicts):
        return ta.tag_result(rows_for("loud", verdicts))

    def test_one_failure_in_ten_is_within_the_limit(self):
        result = self.result(["pass"] * 9 + ["fail"])
        self.assertEqual((result["fail_rate"], result["status"]), (0.1, "ok"))

    def test_unsure_counts_as_a_failure(self):
        result = self.result(["pass"] * 8 + ["fail", "unsure"])
        self.assertEqual((result["fail_rate"], result["status"]), (0.2, "provisional"))
        self.assertEqual(result["failed_ids"], ["loud-8", "loud-9"])

    def test_blank_verdicts_make_the_tag_incomplete(self):
        result = self.result(["pass"] * 9 + [""])
        self.assertEqual((result["status"], result["blank"]), ("incomplete", 1))

    def test_an_unanswered_tag_has_no_failure_rate(self):
        self.assertEqual(self.result([""] * 3)["fail_rate"], 0.0)

    def test_scores_are_reported_per_tag(self):
        rows = rows_for("loud", ["pass"] * 10) + rows_for("crowd", ["fail"] * 3 + ["pass"] * 7)
        report = ta.score(rows)
        self.assertEqual({t: r["status"] for t, r in report["tags"].items()}, {"crowd": "provisional", "loud": "ok"})

    def test_an_unknown_verdict_is_rejected(self):
        path = Path(tempfile.mkdtemp()) / "a.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ta.FIELDS)
            writer.writeheader()
            writer.writerows(rows_for("loud", ["maybe"]))
        with self.assertRaisesRegex(ta.AuditError, "pass, fail or unsure"):
            ta.read_verdicts(path)

    def test_verdicts_are_case_and_space_insensitive(self):
        self.assertEqual(self.result([" PASS ", "Fail"])["fail"], 1)


class MainTests(AuditTestBase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return ta.main(list(argv)), out.getvalue(), err.getvalue()

    def test_sample_then_score_round_trip(self):
        manifest = self.root / "m.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        pack = self.root / "pack"
        code, out, _ = self.run_main("sample", "--manifest", str(manifest), "--cache", str(self.cache), "--out-dir", str(pack))
        self.assertEqual(code, 0)
        self.assertIn("43 excerpts", out)
        with (pack / "audit.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["verdict"] = "pass"
        with (pack / "audit.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ta.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        report_path = self.root / "tag_audit.json"
        code, out, _ = self.run_main("score", "--csv", str(pack / "audit.csv"), "--out", str(report_path))
        self.assertEqual(code, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual({r["status"] for r in report["tags"].values()}, {"ok"})
        self.assertIn("whisper: ok", out)

    def test_errors_are_reported_without_a_traceback(self):
        code, _, err = self.run_main("score", "--csv", str(self.root / "missing.csv"), "--out", str(self.root / "o.json"))
        self.assertEqual(code, 1)
        self.assertIn("error:", err)


class ReviewRobustnessTests(AuditTestBase):
    def test_sound_in_the_last_fraction_of_a_clip_is_in_the_excerpt(self):
        audio = np.zeros(int(20.25 * am.SR), dtype=np.float32)
        audio[-am.SR // 4:] = 0.5
        self.assertGreater(float(np.abs(ta.loudest_excerpt(audio, 20.0)).max()), 0.4)

    def test_a_replaced_cache_file_is_refused(self):
        item = dict(self.items[0], sha256="0" * 64)
        with self.assertRaisesRegex(ta.AuditError, "does not match the manifest sha256"):
            ta.verified_path(item, self.cache)

    def test_a_matching_hash_is_accepted(self):
        item = dict(self.items[0])
        item["sha256"] = ingest.sha256_file(ingest.cache_path(self.cache, item))
        self.assertEqual(ta.verified_path(item, self.cache), ingest.cache_path(self.cache, item))


class CsvShapeTests(AuditTestBase):
    def write_csv(self, rows, name="audit.csv"):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ta.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_a_row_cut_short_is_an_error_naming_its_line(self):
        path = self.root / "short.csv"
        path.write_text(",".join(ta.FIELDS) + "\n001.wav,x,loud,q,-20,\n", encoding="utf-8")
        with self.assertRaisesRegex(ta.AuditError, "line 2 is missing a column"):
            ta.read_verdicts(path)

    def test_an_empty_csv_is_an_error(self):
        with self.assertRaisesRegex(ta.AuditError, "no rows"):
            ta.read_verdicts(self.write_csv([]))

    def test_deleted_rows_are_detected_when_the_pack_list_is_present(self):
        rows = rows_for("loud", ["pass"] * 3)
        path = self.write_csv(rows[:2])
        (self.root / ta.PACK_LIST).write_text(json.dumps({"files": [r["file"] for r in rows]}), encoding="utf-8")
        with self.assertRaisesRegex(ta.AuditError, "missing"):
            ta.read_verdicts(path)

    def test_a_complete_csv_matches_its_pack_list(self):
        rows = rows_for("loud", ["pass"] * 3)
        path = self.write_csv(rows)
        (self.root / ta.PACK_LIST).write_text(json.dumps({"files": [r["file"] for r in rows]}), encoding="utf-8")
        self.assertEqual(len(ta.read_verdicts(path)), 3)

    def test_sample_writes_the_pack_list(self):
        out = self.root / "pack"
        rows = ta.build_pack(self.manifest, self.cache, out)
        listed = json.loads((out / ta.PACK_LIST).read_text(encoding="utf-8"))["files"]
        self.assertEqual(listed, [r["file"] for r in rows])

    def test_score_refuses_to_overwrite_the_verdict_csv(self):
        path = self.write_csv(rows_for("loud", ["pass"]))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = ta.main(["score", "--csv", str(path), "--out", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("overwrite the verdicts", err.getvalue())
        self.assertIn("file,item_id", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
