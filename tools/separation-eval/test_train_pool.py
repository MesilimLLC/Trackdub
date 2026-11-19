import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import ingest
import train_pool


def sha(n: int) -> str:
    return f"{n:064x}"


def rec(item_id: str, n: int, group: str, **over) -> dict:
    base = {"id": item_id, "source": "vctk", "url": "https://example.org/a.wav", "role": "dialogue", "group": group,
            "license_spdx": "CC-BY-4.0", "license_evidence_url": "https://example.org/license",
            "attribution_text": "x", "sha256": sha(n)}
    base.update(over)
    return base


EVAL = [
    rec("vctk-dlg-p225", 1, "vctk:p225"),
    rec("fsorig-sfx-102096", 2, "freesound:moxobna", source="freesound", role="sfx", freesound_id=102096,
        license_spdx="CC0-1.0"),
    rec("jamendo-mus-aaa", 3, "jamendo:tuune", source="openverse:jamendo", role="music", license_spdx="CC-BY-3.0",
        url="https://prod-1.storage.jamendo.com/?trackid=685596&format=mp32"),
]
DROPPED = [rec("fsorig-dia-763980", 4, "freesound:shangusburger", source="freesound", freesound_id=763980,
               license_spdx="CC0-1.0", excluded_from_training=True)]


def check(candidates, excluded=DROPPED):
    manifest = ingest.build_manifest(EVAL)
    return train_pool.check_candidates(candidates, train_pool.load_eval_identity(manifest, excluded))


def codes(verdict) -> list[str]:
    return [r.split(":", 1)[0] for r in verdict.reasons]


class OverlapTests(unittest.TestCase):
    def test_clean_candidate_is_accepted(self):
        (v,) = check([rec("vctk-dlg-p226", 10, "vctk:p226")])
        self.assertEqual(v.reasons, [])

    def test_each_kind_of_eval_identity_is_rejected(self):
        cases = {
            "eval-id": rec("vctk-dlg-p225", 11, "vctk:p999"),
            "eval-audio": rec("vctk-dlg-p227", 1, "vctk:p227"),
            "eval-group": rec("vctk-dlg-p225-b", 12, "VCTK:P225"),
            "eval-upstream": rec("fs-sfx-x", 13, "freesound:someoneelse", source="freesound", freesound_id=102096,
                                 license_spdx="CC0-1.0"),
        }
        for code, candidate in cases.items():
            with self.subTest(code=code):
                (v,) = check([candidate])
                self.assertIn(code, codes(v))

    def test_jamendo_track_id_matches_across_urls(self):
        candidate = rec("jam-x", 14, "jamendo:other", source="openverse:jamendo", role="music",
                        license_spdx="CC-BY-3.0", url="https://mp3d.jamendo.com/download/track/685596/mp32/")
        (v,) = check([candidate])
        self.assertIn("eval-upstream", codes(v))
        self.assertEqual(train_pool.jamendo_track_id("https://example.org/?trackid=685596"), None)
        self.assertEqual(train_pool.jamendo_track_id("https://notjamendo.com/?trackid=685596"), None)
        self.assertEqual(train_pool.jamendo_track_id("https://JAMENDO.com/track/685596"), "685596")

    def test_excluded_records_count_as_eval(self):
        (v,) = check([rec("fs-dia-y", 15, "freesound:ShangusBurger", source="freesound", license_spdx="CC0-1.0")])
        self.assertIn("eval-group", codes(v))
        (v,) = check([rec("fs-dia-y", 15, "freesound:shangusburger", source="freesound", license_spdx="CC0-1.0")],
                     excluded=[])
        self.assertEqual(v.reasons, [])

    def test_freesound_id_in_a_url_matches_an_eval_sound(self):
        for url in ("https://freesound.org/people/someone/sounds/102096/",
                    "https://freesound.org/apiv2/sounds/102096/download/",
                    "https://freesound.org/s/102096/",
                    "https://cdn.freesound.org/previews/102/102096_5121236-hq.mp3"):
            with self.subTest(url=url):
                (v,) = check([rec("fs-sfx-z", 16, "freesound:other", source="freesound", url=url,
                                  license_spdx="CC0-1.0")])
                self.assertIn("eval-upstream", codes(v))
        self.assertIsNone(train_pool.freesound_id_in_url("https://notfreesound.org/s/102096/"))
        self.assertIsNone(train_pool.freesound_id_in_url("https://freesound.org/people/someone/"))

    def test_eval_record_with_unreadable_freesound_id_does_not_crash(self):
        identity = train_pool.load_eval_identity(ingest.build_manifest(EVAL), [{"id": "x-1", "freesound_id": "n/a"}])
        self.assertIn("x-1", identity.ids)


class RecordTests(unittest.TestCase):
    def test_record_problems(self):
        cases = {
            "invalid": rec("vctk-dlg-p230", 20, "vctk:p230", license_spdx="CC-BY-SA-4.0"),
            "no-sha256": rec("vctk-dlg-p231", 21, "vctk:p231", sha256=None),
            "bad-group": rec("vctk-dlg-p232", 22, "p232"),
            "marked-excluded": rec("vctk-dlg-p233", 23, "vctk:p233", excluded_from_training=True),
        }
        for code, candidate in cases.items():
            with self.subTest(code=code):
                (v,) = check([candidate])
                self.assertEqual(codes(v), [code])

    def test_malformed_freesound_id_is_rejected_without_stopping_the_batch(self):
        bad, good = check([rec("fs-sfx-b", 26, "freesound:b", source="freesound", freesound_id="unknown",
                               license_spdx="CC0-1.0"),
                           rec("fs-sfx-c", 27, "freesound:c", source="freesound", freesound_id="00777",
                               license_spdx="CC0-1.0")])
        self.assertEqual(codes(bad), ["bad-freesound-id"])
        self.assertEqual(good.reasons, [])
        self.assertEqual(train_pool.upstream_keys(good.record), {"freesound:777"})

    def test_missing_group_is_a_clean_rejection(self):
        candidate = rec("vctk-dlg-p235", 25, "vctk:p235")
        del candidate["group"]
        (v,) = check([candidate])
        self.assertEqual(codes(v), ["invalid"])

    def test_noncommercial_licence_is_rejected(self):
        (v,) = check([rec("vctk-dlg-p234", 24, "vctk:p234", license_spdx="CC-BY-NC-4.0")])
        self.assertEqual(codes(v), ["invalid"])

    def test_duplicates_within_the_pool_keep_the_first(self):
        first, same_audio, same_id = check([rec("vctk-dlg-p240", 30, "vctk:p240"),
                                            rec("vctk-dlg-p241", 30, "vctk:p241"),
                                            rec("vctk-dlg-p240", 31, "vctk:p242")])
        self.assertEqual(first.reasons, [])
        self.assertEqual(codes(same_audio), ["pool-duplicate-audio"])
        self.assertEqual(codes(same_id), ["pool-duplicate-id"])

    def test_rejected_candidate_does_not_block_a_later_one(self):
        bad, good = check([rec("vctk-dlg-p250", 40, "vctk:p225"), rec("vctk-dlg-p251", 40, "vctk:p251")])
        self.assertEqual(codes(bad), ["eval-group"])
        self.assertEqual(good.reasons, [])

    def test_librivox_is_accepted_with_a_reader_warning(self):
        (v,) = check([rec("librivox-dlg-new-book", 50, "librivox:new_book", source="librivox")])
        self.assertEqual(v.reasons, [])
        self.assertEqual(len(v.warnings), 1)


class CliTests(unittest.TestCase):
    def test_check_writes_pool_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "items.v1.2.manifest.json"
            manifest.write_text(json.dumps(ingest.build_manifest(EVAL)), encoding="utf-8")
            dropped, cands = root / "dropped.jsonl", root / "cand.jsonl"
            dropped.write_text("# dropped\n" + json.dumps(DROPPED[0]) + "\n", encoding="utf-8")
            cands.write_text("\n".join(json.dumps(r) for r in [rec("vctk-dlg-p300", 60, "vctk:p300"),
                                                                rec("vctk-dlg-p301", 61, "vctk:p225")]),
                             encoding="utf-8")
            out, report_path = root / "items.train.jsonl", root / "report.json"
            train_pool.main(["check", "--eval-manifest", str(manifest), "--candidates", str(cands),
                             "--exclude-records", str(dropped), "--out", str(out), "--report", str(report_path)])
            pool = ingest.read_jsonl(out)
            self.assertEqual([r["id"] for r in pool], ["vctk-dlg-p300"])
            self.assertEqual(pool[0]["split"], "train")
            self.assertFalse(pool[0]["excluded_from_training"])
            items_sha = ingest.build_manifest(EVAL)["items_sha256"]
            self.assertEqual(pool[0]["disjointness"]["eval_items_sha256"], items_sha)
            self.assertIn(items_sha, out.read_text(encoding="utf-8").splitlines()[0])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual((report["accepted"], report["rejected"]), (1, 1))
            self.assertEqual(report["rejected_by_reason"], {"eval-group": 1})
            self.assertEqual(report["rejections"][0]["id"], "vctk-dlg-p301")
            self.assertIn(str(dropped), report["exclude_records"])
            self.assertEqual(report["candidate_files"][str(cands)], hashlib.sha256(cands.read_bytes()).hexdigest())
            self.assertEqual(report["eval_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_outputs_that_alias_inputs_or_each_other_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "m.json"
            manifest.write_text(json.dumps(ingest.build_manifest(EVAL)), encoding="utf-8")
            cands = root / "c.jsonl"
            cands.write_text(json.dumps(rec("vctk-dlg-p500", 80, "vctk:p500")), encoding="utf-8")
            before = {p: p.read_bytes() for p in (manifest, cands)}
            cases = {"out is a candidate file": (cands, root / "r.json"),
                     "report is the eval manifest": (root / "o.jsonl", manifest),
                     "out and report are the same": (root / "same.json", root / "same.json"),
                     "out names a candidate another way": (root / "x" / ".." / "c.jsonl", root / "r.json")}
            for label, (out, report) in cases.items():
                with self.subTest(label), self.assertRaises(SystemExit):
                    train_pool.main(["check", "--eval-manifest", str(manifest), "--candidates", str(cands),
                                     "--out", str(out), "--report", str(report)])
            self.assertEqual({p: p.read_bytes() for p in (manifest, cands)}, before)
            self.assertFalse((root / "same.json").exists())

    def test_edited_eval_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = ingest.build_manifest(EVAL)
            data["items"] = data["items"][:-1]
            manifest = root / "m.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            cands = root / "c.jsonl"
            cands.write_text(json.dumps(rec("vctk-dlg-p400", 70, "vctk:p400")), encoding="utf-8")
            with self.assertRaises(SystemExit):
                train_pool.main(["check", "--eval-manifest", str(manifest), "--candidates", str(cands),
                                 "--out", str(root / "o.jsonl"), "--report", str(root / "r.json")])


if __name__ == "__main__":
    unittest.main()
