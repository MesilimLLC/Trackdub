import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import fsd50k_overlap as fo

DEV = "fname,labels,mids,split\n111,Explosion,/m/1,train\n222,Crowd,/m/2,val\n"
EVAL = "fname,labels,mids\n333,Engine,/m/3\n"


def make_zip(path: Path, dev=DEV, eval_csv=EVAL):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("FSD50K.ground_truth/dev.csv", dev)
        zf.writestr("FSD50K.ground_truth/eval.csv", eval_csv)
        zf.writestr("FSD50K.ground_truth/vocabulary.csv", "index,label,mid\n")


ITEMS = [
    {"id": "a", "freesound_id": 111, "role": "sfx", "tags": ["loud"], "split": "test"},
    {"id": "b", "freesound_id": 222, "role": "ambience", "tags": ["crowd"], "split": "dev"},
    {"id": "c", "freesound_id": 333, "role": "rir", "tags": [], "split": "test"},
    {"id": "d", "freesound_id": 999, "role": "sfx", "tags": ["loud"], "split": "test"},
    {"id": "e", "role": "dialogue", "tags": [], "split": "dev"},
]


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.zip = self.root / "gt.zip"
        make_zip(self.zip)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ids_come_from_both_csvs_and_remember_the_set(self):
        self.assertEqual(fo.read_fsd50k_ids(self.zip), {111: "dev", 222: "dev", 333: "eval"})

    def test_non_numeric_names_are_skipped(self):
        make_zip(self.zip, dev="fname,labels\n111,x\nabc,y\n")
        self.assertEqual(fo.read_fsd50k_ids(self.zip), {111: "dev", 333: "eval"})

    def test_a_zip_without_the_csvs_is_an_error(self):
        with zipfile.ZipFile(self.zip, "w") as zf:
            zf.writestr("other.txt", "x")
        with self.assertRaisesRegex(fo.OverlapError, "has no"):
            fo.read_fsd50k_ids(self.zip)

    def test_a_zip_with_no_ids_is_an_error(self):
        make_zip(self.zip, dev="fname,labels\n", eval_csv="fname,labels\n")
        with self.assertRaisesRegex(fo.OverlapError, "no Freesound ids"):
            fo.read_fsd50k_ids(self.zip)

    def test_only_items_with_a_matching_freesound_id_are_flagged(self):
        flagged = fo.flag_items(ITEMS, fo.read_fsd50k_ids(self.zip))
        self.assertEqual([f["id"] for f in flagged], ["a", "b", "c"])
        self.assertEqual(flagged[2]["fsd50k_set"], "eval")

    def test_summary_counts_by_role_tag_split_and_set(self):
        ids = fo.read_fsd50k_ids(self.zip)
        summary = fo.summarize(ITEMS, fo.flag_items(ITEMS, ids))
        self.assertEqual((summary["freesound_items"], summary["in_fsd50k"]), (4, 3))
        self.assertEqual(summary["share_in_fsd50k"], 0.75)
        self.assertEqual(summary["by_role"], {"ambience": 1, "rir": 1, "sfx": 1})
        self.assertEqual(summary["by_tag"], {"(none)": 1, "crowd": 1, "loud": 1})
        self.assertEqual(summary["by_split"], {"dev": 1, "test": 2})
        self.assertEqual(summary["by_fsd50k_set"], {"dev": 2, "eval": 1})

    def test_the_summary_handles_items_with_no_freesound_ids(self):
        self.assertEqual(fo.summarize([{"id": "x", "role": "dialogue"}], [])["share_in_fsd50k"], 0.0)

    def test_main_writes_a_report_and_leaves_the_manifest_alone(self):
        manifest = self.root / "m.json"
        manifest.write_text(json.dumps({"items_sha256": "abc", "items": ITEMS}), encoding="utf-8")
        before = manifest.read_bytes()
        out = self.root / "sub" / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = fo.main(["--manifest", str(manifest), "--fsd50k-zip", str(self.zip), "--out", str(out)])
        self.assertEqual(code, 0)
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((report["items_sha256"], report["fsd50k_ids"], len(report["items"])), ("abc", 3, 3))
        self.assertEqual(manifest.read_bytes(), before)

    def test_main_reports_a_bad_archive(self):
        manifest = self.root / "m.json"
        manifest.write_text(json.dumps({"items": ITEMS}), encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = fo.main(["--manifest", str(manifest), "--fsd50k-zip", str(self.root / "missing.zip"), "--out", str(self.root / "r.json")])
        self.assertEqual(code, 1)
        self.assertIn("cannot open", err.getvalue())


if __name__ == "__main__":
    unittest.main()
