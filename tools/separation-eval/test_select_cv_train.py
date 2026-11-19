import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import build_local_items as bli
import ingest
import select_cv_train as sel

ARCHIVE = "1789493165313-cv-corpus-27.0-2026-09-11-xx.tar.gz"


def clip_name(client: str, n: int) -> str:
    return f"common_voice_xx_{client}_{n}.mp3"


class CvTrainFixture(unittest.TestCase):
    """A fake cv-root: three speakers with 12, 5 and 12 good clips, one of them in the eval corpus."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clients = {"alice": 12, "bob": 5, "evaleve": 12}
        self.write_meta()
        self.write_archive()
        self.eval_manifest = self.root / "items.v1.2.manifest.json"
        evalitem = {"id": "cv-dlg-xx-e", "group": sel.speaker_group("xx", "evaleve").upper(), "role": "dialogue"}
        self.eval_manifest.write_text(json.dumps(ingest.build_manifest([evalitem])), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write_meta(self):
        meta = self.root / "cv-meta" / "xx"
        meta.mkdir(parents=True)
        validated = ["client_id\tpath\tup_votes\tdown_votes\tgender\tage\taccent"]
        durations = ["clip\tduration[ms]"]
        for client, n in self.clients.items():
            for k in range(n):
                validated.append(f"{client}\t{clip_name(client, k)}\t2\t0\tfemale_feminine\tthirties\t")
                durations.append(f"{clip_name(client, k)}\t4000")
        validated.append(f"alice\t{clip_name('alice', 99)}\t1\t0\t\t\t")
        durations.append(f"{clip_name('alice', 99)}\t4000")
        validated.append(f"alice\t{clip_name('alice', 98)}\t3\t0\t\t\t")
        durations.append(f"{clip_name('alice', 98)}\t20000")
        (meta / "validated.tsv").write_text("\n".join(validated) + "\n", encoding="utf-8")
        (meta / "clip_durations.tsv").write_text("\n".join(durations) + "\n", encoding="utf-8")

    def write_archive(self):
        with tarfile.open(self.root / ARCHIVE, "w:gz") as tf:
            for client, n in self.clients.items():
                for k in [*range(n), 98, 99]:
                    data = f"{client}-{k}".encode()
                    info = tarfile.TarInfo(f"cv-corpus-27.0-2026-09-11/xx/clips/{clip_name(client, k)}")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))

    def run_main(self, out: Path, *extra: str) -> dict:
        sel.main(["--cv-root", str(self.root), "--eval-manifest", str(self.eval_manifest), "--out-dir", str(out),
                  "--clips-per-speaker", "8", *extra])
        return json.loads((out / "commonvoice-subset.manifest.json").read_text(encoding="utf-8"))


class SelectionTests(CvTrainFixture):
    def test_only_good_clips_from_non_eval_speakers_with_enough_clips(self):
        clips = sel.good_clips_by_client(self.root / "cv-meta" / "xx")
        self.assertEqual(len(clips["alice"]), 12)
        speakers = sel.select_speakers("xx", clips, sel.eval_groups(self.eval_manifest), sel.Rules(10, 8, "s"))
        self.assertEqual([s.group for s in speakers], [sel.speaker_group("xx", "alice")])
        self.assertEqual(len(speakers[0].clips), 8)
        self.assertEqual(speakers[0].gender_class, "F")

    def test_choice_is_seeded(self):
        clips = sel.good_clips_by_client(self.root / "cv-meta" / "xx")
        pick = lambda seed: [c.path for c in sel.select_speakers("xx", clips, set(), sel.Rules(10, 8, seed))[0].clips]
        self.assertEqual(pick("a"), pick("a"))
        self.assertNotEqual(pick("a"), pick("b"))

    def test_group_matches_the_eval_scheme(self):
        self.assertEqual(sel.speaker_group("ar", "x"), f"commonvoice:ar:{hashlib.sha256(b'x').hexdigest()[:16]}")


class ExtractionTests(CvTrainFixture):
    def test_extracts_clips_and_writes_a_manifest_build_local_items_reads(self):
        out = self.root / "train"
        manifest = self.run_main(out)
        self.assertEqual(manifest["summary"]["xx"]["speakers"], 1)
        self.assertEqual(len(manifest["files"]), 8)
        for entry in manifest["files"]:
            raw = (out / "xx" / "clips" / Path(entry["path"]).name).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(entry["split"], "train")
            self.assertNotIn("alice", entry["group"])
        sources = bli.commonvoice_sources(out)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].group, sel.speaker_group("xx", "alice"))
        self.assertEqual(len(sources[0].clips), 8)
        self.assertEqual(manifest["excluded_eval_items_sha256"],
                         json.loads(self.eval_manifest.read_text(encoding="utf-8"))["items_sha256"])

    def test_min_clips_and_language_filter(self):
        self.assertEqual(self.run_main(self.root / "a", "--min-clips", "5")["summary"]["xx"]["speakers"], 2)
        self.assertEqual(self.run_main(self.root / "b", "--languages", "pt")["files"], [])

    def test_refuses_to_write_into_the_eval_subset_folder(self):
        with self.assertRaises(SystemExit):
            self.run_main(self.root / "commonvoice")


if __name__ == "__main__":
    unittest.main()
