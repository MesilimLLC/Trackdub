import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audiomath as am
import ingest
import noise_screen as ns


class SignalTests(unittest.TestCase):
    def test_long_zero_runs_are_removed_short_ones_kept(self):
        x = np.concatenate([np.ones(100), np.zeros(ns.MIN_SILENCE_RUN), np.ones(50), np.zeros(5), np.ones(10)])
        y = ns.drop_digital_silence(x)
        self.assertEqual(y.size, 100 + 50 + 5 + 10)

    def test_short_audio_is_tiled_to_fill_windows(self):
        # As in the reference scorer: the audio is doubled until it covers a window, then windowed one second apart.
        w = ns.windows(np.arange(1, 1001, dtype=np.float32), max_windows=20)
        self.assertEqual(w.shape[1], ns.WINDOW)
        self.assertGreaterEqual(w.shape[0], 1)
        self.assertEqual(w[0, 1000], 1.0)

    def test_long_audio_is_thinned_evenly(self):
        x = np.ones(ns.WINDOW + 100 * ns.HOP, dtype=np.float32)
        self.assertEqual(ns.windows(x, max_windows=200).shape[0], 101)
        self.assertEqual(ns.windows(x, max_windows=5).shape[0], 5)

    def test_all_silence_is_an_error(self):
        with self.assertRaises(ValueError):
            ns.windows(ns.drop_digital_silence(np.zeros(ns.SR)), max_windows=5)

    def test_published_polynomials_are_applied(self):
        raw = np.array([[1.0, 2.0, 3.0]])
        mos = ns.to_mos(raw, "p835")
        expected = [np.poly1d(c)(v) for c, v in zip(ns.POLYNOMIALS["p835"], raw[0])]
        np.testing.assert_allclose(mos[0], expected)


class ScreenTests(unittest.TestCase):
    """A fake scorer stands in for the ONNX models: background quality falls with the item's noise level."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        rng = np.random.default_rng(0)
        self.records = []
        for name, noise in (("clean", 0.0), ("noisy", 0.2)):
            rec = {"id": f"cv-dlg-xx-{name}", "role": "dialogue", "ext": "wav", "source": "commonvoice"}
            t = np.arange(3 * am.SR) / am.SR
            speech = 0.3 * np.sin(2 * np.pi * 220 * t) + noise * rng.standard_normal(t.size)
            path = ingest.cache_path(self.cache, rec)
            path.parent.mkdir(parents=True, exist_ok=True)
            am.write_wav(path, speech.astype(np.float32).reshape(-1, 1), am.SR)
            self.records.append(rec)
        self.records.append({"id": "fs-sfx-x", "role": "sfx", "ext": "wav", "source": "freesound"})
        self.candidates = self.root / "cand.jsonl"
        self.candidates.write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def fake_scorer(batch: np.ndarray) -> dict:
        noisiness = float(np.abs(np.diff(batch, axis=1)).mean())
        bak = 4.2 - 20.0 * noisiness
        return {"p835": np.array([3.5, bak, 3.4]), "personalized": np.array([3.5, bak + 0.1, 3.4])}

    def run_screen(self, *extra: str) -> dict:
        args = ns.parser().parse_args(["screen", "--candidates", str(self.candidates), "--cache", str(self.cache),
                                       "--models-dir", str(self.root / "models"), "--out", str(self.root / "o.jsonl"),
                                       "--report", str(self.root / "r.json"), "--workers", "1", *extra])
        return ns.run_screen(args, scorer=self.fake_scorer)

    def test_noisy_item_rejected_clean_item_kept_with_scores(self):
        report = self.run_screen()
        kept = ingest.read_jsonl(self.root / "o.jsonl")
        self.assertEqual([r["id"] for r in kept], ["cv-dlg-xx-clean"])
        self.assertGreater(kept[0]["noise_screen"]["bak"], ns.DEFAULT_MIN_BAK)
        reasons = {r["id"]: r["reasons"] for r in report["rejections"]}
        self.assertTrue(reasons["cv-dlg-xx-noisy"][0].startswith("noisy-background"))
        self.assertEqual(reasons["fs-sfx-x"], ["unscored: not a dialogue item"])
        self.assertEqual((report["accepted"], report["rejected"]), (1, 2))
        self.assertEqual(report["bak"]["n"], 2)

    def test_thresholds_are_configurable(self):
        report = self.run_screen("--min-bak", "0", "--min-personalized-bak", "0")
        self.assertEqual(report["accepted"], 2)

    def test_missing_audio_is_reported_not_fatal(self):
        self.records[0]["id"] = "cv-dlg-xx-missing"
        self.candidates.write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")
        report = self.run_screen()
        reasons = {r["id"]: r["reasons"][0] for r in report["rejections"]}
        self.assertTrue(reasons["cv-dlg-xx-missing"].startswith("unscored:"))

    def test_outputs_must_differ_from_the_candidates(self):
        args = ns.parser().parse_args(["screen", "--candidates", str(self.candidates), "--cache", str(self.cache),
                                       "--models-dir", "m", "--out", str(self.candidates), "--report", "r.json"])
        with self.assertRaises(SystemExit):
            ns.run_screen(args, scorer=self.fake_scorer)


if __name__ == "__main__":
    unittest.main()
