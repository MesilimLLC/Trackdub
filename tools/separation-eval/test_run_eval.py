import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import audiomath as am
import mixgen
import run_eval
from test_mixgen import DURATION, make_corpus


def build_corpus(root: Path, recipes=("A1", "A9", "A10"), count=1) -> Path:
    cache = root / "cache"
    items = make_corpus(cache)
    out = root / "corpus"
    mixgen.generate_corpus(items, cache, out, split="dev", recipes=list(recipes), seed=5,
                           items_sha256="0" * 64, count_override=count, duration_override=DURATION, no_codec=True)
    return out


def fake_separator(make_outputs, fail_ids=(), timing_extra=None):
    """Stands in for `Trackdub.Benchmarks separation-eval`: reads the jobs file, writes stems and results."""
    def run(jobs_path: Path, results_path: Path) -> None:
        lines = []
        for index, line in enumerate(jobs_path.read_text().splitlines()):
            job = json.loads(line)
            if job["id"] in fail_ids:
                lines.append({"id": job["id"], "job_index": index, "ok": False, "error": "boom", "wall_ms": 5.0})
                continue
            mix = am.load_audio(job["input"], channels=2, sr=run_eval.SEPARATOR_SR)
            vocals, bed = make_outputs(mix.mean(axis=1, keepdims=True))
            Path(job["vocals_output"]).parent.mkdir(parents=True, exist_ok=True)
            run_eval.write_pcm16(Path(job["vocals_output"]), vocals, run_eval.SEPARATOR_SR)
            run_eval.write_pcm16(Path(job["bed_output"]), bed, run_eval.SEPARATOR_SR)
            secs = mix.shape[0] / run_eval.SEPARATOR_SR
            lines.append({"id": job["id"], "job_index": index, "ok": True, "error": None, "wall_ms": 100.0 * (index + 1),
                          "audio_seconds": secs, "rtf": 0.1 * (index + 1), "working_set_before_bytes": 10,
                          "peak_working_set_bytes": 1000 * (index + 1), "selected_provider": "Cpu",
                          **(timing_extra or {})})
        results_path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return run


def evaluate(root, corpus, separator, **kw):
    return run_eval.evaluate(corpus, root / "work", separator, candidate="test", provider=None,
                             hardware_label="unit-test", **kw)


class EvalTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = build_corpus(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def clip(self, result, cid):
        return next(c for c in result["clips"] if c["clip_id"] == cid)


class HelperTests(unittest.TestCase):
    def test_to_stereo(self):
        mono = np.ones((10, 1), np.float32)
        self.assertEqual(run_eval.to_stereo(mono).shape, (10, 2))
        six = np.zeros((10, 6), np.float32)
        six[:, 2] = 1.0
        out = run_eval.to_stereo(six)
        np.testing.assert_allclose(out[:, 0], 10 ** (-3 / 20), atol=1e-6)
        with self.assertRaises(run_eval.EvalError):
            run_eval.to_stereo(np.zeros((4, 3), np.float32))

    def test_pcm16_roundtrip_returns_quantised_signal(self):
        with tempfile.TemporaryDirectory() as td:
            x = np.array([[0.5], [-0.25], [1.5]], np.float32)
            q = run_eval.write_pcm16(Path(td) / "a.wav", x, 44100)
            back = am.load_audio(Path(td) / "a.wav", channels=1, sr=44100)
            np.testing.assert_allclose(back, q, atol=1e-6)
            self.assertLessEqual(float(q.max()), 1.0)

    def test_length_adjustment_within_and_beyond_tolerance(self):
        est = np.zeros((44100, 1), np.float32)
        out, adj = run_eval.to_reference_domain(est, 48000, 48000)
        self.assertEqual(out.shape, (48000, 2))
        self.assertLessEqual(abs(adj), run_eval.LENGTH_TOLERANCE_SAMPLES)
        with self.assertRaises(run_eval.EvalError):
            run_eval.to_reference_domain(est, 48100, 48000)

    def test_summarise_worst_decile_direction(self):
        values = list(range(1, 11))
        self.assertEqual(run_eval.summarise(values, higher_is_better=True)["worst_decile"], 1.9)
        self.assertAlmostEqual(run_eval.summarise(values, higher_is_better=False)["worst_decile"], 9.1)
        self.assertIsNone(run_eval.summarise([], True))


class EvaluateTests(EvalTestBase):
    def test_no_separation_passes_the_gate_and_leaks_everything(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
        result = evaluate(self.root, self.corpus, sep)
        a1 = self.clip(result, "a1-dev-000")
        self.assertTrue(a1["ok"], a1["error"])
        self.assertTrue(a1["reconstruction"]["passed"], a1["reconstruction"])
        self.assertGreater(a1["leakage"]["dialogue_residual_db"], -6.0)

    def test_reconstruction_scores_the_written_pcm16_stems(self):
        sep = fake_separator(lambda mono: (0.5 * mono, 0.5 * mono))
        result = evaluate(self.root, self.corpus, sep, gate={"threshold_db": -120.0})
        a1 = self.clip(result, "a1-dev-000")
        self.assertGreater(a1["reconstruction"]["max_abs_residual"], 0.0)
        self.assertIsNotNone(a1["reconstruction"]["residual_db"])
        self.assertFalse(a1["reconstruction"]["passed"])

    def test_dialogue_only_output_reconstructs_but_mutes_the_bed(self):
        sep = fake_separator(lambda mono: (mono, np.zeros_like(mono)))
        a1 = self.clip(evaluate(self.root, self.corpus, sep), "a1-dev-000")
        self.assertTrue(a1["ok"], a1["error"])
        self.assertTrue(a1["separator_ok"])
        self.assertTrue(a1["reconstruction"]["passed"])
        self.assertLess(a1["damage"]["si_sdr_db"], -20.0)
        self.assertEqual(a1["leakage"]["status"], "scored")
        self.assertEqual(a1["leakage"]["estimated_bed_active_rms_dbfs"], run_eval.metrics.DB_FLOOR)

    def test_muted_output_fails_reconstruction(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), np.zeros_like(mono)))
        result = evaluate(self.root, self.corpus, sep)
        a1 = self.clip(result, "a1-dev-000")
        self.assertFalse(a1["ok"])
        self.assertIn("reconstruction gate failed", a1["error"])
        self.assertFalse(a1["reconstruction"]["passed"])
        self.assertEqual(result["summary"]["clips_scored"], 3)
        self.assertEqual(len(result["summary"]["clips_failed"]), 3)

    def test_undefined_metrics_are_skipped_not_fatal(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
        result = evaluate(self.root, self.corpus, sep)
        a10 = self.clip(result, "a10-dev-000")
        self.assertTrue(a10["ok"], a10["error"])
        self.assertIn("skipped", a10["leakage"])
        a9 = self.clip(result, "a9-dev-000")
        self.assertIsNone(a9["leakage"]["leakage_to_bed_db"])
        self.assertEqual(a9["leakage"]["status"], "not_applicable_reference_bed_below_floor")
        self.assertGreater(a9["leakage"]["estimated_bed_active_rms_dbfs"], -40.0)
        self.assertIsNone(a9["damage"]["si_sdr_db"])
        accounting = result["summary"]["strata"]["A9"]["bed_leakage_accounting"]
        self.assertEqual(accounting["not_applicable_reference_bed_below_floor"], 1)
        self.assertEqual(accounting["estimated_bed_active_rms_dbfs"]["n"], 1)

    def test_failed_and_missing_jobs_are_recorded_per_clip(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono), fail_ids={"a9-dev-000"})
        result = evaluate(self.root, self.corpus, sep)
        self.assertEqual(result["summary"]["clips_scored"], 2)
        failed = {f["clip_id"]: f["error"] for f in result["summary"]["clips_failed"]}
        self.assertIn("boom", failed["a9-dev-000"])

        def drop_last(jobs_path, results_path):
            fake_separator(lambda mono: (np.zeros_like(mono), mono))(jobs_path, results_path)
            rows = results_path.read_text().splitlines()[:-1]
            results_path.write_text("\n".join(rows) + "\n")
        result = evaluate(self.root, self.corpus, drop_last)
        self.assertIn("no result from separator", {f["error"] for f in result["summary"]["clips_failed"]})

    def test_generated_audio_hash_mismatch_stops_before_running_separator(self):
        clip = self.corpus / "a1-dev-000" / "mixture.wav"
        clip.write_bytes(clip.read_bytes() + b"tampered")
        called = []

        with self.assertRaisesRegex(run_eval.EvalError, "mixture SHA-256 mismatch"):
            evaluate(self.root, self.corpus, lambda *_: called.append(True))
        self.assertEqual(called, [])

    def test_resource_summary_splits_cold_and_warm(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
        res = evaluate(self.root, self.corpus, sep)["summary"]["resources"]
        self.assertAlmostEqual(res["rtf_cold_first_job"], 0.1)
        self.assertAlmostEqual(res["rtf_warm_median"], 0.25)
        self.assertEqual(res["peak_working_set_bytes_max"], 3000)
        self.assertEqual(res["selected_providers"], ["Cpu"])

    def test_strata_and_corpus_provenance(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
        result = evaluate(self.root, self.corpus, sep)
        strata = result["summary"]["strata"]
        self.assertEqual(set(strata), {"all", "A1", "A9", "A10"})
        self.assertEqual(strata["all"]["reconstruction_gate_pass"], 3)
        self.assertEqual(strata["all"]["bed_leakage_accounting"]["scored_clips"], 1)
        self.assertEqual(strata["all"]["bed_leakage_accounting"]["skipped_clips"], 1)
        self.assertEqual(result["corpus"]["items_sha256"], "0" * 64)
        self.assertEqual(len(result["corpus"]["manifest_sha256"]), 64)
        self.assertEqual(result["hardware_label"], "unit-test")
        json.dumps(result)

    def test_limit_and_empty_corpus(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
        self.assertEqual(evaluate(self.root, self.corpus, sep, clip_limit=1)["summary"]["clips_total"], 1)
        manifest = self.corpus / "corpus.manifest.json"
        data = json.loads(manifest.read_text())
        data["clips"] = []
        manifest.write_text(json.dumps(data))
        with self.assertRaises(run_eval.EvalError):
            evaluate(self.root, self.corpus, sep)

    def test_clip_limit_must_be_positive_and_none_keeps_full_scope(self):
        sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
        self.assertEqual(evaluate(self.root, self.corpus, sep, clip_limit=None)["summary"]["clips_total"], 3)
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(run_eval.EvalError, "clip_limit must be positive"):
                evaluate(self.root, self.corpus, sep, clip_limit=limit)


class VariantTests(unittest.TestCase):
    def test_all_a11_formats_round_trip_through_the_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = build_corpus(root, recipes=("A11",), count=4)
            sep = fake_separator(lambda mono: (np.zeros_like(mono), mono))
            result = evaluate(root, corpus, sep, gate={"output_bits": 16, "output_rounding": "nearest"})
            by_variant = {clip["variant"]: clip for clip in result["clips"]}
            self.assertEqual(set(by_variant), {"mono", "sr8000", "sr16000", "5.1"})
            for variant in ("mono", "sr16000", "5.1"):
                self.assertTrue(by_variant[variant]["ok"], f"{variant}: {by_variant[variant]['error']}")
                self.assertLessEqual(by_variant[variant]["reconstruction"]["max_abs_residual"], 1.6e-5)
            # Strict full-band -60 dB acceptance exposes the 8 kHz PCM16 round-trip residual;
            # the declared quantization allowance is diagnostic and does not make this pass.
            self.assertFalse(by_variant["sr8000"]["ok"])
            self.assertIn("full-band residual", by_variant["sr8000"]["error"])


class GateConfigTests(EvalTestBase):
    def test_declared_band_and_precision_reach_the_gate_and_the_results(self):
        band_limited = lambda mono: (np.zeros_like(mono), lowpass_to(mono, 11025.0))
        sep = fake_separator(band_limited)
        strict = evaluate(self.root, self.corpus, sep)
        declared = evaluate(self.root, self.corpus, sep, gate={"band_limit_hz": 11025.0, "output_bits": 16, "output_rounding": "truncate"})
        a1 = self.clip(declared, "a1-dev-000")
        self.assertEqual(a1["reconstruction"]["band_limit_hz"], 11025.0)
        self.assertEqual(declared["gate_config"], {"threshold_db": -60.0, "edge_exclusion_s": 0.05,
                                                    "band_limit_hz": 11025.0, "output_bits": 16,
                                                    "output_rounding": "truncate"})
        self.assertEqual(strict["gate_config"], {"threshold_db": -60.0, "edge_exclusion_s": 0.05})
        self.assertLess(self.clip(declared, "a1-dev-000")["reconstruction"]["in_band_residual_db"],
                        self.clip(strict, "a1-dev-000")["reconstruction"]["residual_db"])
        self.assertIn("bandwidth_retained_db", declared["summary"]["strata"]["all"]["metrics"])

    def test_spleeter_profile_matches_the_engine(self):
        self.assertEqual(run_eval.CANDIDATE_PROFILES["spleeter"], {"band_limit_hz": 11025.0, "output_bits": 16, "output_rounding": "truncate"})

    def test_failed_reconstruction_gate_is_a_failed_cli_result_and_creates_output_parent(self):
        out = self.root / "nested" / "results.json"
        args = ["--corpus", str(self.corpus), "--work", str(self.root / "work"), "--out", str(out),
                "--hardware-label", "unit-test", "--runner", "tool"]
        with patch.object(run_eval, "dotnet_runner", return_value=fake_separator(
                lambda mono: (np.zeros_like(mono), np.zeros_like(mono)))):
            exit_code = run_eval.main(args)

        self.assertEqual(exit_code, 2)
        result = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["clips_scored"], 3)
        self.assertEqual(result["summary"]["strata"]["all"]["reconstruction_gate_pass"], 0)
        self.assertEqual(len(result["summary"]["clips_failed"]), 3)


class RunnerDefaultsTests(unittest.TestCase):
    def test_default_framework_is_portable_off_windows_and_windows_specific_on_windows(self):
        with patch.object(run_eval.platform, "system", return_value="Linux"):
            self.assertEqual(run_eval.default_runner()[5], "net11.0")
        with patch.object(run_eval.platform, "system", return_value="Windows"):
            self.assertEqual(run_eval.default_runner()[5], "net11.0-windows10.0.19041.0")


class ModelProvenanceTests(unittest.TestCase):
    def test_no_metadata_reports_no_executed_roots(self):
        provenance = run_eval.model_provenance([{"id": "a"}])
        self.assertEqual(provenance["executed"], [])
        self.assertIn("vocals.onnx", provenance["pinned_file_sha256"])

    def test_reports_roots_and_validates_hashes_against_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "vocals.onnx").write_bytes(b"vocals")
            (root / "accompaniment.onnx").write_bytes(b"accompaniment")
            timings = [{"id": "a", "metadata": {"model_root": str(root)}}]
            provenance = run_eval.model_provenance(timings)
            self.assertEqual(len(provenance["executed"]), 1)
            self.assertEqual(provenance["executed"][0]["model_root"], str(root))
            self.assertEqual(provenance["executed"][0]["file_sha256"]["vocals.onnx"],
                             hashlib.sha256(b"vocals").hexdigest())
            self.assertFalse(provenance["executed"][0]["matches_pinned"])
            self.assertIsInstance(provenance["pinned_file_sha256"], dict)
            self.assertIn("vocals.onnx", provenance["pinned_file_sha256"])

    def test_deduplicates_identical_roots_and_skips_missing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "vocals.onnx").write_bytes(b"v")
            timings = [{"id": "a", "metadata": {"model_root": str(root)}},
                       {"id": "b", "metadata": {"model_root": str(root)}},
                       {"id": "c", "metadata": {}}]
            provenance = run_eval.model_provenance(timings)
            self.assertEqual(len(provenance["executed"]), 1)
            self.assertNotIn("accompaniment.onnx", provenance["executed"][0]["file_sha256"])


def lowpass_to(mono, cutoff):
    spec = np.fft.rfft(mono.astype(np.float64), axis=0)
    spec[np.fft.rfftfreq(mono.shape[0], 1.0 / run_eval.SEPARATOR_SR) > cutoff] = 0.0
    return np.fft.irfft(spec, n=mono.shape[0], axis=0).astype(np.float32)


class RunnerCommandTests(unittest.TestCase):
    def test_command_line_is_built_from_options(self):
        captured = {}

        def fake_run(cmd, check):
            captured["cmd"] = cmd
            return type("P", (), {"returncode": 0})()

        original = run_eval.subprocess.run
        run_eval.subprocess.run = fake_run
        try:
            run_eval.dotnet_runner(["tool"], "cpu", "models")(Path("j.jsonl"), Path("r.jsonl"))
        finally:
            run_eval.subprocess.run = original
        self.assertEqual(captured["cmd"][:2], ["tool", "separation-eval"])
        for flag, value in (("--provider", "cpu"), ("--model-directory", "models"), ("--model", "spleeter")):
            self.assertEqual(captured["cmd"][captured["cmd"].index(flag) + 1], value)

    def test_crash_exit_code_is_an_error_but_partial_failure_is_not(self):
        def runner_with(code):
            original = run_eval.subprocess.run
            run_eval.subprocess.run = lambda cmd, check: type("P", (), {"returncode": code})()
            try:
                run_eval.dotnet_runner(["tool"], None, None)(Path("j"), Path("r"))
            finally:
                run_eval.subprocess.run = original
        runner_with(0)
        runner_with(2)
        with self.assertRaises(run_eval.EvalError):
            runner_with(1)


if __name__ == "__main__":
    unittest.main()
