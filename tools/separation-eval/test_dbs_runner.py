import unittest

try:
    import torch
except ImportError:  # the repo venv has no torch; run these with the candidates venv
    torch = None

from candidates import dbs_runner as dbs

import run_eval


@unittest.skipIf(torch is None, "needs torch")
class StemTests(unittest.TestCase):
    def test_speech_is_the_third_stem(self):
        out = torch.stack([torch.full((2, 10), float(k)) for k in range(3)], dim=1)  # [channels, sources, samples]
        torch.testing.assert_close(dbs.speech_stem(out, 2), torch.full((2, 10), 2.0))

    def test_estimate_bed_is_mixture_minus_speech(self):
        sep = dbs.Separator.__new__(dbs.Separator)
        sep.torch, sep.device_type, sep.chunk_seconds, sep.hop_seconds, sep.model = torch, "cpu", 6.0, 3.0, None
        mix = torch.randn(2, 1000)
        fake = torch.stack([0.1 * mix, 0.2 * mix, 0.5 * mix], dim=1)
        import sys
        import types
        engine = types.ModuleType("dbs.engine")
        engine.separate = lambda model, mixture, protocol, sample_rate: fake
        saved = sys.modules.get("dbs.engine")
        sys.modules["dbs.engine"] = engine
        try:
            dialogue, bed = sep.estimate(mix)
        finally:
            if saved is not None:
                sys.modules["dbs.engine"] = saved
            else:
                del sys.modules["dbs.engine"]
        torch.testing.assert_close(dialogue, 0.5 * mix)
        torch.testing.assert_close(dialogue + bed, mix)


class ProfileTests(unittest.TestCase):
    def test_candidates_have_float_full_band_profiles(self):
        for name in dbs.MODELS:
            self.assertEqual(run_eval.CANDIDATE_PROFILES[name],
                             {"band_limit_hz": None, "output_bits": 32, "output_rounding": "nearest"})

    def test_window_sizes_follow_the_authors_recipes(self):
        self.assertEqual(dbs.MODELS["dbs-tiger-dnr-residual"][1:], (6.0, 3.0))
        self.assertEqual(dbs.MODELS["dbs-bsrnn-dnr-residual"][1:], (3.0, 1.5))


if __name__ == "__main__":
    unittest.main()
