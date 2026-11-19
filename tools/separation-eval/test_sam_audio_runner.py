import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:  # the repo venv has no torch; run these with the candidates venv
    torch = None

from candidates import sam_audio_runner as sam

import run_eval


class WindowTests(unittest.TestCase):
    def test_short_input_is_one_window(self):
        self.assertEqual(sam.window_starts(100, 300, 50), [0])
        self.assertEqual(sam.window_starts(300, 300, 50), [0])

    def test_windows_cover_the_input_and_end_at_its_end(self):
        starts = sam.window_starts(1000, 300, 50)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 300, 1000)
        for a, b in zip(starts, starts[1:]):
            self.assertLessEqual(b - a, 250)

    def test_aligned_final_window_occurs_once(self):
        self.assertEqual(sam.window_starts(550, 50, 0), list(range(0, 550, 50)))

    @unittest.skipIf(torch is None, "needs torch")
    def test_crossfade_join_reconstructs_a_signal_split_into_windows(self):
        n, window, overlap = 1000, 300, 50
        x = torch.randn(2, n)
        starts = sam.window_starts(n, window, overlap)
        pieces = [x[:, s:s + window] for s in starts]
        torch.testing.assert_close(sam.crossfade_join(pieces, starts, n, overlap), x)

    @unittest.skipIf(torch is None, "needs torch")
    def test_single_window_passes_through(self):
        x = torch.randn(2, 120)
        torch.testing.assert_close(sam.crossfade_join([x], [0], 120, 50), x)


@unittest.skipIf(torch is None, "needs torch")
class SeparateWindowTests(unittest.TestCase):
    def separator(self, output_length):
        sep = sam.Separator.__new__(sam.Separator)
        sep.torch = torch
        sep.device = torch.device("cpu")
        sep.processor = mock.Mock(audio_hop_length=1920)
        stems = [torch.full((output_length,), float(i)) for i in range(1, 5)]
        sep.model = mock.Mock()
        sep.model.separate.return_value = SimpleNamespace(target=stems[:2], residual=stems[2:])
        return sep

    def test_crops_codec_padding_from_targets_and_residuals(self):
        stems = self.separator(49920).separate_window(torch.zeros(2, 48480))
        expected = torch.arange(1, 5, dtype=torch.float32)[:, None].expand(4, 48480)
        torch.testing.assert_close(stems, expected)

    def test_rejects_truncated_or_unexpectedly_padded_output(self):
        for length in (48479, 49921):
            with self.subTest(length=length):
                with self.assertRaises(sam.rc.RunnerError):
                    self.separator(length).separate_window(torch.zeros(2, 48480))


@unittest.skipIf(torch is None, "needs torch")
class EstimateTests(unittest.TestCase):
    """The model is replaced by a fake that returns half of each channel, so the plumbing is tested without weights."""

    def separator(self):
        sep = sam.Separator.__new__(sam.Separator)
        sep.torch = torch
        sep.bed = "residual"
        sep.separate_window = lambda chunk: torch.cat([0.5 * chunk, 0.25 * chunk])  # targets, then residuals
        return sep

    def test_native_bed_is_the_model_residual(self):
        t = torch.arange(sam.rc.SEPARATOR_SR * 3) / sam.rc.SEPARATOR_SR
        mix = 0.1 * torch.stack([torch.sin(2 * torch.pi * 440 * t), torch.sin(2 * torch.pi * 660 * t)])
        sep = self.separator()
        sep.bed = "native"
        dialogue, bed = sep.estimate(mix)
        torch.testing.assert_close(bed, 0.25 * mix, atol=2e-3, rtol=0)
        torch.testing.assert_close(dialogue, 0.5 * mix, atol=2e-3, rtol=0)

    def test_bed_is_the_mixture_minus_the_target_and_lengths_match(self):
        # A tone well inside the band, so the 44.1 -> 48 -> 44.1 kHz round trip is close to exact.
        t = torch.arange(sam.rc.SEPARATOR_SR * 3) / sam.rc.SEPARATOR_SR
        mix = 0.1 * torch.stack([torch.sin(2 * torch.pi * 440 * t), torch.sin(2 * torch.pi * 660 * t)])
        dialogue, bed = self.separator().estimate(mix)
        self.assertEqual(dialogue.shape, mix.shape)
        torch.testing.assert_close(dialogue + bed, mix)
        torch.testing.assert_close(dialogue, 0.5 * mix, atol=2e-3, rtol=0)

    def test_long_input_is_windowed(self):
        sep = self.separator()
        calls = []
        inner = sep.separate_window
        sep.separate_window = lambda chunk: calls.append(chunk.shape[1]) or inner(chunk)
        with mock.patch.object(sam, "WINDOW_SECONDS", 1.0), mock.patch.object(sam, "OVERLAP_SECONDS", 0.25):
            sep.estimate(0.1 * torch.randn(2, int(sam.rc.SEPARATOR_SR * 3.5)))
        self.assertGreater(len(calls), 3)
        self.assertTrue(all(c == sam.NATIVE_SR for c in calls))


class XformersTests(unittest.TestCase):
    def test_stand_ins_import_and_refuse_to_run(self):
        import sys
        import types
        saved = {k: sys.modules.pop(k) for k in ("xformers", "xformers.ops") if k in sys.modules}
        try:
            with mock.patch.dict(sys.modules, {"xformers": types.ModuleType("xformers"),
                                               "xformers.ops": types.ModuleType("xformers.ops")}):
                self.assertTrue(sam.prepare_xformers().startswith("stand-in"))
                from xformers.ops import AttentionBias, fmha
                self.assertTrue(isinstance(AttentionBias, type))
                with self.assertRaises(sam.rc.RunnerError):
                    fmha.memory_efficient_attention(None, None, None)
        finally:
            sys.modules.update(saved)


class ProfileTests(unittest.TestCase):
    def test_candidates_have_float_full_band_profiles(self):
        for name in sam.MODELS:
            self.assertEqual(run_eval.CANDIDATE_PROFILES[name],
                             {"band_limit_hz": None, "output_bits": 32, "output_rounding": "nearest"})

    @unittest.skipIf(torch is None, "needs torch")
    def test_cpu_provider_is_refused(self):
        with self.assertRaises(sam.rc.RunnerError):
            sam.Separator(Path("unused"), "cpu")


if __name__ == "__main__":
    unittest.main()
