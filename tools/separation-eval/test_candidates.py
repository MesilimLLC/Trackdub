import math
import unittest

try:
    import torch
except ImportError:  # the repo venv has no torch; run these with the candidates venv
    torch = None

from candidates import mdxnet_runner as mdx
from candidates import mossformer2_runner as mf2

rc = mdx.rc  # the runners import runner_common by its bare name, so use the same module object


class ConstantsTests(unittest.TestCase):
    def test_mdx_chunking_matches_the_official_vocals_configuration(self):
        self.assertEqual((mdx.N_FFT, mdx.HOP, mdx.DIM_F, mdx.DIM_T), (6144, 1024, 2048, 256))
        self.assertEqual(mdx.CHUNK, 1024 * 511)
        self.assertEqual(mdx.GEN_SIZE, mdx.CHUNK - mdx.N_FFT)
        self.assertEqual(mdx.GEN_SIZE % mdx.HOP, 0)

    def test_the_declared_band_is_the_last_kept_bin(self):
        self.assertAlmostEqual(mdx.BAND_LIMIT_HZ, 14700.0)

    def test_the_pinned_hashes_look_like_hashes(self):
        self.assertRegex(mdx.ARCHIVE_MD5, r"^[0-9a-f]{32}$")
        for value in (mdx.MEMBER_SHA256, mf2.CHECKPOINT_SHA256):
            self.assertRegex(value, r"^[0-9a-f]{64}$")
        for value in (mdx.REPO_REVISION, mf2.CODE_REVISION, mf2.HF_REVISION):
            self.assertRegex(value, r"^[0-9a-f]{40}$")

    def test_mossformer2_decoder_arguments_match_the_official_config(self):
        args = mf2.DECODER_ARGS
        self.assertEqual((args["sampling_rate"], args["win_len"], args["win_inc"], args["fft_len"]), (48000, 1920, 384, 1920))
        self.assertEqual((args["one_time_decode_length"], args["decode_window"]), (20, 4))


class PortabilityTests(unittest.TestCase):
    def test_memory_counters_work_on_this_platform(self):
        peak, current = rc._memory_counters()
        self.assertTrue(isinstance(peak, int) and isinstance(current, int) and peak > 0 and current > 0)

    def test_padding_is_whole_strides_and_zero_when_already_whole(self):
        self.assertEqual(mdx.padding_for(mdx.GEN_SIZE), 0)
        self.assertEqual(mdx.padding_for(2 * mdx.GEN_SIZE), 0)
        self.assertEqual(mdx.padding_for(1), mdx.GEN_SIZE - 1)
        self.assertEqual((1000 + mdx.padding_for(1000)) % mdx.GEN_SIZE, 0)


class CommandLineTests(unittest.TestCase):
    ARGS = ["separation-eval", "--jobs", "j", "--results", "r", "--model-directory", "m"]

    def test_a_known_model_parses(self):
        parsed = rc.parse_command_line([*self.ARGS, "--model", "x"], ["x"])
        self.assertEqual((parsed.model, parsed.provider), ("x", None))

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(SystemExit):
            rc.parse_command_line([*self.ARGS, "--model", "y"], ["x"])


@unittest.skipIf(torch is None, "torch not installed")
class TorchTests(unittest.TestCase):
    def mdx_separator(self):
        sep = mdx.Separator.__new__(mdx.Separator)
        sep.torch = torch
        sep.window = torch.hann_window(mdx.N_FFT, periodic=True)
        return sep

    def test_stft_keeps_the_first_dim_f_bins_with_four_channels(self):
        sep = self.mdx_separator()
        spec = sep.stft(torch.randn(2, 2, mdx.CHUNK))
        self.assertEqual(tuple(spec.shape), (2, 4, mdx.DIM_F, mdx.DIM_T * 2))

    def test_istft_undoes_stft_below_the_band_and_removes_what_is_above_it(self):
        sep = self.mdx_separator()
        t = torch.arange(mdx.CHUNK, dtype=torch.float64) / rc.SEPARATOR_SR  # float32 time loses the tone's phase
        low = (0.3 * torch.sin(2 * math.pi * 1000.0 * t)).float()
        high = (0.3 * torch.sin(2 * math.pi * 18000.0 * t)).float()
        x = torch.stack([low + high, low]).unsqueeze(0)
        y = sep.istft(sep.stft(x))
        middle = slice(mdx.TRIM, -mdx.TRIM)
        self.assertLess(float((y[0, 1, middle] - low[middle]).abs().max()), 2e-3)
        self.assertLess(float((y[0, 0, middle] - low[middle]).abs().max()), 2e-3)

    def test_fix_length_trims_pads_and_refuses_large_differences(self):
        x = torch.ones(2, 10)
        self.assertEqual(rc.fix_length(torch, x, 8).shape[1], 8)
        self.assertEqual(rc.fix_length(torch, x, 12).shape[1], 12)
        with self.assertRaises(rc.RunnerError):
            rc.fix_length(torch, x, 100)

    def test_strict_loading_fills_every_parameter_or_fails(self):
        model = torch.nn.Linear(3, 2)
        good = {"module.weight": torch.ones(2, 3), "bias": torch.zeros(2)}
        mf2.load_strict(model, {"model": good})
        self.assertTrue(bool((model.weight == 1).all()))
        with self.assertRaisesRegex(rc.RunnerError, "not found in the checkpoint"):
            mf2.load_strict(torch.nn.Linear(3, 2), {"model": {"weight": torch.ones(2, 3)}})

    def test_a_wrong_shape_counts_as_missing(self):
        with self.assertRaisesRegex(rc.RunnerError, "bias"):
            mf2.load_strict(torch.nn.Linear(3, 2), {"weight": torch.ones(2, 3), "bias": torch.zeros(5)})


if __name__ == "__main__":
    unittest.main()
