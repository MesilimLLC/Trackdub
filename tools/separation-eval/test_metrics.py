import unittest
import unittest.mock

import numpy as np

import metrics

SR = 16000


def signals(seed=0, seconds=8.0, channels=2, dialogue_active=(0.0, 1.0)):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    d = rng.standard_normal((n, channels)) * 0.1
    lo, hi = int(dialogue_active[0] * n), int(dialogue_active[1] * n)
    d[:lo] = 0.0
    d[hi:] = 0.0
    b = rng.standard_normal((n, channels)) * 0.1
    return d.astype(np.float32), b.astype(np.float32)


def expected_db(leak, bed, d):
    m = np.abs(d).sum(axis=1) > 0
    return 10 * np.log10((leak[m] ** 2).sum() / (bed[m] ** 2).sum())


class BedLeakageTests(unittest.TestCase):
    def test_no_separation_leaks_the_whole_dialogue(self):
        d, b = signals()
        r = metrics.bed_leakage(d, b, d + b, SR)
        self.assertAlmostEqual(r.dialogue_residual_db, 0.0, delta=0.2)
        self.assertGreater(r.leakage_to_bed_db, -3.0)

    def test_exact_estimated_bed_does_not_count_speech_like_reference_bed_as_leakage(self):
        d, _ = signals()
        speech_like_reference_bed = d.copy()
        r = metrics.bed_leakage(d, speech_like_reference_bed, speech_like_reference_bed.copy(), SR)
        self.assertEqual(r.leakage_to_bed_db, metrics.DB_FLOOR)
        self.assertEqual(r.dialogue_residual_db, metrics.DB_FLOOR)
        self.assertEqual(r.status, "scored")

    def test_scaled_leak_matches_expected_level(self):
        d, b = signals(seed=1)
        for alpha in (0.3, 0.05):
            estimated_bed = b + alpha * d
            r = metrics.bed_leakage(d, b, estimated_bed, SR)
            self.assertAlmostEqual(r.leakage_to_bed_db, expected_db(alpha * d, b, d), delta=0.3)
            self.assertAlmostEqual(r.dialogue_residual_db, 20 * np.log10(alpha), delta=0.3)

    def test_primary_ratio_is_normalized_by_reference_bed_energy(self):
        d, b = signals(seed=11)
        est_bed = b + 0.2 * d
        r = metrics.bed_leakage(d, b, est_bed, SR)
        active, _ = metrics._activity_masks(d, SR)
        dialogue_energy = float((d[active].astype(np.float64) ** 2).sum())
        reference_bed_energy = float((b[active].astype(np.float64) ** 2).sum())
        expected = r.dialogue_residual_db + 10 * np.log10(dialogue_energy / reference_bed_energy)
        self.assertAlmostEqual(r.leakage_to_bed_db, expected, delta=0.01)

    def test_primary_ratio_uses_reference_denominator_when_estimated_bed_gain_changes(self):
        d, b = signals(seed=12)
        lower = metrics.bed_leakage(d, b, b + 0.2 * d, SR)
        higher = metrics.bed_leakage(d, b, b + 0.4 * d, SR)
        self.assertAlmostEqual(higher.leakage_to_bed_db - lower.leakage_to_bed_db, 20 * np.log10(2.0), delta=0.1)

    def test_filtered_and_delayed_leak_is_fully_captured(self):
        d, b = signals(seed=2)
        fir = np.array([0.0, 0.0, 0.3, -0.2, 0.1, 0.05])
        shifted = np.stack([np.convolve(d[:, c], fir)[: d.shape[0]] for c in range(2)], axis=1).astype(np.float32)
        r = metrics.bed_leakage(d, b, b + shifted, SR)
        self.assertAlmostEqual(r.leakage_to_bed_db, expected_db(shifted, b, d), delta=0.5)
        self.assertAlmostEqual(r.dialogue_residual_db, expected_db(shifted, d, d), delta=0.5)

    def test_leak_via_opposite_channel_is_captured(self):
        d, b = signals(seed=3)
        swapped = 0.2 * d[:, ::-1]
        r = metrics.bed_leakage(d, b, b + swapped, SR)
        self.assertAlmostEqual(r.leakage_to_bed_db, expected_db(swapped, b, d), delta=0.5)
        self.assertAlmostEqual(r.dialogue_residual_db, 20 * np.log10(0.2), delta=0.5)

    def test_uncorrelated_estimated_bed_has_only_chance_capture(self):
        d, b = signals(seed=4)
        noise = (np.random.default_rng(99).standard_normal(b.shape) * 0.05).astype(np.float32)
        r = metrics.bed_leakage(d, b, b + noise, SR)
        self.assertLess(r.leakage_to_bed_db, -15.0)

    def test_only_dialogue_active_frames_count(self):
        d, b = signals(seed=5, dialogue_active=(0.0, 0.5))
        unscaled_bed = b.copy()
        b[len(b) // 2:] *= 20.0
        leak = 0.2 * d
        r = metrics.bed_leakage(d, b, b + leak, SR)
        baseline = metrics.bed_leakage(d, unscaled_bed, unscaled_bed + leak, SR)
        self.assertAlmostEqual(r.leakage_to_bed_db, baseline.leakage_to_bed_db, delta=0.01)
        self.assertAlmostEqual(r.active_fraction, 0.5, delta=0.02)

    def test_silent_reference_bed_does_not_hide_dialogue_in_estimated_bed(self):
        d, _ = signals(seed=6)
        zero = np.zeros_like(d)
        r = metrics.bed_leakage(d, zero, 0.1 * d, SR)
        self.assertIsNone(r.leakage_to_bed_db)
        self.assertEqual(r.status, "not_applicable_reference_bed_below_floor")
        self.assertAlmostEqual(r.dialogue_residual_db, -20.0, delta=0.3)
        self.assertGreater(r.estimated_bed_active_rms_dbfs, -45.0)

    def test_muted_estimated_bed_remains_scored_against_non_silent_reference(self):
        d, b = signals(seed=6)
        r = metrics.bed_leakage(d, b, np.zeros_like(b), SR)
        self.assertIsNotNone(r.leakage_to_bed_db)
        self.assertEqual(r.status, "scored")
        self.assertEqual(r.estimated_bed_active_rms_dbfs, metrics.DB_FLOOR)

    def test_worst_window_finds_a_burst(self):
        d, b = signals(seed=7, seconds=8.0)
        leak = np.zeros_like(d)
        s = 3 * SR
        leak[s:s + SR] = 0.5 * d[s:s + SR]
        r = metrics.bed_leakage(d, b, b + leak, SR)
        self.assertGreater(r.worst_window_leakage_db, r.dialogue_residual_db + 6.0)
        window = metrics.bed_leakage(d[s:s + SR], b[s:s + SR], b[s:s + SR] + leak[s:s + SR], SR)
        self.assertAlmostEqual(r.worst_window_leakage_db, window.dialogue_residual_db, delta=0.5)
        self.assertAlmostEqual(r.worst_window_leakage_db, 20 * np.log10(0.5), delta=0.5)

    def test_worst_window_is_bounded_when_the_reference_bed_is_near_silent(self):
        d, b = signals(seed=7, seconds=8.0)
        quiet = b.copy()
        s = 3 * SR
        quiet[s:s + SR] *= 1e-4
        leak = np.zeros_like(d)
        leak[s:s + SR] = 0.2 * d[s:s + SR]
        r = metrics.bed_leakage(d, quiet, quiet + leak, SR)
        self.assertAlmostEqual(r.worst_window_leakage_db, 20 * np.log10(0.2), delta=0.5)
        self.assertLessEqual(r.worst_window_leakage_db, 0.5)

    def test_worst_window_ignores_a_barely_audible_dialogue_window(self):
        d, b = signals(seed=7, seconds=8.0)
        s = 3 * SR
        d[s:s + SR] *= 0.03
        noise = np.random.default_rng(3).standard_normal(b.shape).astype(np.float32)
        error = np.zeros_like(b)
        error[s:s + SR] = 0.3 * noise[s:s + SR]
        r = metrics.bed_leakage(d, b, b + error, SR)
        self.assertLess(r.worst_window_leakage_db, -10.0)
        with unittest.mock.patch.object(metrics, "WORST_WINDOW_MIN_DIALOGUE_POWER_RATIO", 0.0):
            unguarded = metrics.bed_leakage(d, b, b + error, SR)
        self.assertGreater(unguarded.worst_window_leakage_db, 10.0)

    def test_worst_window_is_reported_when_the_whole_reference_bed_is_below_the_floor(self):
        d, _ = signals(seed=6)
        r = metrics.bed_leakage(d, np.zeros_like(d), 0.1 * d, SR)
        self.assertEqual(r.status, "not_applicable_reference_bed_below_floor")
        self.assertAlmostEqual(r.worst_window_leakage_db, -20.0, delta=0.5)

    def test_mono_and_multichannel(self):
        for ch in (1, 6):
            d, b = signals(seed=8, channels=ch, seconds=4.0)
            leak = 0.2 * d
            r = metrics.bed_leakage(d, b, b + leak, SR, max_lag_ms=1.0)
            self.assertAlmostEqual(r.leakage_to_bed_db, expected_db(leak, b, d), delta=0.5, msg=f"{ch} ch")
            self.assertAlmostEqual(r.dialogue_residual_db, 20 * np.log10(0.2), delta=0.5, msg=f"{ch} ch")

    def test_misaligned_output_beyond_lag_window_is_reported_as_error_energy_not_hidden(self):
        d, b = signals(seed=9)
        late = np.roll(d, 400, axis=0) * 0.3
        r = metrics.bed_leakage(d, b, b + late, SR, max_lag_ms=5.0)
        self.assertLess(r.dialogue_residual_db, -20.0)
        r_wide = metrics.bed_leakage(d, b, b + late, SR, max_lag_ms=40.0)
        self.assertAlmostEqual(r_wide.dialogue_residual_db, 20 * np.log10(0.3), delta=0.5)

    def test_input_validation(self):
        d, b = signals()
        with self.assertRaises(metrics.MetricError):
            metrics.bed_leakage(d, b, b[:-1], SR)
        bad = b.copy()
        bad[10, 0] = np.nan
        with self.assertRaises(metrics.MetricError):
            metrics.bed_leakage(d, b, bad, SR)
        with self.assertRaises(metrics.MetricError):
            metrics.bed_leakage(np.zeros_like(d), b, b, SR)
        with self.assertRaises(metrics.MetricError):
            metrics.bed_leakage(d[:, 0], b[:, 0], b[:, 0], SR)

    def test_blocked_accumulation_matches_single_block(self):
        d, b = signals(seed=10, seconds=6.0)
        leak = 0.2 * d
        whole = metrics.bed_leakage(d, b, b + leak, SR).leakage_to_bed_db
        old = metrics.CORRELATION_BLOCK
        metrics.CORRELATION_BLOCK = 5000
        try:
            blocked = metrics.bed_leakage(d, b, b + leak, SR).leakage_to_bed_db
        finally:
            metrics.CORRELATION_BLOCK = old
        self.assertAlmostEqual(whole, blocked, delta=0.1)


def noise_like(ref, scale, seed):
    return (np.random.default_rng(seed).standard_normal(ref.shape) * scale).astype(np.float32)


class SiSdrTests(unittest.TestCase):
    def test_exact_match_hits_ceiling(self):
        _, b = signals()
        self.assertEqual(metrics.si_sdr(b, b.copy()), metrics.DB_CEIL)

    def test_known_snr(self):
        _, b = signals(seed=1)
        n = noise_like(b, 0.01, 2)
        expected = 10 * np.log10((b ** 2).sum() / (n ** 2).sum())
        self.assertAlmostEqual(metrics.si_sdr(b, b + n), expected, delta=0.2)

    def test_scale_invariant(self):
        _, b = signals(seed=1)
        est = b + noise_like(b, 0.02, 3)
        self.assertAlmostEqual(metrics.si_sdr(b, est), metrics.si_sdr(b, 3.0 * est), places=3)

    def test_muted_and_uncorrelated_are_bad(self):
        _, b = signals(seed=1)
        self.assertEqual(metrics.si_sdr(b, np.zeros_like(b)), metrics.DB_FLOOR)
        self.assertLess(metrics.si_sdr(b, noise_like(b, 0.1, 4)), -20.0)

    def test_silent_reference_is_undefined(self):
        _, b = signals()
        self.assertIsNone(metrics.si_sdr(np.zeros_like(b), b))


class BedDamageTests(unittest.TestCase):
    def setUp(self):
        self.d, self.b = signals(seed=11, dialogue_active=(0.0, 0.5))

    def test_identical_bed_has_no_damage(self):
        r = metrics.bed_damage(self.d, self.b, self.b.copy(), SR)
        self.assertEqual(r.si_sdr_db, metrics.DB_CEIL)
        self.assertAlmostEqual(r.lsd_inactive_db, 0.0, places=6)
        self.assertGreater(r.lsd_frames, 0)
        self.assertAlmostEqual(r.inactive_fraction, 0.5, delta=0.06)

    def test_uniform_gain_error_gives_constant_lsd(self):
        r = metrics.bed_damage(self.d, self.b, 2.0 * self.b, SR)
        self.assertAlmostEqual(r.lsd_inactive_db, 20 * np.log10(2.0), delta=0.01)
        self.assertEqual(r.si_sdr_db, metrics.DB_CEIL)

    def test_damage_in_inactive_region_shows_in_lsd_and_inactive_sdr(self):
        half = len(self.b) // 2
        est = self.b.copy()
        est[half:] += noise_like(est[half:], 0.05, 5)
        r = metrics.bed_damage(self.d, self.b, est, SR)
        self.assertGreater(r.lsd_inactive_db, 2.0)
        self.assertLess(r.si_sdr_inactive_db, 10.0)
        self.assertEqual(r.si_sdr_active_db, metrics.DB_CEIL)

    def test_damage_in_active_region_is_ignored_by_lsd(self):
        est = self.b.copy()
        est[: len(est) // 4] += noise_like(est[: len(est) // 4], 0.05, 6)
        r = metrics.bed_damage(self.d, self.b, est, SR)
        self.assertAlmostEqual(r.lsd_inactive_db, 0.0, places=6)
        self.assertLess(r.si_sdr_active_db, 15.0)
        self.assertEqual(r.si_sdr_inactive_db, metrics.DB_CEIL)

    def test_lowpassed_bed_has_large_lsd(self):
        from scipy import signal as sg
        sos = sg.butter(8, 2000, fs=SR, output="sos")
        lp = sg.sosfiltfilt(sos, self.b, axis=0).astype(np.float32)
        r = metrics.bed_damage(self.d, self.b, lp, SR)
        self.assertGreater(r.lsd_inactive_db, 15.0)

    def test_guard_excludes_region_next_to_dialogue(self):
        est = self.b.copy()
        edge = len(est) // 2
        guard = metrics.GUARD_FRAMES * int(SR * 0.02)
        est[edge: edge + guard] += noise_like(est[edge: edge + guard], 0.1, 7)
        r = metrics.bed_damage(self.d, self.b, est, SR)
        self.assertAlmostEqual(r.lsd_inactive_db, 0.0, places=6)

    def test_no_inactive_region(self):
        d, b = signals(seed=12)
        r = metrics.bed_damage(d, b, b + noise_like(b, 0.01, 8), SR)
        self.assertIsNone(r.lsd_inactive_db)
        self.assertIsNone(r.si_sdr_inactive_db)
        self.assertEqual(r.lsd_frames, 0)
        self.assertIsNotNone(r.si_sdr_db)

    def test_silent_reference_bed(self):
        r = metrics.bed_damage(self.d, np.zeros_like(self.b), 0.1 * self.b, SR)
        self.assertIsNone(r.si_sdr_db)
        self.assertGreater(r.lsd_inactive_db, 10.0)

    def test_mono_multichannel_and_sample_rates(self):
        for ch, sr in ((1, SR), (6, SR), (2, 8000), (2, 48000)):
            d, b = signals(seed=13, channels=ch, seconds=3.0, dialogue_active=(0.0, 0.5))
            if sr < SR:
                d, b = d[:: SR // sr], b[:: SR // sr]
            elif sr > SR:
                d, b = np.repeat(d, sr // SR, axis=0), np.repeat(b, sr // SR, axis=0)
            r = metrics.bed_damage(d, b, 2.0 * b, sr)
            self.assertAlmostEqual(r.lsd_inactive_db, 20 * np.log10(2.0), delta=0.05, msg=f"{ch}ch {sr}")

    def test_batching_does_not_change_result(self):
        half = len(self.b) // 2
        est = self.b.copy()
        est[half:] += noise_like(est[half:], 0.05, 9)
        whole = metrics.bed_damage(self.d, self.b, est, SR).lsd_inactive_db
        old = metrics.LSD_BATCH_FRAMES
        metrics.LSD_BATCH_FRAMES = 7
        try:
            batched = metrics.bed_damage(self.d, self.b, est, SR).lsd_inactive_db
        finally:
            metrics.LSD_BATCH_FRAMES = old
        self.assertAlmostEqual(whole, batched, places=9)

    def test_input_validation(self):
        with self.assertRaises(metrics.MetricError):
            metrics.bed_damage(self.d, self.b, self.b[:-1], SR)
        bad = self.b.copy()
        bad[5, 0] = np.inf
        with self.assertRaises(metrics.MetricError):
            metrics.bed_damage(self.d, self.b, bad, SR)

    def test_background_only_clip_is_scored_as_wholly_inactive(self):
        silent = np.zeros_like(self.d)
        r = metrics.bed_damage(silent, self.b, self.b.copy(), SR)
        self.assertEqual(r.inactive_fraction, 1.0)
        self.assertIsNone(r.si_sdr_active_db)
        self.assertEqual(r.si_sdr_inactive_db, metrics.DB_CEIL)
        self.assertAlmostEqual(r.lsd_inactive_db, 0.0, places=6)
        damaged = metrics.bed_damage(silent, self.b, 0.5 * self.b, SR)
        self.assertAlmostEqual(damaged.lsd_inactive_db, 20 * np.log10(2.0), delta=0.01)
        active, inactive = metrics._activity_masks(silent, SR, allow_silent=True)
        self.assertFalse(active.any())
        self.assertTrue(inactive.all())
        with self.assertRaises(metrics.MetricError):
            metrics._activity_masks(silent, SR)


class FalseDialogueTests(unittest.TestCase):
    def test_dialogue_stem_energy_relative_to_mixture(self):
        _, b = signals(seed=3)
        self.assertEqual(metrics.false_dialogue(b, np.zeros_like(b)).dialogue_to_mixture_db, metrics.DB_FLOOR)
        r = metrics.false_dialogue(b, 0.1 * b)
        self.assertAlmostEqual(r.dialogue_to_mixture_db, -20.0, places=4)
        self.assertAlmostEqual(r.dialogue_rms_dbfs, 10 * np.log10(float(((0.1 * b.astype(np.float64)) ** 2).mean())), places=4)
        self.assertIsNone(metrics.false_dialogue(np.zeros_like(b), 0.1 * b).dialogue_to_mixture_db)
        with self.assertRaises(metrics.MetricError):
            metrics.false_dialogue(b, b[:-1])


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.d, self.b = signals(seed=21)
        self.mix = self.d + self.b

    def check(self, d, b, **kw):
        return metrics.check_reconstruction(self.mix, d, b, SR, **kw)

    def test_exact_split_passes(self):
        r = self.check(self.d, self.b)
        self.assertTrue(r.passed)
        self.assertEqual(r.reasons, ())
        self.assertLess(r.residual_db, -100.0)

    def test_any_split_that_sums_back_passes(self):
        r = self.check(0.3 * self.mix, 0.7 * self.mix)
        self.assertTrue(r.passed)

    def test_residual_level_is_measured_against_the_gate(self):
        rms = np.sqrt((self.mix ** 2).mean())
        for level, expect in ((-70.0, True), (-50.0, False)):
            err = noise_like(self.mix, rms * 10 ** (level / 20), 22)
            r = self.check(self.d, self.b + err)
            self.assertAlmostEqual(r.residual_db, level, delta=0.3)
            self.assertEqual(r.passed, expect, level)

    def test_custom_threshold(self):
        err = noise_like(self.mix, 1e-3, 23)
        self.assertFalse(self.check(self.d, self.b + err, threshold_db=-80.0).passed)
        self.assertTrue(self.check(self.d, self.b + err, threshold_db=-20.0).passed)

    def test_length_and_channel_mismatch_fail_without_raising(self):
        for bad_d, bad_b in ((self.d[:-1], self.b), (self.d, self.b[:, :1]), (self.d[:, 0], self.b[:, 0])):
            r = self.check(bad_d, bad_b)
            self.assertFalse(r.passed)
            self.assertIsNone(r.residual_db)
            self.assertTrue(r.reasons)

    def test_nonfinite_fails(self):
        bad = self.b.copy()
        bad[3, 1] = np.nan
        r = self.check(self.d, bad)
        self.assertFalse(r.passed)
        self.assertIn("NaN", r.reasons[0])

    def test_delayed_outputs_fail_and_report_the_lag(self):
        late_d = np.roll(self.d, 100, axis=0)
        late_b = np.roll(self.b, 100, axis=0)
        r = self.check(late_d, late_b)
        self.assertFalse(r.passed)
        self.assertEqual(r.lag_samples, 100)
        self.assertTrue(any("shifted by 100" in x for x in r.reasons))

    def test_early_outputs_report_negative_lag(self):
        r = self.check(np.roll(self.d, -40, axis=0), np.roll(self.b, -40, axis=0))
        self.assertEqual(r.lag_samples, -40)

    def test_single_bad_channel_is_visible(self):
        bad = self.b.copy()
        bad[:, 1] += noise_like(bad[:, 1], 0.05, 24)
        r = self.check(self.d, bad)
        self.assertFalse(r.passed)
        self.assertGreater(r.worst_channel_db, r.residual_db)
        self.assertGreater(r.max_abs_residual, 0.05)

    def test_silent_mixture(self):
        z = np.zeros_like(self.d)
        exact = metrics.check_reconstruction(z, z, z, SR)
        self.assertTrue(exact.passed)
        self.assertEqual(exact.silent_judged_residual_rms, 0.0)
        r = metrics.check_reconstruction(z, z, z + 0.01, SR)
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.silent_judged_residual_rms, 0.01, delta=1e-8)

    def test_silent_mixture_uses_absolute_residual_tolerance_inside_declared_band(self):
        z = np.zeros_like(self.d)
        below = metrics.check_reconstruction(z, z, z + 5e-9, SR, band_limit_hz=6000.0)
        self.assertTrue(below.passed)
        self.assertLessEqual(below.silent_judged_residual_rms, metrics.SILENT_RECONSTRUCTION_RMS_TOLERANCE)

        above = metrics.check_reconstruction(z, z, z + 1e-7, SR, band_limit_hz=6000.0)
        self.assertFalse(above.passed)
        self.assertGreater(above.silent_judged_residual_rms, metrics.SILENT_RECONSTRUCTION_RMS_TOLERANCE)
        self.assertIn("absolute tolerance", above.reasons[0])

    def test_silent_declared_band_does_not_gate_out_of_band_residual(self):
        z = np.zeros_like(self.d)
        time = np.arange(self.d.shape[0], dtype=np.float64) / SR
        out_of_band = (1e-3 * np.sin(2.0 * np.pi * 7000.0 * time)).astype(np.float32)[:, None]
        out_of_band = np.repeat(out_of_band, self.d.shape[1], axis=1)
        result = metrics.check_reconstruction(z, z, out_of_band, SR, band_limit_hz=6000.0)
        self.assertTrue(result.passed)
        self.assertGreater(result.max_abs_residual, 1e-4)
        self.assertLessEqual(result.silent_judged_residual_rms, metrics.SILENT_RECONSTRUCTION_RMS_TOLERANCE)

    def test_as_dict_is_json_serialisable(self):
        import json
        json.dumps(self.check(self.d, self.b).as_dict())
        json.dumps(self.check(self.d[:-1], self.b).as_dict())


if __name__ == "__main__":
    unittest.main()
