import hashlib
import json
import math
import struct
import tempfile
import unittest
import zlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import audiomath as am
import ingest
import mixgen

DURATION = 4.0


def speechlike(seconds, rng):
    n = int(seconds * am.SR)
    env = (np.sin(2 * np.pi * 3.0 * np.arange(n) / am.SR) > -0.2).astype(np.float32)
    return (rng.standard_normal(n).astype(np.float32) * 0.1 * env)


def rir(rt60, rng):
    n = int(rt60 * 1.2 * am.SR)
    t = np.arange(n) / am.SR
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[0] = 8.0
    return h.astype(np.float32)


def make_corpus(cache: Path):
    rng = np.random.default_rng(1)
    specs = [
        ("dialogue", "dlg-a", "spk-a", [], speechlike(3.0, rng), 1),
        ("dialogue", "dlg-b", "spk-b", [], speechlike(3.0, rng), 1),
        ("dialogue", "dlg-w", "spk-w", ["whisper"], speechlike(3.0, rng), 1),
        ("music", "mus-i", "trk-i", ["instrumental"], rng.standard_normal((am.SR * 6, 2)).astype(np.float32) * 0.1, 2),
        ("music", "mus-v", "trk-v", ["vocals"], rng.standard_normal((am.SR * 6, 2)).astype(np.float32) * 0.1, 2),
        ("sfx", "sfx-l", "pack-l", ["loud"], rng.standard_normal((am.SR * 2, 2)).astype(np.float32) * 0.2, 2),
        ("ambience", "amb-c", "amb-c", ["crowd"], rng.standard_normal((am.SR * 5, 2)).astype(np.float32) * 0.05, 2),
        ("rir", "rir-1", "rir-1", [], rir(1.0, rng), 1),
        ("rir", "rir-short", "rir-short", [], rir(0.1, rng), 1),
    ]
    items = []
    for role, item_id, group, tags, audio, ch in specs:
        item = {
            "id": item_id, "role": role, "group": group, "tags": tags, "split": "dev",
            "url": f"https://example.org/{item_id}.wav", "license_spdx": "CC0-1.0",
        }
        path = ingest.cache_path(cache, item)
        path.parent.mkdir(parents=True, exist_ok=True)
        am.write_wav(path, audio.reshape(-1, ch), am.SR)
        item["sha256"] = ingest.sha256_file(path)
        items.append(item)
    return items


class MixgenTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.items = make_corpus(self.cache)

    def tearDown(self):
        self.tmp.cleanup()

    def clip(self, recipe_id, index=0, seed=7):
        ctx = mixgen.Context(self.items, self.cache, seed, no_codec=True, duration_override=DURATION)
        return mixgen.generate_clip(mixgen.RECIPES[recipe_id], index, "dev", ctx)


class AudioMathTests(unittest.TestCase):
    def test_lufs_of_full_scale_997hz_sine(self):
        t = np.arange(am.SR * 5) / am.SR
        x = np.sin(2 * np.pi * 997 * t).astype(np.float32)[:, None]
        self.assertAlmostEqual(am.integrated_lufs(x), -3.01, delta=0.1)

    def test_lufs_silence_is_minus_inf(self):
        self.assertTrue(math.isinf(am.integrated_lufs(np.zeros((am.SR * 2, 2), np.float32))))

    def test_rt60_estimate(self):
        h = rir(1.0, np.random.default_rng(3))
        self.assertAlmostEqual(am.estimate_rt60(h), 1.0, delta=0.15)

    def test_empty_rir_has_undefined_rt60(self):
        self.assertTrue(math.isnan(am.estimate_rt60(np.empty(0, dtype=np.float32))))

    def test_ffmpeg_and_wav_fallback_use_the_same_channel_adaptation(self):
        source = np.random.default_rng(3).standard_normal((4096, 6)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "surround.wav"
            wavfile.write(path, 32000, source)
            payload = BytesIO()
            wavfile.write(payload, 32000, source)
            pipe_payload = bytearray(payload.getvalue())
            struct.pack_into("<I", pipe_payload, 4, 0xFFFFFFFF)
            data_chunk = pipe_payload.index(b"data")
            struct.pack_into("<I", pipe_payload, data_chunk + 4, 0xFFFFFFFF)
            with patch.object(am, "have_ffmpeg", return_value=True), patch.object(
                    am.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=bytes(pipe_payload), stderr=b"")):
                ffmpeg_result = am.load_audio(path, channels=2, sr=am.SR)
            with patch.object(am, "have_ffmpeg", return_value=False):
                scipy_result = am.load_audio(path, channels=2, sr=am.SR)
        np.testing.assert_array_equal(ffmpeg_result, scipy_result)


class LoopToLengthTests(unittest.TestCase):
    def test_empty_source_fails_instead_of_looping_forever(self):
        with self.assertRaisesRegex(ValueError, "empty audio source"):
            mixgen.loop_to_length(np.empty((0, 2), np.float32), 100, np.random.default_rng(1))

    def test_single_sample_source_repeats_to_requested_length(self):
        source = np.array([[0.25, -0.5]], np.float32)
        out, offset = mixgen.loop_to_length(source, 100, np.random.default_rng(1))
        self.assertEqual(offset, 0)
        self.assertEqual(out.shape, (100, 2))
        np.testing.assert_array_equal(out, np.repeat(source, 100, axis=0))

    def test_short_source_fills_long_output_with_finite_audio(self):
        source = np.linspace(-0.5, 0.5, 256, dtype=np.float32).reshape(128, 2)
        out, offset = mixgen.loop_to_length(source, 600_000, np.random.default_rng(1))
        self.assertEqual(offset, 0)
        self.assertEqual(out.shape, (600_000, 2))
        self.assertTrue(np.isfinite(out).all())
        crossfade = max(1, min(2400, source.shape[0] // 4))
        np.testing.assert_array_equal(out[: source.shape[0] - crossfade], source[:-crossfade])


class ItemPoolCacheTests(MixgenTestBase):
    def test_lru_cache_evicts_one_old_entry_and_retains_a_hot_entry(self):
        pool = mixgen.ItemPool(self.items, self.cache, "dev", audio_cache_capacity=2)
        first, second, third = self.items[:3]
        calls = []

        def load(path, channels):
            calls.append(Path(path).name)
            return np.array([[len(calls)]], dtype=np.float32)

        with patch.object(mixgen.am, "load_audio", side_effect=load):
            first_audio = pool.load(first, 1)
            pool.load(second, 1)
            self.assertIs(pool.load(first, 1), first_audio)
            pool.load(third, 1)
            self.assertEqual(len(pool._audio), 2)
            self.assertEqual(calls.count(ingest.cache_path(self.cache, first).name), 1)
            pool.load(second, 1)
            self.assertEqual(calls.count(ingest.cache_path(self.cache, second).name), 2)


class RecipeTests(MixgenTestBase):
    def test_duration_override_must_be_finite_and_positive(self):
        for duration in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(duration=duration):
                ctx = mixgen.Context(self.items, self.cache, 7, duration_override=duration)
                with self.assertRaisesRegex(mixgen.GenerationError, "finite and positive"):
                    mixgen.generate_clip(mixgen.RECIPES["A9"], 0, "dev", ctx)

    def test_positive_duration_override_controls_sample_count(self):
        ctx = mixgen.Context(self.items, self.cache, 7, no_codec=True, duration_override=1.5)
        arrays, _, _ = mixgen.generate_clip(mixgen.RECIPES["A9"], 0, "dev", ctx)
        self.assertEqual(arrays["mixture"].shape[0], int(1.5 * am.SR))

    def test_mixture_is_exact_sum(self):
        arrays, sr, _ = self.clip("A1")
        np.testing.assert_allclose(arrays["mixture"], arrays["dialogue"] + arrays["bed"], atol=1e-6)
        self.assertEqual(sr, am.SR)
        self.assertEqual(arrays["mixture"].shape, (int(DURATION * am.SR), 2))

    def test_loudness_normalized(self):
        arrays, _, meta = self.clip("A1")
        self.assertAlmostEqual(am.integrated_lufs(arrays["mixture"]), mixgen.TARGET_LUFS, delta=0.5)
        self.assertLessEqual(float(np.abs(arrays["mixture"]).max()), mixgen.PEAK_LIMIT + 1e-4)

    def test_snr_hits_target(self):
        for rid in ("A1", "A7"):
            _, _, meta = self.clip(rid)
            self.assertAlmostEqual(meta["measured_snr_db"], meta["target_snr_db"], delta=0.3)

    def test_a7_is_low_snr(self):
        _, _, meta = self.clip("A7")
        self.assertLessEqual(meta["measured_snr_db"], 0.3)

    def test_determinism_and_subset_independence(self):
        a, _, _ = self.clip("A1", 0)
        b, _, _ = self.clip("A1", 0)
        self.assertEqual(hashlib.sha256(a["mixture"].tobytes()).hexdigest(),
                         hashlib.sha256(b["mixture"].tobytes()).hexdigest())
        self.clip("A2", 0)
        c, _, _ = self.clip("A1", 0)
        np.testing.assert_array_equal(a["mixture"], c["mixture"])

    def test_different_seed_or_index_differs(self):
        a, _, _ = self.clip("A1", 0, seed=7)
        b, _, _ = self.clip("A1", 0, seed=8)
        c, _, _ = self.clip("A1", 1, seed=7)
        self.assertFalse(np.array_equal(a["mixture"], b["mixture"]))
        self.assertFalse(np.array_equal(a["mixture"], c["mixture"]))

    def test_a9_bed_is_silent_a10_dialogue_is_silent(self):
        nine, _, _ = self.clip("A9")
        ten, _, _ = self.clip("A10")
        self.assertEqual(float(np.abs(nine["bed"]).max()), 0.0)
        np.testing.assert_allclose(nine["mixture"], nine["dialogue"], atol=1e-6)
        self.assertEqual(float(np.abs(ten["dialogue"]).max()), 0.0)
        np.testing.assert_allclose(ten["mixture"], ten["bed"], atol=1e-6)

    def test_a3_uses_in_range_rir_only(self):
        _, _, meta = self.clip("A3")
        self.assertEqual(meta["rir_id"], "rir-1")
        self.assertTrue(0.4 <= meta["rt60_s"] <= 2.0)

    def test_reverb_keeps_timing(self):
        rng = np.random.default_rng(5)
        dry = np.zeros(am.SR, np.float32)
        dry[20000] = 1.0
        wet = mixgen.reverberate(dry, rir(0.5, rng))
        self.assertEqual(int(np.argmax(np.abs(wet))), 20000)

    def test_a6_uses_two_distinct_groups(self):
        _, _, meta = self.clip("A6")
        groups = {s["group"] for s in meta["speakers"]}
        self.assertEqual(len(groups), 2)

    def test_a8_requires_whisper_tag(self):
        _, _, meta = self.clip("A8")
        self.assertIn("dlg-w", {s["id"] for s in meta["sources"]})
        self.assertNotIn("dlg-a", {s["id"] for s in meta["sources"]})

    def test_tag_filters_for_beds(self):
        _, _, m2 = self.clip("A2")
        _, _, m5 = self.clip("A5")
        self.assertIn("sfx-l", {s["id"] for s in m2["sources"]})
        self.assertIn("mus-v", {s["id"] for s in m5["sources"]})

    def test_missing_source_category_is_an_error(self):
        self.items = [i for i in self.items if i["role"] != "ambience"]
        with self.assertRaises(mixgen.GenerationError):
            self.clip("A4")

    def test_a11_variants(self):
        expected = {0: ("mono", 48000, 1), 1: ("sr8000", 8000, 2), 2: ("sr16000", 16000, 2), 3: ("5.1", 48000, 6)}
        for idx, (variant, sr, ch) in expected.items():
            arrays, out_sr, meta = self.clip("A11", idx)
            self.assertEqual(meta["variant"], variant)
            self.assertEqual(out_sr, sr)
            self.assertEqual(arrays["mixture"].shape[1], ch)
            np.testing.assert_allclose(arrays["mixture"], arrays["dialogue"] + arrays["bed"], atol=1e-5)

    def test_splits_are_respected(self):
        for item in self.items:
            item["split"] = "test"
        ctx = mixgen.Context(self.items, self.cache, 7, no_codec=True, duration_override=DURATION)
        with self.assertRaises(mixgen.GenerationError):
            mixgen.generate_clip(mixgen.RECIPES["A1"], 0, "dev", ctx)

    def test_split_counts(self):
        self.assertEqual(mixgen.split_counts(30), {"dev": 9, "test": 21})
        self.assertEqual(mixgen.split_counts(3), {"dev": 1, "test": 2})


class CorpusWriteTests(MixgenTestBase):
    def test_generate_corpus_writes_files_and_manifest(self):
        out = self.root / "corpus"
        manifest = mixgen.generate_corpus(
            self.items, self.cache, out, split="dev", recipes=["A1", "A9"], seed=3,
            items_sha256="x" * 64, count_override=1, duration_override=DURATION, no_codec=True)
        self.assertEqual([c["clip_id"] for c in manifest["clips"]], ["a1-dev-000", "a9-dev-000"])
        clip = manifest["clips"][0]
        for name in ("mixture", "dialogue", "bed"):
            path = out / clip["clip_id"] / f"{name}.wav"
            self.assertEqual(ingest.sha256_file(path), clip["files"][name])
        on_disk = json.loads((out / "corpus.manifest.json").read_text())
        self.assertEqual(on_disk["items_sha256"], "x" * 64)

    def test_unknown_recipe_rejected(self):
        with self.assertRaises(mixgen.GenerationError):
            mixgen.generate_corpus(self.items, self.cache, self.root / "o", split="dev", recipes=["A99"],
                                   seed=1, items_sha256="x", count_override=1)


class LanguageTests(MixgenTestBase):
    GROUPS = {"ar": "spk-ar", "pt": "spk-pt", "ru": "spk-ru"}

    def add_dialogue(self, language):
        item_id = f"dlg-{language}"
        item = {"id": item_id, "role": "dialogue", "group": self.GROUPS[language], "tags": [], "split": "dev",
                "url": f"https://example.org/{item_id}.wav", "license_spdx": "CC0-1.0", "language": language}
        path = ingest.cache_path(self.cache, item)
        am.write_wav(path, speechlike(3.0, np.random.default_rng(zlib.crc32(item_id.encode()))).reshape(-1, 1), am.SR)
        item["sha256"] = ingest.sha256_file(path)
        self.items.append(item)

    def add_all(self):
        for language in self.GROUPS:
            self.add_dialogue(language)

    def test_english_recipes_ignore_non_english_dialogue(self):
        self.add_all()
        for index in range(6):
            _, _, meta = self.clip("A1", index)
            self.assertIn(meta["speakers"][0]["group"], {"spk-a", "spk-b", "spk-w"})
            self.assertNotIn("dialogue_language", meta)

    def test_english_output_is_unchanged_by_extra_languages(self):
        before, _, _ = self.clip("A1", 0)
        self.add_all()
        after, _, _ = self.clip("A1", 0)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_an_explicit_english_language_matches_a_missing_one(self):
        before, _, _ = self.clip("A1", 0)
        for item in self.items:
            if item["role"] == "dialogue":
                item["language"] = "en"
        after, _, _ = self.clip("A1", 0)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_a13_cycles_through_the_non_english_languages_present(self):
        self.add_all()
        languages = [self.clip("A13", index)[2]["dialogue_language"] for index in range(6)]
        self.assertEqual(languages, ["ar", "pt", "ru", "ar", "pt", "ru"])

    def test_a13_uses_only_speakers_of_the_chosen_language(self):
        self.add_all()
        for index in range(3):
            _, _, meta = self.clip("A13", index)
            self.assertEqual(meta["speakers"][0]["group"], self.GROUPS[meta["dialogue_language"]])

    def test_a13_fails_without_non_english_dialogue(self):
        with self.assertRaisesRegex(mixgen.GenerationError, "non-English"):
            self.clip("A13", 0)

    def test_optional_recipes_are_left_out_of_the_default_list(self):
        self.assertIn("A13", mixgen.RECIPES)
        self.assertNotIn("A13", mixgen.DEFAULT_RECIPE_IDS)
        self.assertEqual(len(mixgen.DEFAULT_RECIPE_IDS), 12)


@unittest.skipUnless(am.have_ffmpeg(), "ffmpeg not installed")
class CodecTests(unittest.TestCase):
    def test_codec_roundtrip_is_aligned(self):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((am.SR * 4, 2)) * 0.1).astype(np.float32)
        x = np.convolve(x[:, 0], np.ones(8) / 8, mode="same").astype(np.float32)
        x = np.stack([x, x], axis=1)
        for codec in ("aac", "ac3"):
            y, _ = mixgen.codec_roundtrip(x, codec)
            self.assertEqual(y.shape, x.shape)
            corr = np.corrcoef(x[:, 0], y[:, 0])[0, 1]
            self.assertGreater(corr, 0.9, codec)


if __name__ == "__main__":
    unittest.main()
