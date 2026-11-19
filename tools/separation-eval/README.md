# Separation evaluation corpus tooling

Design and rubric: `docs/audits/separation-eval-corpus-and-rubric.md` (gated repo).

`ingest.py` turns a hand-curated list of candidate items into a verified item manifest. It is stdlib-only (it uses `ffprobe` when on PATH, else `wave` for WAV files).

- Rejects any item whose license is not on the allow-list, is noncommercial or no-derivatives, or is share-alike unless `--allow-sharealike` is passed.
- Downloads over https only, into a cache directory that is **not** committed, and verifies any expected `sha256`.
- Rejects duplicate ids and byte-identical files.
- Assigns `dev` or `test` by hashing the source `group` (speaker, track, SFX pack), so a group never spans both splits.
- `verify` re-checks the manifest hash, split assignment, licenses and cached file hashes.

```bash
python tools/separation-eval/ingest.py ingest --items items.jsonl --cache D:/corpus-cache --out items.manifest.json
python tools/separation-eval/ingest.py verify --manifest items.manifest.json --cache D:/corpus-cache
python -m unittest discover -s tools/separation-eval -p 'test_*.py'
```

The licenses in `items.example.jsonl` are placeholders. Verify each item's license at its `license_evidence_url` before adding it.

## Python environment

The corpus generator, audio metrics, and evaluation runner use pinned NumPy and SciPy dependencies.
From a clean checkout, create an environment and install them before running those tools:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r tools/separation-eval/requirements.txt
```

`ingest.py` and `recipe_coverage.py` only need the Python standard library when no cached RIR audio is
being measured. The full test suite and all DSP generation require the pinned environment above.

## Mixture generator

`mixgen.py` renders Tier A clips (mixture, dialogue reference, bed reference; 48 kHz float32 WAV plus `meta.json`) from a verified item manifest, for recipes A1 to A12 in the design doc.

```bash
python tools/separation-eval/mixgen.py recipes
python tools/separation-eval/mixgen.py generate --manifest items.manifest.json --cache D:/corpus-cache --out D:/corpus-dev --split dev --seed 1234
```

- Refuses to run unless the item manifest verifies against the cache.
- Each clip depends only on `(seed, clip id)`, so any subset regenerates identically.
- Uses only items from the requested split. A recipe fails loudly if the corpus lacks the tagged sources it needs; it never relaxes the tags.
- Item tags the recipes rely on (set `tags` in the items file): `instrumental`, `vocals` (music); `loud` (sfx); `crowd` (ambience); `whisper` (dialogue). RIRs are filtered to RT60 0.4 to 2.0 s.
- Dialogue-to-bed SNR is calibrated on dialogue-active frames, then the mixture is normalized to -23 LUFS with a joint peak guard applied to all three signals, so `mixture == dialogue + bed` exactly (before any codec round trip, which affects the mixture only).
- Reverberant dialogue (A3) keeps the reverb in the dialogue reference; it is what the separator must remove.
- A11 renders the refs and mixture in alternate formats: mono, 8 kHz, 16 kHz, and 5.1 (dialogue on the center channel).
- `--no-codec` skips the AAC/AC-3 round trip; otherwise 25% of clips (except A11) get one, which requires ffmpeg.
- A12 (20 min) holds several full-length float arrays in memory; expect a few GB of RAM.
- `audiomath.py` (loudness, activity mask, RT60, loading) is shared with the upcoming metric library.

## Runner and Spleeter baseline

`run_eval.py` scores a candidate on a corpus from `mixgen.py`. The candidate runs out of process. For Spleeter the command is `Trackdub.Benchmarks separation-eval` (`src/Trackdub.Benchmarks/SeparationEvalRunner.cs`), which drives the real `SpleeterStemSeparationEngine` through the headless composition root in one warm host and records wall time, RTF and peak working set per job.

```bash
python tools/separation-eval/run_eval.py --corpus D:/corpus-dev --work D:/eval-work --out spleeter-dev.json \
    --hardware-label "<machine description>" --provider cpu --model-cache-directory <Trackdub model-cache root>
```

The model-cache root must contain `model-cache-records.json` with a registered Spleeter model whose root contains
`vocals.onnx` and `accompaniment.onnx`. If `--model-cache-directory` is omitted, `--model-directory` is used as the
fallback model-cache root; neither option registers an arbitrary two-file directory as a Spleeter model. For direct
local weights, the runner also honors `TRACKDUB_SPLEETER_ONNX_PATH` when it points to that two-file directory.

- Inputs are prepared the way the pipeline feeds Spleeter: stereo (5.1 is downmixed), 44.1 kHz, PCM16. The engine writes mono 44.1 kHz stems.
- Two domains are scored. The reconstruction gate runs in the separator domain at 44.1 kHz: mono stems (Spleeter's) must add up to the mono downmix of the input, stereo stems to the stereo input. Bed leakage, bed damage and dialogue SI-SDR run in the reference domain: stems are resampled to the clip rate, a mono stem is tiled to stereo and a stereo stem is scored as stereo. For a mono separator the stereo-to-mono collapse and resampling are therefore part of what is measured. Each stem keeps its own layout (`stem_channels` in the clip record); a mono stem paired with a stereo one is tiled, and stems with more than two channels fail the clip. Results schema version 6 made this change: until then every stem was collapsed to mono, which capped even an exact stereo bed at a median bed SI-SDR of 6.6 dB on the v1.1 dev split, so version 6 damage and leakage figures for stereo candidates are not comparable with version 5 ones.
- The reconstruction gate checks the residual against the fixed threshold, judged without the first and last 50 ms of the clip (`edge_exclusion_s`, applied to every candidate, with the mean power of the residual inside each excluded region (low-passed to the declared band when one is declared) still required to stay below -3 dB of the clip's mean mixture power, so neither edge can be dropped outright (each edge is judged separately; the measure is relative to the clip, so a quiet or silent edge does not inflate it and it does not depend on the clip length; a digitally silent clip tolerates at most -80 dBFS RMS of output in its edges) and `edge_residual_db` records the worse edge; clips of 1 s or less are judged whole, so the exclusion is at most 10% of a clip). Separators and their writers leave a few zeroed or unfinished samples at the ends, which is not a separation fault. The whole-clip residuals stay in the results as `residual_with_edges_db`, `in_band_residual_with_edges_db` and (for a silent mixture) `silent_residual_rms_with_edges`, filled on every path, and are never gated. `residual_db`, `in_band_residual_db`, `max_abs_residual`, `worst_channel_db` and `silent_judged_residual_rms` now describe the judged region only: results schema version 4 changed their definition, do not compare them with version 3 results, and schema version 5 added the per-edge backstop (`edge_residual_db`) to the gate, so version 5 verdicts are not comparable with version 4 ones; non-finite samples fail the gate wherever they are. When a candidate declares a band limit (`--band-limit-hz`, 11025 for Spleeter), only residual energy inside that band is judged; above-band residual is reported through `bandwidth_retained_db` and does not change pass/fail. Without a declared limit, the gate uses the full band. If there is no measurable reference energy inside a declared band, the gate uses the full-band mixture energy as the -60 dB reference floor. A silent mixture must produce silence inside the judged band. Stem precision/rounding (`--output-bits`, `--output-rounding`) only record `quantization_allowance_db` as a diagnostic, next to `in_band_residual_db`.
- `worst_window_leakage_db` is the worst qualifying 1 s window of the dialogue-relative leakage (projected error energy over reference-dialogue energy in that window). It is not bed leakage and is not comparable in units with `leakage_to_bed_db`. A window qualifies when at least 25% of its samples are dialogue-active and its mean dialogue power per active sample is at least 1% (-20 dB) of the clip's; windows below that are excluded and cannot set the value, and the field is null when no window qualifies. It stays bounded when the reference bed is near silent, and it is reported even when the whole-clip bed-relative figure is not applicable. Results schema version 3 changed its definition; do not compare it with version 2 results.
- Undefined metrics (for example bed leakage when no dialogue exists) are recorded as skipped, not as failures. A background-only clip (A10, no active dialogue frames) is scored for bed damage as wholly dialogue-inactive (`si_sdr_inactive_db`, `lsd_inactive_db`), and gets `false_dialogue`: dialogue-stem energy relative to the mixture (`dialogue_to_mixture_db`, lower is better) and its RMS; clips with dialogue record it as null. Before schema version 6, A10 had no bed metrics. A failed or missing job fails only its own clip.
- The results JSON records the corpus manifest hash, hardware label, per-clip metrics, per-recipe median and worst-decile aggregates, and cold versus warm RTF. It also records `model_provenance`: every model root the engine actually executed (the `TRACKDUB_SPLEETER_ONNX_PATH` override can supersede the planned model) with the sha256 of its `vocals.onnx` and `accompaniment.onnx`, compared against the pinned revision hashes in `bundled-models.manifest.json`, so results are attributable to the weights that produced them.
- The weights under test must be the pinned revision in `bundled-models.manifest.json`; check their sha256 before trusting a baseline.
- Known failures, Spleeter baseline under the edge-exclusion gate (`edge_exclusion_s` 0.05, results schema 4): none of the 71 dev clips fail (`a4-dev-002`, a quiet mixture with RMS 0.024, passes by 0.03 dB at -60.03 dB in-band), and 2 of the 172 test clips fail: `a5-test-005` (-59.6 dB) and `a5-test-015` (-59.9 dB). Both are quiet mixtures (about -32.5 dBFS RMS) with a modeled PCM16 truncation floor of about -56.5 dB and no edge error, so their residual is spread across the clip. Truncation alone would fail the gate only below about 0.018 RMS (about -35 dBFS), which real quiet dialogue can reach; the quantization allowance is reported but deliberately does not relax the gate. Without the edge exclusion (whole clip) 4 of 71 dev clips and 20 of 172 test clips failed: 18 of the 20 test failures, and `a11-dev-002`, `a7-dev-004` and `a10-dev-001` on dev, were start or end edges (Spleeter writes exact zeros for the first 24 to 43 samples, and a few clips also end unfinished), with in-band residuals of -49.8 to -59.9 dB whole-clip and -61.1 to -72.1 dB once the edges are excluded.

## Other candidates (Bandit v2)

`run_eval.py --candidate <name>` selects a declared-output profile (`CANDIDATE_PROFILES`) and passes the name to the runner as `--model`. `candidates/bandit_v2_runner.py` is an out-of-process runner with the same command line as `Trackdub.Benchmarks separation-eval`, for the Bandit v2 multilingual checkpoint (`checkpoint-multi.ckpt` from Zenodo record 12701995, md5 checked against the listing) and a checkout of kwatcharasupat/bandit-v2. It needs a PyTorch environment (torch and torchaudio, pytorch_lightning, librosa, soundfile) and imports only the model class and the official chunked inference handler from the checkout; the checkpoint loads with `weights_only=True` and `strict=True`. Run it with `--runner "<python> tools/separation-eval/candidates/bandit_v2_runner.py" --model-directory <dir holding the checkpoint and repo/>`.

The model works at 48 kHz, so each job resamples the 44.1 kHz stereo input up, separates each channel as mono, and resamples every stem back (recorded in the per-job metadata, which `model_provenance.executed_metadata` carries into the results together with the checkpoint hashes and code revision). Stems are float32, so no output precision is declared and no band is declared. Two candidates differ only in the bed: `bandit-v2-multi-sum` is the model's own music + sfx estimate (the reconstruction gate then measures the model's additivity), `bandit-v2-multi-residual` is the mixture minus the dialogue estimate, which meets the reconstruction gate by construction, so its verdict says nothing about additivity. Compare them on leakage and damage, and judge the sum form against the gate.

## Finding and listing real items

`discover.py` builds the candidate list from `corpus-sources.v1.json` without downloading any audio. It queries Openverse (Freesound, Jamendo) and archive.org's LibriVox collection, keeps only items whose license maps onto the `ingest.py` allow-list (CC0, public domain, CC BY 3.0 or 4.0), caps items per creator, and writes `items.v1.jsonl`. API responses are cached, because Openverse's anonymous quota is 200 requests a day.

```bash
python tools/separation-eval/discover.py --spec tools/separation-eval/corpus-sources.v1.json --out tools/separation-eval/items.v1.jsonl --api-cache D:/api-cache
python tools/separation-eval/recipe_coverage.py --items tools/separation-eval/items.v1.jsonl --cache D:/corpus-cache
```

`recipe_coverage.py` checks every recipe in both splits and exits non-zero if a source category is thin,
the RIR cache is missing, or no cached RIR has an estimated RT60 in A3's 0.4–2.0 second range.

Caveats that affect what the corpus can show:
- Freesound items are 128 kbps MP3 previews (lossy, band-limited near 16 to 19 kHz); LibriVox is 64 kbps read speech truncated to `range_bytes`. Neither is film-grade audio, so above-band retention is only partly exercised.
- Music, SFX, ambience and whisper tags come from the search query (`tag_basis: search-query`). Listen to a sample of each tag before trusting the A2, A4, A5 and A8 strata.
- Impulse responses are checked for RT60 (0.4 to 2.0 s) from the audio cache before recipe coverage passes; mixgen applies the same range when generating A3.
- CC BY items carry their attribution text in the manifest; keep it with any redistributed copy.
- The group is the creator (or the LibriVox book), so one voice or artist stays on one side of the split.
