# Handoff: Cold-start core + two model pure wins

**Source audit:** `docs/audits/inference-pipeline-cold-start-audit-2026-10-25.md` (`92dc8f00`, landed `9adcddb4`)

**Covers:** C9, C2, C2b, M-H1, M2, M-H2/H3

**Status:** Merged to `origin/main` via PRs [#530](https://github.com/Mesililm-LLC/Trackdub/pull/530), [#533](https://github.com/Mesililm-LLC/Trackdub/pull/533), and their fix-ups. The branch `cutover` this file sits on is unrelated (editing-flow work).

**Next in audit order** (see `docs/audits/inference-pipeline-cold-start-audit-2026-10-25.md#recommended-next-steps`): C4 (highest remaining cold-start value — `ep-smoke` dominated the measured pre-stage cost), then C1/C3, then C7. M-H1/M2 are done; M-H2/H3 are done; M-H4 stays blocked (Opus `optimum::if` export is broken, not a code fix).

## What was shipped and why

All three were the audit's "do first" items for concrete reasons, not just by severity order.

### C9 — No production cold-start measurement existed

No pipeline-start to first-inference number anywhere, so none of the other findings could be ranked. A baseline run then *reordered* the audit: stage work totalled 21 s of a 298 s wall on a 6.3 s clip (86% gaps), so model-code items that look hot on short media are not what slows a short pipeline.

- **Shipped:** `src/Trackdub.Contracts/Diagnostics/ColdStartCapture.cs` and
  `PipelineColdStartTiming.cs`, wiring from `DubbingPipelineEngine.cs`, and the
  `run-manifest.json` `coldStartTiming` field (per-phase totals plus
  `firstInferenceCompletionElapsedMs`).
- **To reproduce:** `trackdub run pipeline --media <clip.mp4> --target-language es --model asr:whisper-small --model translation:opus-en-es --model tts:qwen3-tts-0.6b --output <dir> --progress text` and read `coldStartTiming`.

The second correction the audit should have flagged earlier: M-H4's "pure win" framing. The bundled `decoder_model_merged.onnx` exposes the cache interface, so it looks exploitable; step 1 with `use_cache_branch=true` throws `Starts must be a 1-D array` inside `optimum::if` on every installed pair, so the defect is in the export, not the caller. Documented in the audit as blocked upstream.

### C2 + C2b — The audit's critical cold-start fix

Two separable defects shared a root cause:

1. Below 2x source bytes of free disk, `ShouldCapture` returned false with only a trace-level warning, so every later launch rebuilt every engine with no surfaced diagnostic; re-review made explicit that the state must not be silently stale.
2. Nothing warmed after a model install, so the first real run paid the full build.

Both landed together because the second is the logical follow-on of the first:

- `src/Trackdub.Inference.Onnx/EpContext/EpContextCapture.cs` — surfaced C2a via a readiness warning and a background `EpContextWarmupService` pass; second pass added a `BackgroundEngineWarmupService` and `TRACKDUB_ENGINE_AUTOWARM=0` opt-out for C2b.
- The opt-out was made **process-wide** (`TRACKDUB_ENGINE_AUTOWARM`, `TRACKDUB_ENGINE_AUTOWARM_ENABLED`) to cover the CLI path — per-stage overrides didn't fit the warm path's granularity.

Measured before/after: TensorRT session create **25.4 s to 5.8 s** (4.4x), diarization **4.2 s to 1.6 s**.

### M-H1/M2 — 300 MB re-read + 52k-entry JSON parse per request

`Qwen3AsrEmbedTokens.Load` and `Qwen3AsrTokenizer.LoadAsync` ran inside `TranscribeAsync`; `WhisperTokenizerDecoder.LoadAsync` and `ReadMelBins` re-parsed `preprocessor_config.json` on every transcription and every TRT profile build. All are audio-independent, so they were made keyed sidecar caches:

- `src/Trackdub.Inference.Onnx/Qwen3Asr/Qwen3AsrSidecarCaches.cs` and
  `src/Trackdub.Inference.Onnx/Whisper/WhisperSidecarCaches.cs` — `SidecarCache<T>` per model
  root, invalidated on install.
- `src/Trackdub.Contracts/Diagnostics/ColdStartCapture.cs` already existed; the audit doc cites
  the `SidecarCache` invalidation contract.

### M-H2/H3 — KV-cache churn

The audit flagged quadratic `.ToArray()` on the Qwen3-TTS KV cache. The fix split across two commits:

1. `af3763e1` (opencode, from `cold-start/kv-churn`) — pre-sized the code-predictor KV buffer from 16 positions with a debug invariant; that was the real saving.
2. The follow-on (`8ed50d25`, then `1d715987` fix-up + review response) wired the KV-buffer lifecycle correctly so a throw in prefill dispose doesn't strand four pooled arrays (~290 MB each for the talker at production dims).

Separately verified: the remaining throw window between prefill-session disposal and the generation `try` is OOM-only.

## Subskills used

No `~/.agents/skills/` were used — the work predated the current skill set.

- `audit` — the original inference cold-start audit.
- `inference` — Opus `optimum::if` investigation, KV cache lifecycle and sizing, mel math verification.
- `composition` — DI wiring for the warmup queue and its lifecycle (startup opt-out).

For `inference`-layer follow-ups (C4, C1/C3, C7), load `audit` before `inference` so ranking is evidence-backed — the baseline reordered this audit and would do so again for the next items.

## Open items

- **C4** is the highest remaining cold-start item: `ep-smoke` at ~1.9 s per smoke session build dominated the measured pre-stage cost. Persisting/reusing smoke verdicts or moving smoke off the planning critical path onto the existing background warmup coordinator is the next pass.
- **C1, C3, C7** are the next cheapest cold-start fixes in the audit's order.
- **M-H4** stays blocked upstream (needs an upstream re-export of the Opus merged decoder).
- **M-H1/M2, M-H2/H3** are done; `cold-start/m10-mh5` is tracked separately and is not part of this handoff.
- The current `cutover` branch's uncommitted edits (editing-flow feature, not covered here) were not part of the audited work.

Recommended skills for the next session: `audit` and `inference`.
