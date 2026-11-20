---
name: trackdub-media-cleanup
description: Diagnose and verify media cleanup in Trackdub projects.
version: 0.1.0
author: Trackdub contributors, Hermes Agent; adapted from Mehdi Ksibi
license: MIT
platforms: [windows, macos, linux]
metadata:
  upstream: Mehdi-Ks/video-cleanup
  upstream_package: Video Cleanup-1.0.1-v1
---

# Trackdub media cleanup

Use Trackdub's project spine and existing media/inference services to diagnose speech quality, select the smallest justified intervention, and verify artifacts before delivery. This is an agent procedure, not a new runtime implementation or a promise that every upstream feature is available.

Adapted from **Video Cleanup by Mehdi Ksibi**, distributed as a Claude plugin by `Mehdi-Ks/video-cleanup`. See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE). This is not represented as an Anthropic-authored skill.

## When to use

- Noisy, quiet, echoey, clipped, uneven, or one-sided speech in source media.
- Cleanup before transcription, diarization, speaker-reference selection, or dubbing.
- Comparing source, separated speech, enhanced speech, preview mix, and final export.
- Requests to make a video sound/look better, where available operations must be checked first.
- Don't use this procedure to silently translate, clone voices, alter faces, add branding, or rerender the whole project when only diagnosis was requested.

## Prerequisites and boundaries

1. Read repository `AGENTS.md` and relevant subdirectory instructions. Source/tests win over stale docs.
2. Read the local source and docs for repository guidance; TensorRT-RTX / EP ABI pins come from `runtime/trt-rtx-ep.manifest.json` and the ONNX Runtime pin from `Directory.Packages.props`.
3. Run CLI help through `terminal` before relying on flags. From the public-core repository root, the portable source invocation is `dotnet run --project src/Trackdub.Cli --framework net10.0 -- <args>`. An installed `trackdub` executable is an alternative only after checking its help/version.
4. .NET 10 and Trackdub's media/runtime prerequisites apply. Do not run the upstream Python setup, install pip dependencies, or add Python, Conda, Docker, or CUDA Toolkit as end-user dependencies.
5. Provider registered != model downloaded != stage ran != stage succeeded. Inspect actual readiness, selected backend, fallback reason, and output artifacts.
6. Ask before downloads/installations, stating source, artifact, known size, license, and storage location; say when size is unknown. Cloud processing needs explicit disclosure and consent. Voice cloning needs specific consent; passing `--voice-clone` itself grants session consent, so never add it implicitly.

## Quick reference

Use these through `terminal`, with the repository root as working directory. Substitute actual paths; examples are not a batch to execute indiscriminately.

```text
dotnet run --project src/Trackdub.Cli --framework net10.0 -- --help
dotnet run --project src/Trackdub.Cli --framework net10.0 -- doctor --json
dotnet run --project src/Trackdub.Cli --framework net10.0 -- models status --json
dotnet run --project src/Trackdub.Cli --framework net10.0 -- project create --media "input.mp4" --output "projects/cleanup-v1.trackdub"
dotnet run --project src/Trackdub.Cli --framework net10.0 -- project info --project "projects/cleanup-v1.trackdub"
dotnet run --project src/Trackdub.Cli --framework net10.0 -- check --project "projects/cleanup-v1.trackdub"
dotnet run --project src/Trackdub.Cli --framework net10.0 -- run stage --help
dotnet run --project src/Trackdub.Cli --framework net10.0 -- run pipeline --help
```

`project create` registers source media without automatically starting transcription. It does not by itself prove enhancement ran. `doctor`/`check` diagnose runtime readiness, not perceptual media quality.

## Procedure

### 1. Establish scope and preserve evidence

- Locate the original local media or existing `.trackdub` project; prefer originals over platform-recompressed copies.
- Determine whether the goal is diagnosis, cleanup, transcription improvement, or an explicitly requested dub. Ask only where that distinction changes the action.
- Record source path, SHA-256, duration, stream layout, and relevant timestamps using tools. Keep source files untouched and prior successful artifacts intact.
- For experiments use a new project/output directory per variant. Existing project operations can update the latest artifact references; stage-specific file paths alone are not a complete immutable project snapshot.
- Ask before trimming, channel replacement/downmix, new language/voice, visual changes, subtitles, or branding not already requested.
- **Completion:** scope, source baseline, output location, and permitted changes are recorded.

### 2. Check runtime and project state

- Run `doctor --json`, `models status --json`, and project-aware `check` as relevant. Inspect project history with `project info` before reprocessing.
- Keep preflight enabled. Do not use `--skip-preflight`, `--force-rerun`, or hard GPU pins to disguise missing prerequisites.
- Use a healthy CPU fallback when allowed; record what actually executed rather than claiming GPU readiness from an installed provider.
- Record disabled, skipped, missing-prerequisite, canceled, and failed states separately from successful enhancement. A source-audio fallback is not cleaned audio.
- **Completion:** readiness and artifact reuse/fallback decisions are backed by real output.

### 3. Diagnose the media before choosing processing

- Inspect project quality artifacts under `artifacts/audio/quality` when available. Paths are defined by `src/Trackdub.Contracts/Projects/ProjectArtifactPaths.cs`; do not assume artifacts exist because a directory exists.
- Measure duration, channels, integrated loudness, peaks/clipping, silence, and left/right balance where supported. Label unavailable measurements explicitly.
- Optional independent read-only checks through `terminal`:

```text
ffprobe -v error -show_format -show_streams -of json "input.mp4"
ffmpeg -hide_banner -i "input.mp4" -vn -af ebur128=peak=true -f null -
```

- Extract diagnostic frames/spectrograms into a separate evidence directory only as needed; inspect images through an available vision tool. Do not claim to have watched/heard media unless a tool actually supported that examination.
- Do not call a percentile-derived level gap true SNR or claim that a louder export is necessarily clearer. Note wind, tonal interference, distance/room sound, damaged speech, and compression as hypotheses unless supported by measurements/listening.
- **Completion:** each proposed operation addresses an evidenced defect at stated timestamps.

### 4. Process through the existing Trackdub workflow

- Prefer existing SDK/application project workflows, artifact store, and stage records over unrelated standalone scripts. Inspect the exposed API/settings before invoking enhancement.
- `speech-enhancement` is a domain stage, but the current `run stage` CLI accepted list does **not** expose it directly. Do not invent `run stage --stage speech-enhancement`, `--denoise`, a strength knob, or a cleanup-only CLI command. If the requested host/API path is unavailable, deliver diagnosis and identify the exact execution blocker; do not launch a full dub as a substitute.
- Inspect `SpeechAudioEnhancementStageHandler` and the selected implementation/composition wiring before describing the backend as DeepFilterNet, FFmpeg, or NVIDIA AFX. A fallback implementation or stub is not proof of model execution.
- Preserve routing policy: `SpeechEnhancementGenerationStage` updates VAD and diarization audio; ASR retains its unprocessed source. Never silently redirect ASR to enhanced speech. Evaluate any routing change independently with regression evidence.
- Separation and enhancement are different operations. Preserve original/background stems and final spatial context. Do not apply speech-only mono/EQ processing to the entire stereo mix by default.
- Do not inherit the upstream universal -14 LUFS target, mono fold-down, EQ, denoiser strength, or upscaling recipe. Trackdub's `--match-loudness` is opt-in source-loudness matching, not a -14 LUFS preset. Peak normalization and integrated loudness are distinct.
- Reuse valid artifacts where applicable. Isolate comparisons before an explicitly approved rerun; verify dependent stages are consistent after changed inputs.
- **Completion:** each attempted change has a real stage result, backend/provenance, artifact path, and explicit failure/skip reason where applicable.

### 5. Keep visual cleanup and branding separate

- Watermarks, lower thirds, end cards, deblocking, and upscaling from the upstream plugin are not established Trackdub capabilities merely because this skill describes them.
- Verify live source/help for any requested operation. Unsupported requests need an explicitly scoped implementation task or a separately approved external derivative, never a fictitious pipeline success.
- No generative face/text reconstruction by default. Discuss authenticity and identity changes before any optional visual synthesis. Do not promise to restore compression-destroyed detail.
- **Completion:** requested visual work is either verified as executed or clearly labeled unavailable/not attempted.

### 6. Verify before delivery

- Re-read `project info`; locate the exact output files and verify their existence, readability, expected streams, duration, sample rate/channels, and relevant provenance/stage history.
- Re-measure comparable before/after loudness and peaks. Use matching time ranges/channel layouts; retain raw measurement output. Do not invent noise-improvement numbers where no reliable measurement exists.
- Check stream start times, duration, and timing changes. Container timestamps alone do not establish lip sync; report whether acoustic offset or perceptual sync was actually evaluated. If using 20 ms as a working tolerance, label it a chosen review threshold, not a universally validated guarantee.
- For altered pictures, inspect representative frames and any overlays/end card. For speech, request timestamped human A/B listening when no listening tool is available; do not claim it sounds better based solely on numerical metrics.
- Recompute the original source hash and confirm it is unchanged. Check prior successful artifacts remain available. Report incomplete checks and limitations explicitly.
- Save an evidence report using [templates/cleanup-report.md](templates/cleanup-report.md) outside runtime-managed artifact locations; the report is supplementary, not a replacement for Trackdub stage evidence.
- **Completion:** delivery includes real files, verified changes, skipped/failed work, unresolved limitations, and precise human-review timestamps.

## Pitfalls

- Birdsong, room echo, distant microphones, burned-in captions, clipping, and destroyed detail may remain. Treat these as source-specific limitations, not guaranteed denoiser behavior.
- Aggressive gates, compression, and noise reduction can damage word endings or pump noise. Prefer a conservative comparison; never optimize metrics at the expense of intelligibility.
- A successful CLI exit or existing WAV is insufficient: it may be reused or source fallback. Check stage status and provenance.
- A cleanup-only request does not authorize translation, cloning, lip synthesis, or full dubbing. `--target-language en` is not a cleanup switch.
- Respect `ADR-0012-wave-pcm16-loudness-policy.md`; this workflow does not flip writer defaults or mix policy.

## Verification when changing this skill or implementing missing support

Validate frontmatter, relative links, attribution, and commands against the live CLI. For runtime changes follow `AGENTS.md`, preserve dependency direction, track work in plans/Git, and test success, disabled/skipped, missing-prerequisite, and failure paths with shared fakes. Use controlled-matrix evidence for end-to-end comparisons; do not put BenchmarkDotNet on PR CI. Documentation-only changes do not prove media cleanup or model readiness.
