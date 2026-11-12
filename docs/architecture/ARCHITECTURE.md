# ARCHITECTURE.md

# Trackdub Architecture

Trackdub is a cross-platform, local-first AI dubbing workstation for Windows, macOS, and Linux.

The same pipeline also runs headless through the `trackdub` CLI and the SDK, including on Linux GPU servers in a container. That headless path is the basis of the opt-in cloud render tier (section 3.6).

The product is designed around a staged editorial workflow:

```text
local media
  -> audio preparation
  -> speech detection
  -> speaker analysis
  -> transcript
  -> translation
  -> voice / TTS
  -> preview
  -> mix
  -> export
```

The important architectural idea is that **every stage produces durable artifacts**. Users should be able to reopen a project, inspect what was generated, edit intermediate results, and rerun only the stages that need to change where the implementation supports it.

This document defines the long-term framework choices, pipeline architecture, project boundaries, and recommended directory layout.

---

## 1. Design goals

Trackdub should be:

- **Cross-platform desktop**: built as a real desktop application, not a browser shell, with Windows, macOS, and Linux treated as product targets.
- **Local-first**: local files and local inference are primary paths.
- **Hardware-aware**: use available acceleration where verified, with clear fallback behavior.
- **Stage-aware**: each pipeline stage has explicit inputs, outputs, status, warnings, and artifacts.
- **Resumable**: projects reopen without recomputing completed work.
- **Inspectable**: the user can see transcript, translation, speakers, voices, TTS takes, and mix state.
- **Truthful**: no fake readiness, no hidden provider swaps, no misleading “GPU enabled” claims.
- **License-aware**: model, voice, provider, and binary licenses are tracked explicitly.
- **Agent-friendly**: Codex, Claude Code, and other coding agents should have clear boundaries.

---

## 2. Non-goals

Trackdub is not trying to be:

- a general-purpose nonlinear video editor
- a full DAW
- a one-click magic dubbing service
- a web-first SaaS product (the opt-in cloud render tier in section 3.6 runs the same pipeline on a rented GPU; the desktop app stays local-first)
- a voice-cloning toy
- a Python environment manager for end users
- a perfect lip-sync engine
- a guarantee that all open models are commercially safe

Voice cloning, if supported, must be opt-in, consent-gated, and license-aware.

---

## 3. Recommended framework and runtime stack

### 3.1 Application framework

Recommended desktop framework:

```text
Avalonia
.NET 11
C#
```

Historical WinUI/Windows App SDK material may still appear in old docs, tests, or dead references. Treat it as migration evidence only: if it reveals behavior that was accidentally left behind, verify that the behavior still makes product sense and port it into Avalonia or shared layers. Do not add new WinUI paths.

Why:

- Single desktop shell across Windows, macOS, and Linux.
- Good fit for a local-first, hardware-aware workstation without splitting UI implementation by OS.
- Lets platform-specific runtime and native dependency differences stay behind explicit service seams.
- Keeps UI behavior, layout, keyboard shortcuts, and tests aligned across supported platforms.

### 3.2 Inference runtime

Recommended inference stack for the desktop app:

```text
Windows ML
ONNX Runtime
DirectML
TensorRT-RTX where supported
CPU fallback
```

The Linux container is one runtime image (based on `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04`, with classic TensorRT 10.13) published as two Dockerfile targets. They differ only in `TRACKDUB_GPU_PROFILE`, which tells provisioning (`scripts/docker/seed-and-warm-gpu-worker.sh`) which route to set up:

```text
rtx     consumer RTX GPUs: installs the TensorRT-RTX plugin during provisioning
server  datacenter GPUs (for example AWS G6/L4, G6e/L40S): CUDA and classic TensorRT, no TensorRT-RTX
```

The runtime layer should not assume every model works with every execution provider.

Instead, each model/provider pair must pass:

```text
load test
smoke inference test
basic output sanity check
latency measurement
memory measurement
fallback behavior test
```

Execution provider selection should be treated as a **runtime plan**, not a static user setting.

Example provider priority:

```text
TensorRT-RTX, when supported and validated
DirectML, when supported and validated
CPU, always available fallback
```

Do not make unsupported acceleration a hard blocker. If the app cannot use a GPU path, it should tell the user why and continue with a slower path where practical.

#### 3.2.1 Engine Caching & Invalidation (TensorRT)

Trackdub caches compiled inference engines to reduce startup latency. The default desktop cache location for TensorRT is:
`%LOCALAPPDATA%\Trackdub\EngineCache`

`TRACKDUB_ENGINE_CACHE_ROOT` overrides the location; the Linux container sets it to `/data/engine-cache`. Classic TensorRT engines (server profile) are stored under `classic-trt/<fingerprint>`, where the fingerprint covers the GPU, driver, TensorRT binary and ONNX Runtime identity. On a server, build engines on the target instance type with `trackdub cache warm` and persist them; never copy a cache between GPU models or driver versions.

- **Automatic Invalidation**: TensorRT verifies environment metadata embedded in the cached `.engine` files. It automatically invalidates the cache and triggers recompilation if it detects a mismatch in GPU architecture, NVIDIA Display Driver version, or specific ONNX model shapes/profiles.
- **Manual Cache Clearing Rules**: If persistent inference crashes occur (especially following unexpected shutdowns or driver updates), or if the hardware is changed, users should manually delete all files in `%LOCALAPPDATA%\Trackdub\EngineCache`. The engines will automatically rebuild on their next use.

### 3.3 Media stack

Recommended media stack:

```text
FFmpeg
libmpv primary playback
LibVLC fallback playback
Avalonia rendering and custom controls
```

Suggested responsibilities:

- **FFmpeg**: probing, audio extraction, normalization, muxing, and export verification.
- **libmpv**: primary video preview playback where native dependencies are available.
- **LibVLC**: fallback video preview playback when libmpv is unavailable or fails.
- **Avalonia rendering/custom controls**: waveform, timeline, overlays, subtitles, speaker labels, confidence visuals, and desktop chrome.

Readiness checks must report whether `ffmpeg` and `ffprobe` are already available; they must not silently download tools. Downloads, if offered, should be explicit user actions.

Playback behavior must be validated through the Avalonia shell. For `.mkv` or other unsupported preview formats, prefer an explicit conversion/import-staging workflow unless the current libmpv/LibVLC path supports the format with overlays and fullscreen behavior intact.

### 3.4 Persistence stack

Recommended persistence stack:

```text
SQLite
Dapper
project-local artifact store
global model cache
```

See [ADR-0002](docs/adr/ADR-0002-sqlite-project-persistence.md) for the project-versus-machine persistence split.

SQLite owns structured project state. The filesystem owns large artifacts.

Do not put generated audio, video, model weights, or source media blobs directly into SQLite.

### 3.5 Packaging

Recommended packaging direction:

```text
MSIX or signed installer
self-contained .NET runtime
bundled FFmpeg where licensing permits
model downloads managed by manifest
```

End users should not need:

- Python
- Conda
- Docker
- WSL
- CUDA Toolkit
- manual PATH edits

This applies to the desktop app. The Linux GPU worker is an operator artifact, not an end-user install: it ships as a container image built from the `Dockerfile` `server` target.

### 3.6 Headless and cloud render execution

The `trackdub` CLI (`src/Trackdub.Cli`) runs the full pipeline without the desktop UI:

```text
trackdub run pipeline --media in.mp4 --output ./project --preset <preset> \
  --result-file result.json --progress-file progress.jsonl
```

It writes the same per-stage project artifacts as the desktop app, plus a structured result file and a progress stream that a job runner can parse. `trackdub doctor`, `trackdub models download|verify` and `trackdub cache warm` cover health checks, hash-verified model provisioning and engine builds.

Cloud presets fix the models, variants and per-stage providers for server runs (`src/Trackdub.Cli/PocServerPreset.cs`, `src/Trackdub.Cloud/Jobs/CloudPresets.cs`): `poc-server-v1` (English to Spanish) and `poc-server-zh-v1` (English to Chinese). Clients pick a preset name; they never supply commands, model IDs or paths.

The cloud render tier is opt-in per project and requires explicit user consent before source media leaves the device (see the Stage 0 rule "Do not silently switch to cloud"). The proof of concept with SnapSoft targets this job path:

```text
Cloudflare API (auth, job record) -> SQS -> EC2 GPU worker (trackdub run pipeline) -> S3 (KMS encryption, presigned URLs, 7-day lifecycle)
```

The repository also contains an earlier AWS-only job stack (`Trackdub.Api`, `Trackdub.Cloud.Aws`, `Trackdub.CloudJobAdapter`, plus the `IJobScheduler` interface in `Trackdub.Cloud`) that uses Cognito, DynamoDB and a scheduler interface instead of Cloudflare and SQS. It is outside the POC scope. See [docs/handoff/snapsoft-onboarding.md](../handoff/snapsoft-onboarding.md) for the POC mapping.

---

## 4. Architectural boundaries

The solution is organized around product boundaries.

```text
UI
Application use cases
Domain model
Infrastructure
Media
Inference abstractions
Concrete ONNX inference
Benchmarks/tools
```

The most important dependency rule:

```text
Domain depends on nothing.
```

The domain model must not reference:

- Avalonia UI
- SQLite
- FFmpeg
- Windows ML
- ONNX Runtime
- filesystem paths as implementation details
- cloud providers

---

## 5. Recommended project dependency flow

```text
Trackdub.App.Avalonia
  -> Trackdub.Application
  -> Trackdub.Domain
  -> Trackdub.Contracts

Trackdub.Application
  -> Trackdub.Domain
  -> Trackdub.Contracts
  -> abstractions only

Trackdub.Infrastructure
  -> Trackdub.Application
  -> Trackdub.Domain
  -> Trackdub.Contracts

Trackdub.Media
  -> Trackdub.Application
  -> Trackdub.Domain
  -> Trackdub.Contracts

Trackdub.Inference
  -> Trackdub.Domain
  -> Trackdub.Contracts

Trackdub.Inference.Onnx
  -> Trackdub.Inference
  -> Trackdub.Domain
  -> Trackdub.Contracts

Trackdub.Benchmarks
  -> Trackdub.Inference
  -> Trackdub.Inference.Onnx
  -> Trackdub.Infrastructure
  -> Trackdub.Domain
```

The application layer coordinates work. Concrete implementation layers do not mutate project state directly.

When Infrastructure or Media reference `Trackdub.Application`, that dependency should stay limited to application-defined interfaces or DTOs. Those projects should not call use cases back into the coordinator.

---

## 6. Pipeline overview

The long-term pipeline is staged.

```text
0. Preflight and runtime planning
1. Ingest and project creation
2. Audio preparation
3. Optional stem separation
4. Voice activity detection
5. Speaker diarization
6. Transcription
7. Translation
8. Speaker / voice assignment
9. TTS generation
10. Preview and refinement
11. Mix and export
12. Diagnostics and benchmark reporting
```

Each stage should have:

```text
stage_run_id
stage kind
input artifact IDs
output artifact IDs
model/provider/runtime metadata
settings hash
started_at
completed_at
status
warnings
error classification
```

---

## 7. Pipeline stages

### 7.1 Stage 0: Preflight and runtime planning

Purpose:

- Detect OS, GPU, memory, and supported execution providers.
- Discover installed model cache.
- Determine which models are required.
- Run model/provider smoke tests where needed.
- Produce a plan the UI can show before work starts.

Outputs:

```text
hardware profile
execution provider availability
model availability
runtime plan
warnings
```

Rules:

- Do not show “GPU ready” until at least one real model smoke test passes.
- Do not silently switch to cloud.
- Do not block local CPU fallback unless the stage truly cannot run.

---

### 7.2 Stage 1: Ingest and project creation

Purpose:

- Accept source media.
- Probe container/stream metadata.
- Create project folder and database.
- Register source media.
- Extract a project-local ingest audio artifact without applying downstream normalization rules yet.

Outputs:

```text
MediaAsset
Project
source-reference.json
ingest_audio.wav
media metadata
artifact records
```

Rules:

- Reference original media by default.
- Offer a later option to bundle media into the project.
- Store media fingerprints so moved/changed source files can be detected.
- Do not collapse ingest and normalization into one irreversible step.

---

### 7.3 Stage 2: Audio preparation

Purpose:

- Normalize audio for downstream stages.
- Prepare sample-rate-specific derivatives.
- Generate waveform data.

Typical outputs:

```text
working_audio_48khz.wav
asr_audio_16khz_mono.wav
waveform.json
loudness metadata
```

Rules:

- Internal timing should prefer sample-accurate or integer millisecond representation.
- Avoid floating-point drift in long media.
- Stage 2 owns normalization, derivative generation, and waveform summaries.

---

### 7.4 Stage 3: Optional stem separation

Purpose:

- Estimate vocal and ambiance stems.
- Improve ASR/diarization in noisy content.
- Preserve ambiance/music/SFX for final mix where useful.

Outputs:

```text
vocals.wav
ambiance.wav
separation manifest
warnings
```

Rules:

- Stem separation should not be mandatory for all workflows.
- Never promise perfect dialogue removal.
- Label outputs as estimated stems.

---

### 7.5 Stage 4: Voice activity detection

Purpose:

- Identify speech ranges.
- Reduce wasted diarization/ASR work.
- Create candidate speech segments.

Outputs:

```text
speech ranges
silence ranges
segment candidates
```

Rules:

- VAD is small and may be CPU-friendly.
- Do not over-optimize tiny models before major pipeline stages work.

---

### 7.6 Stage 5: Speaker diarization

Purpose:

- Assign speaker labels to speech regions.
- Create editable speaker turns.
- Extract candidate reference clips where voice cloning is enabled.

Outputs:

```text
speaker turns
speaker records
reference clip candidates
diarization warnings
```

Rules:

- Treat diarization as editable guesses.
- Do not block the pipeline on diarization.
- Support single-speaker and manual assignment workflows.
- Expose merge/split/rename speaker operations.

---

### 7.7 Stage 6: Transcription

Purpose:

- Produce timed transcript segments.
- Attach words/timestamps where supported.
- Attach speaker IDs where available.

Outputs:

```text
TranscriptSegment
WordTimestamp
TranscriptRevision
ASR warnings
```

Rules:

- Transcripts are editable.
- ASR output should not overwrite user edits without explicit action.
- Store model and runtime provenance.

---

### 7.8 Stage 7: Translation

Purpose:

- Translate transcript segments into the target language.
- Preserve segment and speaker context.
- Support manual editing and later regeneration.

Outputs:

```text
TranslationRevision
TranslatedSegment
translation warnings
```

Rules:

- Direct language-pair models are preferred where available.
- Pivot routing must be visible when used.
- Translation uses only commercially licensed models: every translation model in the manifest has `commercial_allowed: true` and `commercial_use_verified: true` (see section 10).
- Translation output is draft material.

---

### 7.9 Stage 8: Speaker / voice assignment

Purpose:

- Map speakers to voices or reference clips.
- Decide whether stock voices or voice cloning is used.
- Enforce consent and commercial-safe rules.

Outputs:

```text
VoiceAssignment
SpeakerVoiceProfile
consent records
voice warnings
```

Rules:

- Stock voice TTS should work before voice cloning is required.
- Voice cloning requires explicit consent flow.
- Manifest authoring (commercial_allowed, commercial_use_verified, lane) is the only gate for unsafe models/providers. There is no runtime CommercialSafeMode toggle.

---

### 7.10 Stage 9: TTS generation

Purpose:

- Generate dubbed speech per segment.
- Store takes as artifacts.
- Allow review and replacement.

Outputs:

```text
TtsTake
dubbed segment audio
duration metadata
timing fit warnings
```

Rules:

- Do not silently time-compress extreme mismatches.
- Store natural duration and fitted duration.
- Allow multiple takes where practical.
- Do not mutate existing takes in place.

---

### 7.11 Stage 10: Preview and refinement

Purpose:

- Preview dubbed output in context.
- Let users identify and fix transcript, translation, timing, voice, and mix issues.

Outputs:

```text
preview mix
selected range render
UI preview state
```

Rules:

- Preview and export should share the same mix plan representation.
- Preview must not invent timing rules that export does not use.

---

### 7.12 Stage 11: Mix and export

Purpose:

- Combine dubbed speech with original/ambiance audio.
- Preserve source video where possible.
- Export final audio/video and subtitles.

Outputs:

```text
MixPlan
final_dubbed_audio.wav
exported video
subtitle files
export manifest
```

Rules:

- Copy original video stream where possible.
- Encode only what is necessary.
- Embed metadata indicating AI-dubbed output where appropriate.
- Preserve export provenance.
- Run export preflight for source media, container support, tool availability, and stale or missing upstream outputs before expensive render/mux work.
- Stage delivery files adjacent to the requested output and atomically replace/rename on success; failed exports must not truncate an existing user output.

---

### 7.13 Stage 12: Diagnostics and benchmarks

Purpose:

- Explain what happened.
- Support bug reports.
- Track model/runtime performance over time.

Outputs:

```text
benchmark rows
diagnostic bundle
stage logs
hardware profile
runtime plan records
```

Rules:

- Diagnostics should be local by default.
- Do not upload diagnostics without explicit user action.
- Redact sensitive paths/tokens where needed.

---

## 8. Persistence model

### 8.1 Project folder

Recommended project layout:

```text
ProjectName.trackdub/
├── trackdub.db
├── manifest.json
├── media/
│   ├── source-reference.json
│   └── ingest_audio.wav
├── artifacts/
│   ├── stems/
│   ├── vad/
│   ├── diarization/
│   ├── transcript/
│   ├── translation/
│   ├── tts/
│   ├── mix/
│   └── export/
├── logs/
└── temp/
```

### 8.2 Machine-local app data

Recommended machine-local layout:

```text
%LocalAppData%/Trackdub/
├── settings.json
├── models/
├── model-cache/
├── benchmarks/
└── logs/
```

Project artifacts and machine-local caches should remain separate. Model cache data and benchmark history should not roam with the user's profile by default.

---

## 9. Core project database tables

Suggested initial SQLite tables for `trackdub.db`:

```text
Projects
MediaAssets
StageRuns
Artifacts
Speakers
SpeakerTurns
TranscriptRevisions
TranscriptSegments
Words
TranslationRevisions
TranslatedSegments
VoiceAssignments
TtsTakes
MixPlans
Exports
ConsentRecords
SchemaVersion
```

Global model cache inventory and cross-project benchmark history belong in the machine-local data root, not in each project's `trackdub.db`.

Every artifact record should include:

```text
artifact_id
artifact_kind
project_relative_path
sha256
size_bytes
duration
sample_rate
created_at
stage_run_id
input_artifact_hashes
model_id
model_revision
execution_provider
settings_hash
warnings_json
```

Use project-relative locations or storage keys in domain records. Infrastructure resolves those keys to absolute filesystem paths at runtime.

---

## 10. Model manifest and licensing

Every model must have a manifest.

Example:

```json
{
  "model_id": "example/model",
  "task": "asr",
  "license": "MIT",
  "commercial_allowed": true,
  "redistribution_allowed": true,
  "requires_attribution": false,
  "requires_user_consent": false,
  "voice_cloning": false,
  "commercial_use_verified": true,
  "source_url": "",
  "revision": "",
  "sha256": "artifact-sha256"
}
```

Rules:

- Trackdub uses only commercially licensed models. A model with an unknown or non-commercial license is not added to the manifest for any pipeline stage.
- `commercial_use_verified: true` requires both commercial-use license confidence and a non-empty SHA-256.
- `commercial_allowed: true` alone is not enough for commercial-safe routing.
- Voice-cloning models require explicit consent flow.
- Model licenses are independent from the app license.
- Commercial safety must be enforceable by code, not just documentation. Today it is read from the manifest fields (`commercial_allowed`, `commercial_use_verified`) by the model inventory, starter-pack and translation provisioning code. There is no `CommercialSafeEvaluator` type and no runtime CommercialSafeMode flag.
- The live manifest is `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json`; `revision` may be `cache-installed` when no upstream revision is pinned.

---

## 11. Directory breakdown

### 11.1 Top-level layout

```text
Trackdub/
├── docs/                  (legal policies live in docs/legal/, e.g. MODEL_LICENSE_POLICY.md, THIRD_PARTY_NOTICES.md)
├── src/
├── tests/
├── tools/
├── assets/
├── resources/
├── scripts/               (scripts/docker/ holds the GPU worker seed and warm scripts)
├── docker/                (cloud preset pins, model lists and TensorRT warm lists for the Linux worker)
├── runtime/               (pinned runtime manifests, e.g. trt-rtx-ep.manifest.json)
├── workers/
├── build/
├── .github/
├── .claude/
├── Dockerfile             (Linux GPU worker image: targets rtx and server)
├── Dockerfile.api
├── Dockerfile.cloud-worker
├── docker-compose.yml
├── docker-compose.aws.yml
├── Directory.Build.props
├── Directory.Packages.props
├── global.json
├── Trackdub.slnx
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
└── AGENTS.md
```

---

## 12. Source projects

### 12.1 `src/Trackdub.App.Avalonia`

Purpose:

Avalonia application shell for Windows, macOS, and Linux.

What belongs here:

- views
- view models
- commands
- navigation
- overlays
- UI resources
- user-facing validation state

What should not go here:

- ONNX model wrappers
- SQL
- FFmpeg command construction
- artifact store implementation
- domain invariants
- pipeline business rules

---

### 12.2 `src/Trackdub.Domain`

Purpose:

Pure domain model.

What belongs here:

- projects
- media assets
- speakers
- segments
- words
- translations
- voice assignments
- TTS takes
- mix plans
- stage runs
- artifact records
- value objects

What should not go here:

- Avalonia or other UI references
- SQLite references
- Windows ML references
- ONNX Runtime references
- FFmpeg references
- file IO implementation

---

### 12.3 `src/Trackdub.Application`

Purpose:

Use cases and orchestration.

What belongs here:

- create/open/resume project
- start pipeline stage
- commit stage run
- invalidate downstream artifacts
- evaluate model/license safety
- request export
- run benchmark use cases

What should not go here:

- XAML
- concrete SQL
- concrete ONNX session code
- FFmpeg process execution

---

### 12.4 `src/Trackdub.Infrastructure`

Purpose:

Persistence, filesystem, settings, logging, diagnostics.

What belongs here:

- SQLite connection factory
- migrations
- repositories
- artifact store
- settings store
- diagnostic bundle creation
- consent storage
- logging setup

What should not go here:

- UI views
- model tensor code
- pipeline strategy decisions

---

### 12.5 `src/Trackdub.Media`

Purpose:

Media processing and timing.

What belongs here:

- media probe
- audio extraction
- audio normalization
- waveform summaries
- playback abstractions
- muxing/export implementation
- sample/time conversion
- drift handling

What should not go here:

- ASR logic
- TTS logic
- translation logic
- SQLite repositories
- UI controls

---

### 12.6 `src/Trackdub.Inference`

Purpose:

Inference abstractions and runtime planning.

What belongs here:

- execution provider selection
- model manifests
- model registry
- model cache planning
- download planning
- inference interfaces
- benchmark abstractions

What should not go here:

- concrete model tensor layouts
- XAML
- project persistence implementation

---

### 12.7 `src/Trackdub.Inference.Onnx`

Purpose:

Concrete ONNX / Windows ML model wrappers.

What belongs here:

- Silero VAD wrapper
- SortFormer wrapper
- Whisper wrapper
- Opus-MT wrapper
- MADLAD wrapper
- Kokoro wrapper
- Chatterbox wrapper
- BS-RoFormer wrapper
- model-specific tokenizers
- model-specific tensor mapping

What should not go here:

- UI state
- SQLite repositories
- pipeline stage commits
- export business rules

---

### 12.8 `src/Trackdub.Benchmarks`

Purpose:

Benchmark harness and performance measurement.

What belongs here:

- model benchmark scenarios
- latency measurement
- real-time factor calculation
- memory measurement
- WER/CER helpers
- benchmark report generation

What should not go here:

- user project state
- UI views
- broad app orchestration

---

### 12.9 `src/Trackdub.Tools`

Purpose:

Developer tools.

What belongs here:

- model manifest builder
- artifact inspector
- database inspection tools
- FFmpeg command experiments
- migration helpers

What should not go here:

- production UI
- production inference orchestration
- user-facing business logic

---

### 12.10 `src/Trackdub.Contracts`

Purpose:

Stable cross-boundary DTOs where needed.

What belongs here:

- pipeline messages
- diagnostic report contracts
- benchmark report contracts
- project import/export contracts

What should not go here:

- domain invariants
- concrete persistence code
- UI classes

### 12.11 Headless and cloud projects

| Project | Purpose |
|---|---|
| `src/Trackdub.Composition` | Composition root shared by the desktop app, CLI and SDK; starter packs (`StarterPacks/*.json`). |
| `src/Trackdub.Sdk` | Headless engine API: run stages, resume, batch processing. |
| `src/Trackdub.Cli` | The `trackdub` CLI: `run pipeline`, `doctor`, `models`, `providers`, `cache warm`, cloud presets. Used in the Linux GPU container. |
| `src/Trackdub.Cloud` | Cloud job contracts, cloud preset list, media validation, the `IJobScheduler` interface. |
| `src/Trackdub.Cloud.Aws` | DynamoDB job store and S3 artifact store. |
| `src/Trackdub.Api` | CPU ASP.NET job API (Cognito JWT). |
| `src/Trackdub.CloudJobAdapter` | One-shot worker wrapper around the CLI (claim, lease heartbeat, artifact upload, outcome write); image `Dockerfile.cloud-worker`. |

`Trackdub.Cloud.Aws`, `Trackdub.Api` and `Trackdub.CloudJobAdapter` form the earlier AWS-only job stack described in section 3.6. It is not part of the SnapSoft POC job path.

---

## 13. Tests

Recommended test layout:

```text
tests/
├── Trackdub.Domain.Tests/
├── Trackdub.Application.Tests/
├── Trackdub.Infrastructure.Tests/
├── Trackdub.Media.Tests/
├── Trackdub.Inference.Tests/
├── Trackdub.Benchmarks.Tests/
├── Trackdub.App.Avalonia.Tests/
├── Trackdub.Architecture.Tests/
├── Trackdub.Composition.Tests/
├── Trackdub.Sdk.Tests/
└── Trackdub.TestDoubles/              (shared source, no csproj)
```

Testing strategy:

- Domain tests should be fast and pure.
- Application tests should use fakes.
- Infrastructure tests can use temporary SQLite files.
- Media tests should use tiny sample files.
- ONNX tests should separate smoke tests from slow benchmarks.
- Integration tests should use minimal media and deterministic artifacts.

---

## 14. Agent workflow

Agents should be used for bounded tasks.

Good tasks:

```text
Implement one SQLite migration.
Implement one repository.
Implement one model manifest parser.
Implement one benchmark scenario.
Implement one ONNX wrapper smoke test.
Refactor one boundary leak.
```

Bad tasks:

```text
Build the whole app.
Implement every model.
Create the entire UI.
Add monetization.
Add cloud providers.
Rewrite the architecture.
```

Agent-generated code should be rejected if it:

- puts inference code in the UI project
- puts SQL in view models
- adds models without license metadata
- adds Python/Docker/WSL requirements to end-user runtime
- silently uploads user data
- silently switches provider route
- invents untested acceleration behavior

---

## 15. Early implementation order

Recommended order:

```text
1. Harness: ONNX/Windows ML model smoke test
2. Harness: benchmark persistence
3. Domain: project/media/artifact/stage-run records
4. Infrastructure: SQLite migrations and repositories
5. Media: probe and normalized audio extraction
6. Application: create/open/resume project
7. Inference: VAD + ASR abstractions
8. Inference.Onnx: first ASR smoke wrapper
9. App: minimal open project + transcript display
10. Pipeline: transcript-only vertical slice
```

Do not start with the full DAW UI.

---

## 16. Architecture review checklist

Before merging a significant change, ask:

- Does this respect the dependency direction?
- Did this add a model or dependency?
- Is the license documented?
- Does the manifest need updated lane, commercial_allowed, or commercial_use_verified fields for any new or changed model?
- Are artifacts persisted immutably?
- Is downstream invalidation handled?
- Does the user see route/fallback changes?
- Is there a test or harness result?
- Did this leak UI concerns into domain/application code?
- Did this leak model/runtime details into UI code?

---

## 17. Summary

Trackdub should behave less like a magic AI demo and more like a reliable workstation.

The architecture should make this possible by keeping:

- UI separate from orchestration
- orchestration separate from concrete runtimes
- domain state separate from persistence implementation
- media processing separate from AI inference
- model licenses visible to the product logic
- user project artifacts durable and inspectable

The guiding principle:

> A dubbing app is only useful if users can understand, edit, and trust the pipeline that produced the result.

# Local Dubbing Pipeline — Architectural Audit & Proposed Fixes (Unified)

> **Audit date:** 2026-07-08  
> **Verified against source:** 2026-07-08 (re-audit pass; see §8 Correction Log)  
> **Scope:** media ingest → playback → full local inference pipeline → export dubbed video  
> **Layers audited:** App.Avalonia, Media/Media.Playback, Application (pipeline orchestration), Inference/Inference.Onnx, Sdk  
> **Layers excluded:** Cloud (Api, Worker, WebhookDelivery), Benchmarks, Tools  
> **Fix posture:** proposed-only — no code changes in this audit  
> **Sources:** Two independent audits merged, then corrected against current tree. Surviving source walk: `docs/Audit/pipeline-full-walk-audit-2026-07-08.md`. Cited `docs/architecture/local-pipeline-audit-2026-07-08.md` is **not present** in the tree (agent-1 content survived only via this merge). Principles supplement: `docs/architecture/pipeline-principles-review-grok-2026-07-08.md`.

---

## Executive Summary

Trackdub's local dubbing spine has one real architectural split and several execution-governance gaps:

1. **Two stage abstractions, but they are stacked — not “unused dual pipelines.”**  
   Leaf work lives in `*StageHandler` classes (12 production handlers). Transcript pre-translation stages are also wrapped as 6 `ITranscriptGenerationStage` implementations, built once inside `TranscriptGenerationService` via `TranscriptPipelineBuilder`. Avalonia and SDK **both** reach that service for VAD / diarization / ASR / text-refinement-ASR / speaker-assignment (and the full transcript chain on initial/stem-triggered regen). Translation, TTS, export, lips, separation, overlap-rescue run through workspace workflows / handlers — **not** through `ITranscriptGenerationStage`. Calling GenerationStages “dead” or “unused by UI/SDK” is wrong.

2. **Scattered execution governance (still true).** SDK resume skip, transcript-pipeline resume skip, and `StageArtifactResumeEvaluator` overlap. SDK `PrerequisiteStages` only blocks on Vad+Asr. Lip sync/synthesis are runnable from Avalonia but are **not** in SDK `DefaultStageOrder` and have **no** `RunStageWorkflowAsync` cases. Stage graphs are hard-coded in multiple places. Model provisioning has multiple entry points.

**Critical gaps (corrected):**
- No `TranslationGenerationStage`, `TtsGenerationStage`, or `ExportGenerationStage` — those lanes are workflow/handler-only by design today (agents 1 & 2; still accurate)
- Lip sync / lip synthesis not in SDK `DefaultStageOrder` and not dispatched in SDK `RunStageWorkflowAsync` (agent 2; still accurate). Avalonia **does** dispatch both via `SpeakerVoice.RunPipelineStage`
- ~~`CompositionRoot` registers only 4 of 6 GenerationStages~~ — **withdrawn**; all 6 are registered (see F5 correction)
- 7 `.devin/skills/` entries in `AGENTS.md` do not exist on disk (still accurate)
- Pipeline **run** dispatch lives in `SpeakerVoice.cs` (~1.6k+ lines) mixed with voice logic; `PipelineUi.cs` owns rows/status (still accurate)
- Resume logic duplicated across 3 systems; snapshot drift can cause wrong skip/rerun (still accurate)

**What works well:** media ingest spine + lazy normalize (`CreateMediaSpineAsync` then `EnsureNormalizedAudio*`), playback (libmpv / LibVLC fallback), model manifest governance, headless dub green path via SDK for the 7-stage default order, pre-flight readiness checker, artifact preservation on skip/fail, `StageRunRecord` lifecycle via `StageRunHelper`.

---

## 1. Audit Scope & Method

### Scope boundary
The audited flow: user opens video → media loads → playback works → full pipeline stages run → dubbed video exports.

### Layers examined
| Layer | Project(s) | What was checked | Agent |
|---|---|---|---|
| Desktop UI | `Trackdub.App.Avalonia` | Pipeline VMs, stage trigger, export VM, import flow | 1 |
| Media & Playback | `Trackdub.Media`, `Trackdub.Media.Playback` | FFmpeg services, playback engine, export/mux | 1 |
| Application | `Trackdub.Application` | Pipeline builders, stage handlers, snapshots, degradations, orchestration | 1 & 2 |
| Inference | `Trackdub.Inference`, `Trackdub.Inference.Onnx` | Runtime planner, EP providers, manifests, readiness | 1 |
| SDK | `Trackdub.Sdk` | DubbingEngine, workspace composition, readiness checker | 2 |
| Domain | `Trackdub.Domain` | StageNames, StageRuns, core entities | 1 & 2 |

### Method
Read-only source code exploration (including 2026-07-08 re-verification of claimed paths/counts). Foundational docs (`AGENT_CONTEXT.md`, `MILESTONE.md`) as contract reference. No builds or tests run in this audit.

---

## 2. Architecture Overview

### 2.1 Stage abstractions: stacked handlers + transcript GenerationStages

```
┌─────────────────────────────────────────────────────┐
│                  Avalonia UI                         │
│  PipelineUi.cs (rows / status / stage keys)          │
│  SpeakerVoice.cs (RunPipelineStage switch)           │
│  → workspace / _workspaceCommands / export UI        │
│  → _runtimeModelCoordinator (pre-flight)             │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   WorkspaceLayer    │  (TranscriptWorkspace)
          │   → ProjectWorkflow                       │
          │   → TranslationWorkflow                   │
          │   → TtsWorkflow / TtsOrchestrationService │
          │   → ExportWorkflow                        │
          │   → LipSyncWorkflow (nullable on ctor)    │
          │   → LipSynthesisWorkflow (nullable ctor)  │
          │   → PreviewMixWorkflow                    │
          └──────┬────────┬─────┘
                 │        │
    ┌────────────▼──┐  ┌──▼───────────────────────┐
    │ StageHandler   │  │ GenerationStage wrappers │
    │ (leaf engines) │  │ ITranscriptGenerationStage│
    │                │  │                          │
    │ StemSeparation │  │ SpeechEnhancementGen     │
    │ SpeechAudioEnh │  │ VadGenerationStage       │
    │ SpeechAudioPrep│  │ SpeakerDiarizationStage  │
    │ Vad / Asr /    │  │ AsrGenerationStage       │
    │ Diarization /  │  │ TextRefinementGeneration│
    │ TextRefine /  │  │ SpeakerAssignmentAnd     │
    │ StartTts /     │  │   PersistenceStage       │
    │ Export /       │  │                          │
    │ OverlapRescue  │  │ Built by TranscriptPipeline│
    │ LipSync /      │  │ Builder inside             │
    │ LipSynthesis   │  │ TranscriptGenerationService│
    │ (12 handlers)  │  │ Used by ProjectWorkflow     │
    │                │  │ (Avalonia + SDK transcript) │
    │                │  │ No Translation / TTS / Export│
    └───────────────┘  └─────────────────────────────┘
```

Most GenerationStages wrap a StageHandler. They are **not** a second independent engine — they are the transcript-graph orchestration path over the handlers. Translation / TTS / export stay on workflow + handler paths without GenerationStage adapters.

### 2.2 Execution layers (SDK → workspace → sub-pipeline → leaf)

1. **Outer orchestrator** (`TrackdubDubbingEngine.ExecuteAsync`): source validation, snapshot capture, preflight, loop over resolved stage order, resume skip, prerequisite blocking, dispatch.
2. **Workspace layer** (`TranscriptWorkspace`): holds workflows with `_pipelineGuard` serialization semaphore.
3. **Transcript sub-pipeline** (`TranscriptGenerationService` → `TranscriptGenerationPipeline`): chain of 6 stages — SpeechEnhancement → VAD → Diarization → ASR → TextRefinement (ASR) → SpeakerAssignment. Also `ResolveTranscriptStages` for single-stage re-runs (Vad / Diarization / Asr / TextRefinementAsr only).
4. **Leaf implementations**: handlers + GenerationStage wrappers + workflows for later stages.

### 2.3 Defined stages
All **18** constants from `src/Trackdub.Domain/StageRuns/StageNames.cs`:

`vad`, `asr`, `diarization`, `speaker-assignment`, `translation`, `text-refinement`, `text-refinement-asr`, `text-refinement-translation`, `tts`, `separation`, `speech-enhancement`, `audio-preparation`, `preview-mix`, `voice-cloning`, `export`, `lip-sync`, `lip-synthesis`, `overlap-rescue`.

SDK `DefaultStageOrder`: `[Separation, Vad, Diarization, Asr, Translation, Tts, Export]` — **7** of 18. Not in that order: speech-enhancement, audio-preparation, text-refinement*, speaker-assignment, preview-mix, voice-cloning, overlap-rescue, lip-sync, lip-synthesis (among others).

Speech enhancement commonly runs via `ProjectWorkflow.TryPrepareSpeechAudioAsync` (and as first GenerationStage when the transcript pipeline runs) — not as its own SDK outer-loop stage.

---

## 3. Finding Catalog

### A. Architecture & Structure

#### F1. Dual Stage Abstractions — Incomplete Unification [Agent 1, Agent 2#7] *(reframed)*
**Severity:** High (was Critical; downgrade — GenerationStages are live, not orphaned)

Two abstractions with partial overlap. Handlers are the leaf implementations for nearly everything. Six GenerationStages orchestrate the **transcript** slice only, via `TranscriptGenerationService`. Avalonia and SDK **do** use that path for transcript stages. Translation / TTS / Export / lips / separation have no GenerationStage adapters. Duplicate wrapper logic exists for stages that have both a handler and a GenerationStage.

**Proposed fix:** Either (a) extend `ITranscriptGenerationStage` to the full product graph and drive SDK + UI from one registry, or (b) stop pretending GenerationStages are a general pipeline and document them as the transcript-subgraph API only — then delete any unused wrapper duplication carefully. Do **not** “retire” GenerationStages as dead code; they are on the green path.

#### F2. Stage Graph Fragmented and Stringly-Typed [Agent 2#7]
**Severity:** High

Stage definitions hard-coded in multiple places: SDK `DefaultStageOrder`, `TranscriptPipelineBuilder` add-stage chain, `RunStageWorkflowAsync` switch, Avalonia `RunPipelineStage` switch, `ResolveTranscriptStages`, `StageRunHelper.StartKnownStageRun` (record start mapping). Coarse SDK outer stages vs fine-grained internal stages. Enable flags not uniformly applied.

**Proposed fix:** Centralize stage graph definition (one list + metadata for prerequisites, artifact contracts, resume eligibility). Drive builder, dispatcher, CLI filters, and UI from it.

#### F3. Lip Stages Not Executed in Main SDK Pipeline [Agent 2#1]
**Severity:** Critical *(for headless/SDK)*; Medium for Avalonia *(UI already dispatches)*

`DefaultStageOrder` is `[Separation, Vad, Diar, Asr, Translation, Tts, Export]` — no lip-sync or lip-synthesis. Lip runtime selections may be prepared in selections/snapshot, but SDK `RunStageWorkflowAsync` has **no** cases for them (`default` → skip). Avalonia `SpeakerVoice.RunPipelineStage` **does** call `RunLipSyncStageAsync` / `RunLipSynthesisStageAsync`. `LipSyncWorkflow` / `LipSynthesisWorkflow` are nullable on the `TranscriptWorkspace` constructor; CompositionRoot registers both as scoped when building the full app graph.

**Proposed fix:** Add lipsync/lipsynthesis (conditionally, when video source + model prefs present) to SDK `DefaultStageOrder` and dispatch switch. Keep Avalonia path honest about optional workflows.

#### F4. Missing Translation/TTS/Export GenerationStages [Agent 1#3-5]
**Severity:** Medium *(context-dependent — only Critical if goal is one GenerationStage graph)*

Translation, TTS, and Export exist as `StageNames` + workspace methods / handlers but have **no** `ITranscriptGenerationStage` implementation. They cannot participate in `TranscriptPipelineBuilder` today. That is accurate and intentional for the current split architecture.

**Proposed fix:** Only if unifying on GenerationStages — add those adapters. If keeping transcript-subgraph-only, document the boundary instead of treating this as a broken omission.

#### F5. CompositionRoot Partial GenerationStage Registration [Agent 1#6] — **WITHDRAWN**
**Severity:** n/a

**Original claim:** DI registers only Vad, SpeechEnhancement, Asr, TextRefinement — missing SpeakerDiarization and SpeakerAssignment.

**Current truth:** `CompositionRoot` registers all six concrete stages (`VadGenerationStage`, `SpeechEnhancementGenerationStage`, `SpeakerDiarizationStage`, `AsrGenerationStage`, `TextRefinementGenerationStage`, `SpeakerAssignmentAndPersistenceStage`) plus `TranscriptGenerationService`. Nuance: only `SpeechEnhancementGenerationStage` is also bound as `ITranscriptGenerationStage`; the rest are injected by concrete type into `TranscriptGenerationService`.

**No registration fix required.** Optionally normalize so every stage is also registered as `ITranscriptGenerationStage` for discoverability.

#### F6. Stale AGENTS.md Skill Table [Agent 1#7]
**Severity:** Medium

`AGENTS.md` still points agents at 7 `.devin/skills/` entries that are not present: `pipeline-stage`, `pipeline-stage-impl`, `test-double`, `domain-entity`, `degradation-write`, `stage-handler-test`, `audit-pipeline`. Actual skill dirs live under `.agents/skills/` and `.cursor/skills/` with a different set.

**Proposed fix:** Create the skills or correct the table to match what exists.

#### F7. Monolithic UI Pipeline Dispatch [Agent 1#8]
**Severity:** Medium

`RunPipelineStage` / `RunAllPipelineStagesAsync` live in `AvaloniaMainWindowViewModel.SpeakerVoice.cs` (1600+ lines) mixed with voice assignment / TTS helpers. `PipelineUi.cs` owns stage-row chrome and keys, not the run switch.

**Proposed fix:** Extract pipeline execution into a dedicated coordinator (e.g. `PipelineExecutionCoordinator`).

---

### B. Execution Governance

#### F8. Resume Logic Duplicated and Fragile [Agent 2#3]
**Severity:** High

Three cooperating systems:

| Mechanism | Location |
|---|---|
| `HasValidExistingArtifactsAsync` | `TrackdubDubbingEngine` (SDK outer; uses evaluator) |
| `ShouldResumeSkipStage` | `TranscriptGenerationPipeline` inside `TranscriptPipelineBuilder.cs` |
| `StageArtifactResumeEvaluator` | shared Application evaluator |

Snapshot includes model prefs; changing a model alias can cause wrong skip or unnecessary rerun. Speaker assignment hydration depends on prior ASR + diarization artifacts.

**Proposed fix:** Unify under a single `IStageResumeEvaluator` policy for outer + inner loops. Versioned snapshot hash. Tests for snapshot drift.

**Status (2026-07-09):** Never implemented. No `IStageResumeEvaluator`, `StageResumeEvaluator`, or `StageResumeRequest` type exists in the repo. The duplication described above is unchanged — `StageArtifactResumeEvaluator` (static, not DI-based) is the only shared piece; the three call sites still gate stage eligibility independently, and there are still two separately-written execution-snapshot builders (`TrackdubDubbingEngine.CaptureExecutionSnapshot`, `TranscriptGenerationService.BuildExecutionSnapshot`).

#### F9. Prerequisite Blocking Too Narrow [Agent 2#2]
**Severity:** High  
**Verified:** `TrackdubDubbingEngine.PrerequisiteStages` = `{ Vad, Asr }` only.

Diarization failure does not block Translation/TTS/Export in the SDK outer loop.

**Proposed fix:** Expand `PrerequisiteStages` to include `Diarization` (and consider `Translation` before TTS/Export).

#### F10. Purely Sequential Execution [Agent 2#4]
**Severity:** Medium

Transcript pipeline and SDK outer loop are sequential. Per-speaker TTS and per-region ASR paths are serial. Bottleneck for long-form content.

**Proposed fix:** Bounded parallelism for TTS across speakers and (where safe) segment batches.

#### F11. Speech Enhancement Ownership Confused [Agent 2#5] *(corrected)*
**Severity:** Medium

**Not** implemented inside `StemSeparationStageHandler` (that file has no enhancement calls). Real dual paths:

1. **Prep path:** `ProjectWorkflow.TryPrepareSpeechAudioAsync` → `SpeechAudioEnhancementStageHandler` (also used from cleanup / stem-rerun with `regenerateTranscript`, before or around `GenerateTranscriptAsync`).
2. **Transcript pipeline path:** first stage `SpeechEnhancementGenerationStage` wrapping the same handler family.

SDK extracts speech-enhancement degradations after `RunStemSeparationAsync` because stem-rerun can trigger prep + transcript regen that records enhancement stage runs — **not** because enhancement is coded inside the stem handler. Artifact reuse checks reduce (but do not eliminate) double-enhancement risk.

**Proposed fix:** Make speech enhancement a single first-class pre-VAD stage with one enable flag, one degradation path, and documented skip-if-artifact-present semantics. Fix misleading SDK comments that say “internal sub-step of RunStemSeparationAsync.”

#### F12. Mixing Buried Inside Export [Agent 2#6]
**Severity:** Medium

No top-level `preview-mix` in SDK `DefaultStageOrder`. `MixPlanBuilder` builds the mix plan inside `ExportStageHandler`. Interactive `PreviewMixWorkflow` is a separate path. Warning codes (MissingTake, StaleTake, LipSyncArtifactMissing) are collected but not strongly enforced.

**Proposed fix:** Promote an explicit Mix stage that export consumes, with explicit success/failure/warning propagation.

#### F13. Cancellation and Resource Hygiene [Agent 2#9]
**Severity:** Low

Cancellation between outer stages works; long-running ONNX/FFmpeg ops may not be promptly interruptible. Stem temps cleaned mainly at handler start. Single workspace semaphore serializes pipeline activity.

**Proposed fix:** Improve cooperative cancellation inside long-running engines. Add hard timeouts with cleanup.

#### F14. Model Provisioning Scattered [Agent 2#10, Agent 1#12]
**Severity:** Low

Multiple entry points: import ensures, per-stage ensure, `RuntimeModelSetupWorkflow`, deferred download inside some handlers. Risk of inconsistent readiness state.

**Proposed fix:** Centralize a single readiness gate for every stage path.

#### F15. Edge Cases Around Empty/Failed Early Stages [Agent 2#11]
**Severity:** Low

ASR handles VAD-empty gracefully. Downstream assignment / TTS / mix / export vary. Export can be reached with incomplete prior artifacts on the SDK order when prerequisites did not block.

**Proposed fix:** Strengthen guards; fail fast or clear degradation — do not pretend success.

---

### C. Positive Findings (no action needed)

#### P1. Playback + Ingest Well-Wired [Agent 1#10] *(nuance)*
`CreateMediaSpineAsync` registers source only. Normalization is a **separate** API (`EnsureNormalizedProjectAudioAsync` / `EnsureNormalizedAudioForPipelineAsync` → `EnsureNormalizedAudioAsync`), called lazily (Avalonia background load, pipeline preflight, etc.). Still correctly consumed by Avalonia, SDK, and CLI. Playback abstraction (`IMediaPlaybackBackend`, `AvaloniaDubPlaybackCoordinator`) is sound.

#### P2. Media Spine + Artifact Preservation [Agent 2 "happy path"]
Prior usable artifacts left in place on skips/failures. VAD-empty ASR degrades instead of fake success. `StageRunHelper` keeps `StageRunRecord` lifecycle consistent (note: `StartKnownStageRun` starts records — it does not execute stages).

---

## 4. Cross-Cutting Concerns

### 4.1 Test Coverage
- `tests/Trackdub.Architecture.Tests` enforces layer boundaries — run before structural refactors.
- `tests/Trackdub.Application.Tests` covers many stage handlers — verify GenerationStage wrappers too where behavior differs from raw handlers.
- `tests/Trackdub.TestDoubles` currently holds **72** shared `.cs` sources (68 `Fake*.cs` + 4 helpers: `NullModelHashVerifier.cs`, `SortFormerTestFixtures.cs`, `RecordingModelCacheRegistrar.cs`, `RequiresBundledModelFactAttribute.cs`). Check coverage for GenerationStage-facing fakes as stages evolve.
- Recommended: integration test for full SDK order + resume + injected failures + lips (when fixtures available).

### 4.2 Resumability
Still fragmented (F8). Unify evaluator + versioned snapshots.

### 4.3 Per-Segment Status
AGENT_CONTEXT: partial segment success must surface as partial, not blanket success. Verify handler and GenerationStage paths.

### 4.4 Artifact Preservation
Contract: original media + prior successful artifacts sacred. Both prep and GenerationStage paths should respect reuse/skip semantics.

### 4.5 Degradation Records
`PipelineDegradationWriter` lives at `src/Trackdub.Application/Transcripts/PipelineDegradationWriter.cs` (not under `Pipeline/`). Audit writers on every GenerationStage skip/fail/low-confidence path.

---

## 5. Proposed Fixes Roadmap

### Phase 0: Documentation (immediate, no code risk)
| ID | Fix | Effort |
|---|---|---|
| F6 | Fix stale AGENTS.md skill table | 30 min |
| — | Keep this unified audit as the corrected canonical note; fix any wiki links to missing agent-1 path | 15 min |

### Phase 1: Clarify Transcript Graph Boundary (or Extend It)
| ID | Fix | Effort |
|---|---|---|
| F1 | Document or unify stacked abstractions; do not delete live GenerationStages | 0.5–2 d |
| F4 | Only if unifying: add Translation/TTS/Export GenerationStages | 2–4 h each |
| F11 | First-class speech enhancement ownership + accurate SDK comments | 2–3 h |
| F12 | Promote explicit Mix stage | 3–5 h |
| ~~F5~~ | ~~Register missing GenerationStages~~ — already done | — |

### Phase 2: Unify Stage Graph and Execution
| ID | Fix | Effort |
|---|---|---|
| F2 | Centralize stage graph registry + metadata | 1–2 d |
| F8 | Unify resume under single evaluator | 3–5 h |
| F7 | Extract pipeline execution from SpeakerVoice.cs | 2–3 h |

### Phase 3: Fix SDK Execution Gaps
| ID | Fix | Effort |
|---|---|---|
| F3 | Add lip stages to SDK DefaultStageOrder + dispatch (conditional) | 1–2 h |
| F9 | Expand PrerequisiteStages (at least Diarization) | 30 min |
| F10 | Bounded parallelism for TTS/ASR | 4–8 h |

### Phase 4: Quality & Verification
| ID | Fix | Effort |
|---|---|---|
| F13 | Cancellation / timeout audit on long engines | 2–3 h |
| F14 | Centralize model provisioning gate | 2–3 h |
| F15 | Downstream guards for missing prior artifacts | 2–3 h |
| — | E2E: full SDK order + resume + failures + lips | 4–6 h |
| — | Run Architecture.Tests after structural phases | 5 min/phase |

### Total estimated effort: 5–9 days (F5 removed from plan)

---

## 6. Principles-Validation Audit (Grok)

Third note: [pipeline-principles-review-grok-2026-07-08.md](pipeline-principles-review-grok-2026-07-08.md) — documentation-based principles alignment (no source tree in that author's workspace). Forward-looking enhancements, not breakage reports. Supplement to code findings above.

### Grok recommendations — alignment with code findings

| Grok # | Recommendation | Related code finding | Notes |
|---|---|---|---|
| 1 | Immutable `PipelinePlan` record | F2 | `TranscriptGenerationContext` is mutable run context, not a plan |
| 2 | `capabilities` array in manifest schema | F1 | Adapter flags (e.g. phoneme timings) exist; not full manifest capabilities matrix |
| 3 | `IArtifact` base with declared I/O kinds | F1, F2 | `ArtifactKind` exists; stages don't formally declare I/O contracts |
| 4 | `ReadinessReport` persisted with snapshot | F14 | Complements pre-flight checker |
| 5 | Mandatory fake↔real contract tests | — | Quality dimension not fully verified here |
| 6 | `ITranscriptSegmentView` projection | F1, F7 | Decouples segment evolution from downstream stages |
| 7 | Document “new stage” checklist in AGENTS.md | F6 | Deleted `.devin/skills/` were meant to serve this |
| 8 | Prioritize M20 performance completion | — | MILESTONE Priority 2 |

Aspirational — none are current production breakages. Consider in Phase 2+, not as blockers.

---

## 7. Key File Index

### Application layer (pipeline orchestration)
```
src/Trackdub.Application/Transcripts/Pipeline/ITranscriptGenerationStage.cs
src/Trackdub.Application/Transcripts/Pipeline/TranscriptPipelineBuilder.cs
src/Trackdub.Application/Transcripts/Pipeline/TranscriptGenerationContext.cs
src/Trackdub.Application/Transcripts/Pipeline/TranscriptPipelineResumeHydrator.cs
src/Trackdub.Application/Transcripts/TranscriptGenerationService.cs
src/Trackdub.Application/Transcripts/TranscriptWorkspace.cs
src/Trackdub.Application/Transcripts/ProjectWorkflow.cs
src/Trackdub.Application/Transcripts/Stages/
  VadGenerationStage.cs
  SpeechEnhancementGenerationStage.cs
  SpeakerDiarizationStage.cs          ← not *GenerationStage.cs filename
  AsrGenerationStage.cs
  TextRefinementGenerationStage.cs
  SpeakerAssignmentAndPersistenceStage.cs
src/Trackdub.Application/Transcripts/*StageHandler.cs   ← 10 under Transcripts/
src/Trackdub.Application/LipSync/LipSyncStageHandler.cs
src/Trackdub.Application/LipSynthesis/LipSynthesisStageHandler.cs
src/Trackdub.Application/Pipeline/PipelinePreFlightChecker.cs
src/Trackdub.Application/Transcripts/PipelineDegradationWriter.cs   ← not under Pipeline/
src/Trackdub.Application/Projects/ProjectMediaIngestService.cs
src/Trackdub.Application/Projects/SegmentStageRunProvenanceStore.cs
src/Trackdub.Application/Mixing/MixPlanBuilder.cs
src/Trackdub.Application/Mixing/MixPlanStore.cs
src/Trackdub.Application/Transcripts/StageRunHelper.cs
src/Trackdub.Application/Transcripts/StageArtifactResumeEvaluator.cs
```

### Domain layer
```
src/Trackdub.Domain/StageRuns/StageNames.cs              ← 18 stage constants
```

### Composition
```
src/Trackdub.Composition/CompositionRoot.cs              ← GenerationStages ~L397–404 (all 6)
```

### SDK (headless)
```
src/Trackdub.Sdk/TrackdubDubbingEngine.cs                ← DefaultStageOrder, dispatch, resume
src/Trackdub.Sdk/TrackdubPipelineStages.cs               ← RequiresSourceMedia helper only
src/Trackdub.Sdk/TrackdubPipelineReadinessChecker.cs
src/Trackdub.Sdk/TrackdubProjectContextResolver.cs
src/Trackdub.Sdk/StageOutcome.cs
```

### Avalonia UI
```
src/Trackdub.App.Avalonia/ViewModels/AvaloniaMainWindowViewModel.PipelineUi.cs
src/Trackdub.App.Avalonia/ViewModels/AvaloniaMainWindowViewModel.SpeakerVoice.cs  ← RunPipelineStage
src/Trackdub.App.Avalonia/ViewModels/AvaloniaMainWindowViewModel.ProjectImport.cs
src/Trackdub.App.Avalonia/ViewModels/ExportMixViewModel.cs
src/Trackdub.App.Avalonia/ViewModels/PipelineStageModelCatalog.cs
src/Trackdub.App.Avalonia/Playback/AvaloniaDubPlaybackCoordinator.cs
```

### Inference layer
```
src/Trackdub.Inference/Runtime/Planning/RuntimePlanner.cs
src/Trackdub.Inference/Runtime/ModelManifest/ModelManifestLoader.cs
src/Trackdub.Inference/Runtime/ModelManifest/BundledModelManifestRegistry.cs
```

### Docs referenced
```
AGENT_CONTEXT.md
MILESTONE.md
AGENTS.md                                  ← stale skill table (TestDoubles count fixed 2026-07-09)
docs/Audit/pipeline-full-walk-audit-2026-07-08.md
docs/architecture/pipeline-principles-review-grok-2026-07-08.md
docs/architecture/P0-pipeline-audit-2026-06-01.md
tests/Trackdub.Architecture.Tests/
```

---

## 8. Correction Log (2026-07-08 re-verification)

| Original claim | Verdict | Correction |
|---|---|---|
| GenerationStages unused by Avalonia/SDK | **False** | Both use via `ProjectWorkflow` → `TranscriptGenerationService` for transcript stages |
| Dual independent working vs dead pipelines | **Misleading** | Stacked: GenerationStages wrap handlers for transcript subgraph |
| CompositionRoot registers 4/6 GenerationStages | **False** | All 6 registered |
| StageNames = 17 | **False** | **18** constants |
| Speech enhancement inside `StemSeparationStageHandler` | **False** | `TryPrepareSpeechAudioAsync` + GenerationStage; stem handler clean |
| Lip stages missing everywhere | **Partial** | Missing from **SDK** only; Avalonia dispatches |
| `CreateMediaSpineAsync` → normalize inline | **Overstated** | Spine + lazy `EnsureNormalized*` APIs |
| `PipelineDegradationWriter` under `Pipeline/` | **False** | Under `Transcripts/` |
| TestDoubles = 43 | **Stale** | **68** `Fake*.cs` + **4** helpers (**72** `.cs` total) |
| Agent-1 source at `docs/architecture/local-pipeline-audit-2026-07-08.md` | **Missing** | Only this unified file + `docs/Audit/pipeline-full-walk-…` |
| F5 “register missing stages” in roadmap | **Obsolete** | Removed from Phase 1 |

---

> **Audit methodology:** Read-only source scan. Independent agent audits merged, then spot-verified against current sources (handlers, CompositionRoot, StageNames, DubbingEngine switch, ProjectWorkflow stem/enhancement path, Avalonia RunPipelineStage, file index). No builds or tests run.  
> **Next step:** `dotnet test tests/Trackdub.Architecture.Tests` before any structural code changes.

# Trackdub P0 Pipeline Audit — 2026-06-01

**Author:** Claude (Opus 4.8), read-only investigation. **Branch:** `main` (commit `2f0fb44da`).
**Scope:** `Trackdub.Application` pipeline engine/stages/prerequisites/stage-run persistence, the
`Trackdub.Inference` runtime planner where it gates pipeline execution, `Trackdub.Sdk`
single-stage path, and DI wiring in `Trackdub.Composition`. **No code was changed.**

**Purpose:** Re-validate the backlog **P0-1** premise ("none of the pipeline stages work") and the
prior write-ups (`Opus Arch Audit 5.23.md`, `codex Arch Review 5.24.md`) against current `main`, then
hand Cursor an ordered, evidence-based fix plan. The prior audits cite a stale `D:\Dev\Trackdub`
checkout; every claim below is re-cited against the current tree with `path:line`.

---

## 1. Executive summary

**The backlog's stated P0-1 root causes are already fixed.** The Opus 5.23 Tier-1 items that the
backlog points at — wrong `StageName`, `DiarizationStageHandler` bypass, missing pre-flight
implementation, fabricated diarization `Guid`, no per-stage timeout — are all repaired in current
`main` (see §3). In-process orchestration is healthy: the two full-pipeline tests
(`TranscriptProjectServiceTests`, `TranscriptWorkspacePipelineGuardTests`) build VAD→Diarization→ASR→
SpeakerAssignment with fakes and **pass 125/125** (verified, §6). So P0-1 is **not** a stage-wiring
or orchestration bug anymore.

**The residual P0-1 cluster is "the first stage's model is never provisioned + readiness is reported
dishonestly," not stage wiring:**

1. **VAD is never provisioned on the transcription path (the user-path blocker).** The only ensure
   that covers the VAD model — `RuntimeModelSetupCoordinator.EnsureImportModelsAvailableAsync`
   (`CreateImportRequests` includes VAD) — has **no live caller** in `src/`. The Avalonia "transcribe"
   action ensures **ASR only** (`EnsureAsrModelAvailableAsync`), then runs the full VAD→Diarization→ASR
   trio. VAD is the **first** stage and a hard prerequisite, and — unlike diarization, which
   auto-downloads inside its own handler — VAD has **no** in-handler download. So unless the Model
   Manager already put `silero-vad` on disk, VAD fails first and the whole transcription fails →
   "nothing works." (Finding **A7**.)
2. **Readiness is reported as a failure, not a block.** `RuntimePlannerPreFlightChecker` only throws on
   planner `Blocked`; a missing model yields `DownloadRequired`, which passes pre-flight. The routed
   engine then throws a generic `InvalidOperationException` ("Model setup required before running …"),
   recorded as a bucketed `PIPELINE_STAGE_UNHANDLED_EXCEPTION` **failure** instead of an honest,
   pre-run "download required" block. So even the VAD failure above looks like a crash, not a
   provisioning problem. (Finding **A1**.)
3. **"Run one stage" is not one stage.** Requesting VAD, ASR, or Diarization individually re-runs the
   whole transcript sub-pipeline (and, on first run, project creation). There is no isolated single-
   stage execution for the three transcript stages. (Finding **A2**.)
4. **No resume.** The resume check is a `return false;` stub, so every run restarts at VAD. (Finding
   **A3**.) And **real-model end-to-end is unproven by this static audit** (read-only; A5).

**Recommended Cursor sequence (Lane A):**
**Slice 1 — "one honest green stage"** = (a) provision VAD on the transcription path — call the
orphaned `EnsureImportModelsAvailableAsync` (or add a VAD ensure) before `RunInitialTranscriptionAsync`
(A7); **and** (b) make `DownloadRequired` a distinct pre-run blocked/needs-setup state in pre-flight so
a still-missing model blocks honestly instead of failing late (A1). Add a fake-backed test proving VAD
reports running→succeeded with an artifact on the happy path and **blocked (not failed)** when the
model is absent. **Slice 2** = true single-stage path / rename the conflated one (A2). **Slice 3** =
stale-`Running` reaper + thread the logger (A4). Resume (A3) is a follow-up. **P0-3 as written is not
reproducible** — see §4 open question.

---

## 2. Two execution surfaces (orientation)

There are **two** stage-execution paths; conflating them is the source of most confusion in the prior
audits.

| Surface | Entry | Stages |
|---|---|---|
| **Transcript sub-pipeline** | `TranscriptGenerationService.GenerateTranscriptAsync` builds 4 `ITranscriptGenerationStage`s | VAD → Diarization → ASR → SpeakerAssignment ([TranscriptGenerationService.cs:42](../../src/Trackdub.Application/Transcripts/TranscriptGenerationService.cs):42-47) |
| **Workspace workflows** | `TranscriptWorkspace` / `ProjectWorkflow` methods, driven by the Avalonia VM or SDK | Separation, AudioPreparation/SpeechEnhancement ("Cleanup"), Translation, TTS, PreviewMix, Export |

The `*StageHandler` classes (`VadStageHandler`, `AsrStageHandler`, `DiarizationStageHandler`,
`StemSeparationStageHandler`, `SpeechAudioPreparationStageHandler`, `SpeechAudioEnhancementStageHandler`,
`StartTtsStageHandler`, `ExportStageHandler`) are the leaf workers. The four `ITranscriptGenerationStage`
wrappers (`Vad/Asr/SpeakerDiarization/SpeakerAssignmentAndPersistence` in `Transcripts/Stages/`) are
distinct from them. **Every `*StageHandler` is reachable** — none is registered-but-orphaned (the Opus
#2 concern; see §3).

---

## 3. Status of prior-audit findings (Opus 5.23 + codex 5.24)

| Prior ID | Claim | Now | Evidence (current `main`) |
|---|---|---|---|
| Opus #1 | `SpeakerAssignmentAndPersistenceStage.StageName` returns `Diarization` | **FIXED** | Returns `StageNames.SpeakerAssignment` — [SpeakerAssignmentAndPersistenceStage.cs:22](../../src/Trackdub.Application/Transcripts/Stages/SpeakerAssignmentAndPersistenceStage.cs):22 |
| Opus #2 | `DiarizationStageHandler` registered but bypassed by the pipeline | **FIXED** | `SpeakerAssignmentService` takes it as a **required** ctor dep ([SpeakerAssignmentService.cs:23,33](../../src/Trackdub.Application/Transcripts/SpeakerAssignmentService.cs):23) and `CreateDiarizationAsync` calls `diarizationStageHandler.DiarizeAsync` ([:302](../../src/Trackdub.Application/Transcripts/SpeakerAssignmentService.cs):302); registered at [CompositionRoot.cs:280](../../src/Trackdub.Composition/CompositionRoot.cs):280. Diarization's auto-download path is therefore now on the main pipeline. |
| Opus #3 | `IPipelinePreFlightChecker` has no impl, is nullable, never runs | **FIXED (with new gap A1)** | Real impl `RuntimePlannerPreFlightChecker` ([RuntimePlannerPreFlightChecker.cs](../../src/Trackdub.Composition/Pipeline/RuntimePlannerPreFlightChecker.cs)); registered [CompositionRoot.cs:290](../../src/Trackdub.Composition/CompositionRoot.cs):290; injected **non-nullable** and actually invoked ([TranscriptGenerationService.cs:25,37,59-64](../../src/Trackdub.Application/Transcripts/TranscriptGenerationService.cs):59). |
| Opus #4 | Diarization fabricates `Guid.NewGuid()` → orphan stage-run id | **FIXED** | Now `?? Guid.Empty` with a comment, and the real id is threaded via `SpeakerTurn.Create(..., stageRun.Id)` ([SpeakerAssignmentService.cs:353](../../src/Trackdub.Application/Transcripts/SpeakerAssignmentService.cs):353). The `Guid.Empty` branch is dead on the success path: empty turns → `CreateDiarizationAsync` returns `null` → no artifact write ([SpeakerDiarizationStage.cs:43-56](../../src/Trackdub.Application/Transcripts/Stages/SpeakerDiarizationStage.cs):43). |
| Opus #5 | `StageRunHelper` terminal persistence writes to `Trace`; stale `Running` rows lie | **PARTIAL → A4** | `Trace`→`IApplicationLogger` ([StageRunHelper.cs:235](../../src/Trackdub.Application/Transcripts/StageRunHelper.cs):235). But the logger is optional and **not passed** on the hot path (`RunStageAsync` called without it — [VadStageHandler.cs:35-58](../../src/Trackdub.Application/Transcripts/VadStageHandler.cs):35), and there is still **no startup orphan-reaper** for stale `Running` rows (none found in repo). |
| Opus #6 | No per-stage timeout, cancellation filter too broad, no resume | **PARTIAL → A3** | Per-stage **timeout exists** (`StageOptions.Timeout`, applied with a linked CTS — [TranscriptPipelineBuilder.cs:50-83](../../src/Trackdub.Application/Transcripts/Pipeline/TranscriptPipelineBuilder.cs):50; [StageOptions.cs](../../src/Trackdub.Application/Transcripts/Pipeline/StageOptions.cs)). Cancellation filter tightened to `stageToken.IsCancellationRequested`. **Resume still missing** (A3). |
| Opus #13 | Each handler re-implements start/complete/cancel/fail boilerplate | **FIXED** | Canonical `StageRunHelper.RunStageAsync<T>` exists ([StageRunHelper.cs:44](../../src/Trackdub.Application/Transcripts/StageRunHelper.cs):44); VAD/ASR/diarization-prep/AudioPreparation use it ([VadStageHandler.cs:35](../../src/Trackdub.Application/Transcripts/VadStageHandler.cs):35, [AsrStageHandler.cs:37](../../src/Trackdub.Application/Transcripts/AsrStageHandler.cs):37). |
| codex | `GenerateCandidatesHandler` builds `TtsSynthesisRequest` without options → bypasses `CommercialSafeMode` | **MOOT** | `InferenceRequestOptions` has **no `CommercialSafeMode` field** ([InferenceRequestOptions.cs](../../src/Trackdub.Contracts/Pipeline/InferenceRequestOptions.cs)); commercial safety is manifest-only per AGENTS.md. Handler now passes `InferenceRequestOptions.Default` ([GenerateCandidatesHandler.cs:88](../../src/Trackdub.Application/Transcripts/GenerateCandidatesHandler.cs):88). |
| codex | ASR runs on zero VAD regions and records success | **FIXED** | `AsrGenerationStage` skips with `VAD_NO_REGIONS` when `RegionPlan.Regions.Count == 0` ([AsrGenerationStage.cs:32-66](../../src/Trackdub.Application/Transcripts/Stages/AsrGenerationStage.cs):32). |
| codex | `RuntimePlanFactory` reports CPU `Ready` on file-check only | **TRUE (by design) → A1 context** | CPU plan = `Ready` after required-files-exist check ([RuntimePlanFactory.cs:69-85](../../src/Trackdub.Inference/Runtime/Planning/RuntimePlanFactory.cs):69); non-CPU = `Verified` after smoke test ([:110-128](../../src/Trackdub.Inference/Runtime/Planning/RuntimePlanFactory.cs):110). `Ready` ≠ `Verified` is now explicit. The honesty gap is the *consumer* side (A1), not the planner. |

**Net:** the prior audits are ~70% stale on Tier-1. Their lower tiers (TTS god-service, ingest rollback,
dual WAV parsers, translation segment-drop) were **not** re-verified here (out of P0 scope) and may
still hold — flagged for a later Lane-A pass, not P0.

---

## 4. P0 finding table (current `main`)

| ID | Sev | Symptom | Root cause | Evidence | Suggested fix | Backlog | Effort |
|---|---|---|---|---|---|---|---|
| **A7** | blocker | On the live Avalonia path, transcription fails at the **first** stage (VAD) unless the Model Manager already downloaded `silero-vad`; user sees "no stage works." | The only ensure covering the VAD model (`EnsureImportModelsAvailableAsync`, whose `CreateImportRequests` includes VAD) has **no live caller**. The transcribe action ensures **ASR only**, then runs VAD→Diar→ASR. VAD has no in-handler auto-download (diarization does); so VAD is never provisioned by the run path. | Orphan: `EnsureImportModelsAvailableAsync` defined [RuntimeModelSetupCoordinator.cs:12](../../src/Trackdub.Application/Transcripts/RuntimeModelSetupCoordinator.cs#L12), no caller in `src/`. VAD in import set: [RuntimeModelRequestFactory.cs:102](../../src/Trackdub.Application/Transcripts/RuntimeModelRequestFactory.cs#L102). ASR-only ensure before full trio: [PipelineUi.cs:1379-1398](../../src/Trackdub.App.Avalonia/ViewModels/AvaloniaMainWindowViewModel.PipelineUi.cs#L1379). VAD handler has no download: [VadStageHandler.cs](../../src/Trackdub.Application/Transcripts/VadStageHandler.cs). | Provision VAD on the transcription path: call `EnsureImportModelsAvailableAsync` (or add an explicit VAD ensure) before `RunInitialTranscriptionAsync`. Pairs with A1 so a still-missing model blocks honestly. | P0-1 | S |
| **A1** | blocker | Even when a model is missing, the stage **fails** mid-run with a generic exception instead of being cleanly blocked pre-run — so A7 looks like a crash, not a setup gap. | Pre-flight only throws on planner `Blocked`; a missing model is `DownloadRequired`, which passes pre-flight. The routed engine then hard-throws `InvalidOperationException("Model setup required …")`, caught by the pipeline as `PIPELINE_STAGE_UNHANDLED_EXCEPTION`. Affects both the VM path (A7) and the SDK/headless path. | Pre-flight: [RuntimePlannerPreFlightChecker.cs:32](../../src/Trackdub.Composition/Pipeline/RuntimePlannerPreFlightChecker.cs#L32). Planner returns `DownloadRequired`: [RuntimePlanner.cs:191](../../src/Trackdub.Inference/Runtime/Planning/RuntimePlanner.cs#L191), [RuntimePlanFactory.cs:199](../../src/Trackdub.Inference/Runtime/Planning/RuntimePlanFactory.cs#L199). Engine throw: [InferenceEngineAdapterSelector.cs:23](../../src/Trackdub.Inference.Onnx/Runtime/Routing/InferenceEngineAdapterSelector.cs#L23). | Treat `DownloadRequired` (any non-runnable plan) as a distinct **pre-run blocked/needs-setup** outcome in `RuntimePlannerPreFlightChecker`, with a structured reason; fail fast as `MODEL_DOWNLOAD_REQUIRED` rather than letting the engine throw. | P0-1 | M |
| **A2** | high | "Run VAD/ASR/Diarization" runs the whole transcript sub-pipeline (and project creation), not the one stage. Breaks the P0-1 acceptance "run **at least one** stage" and the per-stage-Run UX target. | SDK maps all three to `CreateProjectAsync`; VM ASR run calls `RunInitialTranscriptionAsync` when no transcript exists. | SDK: [TrackdubDubbingEngine.cs:372-389](../../src/Trackdub.Sdk/TrackdubDubbingEngine.cs):372. VM: [AvaloniaMainWindowViewModel.PipelineUi.cs:1392-1402](../../src/Trackdub.App.Avalonia/ViewModels/AvaloniaMainWindowViewModel.PipelineUi.cs):1392. | Add a true single-stage workflow (or at minimum rename the conflated "stage" to "transcribe" and document it). For P0-1 acceptance, prove VAD-only or the whole transcribe step reaches a green status with an artifact. | P0-1 | M |
| **A3** | high | Every run restarts at VAD even when artifacts exist; no resume. | `HasValidExistingArtifacts` is a `return false;` TODO stub; `GenerateTranscriptAsync` has no resume-from-artifacts. | [TrackdubDubbingEngine.cs:516-532](../../src/Trackdub.Sdk/TrackdubDubbingEngine.cs):516; [TranscriptGenerationService.cs:49-84](../../src/Trackdub.Application/Transcripts/TranscriptGenerationService.cs):49. | Implement the documented resume check (query latest successful `StageRunRecord`, compare runtime info to snapshot, verify artifact files). Follow-up after A1/A2. | P0-1 (Opus #6) | L |
| **A4** | medium | A stage-run row can stay `Running` forever if both terminal writes fail, with no log and no reaper. | `PersistTerminalAsync` logs via `IApplicationLogger?` but the logger is usually `null` on the hot path; no startup pass ages stale `Running`→`Failed`. | [StageRunHelper.cs:201-237](../../src/Trackdub.Application/Transcripts/StageRunHelper.cs):201 (logger param unset by [VadStageHandler.cs:35](../../src/Trackdub.Application/Transcripts/VadStageHandler.cs):35 / [AsrStageHandler.cs:37](../../src/Trackdub.Application/Transcripts/AsrStageHandler.cs):37). No reaper in repo. | Thread `IApplicationLogger` into `RunStageAsync` callers; add a startup pass that ages `Running` rows older than N min into `Failed` with reason `process_crashed_or_persist_failed`. | P0-1 (Opus #5) | S |
| **A5** | high (honesty) | Static audit cannot prove a real model produces a real artifact end-to-end; the model-manager/download path (MILESTONE P1) is explicitly "not yet reliable." | No real-model run was performed (read-only; models may be absent). Recent churn here: `f387353e9 fix: Qwen3 ASR split ONNX manifest`. | MILESTONE.md Priority 1 ("Model manager and downloader"); git log `f387353e9`. | First Cursor slice must include a real-model run of one stage on a sample project and confirm honest status, not just unit tests. | P0-1 | M |
| **A6** | low | UI/settings label provider-*loadability* as "Ready" (codex's naming nit). | Provider-loadable ≠ model+runtime+smoke ready. | (codex) `RuntimeSelectionService` / `SettingsWindowViewModel` — not re-verified this pass. | Rename to "provider loadable"; reserve "Ready"/"Verified" per `StageRuntimePlanStatus`. Cosmetic; defer. | — | S |

### Commercial safety (Q8) — clean

Commercial routing is manifest-only: there is no runtime `CommercialSafeMode` flag
([InferenceRequestOptions.cs](../../src/Trackdub.Contracts/Pipeline/InferenceRequestOptions.cs)), matching
AGENTS.md. The **Mrx cocktail-fork
is present and `commercial_allowed: true` (MIT)** ([manifest:144-160](../../src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json):144) — not a commercial *violation*, but P3-2 still slates it
for removal. No accidental Mrx mis-routing found.

---

## 5. Answers to the eight investigation questions

1. **Stage identity** — *No current mis-attribution.* The Opus #1 `StageName` lie is fixed
   (SpeakerAssignmentAndPersistenceStage.cs:22). `StageRunHelper.StartKnownStageRun` maps every
   `StageNames.*` constant explicitly (StageRunHelper.cs:22-42).
2. **Handler wiring** — *No orphaned stage handler,* but **one orphaned ensure method.*
   `DiarizationStageHandler` is now a required dep of `SpeakerAssignmentService` and is invoked via
   `CreateDiarizationAsync`→`DiarizeAsync` (SpeakerAssignmentService.cs:33,302); all eight
   `*StageHandler`s are reachable (§2). **However**, `RuntimeModelSetupCoordinator.EnsureImportModelsAvailableAsync`
   — the only ensure that provisions the **VAD** model — has no live caller (RuntimeModelSetupCoordinator.cs:12;
   no caller in `src/`). This is the registered-but-unused finding (A7).
3. **Model download** — *Asymmetric and VAD is uncovered.* Diarization auto-downloads inside its
   handler on the main path; **VAD/ASR do not** auto-download in-pipeline. The Avalonia VM ensures
   models via `RuntimeModelSetupCoordinator`, but **per clicked stage only** — `EnsureAsrModelAvailableAsync`
   ensures ASR alone (PipelineUi.cs:1379) before running the full VAD→Diar→ASR trio, and the VAD-covering
   `EnsureImportModelsAvailableAsync` is never called (A7). So VAD's model is provisioned by **nothing**
   on the run path; readiness depends entirely on the Model Manager having pre-downloaded `silero-vad`.
   The SDK/headless `GenerateTranscriptAsync` path is worse — only the `Blocked`-only pre-flight runs (A1).
4. **Prerequisites (P0-3)** — *"Cleanup requires ASR" is not reproducible in current `main`.* The
   Cleanup row is runnable on `HasProject && HasMedia` ([PipelineUi.cs:388](../../src/Trackdub.App.Avalonia/ViewModels/AvaloniaMainWindowViewModel.PipelineUi.cs):388) and
   `ProjectWorkflow.RunSpeechAudioPreparationAsync` requires only `NormalizedAudio` (optionally a vocal
   stem), never an ASR artifact ([ProjectWorkflow.cs:248-267](../../src/Trackdub.Application/Transcripts/ProjectWorkflow.cs):248). See open question §8.
5. **Fake "ready" UI** — *Status is derived from real `StageRunRecord`s,* not fabricated:
   `DescribeStageRun(GetLatestStageRun(state.StageRuns, …))` (PipelineUi.cs:378-403,484). The honesty
   gaps are A1 (a missing model becomes a generic *failure*, not a clean *blocked*) and A6 (provider
   "loadable" labelled "Ready").
6. **Stage-run integrity** — Dangling `StageRunId` largely resolved (Opus #4 fixed). **Stale `Running`
   rows remain possible** with no reaper and a usually-null logger (A4).
7. **Timeouts / resume** — Per-stage **timeout exists** (StageOptions/TranscriptPipelineBuilder.cs:50-83).
   **Resume is a stub** (A3). Cancellation is recorded as `Canceled` via `RunStageAsync` (StageRunHelper.cs:73-85).
8. **Commercial safety** — Manifest-only; clean (see §4 box).

---

## 6. Verification performed

```powershell
dotnet test tests/Trackdub.Application.Tests --no-restore -m:1 `
  --filter "FullyQualifiedName~TranscriptProjectServiceTests|FullyQualifiedName~TranscriptWorkspacePipelineGuardTests"
# Passed!  Failed: 0, Passed: 125, Skipped: 0  (net10.0)
```

These tests construct the full VAD→Diarization→ASR→SpeakerAssignment pipeline with `TestDoubles` fakes
(TranscriptProjectServiceTests.cs:2995-2999, TranscriptWorkspacePipelineGuardTests.cs:658-662). Their
green state is the evidence that **orchestration works in-process** — so P0-1 is a real-model/run-trigger
problem (A1/A2/A5), not stage wiring. No real-model or whole-solution run was performed (read-only; A5).

---

## 7. Implementation plan for Cursor (vertical slices, Lane A)

> Branch suggestion per slice: `agent/cursor/p0-1-<slug>`. One owner on P0-1 until a stage is green.

**Slice 1 — "one honest green stage" (do first).** Two coupled fixes: provision VAD, and make a still-
missing model a *blocked* outcome rather than a *failure*.
- **A7 (provision VAD):** Call `RuntimeModelSetupCoordinator.EnsureImportModelsAvailableAsync` (or add an
  explicit VAD ensure) before `RunInitialTranscriptionAsync` so the first stage's model is downloaded.
  Files: the transcribe trigger (`AvaloniaMainWindowViewModel.PipelineUi.cs` `RunAsrStageAsync` — **note
  this is Cursor's own lane, coordinate**) or, preferably, ensure inside the Application import/initial-
  transcription path so the SDK benefits too. Verify the orphaned `EnsureImportModelsAvailableAsync` is
  the right seam (it already builds the VAD+ASR import request set).
- **A1 (honest block):** In `src/Trackdub.Composition/Pipeline/RuntimePlannerPreFlightChecker.cs`,
  throw/return a structured blocked result on `DownloadRequired`, not only `Blocked`; optionally map a
  needs-setup pre-flight to a skipped/blocked stage status in
  `src/Trackdub.Application/Transcripts/TranscriptGenerationService.cs` rather than letting the engine
  throw. Do **not** fake readiness — block, don't auto-pass.
- Tests: `tests/Trackdub.Composition.Tests/Pipeline/PipelinePreFlightCheckerTests.cs` (add a
  `DownloadRequired`→blocked case); a fake-backed Application test proving VAD reports
  `Running`→`Succeeded` with an artifact on the happy path and `Blocked` (distinct from `Failed`) when the
  model is absent.
- Acceptance (P0-1): one stage (VAD/transcribe) runs green on a sample project with the model present;
  "UI shows distinct states (running/succeeded/failed/skipped), not fake ready" when it is absent.

**Slice 2 — true single-stage run (A2).** Give the SDK/workspace an isolated VAD (or ASR) execution that
does not trigger `CreateProjectAsync`/`RunInitialTranscriptionAsync`, or explicitly rename and document the
conflated "stage." Files: `src/Trackdub.Sdk/TrackdubDubbingEngine.cs` (RunStageWorkflowAsync); a workspace
command. Tests: SDK `run-stage` test that asserts a single stage executed (codex flagged the existing
tests are parse-only).

**Slice 3 — stale-`Running` hygiene (A4).** Thread `IApplicationLogger` into `RunStageAsync` callers; add
a startup reaper that ages stale `Running` rows to `Failed`. Files:
`src/Trackdub.Application/Transcripts/StageRunHelper.cs`, the VAD/ASR/handler call sites, and the
SQLite stage-run store / app startup. Tests: extend `StageRunHelperTests` (codex noted cancel/skip/
partial/fallback are uncovered).

**Slice 4 (follow-up) — resume (A3).** Implement `HasValidExistingArtifacts` and/or
`TryResumeContextFromArtifactsAsync`. Larger; after a stage is reliably green.

---

## 8. Test plan

Run after each slice (verification ladder, MILESTONE.md):

```powershell
# Slice 1
dotnet test tests/Trackdub.Composition.Tests --no-restore -m:1 --filter "FullyQualifiedName~PipelinePreFlightChecker"
dotnet test tests/Trackdub.Application.Tests --no-restore -m:1 --filter "FullyQualifiedName~TranscriptProjectServiceTests"
# Slice 2
dotnet test tests/Trackdub.Sdk.Tests --no-restore -m:1
# Slice 3
dotnet test tests/Trackdub.Application.Tests --no-restore -m:1 --filter "FullyQualifiedName~StageRunHelper"
# Regression gate before any merge
dotnet build Trackdub.sln -m:1 ; dotnet test --no-build
```

**Tests that currently assert behavior worth knowing about:**
- The full-pipeline fakes (TranscriptProjectServiceTests:2995, TranscriptWorkspacePipelineGuardTests:658)
  pass today — do not let A1/A2 changes regress them.
- `PipelinePreFlightCheckerTests` currently only covers `Blocked` (throws) vs not-blocked (passes) — it
  does **not** assert anything for `DownloadRequired`; that absence is the A1 test gap.
- `FakePipelinePreFlightChecker` records called stage names but no readiness outcome — fine for
  orchestration tests, insufficient to prove A1; add a fake that can return a blocked outcome.
- No test asserts a wrong stage name today (Opus #1 is fixed), so none needs correcting on that axis.

---

## 9. Parallel safety

This audit and its recommended slices are **safe alongside Cursor worktree
`agent/cursor/wave1-ui-polish`** (P1-6, P1-1, P1-3, P1-4/5, P3-1 — AXAML only). No file overlap is
expected on the Lane-A targets:
`src/Trackdub.Composition/Pipeline/RuntimePlannerPreFlightChecker.cs`,
`src/Trackdub.Application/Transcripts/{TranscriptGenerationService,StageRunHelper}.cs`,
`src/Trackdub.Sdk/TrackdubDubbingEngine.cs`, and the corresponding `tests/**`.

**One caution:** A2/A6 reference Avalonia VM files
(`AvaloniaMainWindowViewModel.PipelineUi.cs`) for *evidence only* — Lane A must **not edit**
`src/Trackdub.App.Avalonia/**`. If A1's blocked-state surfacing needs a UI binding, coordinate with
Cursor; the Application/Composition change can land independently and the VM already reads
`StageRunRecord` status, so no AXAML change is required for the pipeline fix itself.

---

## 10. Open questions — **ANSWERED 2026-06-01**

1. **P0-3 intent.** ✅ Observed as a UI label ("audio not ingested yet") when the pipeline was broken,
   not a real prerequisite in the running code. **Decision: close P0-3 as "not reproduced in current
   main." Fold the desired cleanup→ASR ordering improvement into P3-4.**
2. **A1 product call.** ✅ **(b) fail fast** with `MODEL_DOWNLOAD_REQUIRED`. Pipeline service does not
   auto-download — Model Manager remains sole download owner. **Additional requirement:** a headless CLI
   path for model provisioning must exist so users have a way to download models before running headless.
   `IModelDownloadOrchestrator.DownloadAsync`/`VerifyAsync` is already UI-free and registered in
   Composition — a `models download <id>` command in `Trackdub.Cli` wraps it directly, no new abstraction
   needed.
3. **A2 scope.** ✅ **True isolated single-stage run required** for P0-1 acceptance. Not just "transcribe
   step." Slice 2 must give each pipeline stage its own execution path.

---

### Revised slice sequence (post-answers)

**Slice 0 — headless model CLI** *(new, prerequisite for honest fail-fast)*
Before Slice 1 can "fail fast with MODEL_DOWNLOAD_REQUIRED" and not strand users, a headless path to
provision models must exist. Add `models download <model-id>` and `models status` commands to
`Trackdub.Cli`. Files: `src/Trackdub.Cli/Commands/` (new); `IModelDownloadOrchestrator.DownloadAsync` is
the seam ([IModelDownloadOrchestrator.cs](../../src/Trackdub.Contracts/IModelDownloadOrchestrator.cs)).
Registered in Composition already. Progress via `IProgress<ModelDownloadProgress>` → console output.
Tests: `tests/Trackdub.Sdk.Tests` or a new `Trackdub.Cli.Tests`.

**Slice 1 — provision VAD + honest block** *(as before, A7 + A1)*

**Slice 2 — true isolated single-stage run** *(now required, A2)*
Each stage in `TrackdubDubbingEngine.RunStageWorkflowAsync` must dispatch independently. VAD, Diarization,
and ASR currently all route to `CreateProjectAsync` (TrackdubDubbingEngine.cs:372-389). Each needs its own
workspace workflow method. VAD = run just the VAD+SpeechRegion step and surface a VAD `StageRunRecord`.
Diarization = assume VAD artifact present, run diarization only. ASR = assume VAD artifact present, run
ASR + SpeakerAssignment only (or allow re-run of the full trio from existing state).

**Slice 3 — stale-`Running` hygiene** *(A4, after above)*

---

### Handoff (backlog format)

```markdown
## Backlog: P0-1 None of the pipeline stages work
**Owner:** cursor
**Branch:** agent/cursor/p0-1-provision-vad-honest-block
**Status target:** done (one stage green with honest status)
**Acceptance:** Run one stage on a sample project; UI shows distinct running/succeeded/failed/skipped, not fake ready; relevant dotnet test pass.
**Depends:** —
**Touched (planned):**
  Slice 0: src/Trackdub.Cli/Commands/ (models download/status); IModelDownloadOrchestrator is the seam.
  Slice 1: Application/Transcripts/ProjectWorkflow.cs (EnsureImportModelsAvailableAsync caller); Composition/Pipeline/RuntimePlannerPreFlightChecker.cs (DownloadRequired→block).
  Slice 2: Sdk/TrackdubDubbingEngine.cs (per-stage dispatch); Application workspace methods.
  (VM file Avalonia/AvaloniaMainWindowViewModel.PipelineUi.cs — Cursor Wave-1 lane, coordinate if needed.)
**Verified:** Orchestration 125/125 (Application.Tests full-pipeline fakes). Real-model E2E NOT verified.
**Not verified:** real ONNX model run; whole-solution build/test.
**Audit:** docs/architecture/P0-pipeline-audit-2026-06-01.md (open questions answered 2026-06-01)
**Open questions:** CLOSED (see §10).
**P0-3 status:** CLOSED as not-reproduced; fold ordering into P3-4.
**First task:** Slice 0 — headless `models download/status` CLI commands using IModelDownloadOrchestrator.
**Log:** %LOCALAPPDATA%\Trackdub\trackdub.log
**Next ID:** P0-1 Slices 1→2→3 → then P0-4 (playback) in Lane C.
```

\*\*Executive Summary: Deep Architectural Audit of TrackDub Core Media Pipeline\*\*



This audit examines the core media processing pipeline (ingest → normalized audio → VAD → diarization → ASR → translation/glossary → TTS → timing reconciliation → phoneme-aligned lip sync (M22) → preview/export) based on the authoritative documentation: LONGTERM-ROADMAP.md, AGENT\_CONTEXT.md, and AGENTS.md. The source tree is not present in the current workspace, so all findings are documentation-driven with explicit “requires source verification” caveats. Where the docs make concrete claims (especially M22 “wired and real-model verified (2026-06-12)”), they are treated as the current truth pending code inspection.



\*\*Overall Assessment\*\*  

The pipeline architecture is \*\*highly aligned\*\* with TrackDub’s core principles: honest readiness (multiple orthogonal states, never a single boolean), manifest-driven providers, fake-backed design, immutable snapshots, artifact preservation with provenance, and clear skip/failure paths that never destroy prior usable work. Stages are intentionally loosely coupled through contracts and artifacts rather than direct calls. The orchestration hub (Application + Composition + RuntimePlanner) carries the expected higher coupling. Modularity and testability are strong. Extensibility for new providers is excellent; adding entirely new stages is supported but carries non-trivial wiring cost. Performance/memory characteristics are visible in session pooling and the hardware profiler but lack complete end-to-end evidence (M20/M20a status). The main architectural risk is the experimental visual dubbing arc (M23-M26) potentially leaking into the stable audio pipeline if lane separation is not rigorously maintained.



The design supports the product spine and M22 positioning without obvious violations of the “keep the nouns sharp” rule (speech enhancement ≠ stem separation ≠ overlap rescue).



\*\*Significant Findings – Detailed\*\*



\### 1. Stage Orchestration and Runtime Planning

\*\*Finding\*\*: Runtime planning is manifest- and hardware-evidence-driven with explicit separation of “provider registered” vs “model runnable on current hardware.” The planner feeds StageRuntimeRequirementsCatalog and respects per-stage allow-lists. This is a strength, but the orchestration surface (RuntimePlanner + RuntimePlanningPreferencesService + StageRunHelper + Composition) is the primary coupling point.



\- \*\*Severity\*\*: Medium  

\- \*\*Evidence\*\*: LONGTERM-ROADMAP M16a (runtime route selection, planner state distinguishes registered vs verified runnable), M19 (Hardware Profiler integration, IRuntimePlanningPreferences, StageRunHelper persists BenchmarkEvidenceId), AGENT\_CONTEXT “Pipeline + stage rules” (explicit prerequisite contract, immutable snapshot at run start), AGENTS.md dependency graph (Application owns orchestration). M22 specific: RuntimeStage.LipSync added to catalog; RoutedForcedAligner honors PreferredModelAlias + RequirePhonemeTimings capability gate.  

\- \*\*Impact\*\*: Good isolation of decision logic from execution. However, as more stages (overlap-rescue, lip synthesis, portrait) and execution providers (TRT RTX plugin, catalog EPs, future cloud) are added, planner complexity and test surface will grow. Mutable preferences vs immutable snapshot boundary must stay crisp or UI-driven changes can affect running pipelines.  

\- \*\*Recommended Improvement + Trade-offs\*\*: Introduce a pure `PipelinePlan` immutable record produced once at run start (or on explicit re-plan) that captures the fully resolved provider/model/quant/device + skip reasons for every stage/segment. Store it alongside the snapshot. Trade-off: small additional serialization cost vs major gain in auditability, resume correctness, and UI transparency. Low risk; aligns with existing immutable-snapshot philosophy. Verify in source whether this already exists in PipelineSnapshot.



\### 2. Model/Provider Abstraction and Manifest System

\*\*Finding\*\*: The manifest system (model\_id, provider\_id, task, engine\_family, expected\_runtime, input/output\_contract, commercial\_allowed, checksum, quality\_caveats, known\_failure\_modes, etc.) plus ModelDownloadOrchestrator is the single source of truth for every real model route. Commercial lane is enforced at manifest load/validation time, not runtime toggle. This is one of the strongest parts of the architecture.



\- \*\*Severity\*\*: Low (strength)  

\- \*\*Evidence\*\*: LONGTERM-ROADMAP model governance JSON example + “Every real model or provider route must be manifest-driven”, M21 (ModelManagerViewModel, IMdlDownloadOrchestrator, states: missing/downloading/corrupt/installed/blocked/ready), AGENT\_CONTEXT provider/model governance and model lanes (commercial/non-commercial/experimental), bundled-models.manifest.json location. M22: wav2vec2-lv60-espeak-cv-ft-onnx manifest entry includes vocab.json with SHA-256; composition fix resolved model id mismatch.  

\- \*\*Impact\*\*: Enables honest readiness, license auditing, and safe addition of future providers (SepFormer overlap-rescue, AudioShake premium, cloud). Prevents “repo license = commercial safe” mistakes.  

\- \*\*Recommended Improvement + Trade-offs\*\*: Add an explicit `capabilities` array or bitflags (e.g., SupportsPhonemeTimings, RequiresSourceTranscript, ProducesOverlapSources) to the manifest schema so the planner and stage can query without hard-coded knowledge of model\_id. Trade-off: schema version bump + migration for existing manifest entries (small, one-time). This would have prevented or made explicit the M22 routing gap that required a later fix. High value, low cost.



\### 3. Stage Communication and Artifact Passing

\*\*Finding\*\*: Stages communicate exclusively through declared inputs/outputs on immutable snapshots/artifacts routed by the Application layer. No direct stage-to-stage method calls. Each stage declares prerequisites; the planner/snapshot enforces them. Artifacts carry provenance (creator stage, source ids, provider/version, license state, checksum where applicable).



\- \*\*Severity\*\*: Low (strength)  

\- \*\*Evidence\*\*: AGENT\_CONTEXT “Artifact rules” (every generated artifact records what created it, source ids/paths, provider, stage, non-commercial/experimental flag, metadata to resume/explain; distinguish skipped/fallback from new), “Pipeline + stage rules” (declared inputs/outputs, per-segment/artifact status), LONGTERM-ROADMAP M22 (SourceSegmentTranscriptMap used so source alignment uses original TranscriptSegment text while TTS alignment uses translated text; missing source text skips cleanly to Partial).  

\- \*\*Impact\*\*: Excellent modularity and resumability. Failure or skip in lip sync preserves the TTS take. Overlap-rescue can run on suspected regions without contaminating the main stem path.  

\- \*\*Recommended Improvement + Trade-offs\*\*: Formalize an `IArtifact` base with strongly typed `ArtifactKind` (NormalizedAudio, Transcript, TranslatedTranscript, TtsTake, PhonemeTimingPlan, OverlapSources, etc.) and require every stage to declare `RequiredInputKinds` and `ProducedOutputKinds`. Add compile-time or test-time validation that a stage only consumes what prior stages can produce. Trade-off: slightly more boilerplate in new stage handlers vs elimination of subtle data-flow bugs when adding stages like overlap-rescue or lip synthesis. Worth doing before M23 work begins.



\### 4. Error Handling, State Management, and Honest Readiness

\*\*Finding\*\*: Honest readiness is implemented as a set of orthogonal states rather than a single `IsReady` boolean. Statuses distinguish Disabled / NotRun / SkippedLowConfidence / SkippedLicenseGate / SkippedRuntimeUnavailable / SkippedNoPhonemes / Succeeded / Failed / Partial. UI reflects these states; it does not own or mutate pipeline truth. Failures and skips preserve prior usable artifacts.



\- \*\*Severity\*\*: Low (exemplary alignment)  

\- \*\*Evidence\*\*: AGENT\_CONTEXT “Central rule: no fake readiness” (lists 15+ orthogonal states including provider registered, model downloaded+checksummed, license reviewed, commercial mode allowed, hardware available, stage enabled in snapshot, stage ran, produced usable output, skipped safely, failed), “UI rules” (UI reflect app state; distinguish disabled/skipped/failed/succeeded; show non-commercial/experimental warnings), LONGTERM-ROADMAP “do not fake readiness”, M22 skip reasons (SkippedLowConfidence, SkippedInventoryMismatch, SkippedUnsafeStretchRatio, SkippedLicenseGate). DegradationRecord / PipelineDegradationRecord mentioned in dev skills.  

\- \*\*Impact\*\*: Directly fulfills the roadmap’s highest principle. Users and tests see exactly why something did not happen. Lip sync can safely Partial-skip and still produce a usable export with original timing.  

\- \*\*Recommended Improvement + Trade-offs\*\*: Expose a machine-readable `ReadinessReport` (or per-stage `StageReadiness` record) that aggregates all prerequisite states for a given stage/segment at plan time. Persist it with the snapshot. This makes “why is lip sync disabled?” queryable without running the stage. Trade-off: modest additional state to maintain. High diagnostic value for support and for the hardware profiler / preset system. Low risk.



\### 5. Fake vs Real Model Execution Paths

\*\*Finding\*\*: Fake-backed architecture is mandated before any real provider. Composition wires fakes for tests; real providers are only used when manifest + cache + checksum + runtime + license gates pass. Stage handler tests are required to cover success, disabled, missing-prerequisite, skip (all reasons), failure, and cancellation paths. Integration tests exist for real models (e.g., LipSyncRealAlignerIntegrationTests, Wav2Vec2CtcForcedAlignerIntegrationTests).



\- \*\*Severity\*\*: Low (strength)  

\- \*\*Evidence\*\*: AGENTS.md “Testing” and “Key rules” (fakes first; new stage/provider → test enabled/disabled/missing manifest/non-commercial blocked/low confidence skip/runtime unavailable skip/artifact preservation skip/failure; fakes in tests/Trackdub.TestDoubles/), AGENT\_CONTEXT “Testing rules” and review checklist (“Fakes deterministic. Tests cover disabled, missing-prerequisite, skip, failure, success”), “Start any task” (add/update fakes before real providers), M22 (fake aligner and fake stretch service; real model verified after architecture landed). Pipeline-stage and test-double skills exist.  

\- \*\*Impact\*\*: High testability and safety. Real model bugs cannot break the pipeline contract. Composition tests and P0-5 DI regression coverage mentioned for model manager.  

\- \*\*Recommended Improvement + Trade-offs\*\*: Add a mandatory `Fake\*` implementation + handler test as part of the definition of done for any new stage or provider (already close to current practice). Consider a small “contract test” that verifies every real provider implementation satisfies the same skip/failure behaviors as its fake under equivalent conditions. Trade-off: extra test maintenance vs prevention of divergence between fake and real semantics. Strongly recommended before M23 experimental providers are introduced.



\### 6. Data Flow: VAD → Diarization → ASR → Translation → TTS → Lip Sync → Export

\*\*Finding\*\*: The flow is intentionally linear with well-defined skip points. M22 inserts phoneme-aligned lip sync after TTS timing reconciliation and before preview/export. Special mapping (SourceSegmentTranscriptMap) bridges original transcript text (for alignment) and translated text (for TTS). Overlap-rescue and speech-enhancement lanes are intentionally kept separate from the main stem path.



\- \*\*Severity\*\*: Medium (minor coupling hotspot)  

\- \*\*Evidence\*\*: LONGTERM-ROADMAP product spine and M22 position (“TTS generation → TTS timing reconciliation → phoneme-aligned audio lip sync → preview mix / export”), M22 transcript split handling, audio separation strategy table (overlap speech rescue lane distinct from stem separation; do not call SepFormer outputs stable speakers), AGENT\_CONTEXT M22 details (IForcedAligner, PhonemeTimingPlan, conservative stretch service, per-segment lip-sync status).  

\- \*\*Impact\*\*: The mapping layer adds a small but real coupling between ASR/translation output shape and lip-sync input expectations. If transcript segment structure changes, lip sync (and potentially export) can be affected. Overlap-rescue is correctly not yet wired into the main spine (manifest exists; dedicated stage lane does not).  

\- \*\*Recommended Improvement + Trade-offs\*\*: Introduce an explicit `ITranscriptSegmentView` or projection layer so lip sync (and future stages) consume a stable view rather than raw ASR/translation entities. This decouples segment evolution from downstream timing/alignment logic. Trade-off: one extra abstraction vs reduced ripple when ASR or translation providers change output shape. Worth the cost given the roadmap’s emphasis on adding more stages without creating “soup.”



\### Cross-Cutting Evaluations



\*\*Modularity and Testability\*\*: High. Interface-based providers (IForcedAligner, etc.), immutable records in Domain, snapshot isolation, fake-first mandate, and explicit path coverage in stage tests create a testable pipeline. New stage can be added with bounded work (pipeline-stage skill exists).



\*\*Coupling Between Stages\*\*: Low at the execution level (artifacts + contracts). Medium-to-high at the orchestration level (RuntimePlanner, Composition, StageRuntimeRequirementsCatalog, snapshot builders). This is acceptable and expected for a pipeline product; the risk is if Composition or the planner becomes a god object. No evidence of direct stage-to-stage coupling in the docs.



\*\*Support for Adding New Providers or Stages\*\*: Excellent for providers (new manifest entry + adapter implementing existing I\* interface + Composition registration + fake + tests). Good but higher cost for entirely new stages (must integrate with planner catalog, snapshot schema, per-segment status UI, artifact kinds, provenance rules, and usually a new DegradationRecord path). The architecture does not make new stages “free,” which is honest.



\*\*Performance and Memory Characteristics (visible in architecture)\*\*: InferenceSessionPool + SessionPoolKey + warm session factory are present for ONNX reuse (good). Hardware profiler (M19) + quality presets + benchmark evidence persistence exist. However, M20 is only partial (no full profiling-report.md, no export throughput targets, no memory ceiling test evidence in docs) and M20a (full pipeline benchmark suite against representative projects) is long-term. Session pooling helps warm starts; full end-to-end memory behavior under concurrent segments or large projects is not yet visible in the documented architecture.



\*\*Alignment with Honest Readiness and Provenance Principles\*\*: Very high. The central rule from AGENT\_CONTEXT is reflected in status design, artifact rules, manifest gates, UI responsibilities, and testing requirements. M22 implementation followed the rules (fake architecture first, real model after gates, explicit skip reasons that preserve prior TTS take, capability gating so lip sync never lands on word-level aligner). Provenance is explicitly required on every artifact.



\*\*Summary Table of Key Findings\*\*



| # | Area | Severity | Key Strength / Risk | Primary Evidence | Recommended Action |

|---|------|----------|---------------------|------------------|--------------------|

| 1 | Orchestration / Planner | Medium | Strength: manifest + hardware driven; Risk: growing complexity | RuntimePlanner, StageRuntimeRequirementsCatalog, M16a/M19/M22 | Add immutable PipelinePlan record |

| 2 | Manifest System | Low | Major strength: single source of truth, commercial lane enforcement | bundled manifest, ModelDownloadOrchestrator, M21/M22 | Add capabilities array to schema |

| 3 | Artifact / Data Flow | Low | Strength: loose coupling via artifacts | Artifact rules, SourceSegmentTranscriptMap (M22) | Add IArtifact + kind declarations + validation |

| 4 | Honest Readiness / State | Low | Exemplary | Central rule + 15+ states, skip enums, UI rules | Expose ReadinessReport persisted with snapshot |

| 5 | Fake vs Real | Low | Strength: mandated, comprehensive path coverage | Test rules, Composition, M22 fake + real tests | Mandatory contract test between fake and real |

| 6 | VAD→Export Flow Coupling | Medium | Minor hotspot at transcript/lip-sync boundary | M22 transcript split, overlap-rescue lane separation | Stable projection/view for transcript segments |

| 7 | Extensibility (new stage) | Medium | Good for providers; non-trivial for new stages | pipeline-stage skill, planner catalog integration | Document full “new stage” checklist in AGENTS.md |

| 8 | Perf/Memory Visibility | Medium | Partial evidence only | Session pool present; M20/M20a incomplete | Prioritize M20 completion after pipeline reliability P0 |



\*\*Overall Alignment with Roadmap Vision\*\*  

The pipeline design is one of the most faithful implementations of the LONGTERM-ROADMAP and AGENT\_CONTEXT principles visible in the documentation. Honest readiness is not a slogan—it shapes status enums, planner behavior, artifact rules, UI responsibilities, and testing mandates. Provenance and artifact preservation are explicit requirements. The M22 work demonstrates the architecture catching and fixing routing/composition/manifest issues before they became production problems. The main forward risk is ensuring the experimental M23-M26 visual dubbing work respects the same lane separation and fake-backed discipline that the audio pipeline currently follows.



\*\*Next Steps / Questions for Source-Level Verification\*\*



Because this audit is documentation-based, the following are required before declaring any finding closed:



1\. Share (or allow read access to) the following files for verification of claims:

&#x20;  - RuntimePlanner.cs / StageRuntimeRequirementsCatalog.cs (or equivalent)

&#x20;  - RoutedForcedAligner.cs + IForcedAlignerAdapter implementation (especially SupportsPhonemeTimings, RequirePhonemeTimings, PreferredModelAlias handling, SourceSegmentTranscriptMap usage, and skip-to-Partial logic)

&#x20;  - LipSyncStageHandler.cs + LipSyncStageHandlerTests.cs (success, skip reasons, artifact preservation)

&#x20;  - Composition root (how fakes vs real providers and stage handlers are registered)

&#x20;  - Any PipelineSnapshot or IArtifact definition

&#x20;  - bundled-models.manifest.json entry for the wav2vec2 phoneme aligner



2\. Confirm whether an immutable `PipelinePlan` or equivalent already exists, or whether the planner mutates state after the initial snapshot.



3\. Status of overlap-rescue stage wiring (manifest exists per roadmap; dedicated stage lane wiring status?).



This audit provides the deep layer view of the pipeline. Once the above files are available, we can move to a targeted code-level review of one or two critical seams (e.g., M22 lip-sync path or the planner) with concrete diffs or findings. No refactoring is proposed yet—this remains diagnostic.



Ready for the verification slice or the next bounded module.

