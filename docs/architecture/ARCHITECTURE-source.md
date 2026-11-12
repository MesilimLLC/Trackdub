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
- Demucs/HTDemucs is non-commercial only (lane: non-commercial, commercial_allowed: false) and must never be routed in any commercial pipeline path. This is enforced from the manifest fields (see section 10), not via a runtime toggle.
- Non-commercial mode may prefer Demucs while output quality is being evaluated.
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
- Non-commercial translation models (commercial_allowed: false) must never be routed in the bundled pipeline. This is enforced from the manifest fields (see section 10).
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

- Unknown-license models are not commercial-safe.
- Non-commercial models are not commercial-safe.
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
