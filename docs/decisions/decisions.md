# ADR-0001: WinUI 3 + Windows ML

- Status: Draft
- Date: 2026-04-19

## Context

Trackdub is intentionally Windows-only, local-first, and hardware-aware. The app needs a desktop shell that can handle media preview, Windows packaging, composition-heavy UI, and long-running local AI workflows without pretending to be cross-platform.

The repository also has explicit boundary rules:

- inference code must not live in the WinUI project
- runtime/provider changes must remain visible to the user
- model execution has to be proven in a harness before the full editor depends on it

Current platform docs also support this direction:

- Windows App SDK exposes the Windows desktop app runtime and startup/deployment APIs used by WinUI 3 applications.
- ONNX Runtime's Windows guidance recommends the WinML path for Windows development, supports C# packages, and documents Windows execution-provider setup such as DirectML.

## Decision

Trackdub will use WinUI 3 on the Windows App SDK for the desktop shell and use a Windows-native ONNX inference stack for local model execution.

More specifically:

- `src/Trackdub.App` owns the WinUI 3 shell, navigation, resources, composition helpers, and user-facing state.
- `src/Trackdub.Inference` owns runtime planning, model manifests, provider selection, and inference interfaces.
- `src/Trackdub.Inference.Onnx` owns concrete ONNX model wrappers and Windows-specific runtime integration.
- The UI must never host model-wrapper code, session construction, or execution-provider policy.
- ONNX is the primary model interchange format for the first implementation slices.
- Provider selection is a runtime plan validated per model/provider pair, not a static "GPU on" toggle.
- **Superseded in part by [ADR-0002](ADR-0002-windows-ml-provider-strategy.md):** Windows ML is the primary Windows ONNX surface; catalog EPs are preferred where compatible; DirectML is legacy GPU fallback; CPU is terminal fallback. The bullets below remain historically accurate for the first WinUI slice but must not be read as current provider strategy.
- DirectML and CPU fallback were the initial baseline paths on Windows.
- TensorRT-RTX remains optional until it passes the benchmark harness on real target hardware (now integrated via the standalone ORT EP ABI plugin where available).
- If WinML-specific capabilities materially simplify Windows scenarios, they are consumed behind the `Trackdub.Inference.Onnx` boundary rather than exposed to the UI layer.

## Consequences

Positive:

- The UI stack matches the product's Windows-only scope.
- The app can use Windows-native packaging, lifecycle, composition, and media primitives without cross-platform abstraction tax.
- Inference remains testable and replaceable because the shell only depends on application and inference abstractions.
- The benchmark harness can validate provider behavior before the product claims GPU readiness.

Negative:

- This deliberately gives up cross-platform UI portability.
- Windows App SDK packaging/runtime behavior becomes part of the deployment surface that must be tested.
- Some models will still fail on specific providers, so benchmark evidence is required before enabling fast paths by default.

## Alternatives considered

### WPF

Rejected because it is mature but less aligned with the intended modern Windows app stack, composition direction, and long-term Windows App SDK packaging story.

### Avalonia or .NET MAUI

Rejected for the first product slice because Trackdub is not trying to be cross-platform yet, and the extra abstraction cost does not reduce the main project risks.

### Put inference directly in the WinUI project

Rejected because it would violate the repo boundary rules, make runtime policy harder to test, and encourage UI-driven provider shortcuts.

## References

- [Windows App SDK API reference](https://learn.microsoft.com/en-us/windows/windows-app-sdk/api/winrt/)
- [ONNX Runtime install and Windows guidance](https://onnxruntime.ai/docs/install/)
- [ONNX Runtime C# getting started](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [ONNX Runtime on Windows](https://onnxruntime.ai/docs/get-started/with-windows.html)

# ADR-0002: SQLite project persistence

- Status: Draft
- Date: 2026-04-19

## Context

Trackdub is built around durable, inspectable pipeline artifacts. Users need to reopen a project and understand:

- what media the project points at
- which stages ran
- which model/provider/settings produced each artifact
- which edits superseded earlier machine output

That state is structured, relational, and local to a single project. At the same time, the workload also produces large binaries such as source media, extracted audio, stems, TTS takes, preview renders, and exports.

SQLite's current documentation still fits this shape well:

- it is a serverless, file-based, ACID database
- WAL mode is available when concurrent readers and writers matter
- SQLite explicitly documents the tradeoff between storing large blobs internally versus keeping them as external files

## Decision

Each `.trackdub` project will own a project-local SQLite database named `trackdub.db` for structured state, while the filesystem will own large media and generated artifacts.

More specifically:

- `trackdub.db` stores project metadata, stage runs, speakers, transcript revisions, translation revisions, voice assignments, TTS take metadata, mix plans, exports, consent records, and artifact metadata.
- Source media, extracted audio, stems, preview renders, final exports, and other large binaries live under the project folder's `media/` and `artifacts/` directories.
- Artifact rows store project-relative paths or storage keys plus hashes and media metadata, not absolute paths and not raw binary payloads.
- Machine-local state that is not part of one project, such as model cache inventory, benchmark history, settings, and logs, lives outside the project under a machine-local app data root.
- `src/Trackdub.Infrastructure` owns SQLite connection management, migrations, transactions, repository implementations, and path resolution.
- `src/Trackdub.Domain` may reason about artifact identity and provenance, but not about absolute filesystem locations.
- A thin mapper such as Dapper is acceptable inside Infrastructure, but the architectural decision is SQLite plus explicit SQL ownership, not a heavyweight ORM.

## Consequences

Positive:

- A project remains portable because its structured state travels with the project folder.
- Backups, copies, and bug-report bundles stay straightforward.
- SQLite fits the single-user local desktop model without introducing a service dependency.
- Large media files remain streamable and inspectable on disk without bloating the database.

Negative:

- Migrations and integrity checks become part of the app lifecycle.
- The app needs a reliable path-resolution layer for project-relative artifact locations.
- Careless direct file deletion can still orphan artifact records unless cleanup rules are enforced.

## Alternatives considered

### Store everything as JSON files

Rejected because stage history, artifact provenance, revisions, and downstream invalidation are relational enough that ad hoc JSON files would create harder consistency problems.

### Store large media blobs inside SQLite

Rejected for the default design because the project already has a natural artifact directory structure and the app benefits from keeping heavy binaries as normal files.

### Use one global database for all projects

Rejected because it weakens project portability and makes export, backup, and support workflows more fragile.

### Use a client/server database

Rejected because Trackdub is local-first and should not require a separate database service for its primary workflow.

## References

- [SQLite documentation index](https://www.sqlite.org/docs.html)
- [SQLite PRAGMA reference](https://www.sqlite.org/pragma.html)
- [SQLite WAL overview](https://www.sqlite.org/wal.html)

# ADR-0002: Windows ML-first execution provider strategy

- Status: Accepted
- Date: 2026-05-23
- Supersedes in part: [ADR-0001](ADR-0001-winui3-windows-ml.md) (DirectML-as-baseline wording only)

## Context

Microsoft positions **Windows ML** as the forward path for Windows ONNX Runtime deployments. The ONNX Runtime shipped with Windows ML supports explicit execution provider (EP) selection and optional device policies (`MAX_PERFORMANCE`, `PREFER_NPU`, `MAX_EFFICIENCY`, and others). Microsoft recommends starting with **explicit EP selection** for predictability, then experimenting with device policies.

**DirectML** remains supported under sustained engineering, but new feature work has moved to Windows ML. Windows ML still includes CPU and DirectML legacy providers, and adds dynamic provider acquisition and registration for certified vendor EPs (for example AMD MIGraphX, Intel OpenVINO, Qualcomm QNN, and AMD VitisAI). NVIDIA TensorRT RTX is handled separately as the standalone ONNX Runtime EP ABI plugin.

Trackdub already implements much of this layering in `Trackdub.Inference.Onnx` (Windows ML bootstrap, catalog EP registration, TRT RTX plugin registration, smoke tests, fallback chains). Product language and some planning documents still describe DirectML as the centerpiece Windows GPU strategy. That mismatch confuses agents, contributors, and users.

## Decision

On Windows, Trackdub treats **Windows ML as the ONNX integration surface**, not DirectML alone.

1. **Windows ML layer** — Bootstrap, certified catalog EP registration, and `OrtEnv.GetEpDevices()` device enumeration live in `Trackdub.Inference.Onnx` (for example `WindowsMlExecutionProviderBootstrapper`, `WindowsMlProviderRegistrationPolicy`, `WindowsExecutionProviderBootstrapper`). UI and Application layers must not assemble EP policy.

2. **TensorRT RTX plugin route** — `ExecutionProviderKind.TensorRTRtx` is the standalone ORT EP ABI plugin route on Windows. Trackdub locates a bundle containing `onnxruntime_providers_nv_tensorrt_rtx.dll`, `tensorrt_rtx_1_7.dll`, and `tensorrt_onnxparser_rtx_1_7.dll`; registers the plugin with `OrtEnv.RegisterExecutionProviderLibrary`; enumerates `OrtEnv.GetEpDevices()`; and appends the `NvTensorRTRTXExecutionProvider` GPU device explicitly. It does not call Windows ML `ExecutionProvider.TryRegister`, `EnsureAndRegisterCertifiedAsync`, or `SetEpSelectionPolicy` for TRT RTX.

3. **Preferred acceleration** — Where a model and stage allow it, prefer **TensorRT RTX plugin** or certified catalog execution providers (for example `Migraphx`) before generic GPU fallback. Global milestone probe order in `Milestone5PlanningPolicy.SupportedProvidersThisMilestone` reflects this: TensorRT RTX → MIGraphX → native TensorRT/CUDA (Linux or advanced Windows) → DirectML → CPU.

4. **DirectML as legacy GPU fallback** — `ExecutionProviderKind.DirectMl` uses the Windows ML **packaged DirectML route** (`WindowsMlProviderRegistrationRoute.PackagedDirectMl`). It is a supported compatibility fallback when catalog/plugin/vendor EPs are unavailable or fail smoke tests—not the primary forward acceleration strategy.

5. **Explicit selection and gates** — Runtime routing stays **manifest-driven, stage-scoped, and smoke-test verified** per model/provider pair. Provider registration does not imply model readiness. Per-model allowed-provider lists (for example Kokoro CPU-only) remain authoritative.

6. **User-visible honesty** — Discovery messages, Model Manager hints, and hardware override labels should describe TRT RTX as a plugin route, Windows ML catalog EPs as catalog routes, and DirectML as legacy GPU fallback where shown explicitly.

### Mapping to code

| Concept | Location |
|--------|----------|
| Provider enum | `ExecutionProviderKind` in `Trackdub.Domain/Common/RuntimePlanning.cs` |
| Milestone probe order | `Milestone5PlanningPolicy` in `StageRuntimeRequirements.cs` |
| Packaged DirectML vs catalog EP | `WindowsMlProviderRegistrationRoute` in `WindowsMlProviderRegistrationPolicy.cs` |
| TRT RTX plugin location/registration | `TensorRtRtxPluginLocator.cs`, `TensorRtRtxPluginService.cs` |
| Bootstrap and fallback | `WindowsExecutionProviderBootstrapper.DetermineFallbackProviderAsync` |
| Discovery | `OnnxExecutionProviderDiscovery` |
| Session creation | `OnnxExecutionSessionFactory` |

## Consequences

Positive:

- Aligns documentation and UI copy with Microsoft's Windows ML direction and with existing implementation seams.
- Preserves predictability: explicit EP selection and smoke tests before claiming GPU readiness.
- Keeps vendor EP investment (TensorRT RTX plugin, MIGraphX catalog) as first-class without pretending DirectML is the modern default.

Negative:

- Contributors must learn the layered model (TRT RTX plugin / Windows ML catalog EP → DirectML → CPU), not a single "GPU on" switch.
- Bundled manifest `expected_runtime` tokens use `trt-rtx|windows-ml|onnxruntime-migraphx|onnxruntime-directml` for standard ONNX models where appropriate.

## Non-goals (this ADR)

- Changing planner probe order (global order was already correct; Phase 2 aligned stage allow-lists).
- Device policies are optional via advanced studio setting (Phase 3, 2026-05-23); default remains explicit EP selection.
- Adding OpenVINO or QNN catalog routes before product prioritization.

## Device policies (Phase 3)

Optional advanced studio setting `WindowsMlExecutionDevicePolicy` (default `Explicit`). Non-default values call `SessionOptions.SetEpSelectionPolicy` for Windows ML catalog/device-policy routes (`DirectMl`, `Migraphx`) and skip explicit catalog-device append. TensorRT RTX plugin and native CUDA/TensorRT on Windows ignore device policy. Details: [windows-ml-phase-3-device-policies.md](../reference/windows-ml-phase-3-device-policies.md).

## Future work

Deferred items; not required for accepting this strategy ADR:

1. **Additional catalog EPs** — OpenVINO, QNN, and other Windows ML-registered vendor EPs when manifest and smoke coverage exist. Phase 5 documents gates and adds compile-time stubs only (see below).

## Phase 5 — Catalog EP expansion (OpenVINO, QNN)

**Status (5c):** Documentation, operator checklist, and **thin code stubs** only. No milestone probe-order change and no fake readiness.

### OpenVINO dual path

- **Standalone ExecutionProviderKind.OpenVino** — existing Linux/optional Windows install path via IOpenVinoAvailabilityProvider and Infrastructure component downloader. Registration ≠ readiness; smoke still required per model.
- **Future ExecutionProviderKind.OpenVinoCatalog** — WinML catalog Intel EP on Windows, distinct enum value so planner, discovery, and session code do not conflate with standalone OpenVINO. Same honesty rules as MIGraphX: catalog registration and smoke before allow-list or probe-order changes.

### QNN / NPU path

- **Future ExecutionProviderKind.Qnn** — Qualcomm catalog EP registered through Windows ML when certified packages are present.
- **WindowsMlExecutionDevicePolicy.PreferNpu** is a *session selection hint* only. It must not be interpreted as “QNN ready” or “NPU model ready.” Discovery reports QNN as unavailable until Phase 5 gates pass.

### Gating checklist (before probe-order or allow-list changes)

1. Catalog provider id confirmed against ExecutionProviderCatalog / Microsoft docs for the target Windows ML SDK.
2. WindowsMlProviderRegistrationPolicy + OnnxExecutionSessionFactory append path implemented and covered by tests.
3. Per-stage smoke on representative hardware (matrix rows in [windows-ml-stage-provider-matrix.md](../reference/windows-ml-stage-provider-matrix.md)).
4. Bundled manifest expected_runtime tokens updated where product commits to a catalog route.
5. Model Manager / settings copy states registration vs download vs smoke vs stage success separately.
6. Planner smoke failure → fall through to next provider (no stuck “ready” state).

### Explicit non-goals for 5c

- No change to Milestone5PlanningPolicy.SupportedProvidersThisMilestone probe order until matrix smoke passes for the new EP on target hardware.
- Stubs must report **unavailable** in discovery and **not enabled** in registration/session paths — never “GPU ready” or “catalog registered therefore ready.”

Operational detail: [windows-ml-phase-5-catalog-eps.md](../reference/windows-ml-phase-5-catalog-eps.md).

## References

- [DirectML overview](https://learn.microsoft.com/windows/ai/directml/dml) (sustained engineering; legacy GPU path)
- [Windows ML overview](https://learn.microsoft.com/windows/ai/new-windows-ml/overview)
- [Windows ML supported execution providers](https://learn.microsoft.com/windows/ai/new-windows-ml/supported-execution-providers)
- [Windows ML provider selection](https://learn.microsoft.com/windows/ai/new-windows-ml/select-execution-providers)
- [Installing and registering Windows ML EPs](https://learn.microsoft.com/windows/ai/new-windows-ml/initialize-execution-providers)
- [ONNX Runtime TensorRT RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)
- [ONNX Runtime plugin EP library usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)
- Seams: [`docs/reference/migraphx-phase0-seams.md`](../reference/migraphx-phase0-seams.md)

## Phase 4 closeout (2026-05-23)

ORT native load order prefers the managed ONNX Runtime package before app-base DLLs; session pool eviction and policy-cache invalidation run when hardware settings change. Operational checklist: [windows-ml-phase-4-closeout.md](../reference/windows-ml-phase-4-closeout.md).

## Phase 5 licensing policy (2026-05-23)

Vendor EPs and plugin bundles carry third-party licenses that require explicit user acceptance before installation/registration. The following policy is now enforced:

- **License families:** Four families gate the certified EP/plugin families — `AmdRyzenAi` (MIGraphX + VitisAI), `NvidiaTensorRtRtx`, `IntelOpenVino`, `QualcommQnn`.
- **One-time per-machine acceptance** is persisted in `StudioSettings` (flags `AmdRyzenAiLicenseAccepted`, `NvidiaTensorRtRtxLicenseAccepted`, `IntelOpenVinoLicenseAccepted`, `QualcommQnnLicenseAccepted`).
- **`ILicenseConsentService`** (`Trackdub.Application.Runtime`) is the app-layer contract. The Avalonia shell provides `AvaloniaLicenseConsentService`, which shows `EpVendorLicenseDialog` on the UI thread and persists acceptance on confirmation.
- **Per-EP install commands** in `ModelManagerViewModel` call `EnsureAcceptedAsync` before beginning installation. If the user declines, the install aborts and a status message is shown.
- **Bulk install** (`InstallAllCertifiedCatalogAsync`) iterates all four families and requires each to be accepted before proceeding.
- **"View license" commands** (`ViewMigraphxLicenseCommand`, `ViewTensorRtRtxLicenseCommand`, `ViewOpenVinoLicenseCommand`, `ViewQnnLicenseCommand`, `ViewVitisAiLicenseCommand`) open the dialog in informational mode without modifying stored flags.
- License metadata and external links are provided by `LicenseMetadataProvider` (in `EpVendorLicenseDialog.axaml.cs`). Third-party notices are documented in `THIRD_PARTY_NOTICES.md`.

# ADR-0003: Whisper (onnx-community) license classification

- Status: Accepted
- Date: 2026-04-29

## Context

`src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json`
has two entries for `onnx-community/whisper-tiny` (local bundle and the
`whisper-tiny-onnx` variant).

The previous draft of this ADR kept those entries blocked because the
`onnx-community/whisper-tiny` Hugging Face repo does not publish its own
`license:*` metadata tag. That was a conservative interpretation of AGENTS.md
rule 8: "Do not invent commercial safety - unknown license = unsafe."

The project owner has now made the repo-local policy decision that this
specific artifact should be classified as commercial-safe because it is a
direct ONNX-format conversion of `openai/whisper-tiny`, not a new independently
licensed model family.

## Decision

Classify the bundled `onnx-community/whisper-tiny` artifacts as Apache-2.0 and
commercial-safe.

The accepted manifest values are:

```json
"license": "Apache-2.0",
"commercial_allowed": true,
"redistribution_allowed": true,
"commercial_safe_mode": true
```

This decision applies only to the `onnx-community/whisper-tiny` entries that
identify `openai/whisper-tiny` as their base model and do not add conflicting
license or usage terms. It does not weaken the default rule for unrelated model
artifacts: unknown license still remains unsafe unless a project-specific ADR
or manifest evidence explicitly resolves the artifact.

## Evidence

Current public Hugging Face metadata supports this classification:

- `openai/whisper-tiny` declares `license:apache-2.0` in its Hugging Face tags
  and `cardData.license == "apache-2.0"`.
- `onnx-community/whisper-tiny` identifies `openai/whisper-tiny` as its base
  model through both Hugging Face metadata (`base_model:openai/whisper-tiny`)
  and the model page's model tree.
- The `onnx-community/whisper-tiny` model card describes the repository as ONNX
  weights for compatibility with Transformers.js and links back to
  `openai/whisper-tiny`.
- No non-commercial, custom, or otherwise restrictive license or usage term is
  declared on the `onnx-community/whisper-tiny` repo.

## Policy interpretation

For Trackdub, a format-shifted ONNX artifact can inherit the commercial
classification of its Apache-2.0 base model when all of the following are true:

1. The artifact metadata identifies the Apache-2.0 base model.
2. The artifact is a conversion or quantized conversion of that base model, not
   a separately trained model with unknown provenance.
3. The artifact repository does not declare conflicting or more restrictive
   terms.
4. The manifest entry is covered by an ADR or equivalent repo-local evidence.

If any of those conditions changes, the manifest must be re-reviewed before the
entry is used in commercial-safe mode.

## Consequences

Positive:

- Commercial-safe ASR planning can select the bundled ONNX Whisper-tiny runtime
  path.
- The manifest and policy documentation now agree, so future review bots should
  not treat the commercial-safe flip as an unexplained contradiction.
- The stricter default remains intact for unrelated models.

Negative:

- This decision relies on base-model inheritance for a conversion repo whose HF
  metadata still omits a direct license tag. If the repo later adds explicit
  conflicting terms, Trackdub must update the manifest immediately.

## References

- AGENTS.md rule 8: "Do not invent commercial safety - unknown license =
  unsafe."
- `MODEL_LICENSE_POLICY.md`
- [Hugging Face API: `openai/whisper-tiny`](https://huggingface.co/api/models/openai/whisper-tiny)
- [Hugging Face API: `onnx-community/whisper-tiny`](https://huggingface.co/api/models/onnx-community/whisper-tiny)
- [Hugging Face model page: `onnx-community/whisper-tiny`](https://huggingface.co/onnx-community/whisper-tiny)

# ADR-0004: Decompose TranscriptProjectService into bounded workflow services

- Status: Accepted
- Date: 2026-04-24

## Context

`src/Trackdub.Application/Transcripts/TranscriptProjectService.cs` was the
single workspace service the WinUI shell talked to for project lifecycle,
transcript workflows, translation workflows, language persistence, and
artifact provenance. The M6 and M7 completion notes described it that way on
purpose: one service kept the vertical slice small.

Milestone 7 is now complete and the next milestones will each add more
cross-cutting responsibilities to the same service:

- **M8 (playback and settings):** hybrid playback seam, segment editor,
  project management.
- **M10 (diarization):** speaker segmentation must update transcript
  revisions and downstream translation invalidation.
- **M11 (TTS):** per-segment voice assignments, take lifecycle, and more
  revision state to track.
- **M14 (mix):** mix plans and ducking state bound to segments and takes.
- **M15 (export):** export provenance and commercial-safety gating.

Each of these milestones would naturally extend `TranscriptProjectService`
unless we draw boundaries now. Continuing to grow a single workspace service
makes it harder to test with focused fakes, harder to reason about concurrency
and invalidation, and harder to hold to the repo's "bounded agent tasks" rule
from AGENTS.md.

## Decision

Decompose `TranscriptProjectService` into bounded workflow services. The target
shape is:

- `ProjectSessionService` — create / open / close a project, manifest
  persistence, transcript-language setting, top-level session state. Owns the
  `.trackdub` project lifecycle and nothing stage-specific.
- `TranscriptWorkflowService` — VAD and ASR orchestration, transcript
  revision management, manual segment editing, transcript-level artifact
  provenance.
- `TranslationWorkflowService` — translation generation from the current
  transcript revision, manual translation-save flow, translation revision
  management, `needs refresh` derivation, translation-level artifact
  provenance.
- `ProjectStateCoordinator` — mediates cross-cutting state, specifically
  "transcript revision changed → translation is now refresh-needed", and any
  analogous invalidation edges M8–M15 add (segment edits, diarization changes,
  TTS takes, mix plans).

Each service is testable against its own fakes. During transition,
`TranscriptProjectService` may remain as a thin façade that composes these
services so the shell does not break in a single commit.

## Consequences

Positive:

- Boundaries match the milestone roadmap: each upcoming stage (diarization,
  TTS, mix, export) has an obvious home that is not "the single workspace
  service."
- Each service is independently testable with its own fakes, which aligns
  with the "Application tests use fakes" guidance in AGENTS.md.
- Invalidation rules live in one mediator instead of being scattered across
  ad hoc checks inside a large service.
- Keeps individual agent tasks bounded (one workflow, one service) as AGENTS.md
  recommends.

Negative:

- More files to navigate and more DI registrations.
- The transition must be staged so the shell (`Trackdub.App.Avalonia`) keeps working
  through the split; the facade step adds temporary indirection.
- Cross-cutting coordination logic that used to be implicit inside one service
  now needs an explicit contract on `ProjectStateCoordinator`.

## Alternatives considered

### Keep `TranscriptProjectService` as the single workspace service

Rejected because every upcoming milestone (M8, M10, M11, M14, M15) adds more
state and more invalidation edges. Not decomposing now pushes all of that into
one class and makes it the main bottleneck for focused fakes, concurrency
reasoning, and PR review scope.

### Split later, when the service actually hurts

Rejected because the M6–M7 shape already mixes lifecycle, transcript, and
translation concerns. Deferring the split until after M10 / M11 means the
refactor has to be done alongside speaker, take, and mix work, which is
exactly when bounded services are most valuable.

### Extract only a state coordinator, leave everything else in one service

Rejected because the coordinator alone doesn't address the growth of workflow
code inside the service. The workflow services are the ones that need their
own fakes and their own tests; pulling out only the mediator leaves the main
service still owning everything.

## References

- `src/Trackdub.Application/Transcripts/`
- `docs/archive/milestones/completions/MILESTONE-6-COMPLETION.md`
- `docs/archive/milestones/completions/MILESTONE-7-COMPLETION.md`
- `docs/archive/roadmaps/MILESTONE-legacy-2026-04.md`
- `MILESTONE.md`
- AGENTS.md — "Prefer bounded agent tasks" guidance


# ADR-0005: Kokoro-82M ONNX TTS Architecture for M11

- Status: Accepted
- Date: 2026-04-24

## Context

Milestone 11 introduces stock-voice text-to-speech synthesis. The goal is to
produce per-segment dubbed audio from the translated transcript, using a locally
runnable model with no external API dependency.

The candidate model is **Kokoro-82M** — a lightweight, high-quality English TTS
model available as ONNX via `onnx-community/Kokoro-82M-v1.0-ONNX` (Apache-2.0).

Key constraints:
- The model must run offline on a developer laptop with no GPU required.
- The inference stack must integrate with the existing `IRuntimePlanner` /
  `BenchmarkModelPathResolver` pipeline already used by Whisper, Madlad, and
  SortFormer.
- G2P (grapheme-to-phoneme) must produce IPA phonemes for Kokoro's character-
  level tokenizer.
- Phase A and B deliverables must ship without the model on disk (test doubles
  replace the engine; bundled-model tests skip when model is absent).

---

## Decision 1 — Model source: `onnx-community/Kokoro-82M-v1.0-ONNX`

`hexgrad/Kokoro-82M` ships weights as PyTorch `.pt` files only. The ONNX
community mirror (`onnx-community/Kokoro-82M-v1.0-ONNX`) provides a
pre-converted `model.onnx` / `model_quantized.onnx` with three inputs
(`input_ids`, `style`, `speed`) and one `audio` output. This is the correct
source for ONNX Runtime inference.

The upstream model ID is preserved in the manifest (`model_id`) and cited in
logs. The local manifest alias used by code and tests is `kokoro-onnx` (with
`kokoro` as a shorter fallback); the on-disk directory is `models/kokoro-onnx/`.
Versioned aliases (e.g. `kokoro-v2`) may be added alongside the existing alias
when a new Kokoro revision is adopted, rather than rewriting existing alias
references.

## Decision 2 — DirectML excluded (CPU only for M11)

During the M11 spike the DirectML execution provider returned `0x80070057` for
the `ConvTranspose` operator in Kokoro's mel decoder. This is a known upstream
ONNX Runtime DirectML issue unrelated to Trackdub. CPU execution completes
successfully.

`StageRuntimeRequirements` for `RuntimeStage.Tts` therefore lists only
`ExecutionProviderKind.Cpu` in `AllowedProvidersThisMilestone`. This restriction
will be revisited when the upstream fix ships.

## Decision 3 — G2P approach: espeak-ng subprocess

Kokoro requires IPA phonemes as input. Two viable approaches were evaluated:

| | **subprocess `espeak-ng.exe`** | **KokoroSharp 0.6.7 (P/Invoke DLL)** |
|---|---|---|
| Latency | ~10–50 ms per call | < 5 ms |
| Self-contained | No — requires `espeak-ng` on PATH or bundled separately | Yes — `libespeak-ng.dll` ships inside NuGet |
| License | GPL-3.0-or-later (process boundary = mere aggregation) | **GPL-3.0-or-later propagates in-process** |
| Commercial safe | Yes (mere aggregation) | **No — combined work under FSF GPL rules** |

**KokoroSharp 0.6.7 bundles `libespeak-ng.dll` (GPL-3.0-or-later) and loads
it in-process via P/Invoke.** The FSF treats dynamic linking as a combined work.
Shipping Trackdub with that DLL present would force the entire application
binary to re-license under GPL-3.0-or-later.

**Decision: spawn `espeak-ng.exe` as a child process.** The process boundary
constitutes "mere aggregation" under the GPL and does not propagate copyleft
to Trackdub. The `EspeakNgPhonemizer` class encapsulates this and accepts a
configurable executable path to support bundled or PATH-resolved installations.

## Decision 4 — Voicepack style vector slicing

Kokoro ships 56 voicepack `.bin` files (e.g. `af_heart.bin`) under a `voices/`
subdirectory. Each file stores a matrix of shape `(N, 256)` as raw little-endian
float32. At inference, the style tensor `[1, 256]` is the row at index
`phonemeTokenCount` — the raw phoneme token count **before** BOS/EOS padding,
matching upstream `ref_s = voices[len(tokens)]` in `hexgrad/Kokoro-82M/kokoro.py`.
Because `KokoroTokenizer.Encode` wraps the sequence as `[BOS, …phonemes…, EOS]`,
the engine slices with `inputIds.Length - 2`. `KokoroVoicepackLoader` implements
the row read given this pre-padding token count.

## Decision 5 — Tokenizer: character-level from `tokenizer.json`

The `tokenizer.json` bundled with the model contains a `model.vocab` map of
Unicode characters (IPA symbols, ASCII, punctuation) to integer token IDs. A
`$` token (ID 0) wraps both ends of the sequence per the post-processor. The
`KokoroTokenizer` class loads this map at synthesis time and truncates at 512
tokens.

## Decision 6 — Phase C pins a lazy ONNX session

`KokoroTtsEngine` now lazy-loads and pins one `InferenceSession` per resolved
model path and execution provider. Consecutive `SynthesizeAsync` calls reuse
that session instead of cold-loading the ONNX graph per segment.

The engine serializes access to the pinned session with a `SemaphoreSlim`. This
is intentionally conservative for M11: it avoids concurrent mutation risk in
provider-specific session state while still removing the dominant per-segment
load cost. If a future milestone proves concurrent `Run` calls safe for the
selected provider set, this can be relaxed behind a benchmark.

---

## Consequences

- `EspeakNgPhonemizer` resolves `espeak-ng.exe` from a bundled installer path
  first, then falls back to `PATH`. Developer builds can still use
  `winget install eSpeak-NG.eSpeak-NG`.
- Supported bundled binary locations are:
  - `tools/espeak-ng/espeak-ng.exe`
  - `runtimes/win-x64/native/espeak-ng/espeak-ng.exe`
  - `runtimes/win-x64/native/espeak-ng.exe`
  - `espeak-ng/espeak-ng.exe`
- DirectML will not accelerate Kokoro until the upstream ONNX ConvTranspose fix
  lands. Latency on CPU for a 5-second segment is ~200–400 ms on a modern
  laptop.
- **KokoroSharp must not be added as a dependency without first resolving the
  GPL contamination** (strip `libespeak-ng.dll` from the published artifact or
  switch to GPL-only licensing).
- The `KokoroVoiceCatalog` naming parser supports `a`/`b`/`e`/`f`/`h`/`i`/`j`/
  `k`/`p`/`r`/`z` locale prefixes; unknown prefixes map to `"unknown"` and
  remain discoverable.

# ADR-0006: Chatterbox Commercial Use Verification

- Status: Superseded
- Date: 2026-04-29
- Superseded: 2026-05-09 by the manifest hash-integrity rule in `MODEL_LICENSE_POLICY.md`

## Context

Milestone 15 uses Chatterbox ONNX models for consent-gated voice cloning. Commercial-safe mode only allows model routes whose manifests declare `commercial_use_verified: true`.

Current implementation note: `commercial_use_verified: true` now requires both
commercial-use license confidence and a non-empty SHA-256 for artifact integrity.
The Chatterbox entries may still be likely commercial-safe by license review, but
they must not be routed in commercial-safe mode until their manifest entries have
verified hashes.

The Chatterbox ONNX repositories used by Trackdub currently declare MIT licenses on their Hugging Face model pages:

- `ResembleAI/chatterbox-turbo-ONNX`: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
- `onnx-community/chatterbox-ONNX`: https://huggingface.co/onnx-community/chatterbox-ONNX

The model cards also document the expected ONNX graph package layout used by the Trackdub downloader and Chatterbox wrapper.

## Decision

Trackdub treats the two Chatterbox ONNX entries as license candidates that
must remain blocked from commercial-safe routing until artifact hashes are
verified:

- `ResembleAI/chatterbox-turbo-ONNX`
- `onnx-community/chatterbox-ONNX`

Both entries remain voice-cloning models. They must continue to set:

- `voice_cloning: true`
- `requires_user_consent: true`

Commercial-safe mode must not route to these entries while `commercial_use_verified`
is false. If they are later restored to `commercial_use_verified: true`, the
per-session voice-cloning consent gate remains mandatory and non-bypassable.

## Consequences

Positive:

- The ADR records the upstream MIT license evidence that made Chatterbox a plausible commercial-safe candidate.
- The stricter manifest gate avoids presenting license confidence as full commercial readiness before artifact integrity is verified.

Negative:

- The manifest now depends on both upstream license declarations and local artifact hashes. If either source changes license or adds restrictions, the manifest must be revised immediately.
- Commercial-use verification does not verify consent from a voice subject and does not reduce the user's legal responsibility for voice cloning.

# ADR-0007: Managed glossary analyzers

- Status: Draft
- Date: 2026-05-07

## Context

Milestone 17 adds project glossary matching for languages where simple text scanning is not enough. The advanced analyzer pilot needs Japanese, Chinese, and Arabic tokenization without adding native runtimes, sidecars, Python, Java, GPL/LGPL dictionaries, or model manifest changes.

Lucene.NET provides managed analyzers for these languages, but the available analyzer packages are still `4.8.0-beta00017`. Their lockfile closure includes ICU4N alpha packages and older `Microsoft.Extensions.*` transitive dependencies. That is acceptable for a backend-only managed pilot, but it is a runtime dependency risk rather than just a license-notice item.

## Decision

Trackdub will use Lucene.NET managed analyzer packages only behind the Infrastructure glossary analyzer adapters:

- `Lucene.Net.Analysis.Kuromoji` for Japanese.
- `Lucene.Net.Analysis.SmartCn` for Chinese.
- `Lucene.Net.Analysis.Common` `ArabicAnalyzer` for Arabic.
- Korean and unsupported languages remain on the application-layer morphology-lite fallback matcher.

The package versions must stay centrally pinned in `Directory.Packages.props`. Do not add native tokenizer runtimes, Java, Python, MeCab, Nori sidecars, or GPL/LGPL dictionary binaries as part of this pilot.

Runtime guardrails:

- Infrastructure analyzer tests must instantiate the managed analyzers and tokenize representative Japanese, Chinese, and Arabic text.
- Composition tests must resolve `IGlossaryTermMatcher` through DI and exercise analyzer-backed tokenization to catch missing assemblies or loader mismatches.
- The Windows CI build runs the solution test suite in Release, so these smoke tests run against the main product dependency graph.

## Consequences

Positive:

- The analyzer layer improves glossary matching without changing Domain, Contracts, persistence schema, or UI.
- The implementation remains Windows-friendly and dependency-light compared with native or sidecar tokenizers.
- The matcher still has a deterministic morphology-lite fallback when an analyzer cannot produce usable spans.

Negative:

- Beta Lucene.NET and alpha ICU4N transitives may have runtime binding, trimming, or patch-cadence risk.
- Future package upgrades need focused analyzer and DI smoke validation, not only compile-time checks.
- This pilot does not provide full morphological coverage or target-language inflection.

# ADR-0008: Inference Retry / Circuit Breaker

- Status: Draft
- Date: 2026-05-10

## Context

Trackdub runs ONNX inference through a routing layer (`RoutedTtsEngine`, `RoutedAudioTranscriptionEngine`, etc.) that selects a concrete engine adapter (Kokoro, Whisper, Demucs, etc.) based on a `StageRuntimePlan`. Each adapter calls into `InferenceSession` for model execution.

Currently, **no retry or circuit-breaker logic exists** anywhere in the inference path:

1. **Transient failures** (GPU kernel timeout, driver TDR, `OrtException` with ephemeral causes) propagate directly to the pipeline caller, failing the entire stage.
2. **Model-load failures** (corrupt file, EP mismatch, OOM during `InferenceSession` construction) are not distinguished from runtime failures, so repeated pipeline attempts retry a doomed load.
3. **No backpressure mechanism** ΓÇö a failing EP (e.g., TensorRT-RTX with an unsupported opset) is retried immediately on every pipeline run without cooling off.
4. **`PipelineDegradationRecord`** is written for general degradation but has no structured inference-fault taxonomy or circuit state.

This is a reliability gap. A single transient GPU glitch or one corrupt model download should not require a process restart or manual intervention.

## Decision

Introduce a two-layer inference reliability system:

### Layer 1 ΓÇö Retry Policy (transient failures)

Wrap each engine adapter's `RunAsync` (or equivalent Tensor-based inference call) in a retry handler that catches known transient ONNX exceptions:

| Exception / Signal | Classification | Action |
|---|---|---|
| `OrtException` with `OrtErrorCode.OrtErrorCode.RUNTIME_EXCEPTION` | Transient | Retry up to 2├ù with exponential backoff (100ms base, 2├ù factor, capped at 2 s) |
| `OrtException` with `OrtErrorCode.OrtErrorCode.ENGINE_ERROR` | Possibly transient | Retry once after 500 ms |
| `OrtException` with `OrtErrorCode.OrtErrorCode.MODEL_LOAD` | Permanent (model) | Route to circuit breaker |
| `OutOfMemoryException` / GPU OOM | Budget signal | Route to circuit breaker + GPU budget planner |
| `OperationCanceledException` | No retry | Propagate immediately |
| Unknown `OrtException` | Presumed permanent | Route to circuit breaker |

The retry handler is **injected as a decorator** around the adapter, not woven into each engine. This keeps engines testable without retry infra.

### Layer 2 ΓÇö Circuit Breaker (persistent/permanent failures)

A process-wide `InferenceCircuitBreaker` tracks failure state **per key** `(modelAlias, executionProviderKind)`:

```
State machine:

    ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
    Γöé                                          Γöé
    Γû╝    failure count ΓëÑ N (default: 3)        Γöé
  ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ  within sliding window (5 min)  ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
  Γöé      Γöé ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû║ Γöé      Γöé
  Γöé OPEN Γöé                                 ΓöéHALF- Γöé
  Γöé      Γöé ΓùäΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ Γöé OPEN Γöé
  ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ    cooldown timer expires       ΓööΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÿ
    Γû▓         (default: 60 s)                 Γöé
    Γöé                                          Γöé
    Γöé        ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
    Γöé        Γöé  probe (adapter call) succeeds
    Γöé        Γû╝
    Γöé      ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
    ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöé      Γöé
           ΓöéCLOSEDΓöé
           Γöé      Γöé
           ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
```

- **CLOSED**: Normal operation. Failures increment a sliding-window counter.
- **OPEN**: All inference requests for this key are rejected immediately with `InferenceCircuitBreakerOpenException` without touching the model. The pipeline can log a `PipelineDegradationRecord` and attempt a fallback EP or degrade gracefully.
- **HALF-OPEN**: After cooldown, a single probe request is allowed through. Success ΓåÆ CLOSED (reset counter). Failure ΓåÆ OPEN (reset cooldown timer, possibly with backoff).
- **Manual reset**: The breaker exposes a `ResetAsync(key)` method for admin UI or model-reload scenarios.

The breaker state is **in-memory only** (no persistence across restarts). A restart resets all breakers to CLOSED.

### Layer 3 ΓÇö Budget-Aware Backoff (cross-cutting)

The retry and circuit-breaker layers consult an `IExecutionBudgetProvider` (designed in ADR-00XX GPU Memory Budget Planner) before attempting retries:

- If the GPU memory budget is exhausted, retry backoff doubles.
- If the budget indicates an OOM condition, the circuit breaker transitions to OPEN immediately for the affected model.
- Budget recovery (model unload) triggers `ResetAsync` on the affected circuit breaker keys.

### Integration Points

1. **Adapter Decorator**: `RetryingTtsEngineAdapter` wraps `ITtsEngineAdapter` with retry + circuit breaker before delegating to the real adapter.
2. **Registration**: Decorators are registered in DI via `TryDecorate` (Scrutor) or manual `AddSingleton<ITtsEngineAdapter>(sp => new RetryingAdapter(inner, breaker, budget))`.
3. **DegradationRecord**: When a circuit breaker opens, a `PipelineDegradationRecord` with code `INF_CIRCUIT_OPEN` and the breaker key is written. When a retry is exhausted, `INF_RETRY_EXHAUSTED` is written.
4. **Fallback**: The circuit breaker does not itself pick fallback EPs ΓÇö that remains the `RuntimePlanner`'s job. The breaker merely prevents immediate retry of a known-bad path.
5. **Pool integration**: `InferenceSessionPool` evicts sessions for a key when the circuit breaker opens for that key, ensuring stale sessions are not returned to callers.

## Consequences

### Positive

- Transient GPU/ONNX failures are absorbed transparently during pipeline runs.
- Persistent failures (corrupt model, incompatible EP) are detected and blocked from wasting resources within seconds.
- The circuit breaker state machine prevents repeated expensive model loads after a failure.
- The decorator pattern keeps engine implementations testable without retry/circuit infrastructure.
- Integration with `PipelineDegradationRecord` gives users visibility into inference reliability issues.

### Negative

- Adds complexity to the adapter registration path (decorator wiring).
- In-memory circuit breaker state is lost on process restart ΓÇö recovery after a permanent failure requires re-attempting the failing operation.
- Retries increase tail latency for the pipeline stage (max ~3.5 s additional latency in worst case before full exhaustion).
- Half-open probe requests that fail consume resources that a strictly-open breaker would have saved.

### Neutral

- The budget-aware backoff adds a dependency on the GPU memory budget planner (ADR-0012).
- Metrics (breaker state transitions, retry counts) should be exposed for diagnostics but no monitoring infra exists yet.
- The sliding window for failure counting requires a time-source dependency (use `ITimeSystem` or equivalent abstraction for testability).

# ADR-0009: GPU memory budget planner

- Status: Draft
- Date: 2026-05-10

## Context

Trackdub loads multiple ONNX models into GPU memory during a session — at minimum a VAD model, an ASR
model, a diarization model, and a TTS model. Each model wrapper (`IInferenceEngineAdapter`) allocates GPU
resources when constructed and frees them when disposed.

Currently there is no coordination between model allocations. If the combined model working set exceeds
available GPU memory, one of three things happens:

1. DirectML falls back to CPU for one or more models, silently degrading performance.
2. The GPU driver triggers a TDR (timeout-detection-recovery) that stalls the pipeline for seconds.
3. An allocation fails with an `OutOfMemoryException` that propagates as a hard pipeline failure.

None of these outcomes is surfaced to the user as a memory-planning decision. The user cannot tell whether
their GPU can run the selected model combination, or which model would be swapped to CPU if memory runs low.

## Decision

Introduce a **GPU memory budget planner** — a service that estimates the GPU memory footprint of a set of
models before any of them are loaded, and either confirms the plan fits or recommends a degradation strategy.

### Components

**`GpuMemoryBudgetEstimator`** — reads model manifest entries (`ModelManifestEntry`) to obtain per-model
VRAM requirements. Each manifest entry gains an optional `EstimatedVramMb` field (integer, nullable).
If a model has no estimate, the planner uses a worst-case default (e.g. 2 GB for ASR, 512 MB for VAD).

**`IGpuMemoryBudgetPlanner`** — the public interface.

```
bool TryPlan(
    IReadOnlyList<ModelManifestEntry> models,
    long availableVramBytes,
    out GpuMemoryPlan? plan);
```

**`GpuMemoryPlan`** — describes which models fit on GPU, which must run on CPU, and whether the plan
is degraded.

```
public sealed record GpuMemoryPlan(
    bool IsDegraded,
    IReadOnlyList<ModelMemoryAssignment> Assignments);

public sealed record ModelMemoryAssignment(
    string ModelId,
    long EstimatedBytes,
    ExecutionProvider AssignedProvider);
```

**`IVramQuery`** — a small abstraction over DXGI / CUDA API calls that returns the current available
dedicated GPU memory in bytes.

```
public interface IVramQuery
{
    long GetAvailableDedicatedBytes();
}
```

### Integration points

1. `ModelManifestEntry` gains `EstimatedVramMb` (optional, nullable).
2. `GpuMemoryBudgetEstimator` reads manifest entries and computes assignments.
3. The budget planner is called **once per session** during `SessionWorkflowCoordinator` initialization,
   before any engine adapter is constructed.
4. The resulting `GpuMemoryPlan` is passed as part of the pipeline's immutable execution snapshot so stages
   can select the correct `IInferenceEngineAdapter` (GPU or CPU) per model.
5. If `IVramQuery` cannot determine available memory (unsupported API, driver issue), the planner assumes
   a conservative default (e.g. 2 GB) and logs a warning.

### Surface

- A degraded plan is logged and exposed through `PipelineDegradationRecord` (one record per model that was
  bumped to CPU).
- A dedicated "GPU Memory" section in the diagnostic overlay shows the plan, per-model assignment, and
  total estimated vs available.

## Consequences

Positive:

- The user gets a predictable model-loading experience — no surprise mid-pipeline fallback to CPU.
- Degradation is recorded in the pipeline audit trail, making it diagnosable post-hoc.
- The planner decouples VRAM measurement from model loading, keeping engine adapters simple.

Negative:

- `EstimatedVramMb` in the manifest is a static estimate; actual VRAM usage varies by input size and
  runtime state. The estimate should be conservative (overestimate by ~20%).
- `IVramQuery` depends on OS-level GPU APIs that may not be available on all Windows configurations or
  may report inaccurate values under dynamic load.
- The planner adds a new service interface and implementation that must be maintained alongside the
  inference engine adapters.

## Alternatives considered

### Dynamic VRAM tracking (measure actual usage at runtime)

Rejected because:
- DXGI `QueryVideoMemoryInfo` reports total budget / current usage, not per-process usage.
- CUDA `cudaMemGetInfo` reports free memory but race-conditions with concurrent model loads make it
  unreliable for planning.
- Actual measurements come too late — by the time we know memory is tight, models are already loaded.

### No planner — rely on DirectML CPU fallback

Rejected because silent CPU fallback is the current broken behavior. The user needs visibility into
when and why fallback occurs.

## References

- [DXGI.QueryVideoMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo)
- [ONNX Runtime memory-optimization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/memory.html)
- Existing `ModelManifestEntry` — `src/Trackdub.Inference/Runtime/ModelManifest/`

# ADR-0010: Event-sourced pipeline

- Status: Draft
- Date: 2026-05-10

## Context

The current transcript-generation pipeline (TranscriptPipelineBuilder) runs stages sequentially. Each stage
receives a PipelineContext, mutates it, and passes it to the next stage. The pipeline result is the final
state of PipelineContext after all stages complete.

This mutable-passing design has several drawbacks:

1. Mid-run corruption - If a UI action or provider change mutates shared state while a stage is running,
   the pipeline context can end up in an inconsistent state. The recent architectural decision to prefer
   immutable execution snapshots tried to mitigate this at the session level, but the per-stage context
   remains mutable.

2. Lost intermediate state - There is no record of what each stage produced, only the final aggregated
   result. Debugging a failed stage requires re-running the entire pipeline.

3. Audit gap - PipelineDegradationRecord captures skip/failure reasons at specific points, but there is
   no ordered event log that answers "what happened during this run, stage by stage?"

4. Replay impossibility - Because intermediate state is overwritten, the pipeline cannot resume from a
   specific stage after a crash or cancellation.

## Decision

Replace the mutable PipelineContext pass-through with an event-sourced pipeline that appends immutable
stage-completion events to an ordered event log.

### Architecture

TranscriptRunEvent - a discriminated union of all possible stage outcomes:

```
public abstract record TranscriptRunEvent;
public sealed record StageStarted(string StageName, Instant Timestamp) : TranscriptRunEvent;
public sealed record StageCompleted(
    string StageName, StageResult Result, Instant Timestamp) : TranscriptRunEvent;
public sealed record StageSkipped(
    string StageName, string Reason, Instant Timestamp) : TranscriptRunEvent;
public sealed record StageFailed(
    string StageName, PipelineDegradationRecord Degradation, Instant Timestamp) : TranscriptRunEvent;
```

TranscriptRunJournal - in-memory ordered collection of TranscriptRunEvent items:

- Appends are thread-safe (lock around the list).
- The journal is part of the immutable execution snapshot so it is not visible to stages until after they
  complete.
- The journal is serializable to JSON for crash recovery and diagnostic export.

Pipeline execution change:

- Each stage reads its input from the immutable execution snapshot (not from a mutable PipelineContext).
- Each stage writes its output as a TranscriptRunEvent appended to the journal.
- The final pipeline result is derived by folding the journal: fold over StageCompleted events to produce
  the aggregate result (previously done via PipelineContext mutation).
- PipelineContext is removed entirely; its responsibilities are absorbed by the execution snapshot
  (input) and the journal fold (output).

### Migration

1. Add TranscriptRunEvent types to Trackdub.Domain.
2. Add TranscriptRunJournal to Trackdub.Application transcript pipeline.
3. Replace PipelineContext references in TranscriptPipelineBuilder with journal + snapshot.
4. Migrate each stage handler one at a time: the handler reads from snapshot fields instead of
   PipelineContext, and the caller appends a StageCompleted event.
5. Remove PipelineContext after all handlers are migrated.

## Consequences

Positive:

- Full audit trail: every stage outcome is recorded in order, including skips and degradations.
- Crash recovery: the journal can be persisted and used to resume from the last completed stage.
- Thread safety: because the journal is append-only and stages cannot see uncommitted events, there is
  no risk of mid-run context corruption.
- Debugging: a developer can inspect the journal to see exactly what happened in a run.

Negative:

- More allocation per stage: each stage completion allocates a record object instead of mutating in place.
  This is acceptable because the pipeline runs at most once per user action.
- Existing stage handlers reference PipelineContext; the migration requires touching every handler.
- The event types introduce a new abstraction layer (events) on top of the existing StageResult types.

## Alternatives considered

### Keep mutable PipelineContext, add deep-copy snapshots

Rejected because deep-copy snapshots are fragile and expensive for large context objects. The event-sourced
approach is more idiomatic for .NET and provides a clearer audit trail.

### Use System.Reactive (IObservable) for event streaming

Rejected because Reactive Extensions add a significant dependency for what is fundamentally a simple
append-only list. The journal pattern is explicit, testable, and has zero external dependencies.

## References

- Event sourcing pattern (Martin Fowler) - https://martinfowler.com/eaaDev/EventSourcing.html
- Existing TranscriptPipelineBuilder - src/Trackdub.Application/Transcripts/Pipeline/TranscriptPipelineBuilder.cs
- Existing ITranscriptGenerationStage - src/Trackdub.Application/Transcripts/Pipeline/ITranscriptGenerationStage.cs

# ADR-0011: Intentional Trackdub.Contracts → Trackdub.Domain project reference

- Status: Accepted
- Date: 2026-06-02

## Context

The repository's layering guidance treats `Trackdub.Domain` as the innermost
project with no outward dependencies. `Trackdub.Contracts` was introduced as a
shared contract surface for pipeline stages, execution snapshots, and
cross-layer DTOs.

Over time, contract types began reusing domain value objects, enums, and
records directly instead of duplicating parallel contract-only shapes. That
reuse is encoded as a real `ProjectReference` from `Trackdub.Contracts` to
`Trackdub.Domain` in `src/Trackdub.Contracts/Trackdub.Contracts.csproj`.

This coupling predates the stricter "Contracts has zero project dependencies"
wording that appeared in CI prompts, `CLAUDE.md`, and some review checklists.
The canonical dependency diagram in `AGENTS.md`, the architecture test
`ContractsReferencesOnlyDomain`, and this ADR now describe the **actual**
allowed edge: Contracts may depend on Domain and on nothing else.

Removing the reference today would touch 30+ files across Contracts,
Application, Inference, and tests. That is true architectural debt, not a
quick hygiene fix.

## Decision

Keep the `Trackdub.Contracts → Trackdub.Domain` project reference as an
**intentional, documented exception** to the "Domain is the only leaf" ideal.

Enforcement rules:

- `Trackdub.Domain` remains dependency-free.
- `Trackdub.Contracts` may reference **only** `Trackdub.Domain`.
- No other project may treat Contracts as a substitute for Domain when pure
  domain invariants are required; Domain stays the source of pipeline truth.

CI, the dependency-graph audit script, and architecture tests must all encode
this rule consistently. Stale "zero deps on Contracts" checks are bugs in
documentation/automation, not signals to delete the reference without a
planned refactor.

## Consequences

Positive:

- Contract types stay aligned with domain invariants instead of drifting
  duplicate models.
- The allowed edge is explicit, testable, and reviewable.
- Agents and contributors stop fighting a fictional zero-dependency Contracts
  project.

Negative:

- Contracts is no longer a pure outward-facing leaf; it cannot be published or
  reused independently of Domain without pulling Domain types along.
- The ideal DDD onion (Contracts above Domain with no back-edge) remains
  unmet until a deliberate extraction effort lands.

## Alternatives considered

### Remove the reference now and duplicate types in Contracts

Rejected for this milestone: large, risky churn across many files with little
immediate product value and high merge conflict risk on active branches.

### Move shared types into Contracts and delete Domain usage from Contracts

Rejected as a single PR for the same scope reason. Some shared shapes are
genuinely domain entities/value objects; blindly moving them would blur the
Domain/Contracts boundary in the opposite direction.

### Introduce a SharedKernel (or similar) project below Contracts

Preferred **future** remediation path when the team budgets a focused refactor:

1. Identify types referenced by both Contracts and Domain consumers.
2. Extract stable, dependency-free primitives into `Trackdub.SharedKernel`
   (name TBD) or fold portable primitives into Contracts without referencing
   full Domain entities.
3. Rewire Contracts to depend on SharedKernel only; Domain may also depend on
   SharedKernel for shared primitives.
4. Delete `Contracts → Domain` once Contracts compiles without Domain types.

Until that work ships, the existing reference stays.

## Remediation checklist (future)

- Inventory Contracts files that `using Trackdub.Domain` or reference domain
  entity types.
- Classify each usage: true domain invariant vs. portable DTO/enums.
- Extract portable primitives to SharedKernel or duplicate thin contract DTOs
  where duplication is cheaper than shared entity coupling.
- Update `AGENTS.md`, architecture tests, and `verify-dependency-graph.py`
  if the graph changes.
- Remove the project reference only when `ContractsReferencesOnlyDomain` can
  be replaced by `ContractsHasNoProjectReferences` (or SharedKernel-only).

## References

- `tests/Trackdub.Architecture.Tests/DependencyGraphTests.cs` —
  `ContractsReferencesOnlyDomain`
- `tools/ci/verify-dependency-graph.py` — canonical allowed edges
- `AGENTS.md` — dependency diagram (`Contracts → Domain`)

# ADR-0012: WavePcm16 loudness policy — per-call-site opt-in + Roslyn analyzer

- Status: Accepted
- Date: 2026-07-23

## Context

`WavePcm16.WriteSamplesAsync` defaults `normalizePeak: false` to preserve
historical per-sample hard-clip semantics on cumulative mixes. The 5-arg
overload forwards to the 6-arg overload with `normalizePeak: false`; the
6-arg overload's bool flag is opt-in.

A recent bug in `PreviewRangeRenderer` — the only production caller writing
post-multichannel-downmix output — revealed that hot 5.1 mixes silently
hard-clipped to `short.MaxValue`. The bug was detected because a regression
test pinned the expected post-scale values (L ~1.0 / R ~0.686). Without
that test the bug would have shipped silently. The fix added
`normalizePeak: true` to the call site without touching the writer default.

Inventory of `WavePcm16.WriteSamplesAsync` / `WriteMonoAsync` call sites in
`src/` (per the wave-pcm16 hygiene grep):

- `src/Trackdub.Media/Mixing/PreviewRangeRenderer.cs:171` — **HIGH risk**
  (multichannel→stereo additive downmix; cumulative peaks reach ~2.06).
  Opted in (`normalizePeak: true`) this lane (commit `912ae905`).
- `src/Trackdub.Media/Tts/TtsAudioPostProcessor.cs:47` — **NARROW risk**
  (single mono TTS engine output; bounded by the TTS model). No change.
- `src/Trackdub.Media/Waveforms/Pcm16ReferenceClipTrimmer.cs:87` —
  **LOW-MEDIUM risk** (silence-trim pass-through; doesn't additively mix).
  Not migrated this lane.
- `src/Trackdub.Media/Stretch/WsolaPhonemeStretchService.cs:119` —
  **LOW-MEDIUM risk** (WSOLA overlap-add; can theoretically exceed |1| in
  pathological cases). Not migrated this lane.

The bug class — silent hard-clip on hot cumulative output — is reproducible
by any future caller that:

1. mixes samples from multiple sources; OR
2. applies a transform whose output can exceed the input's dynamic range; AND
3. forgets to opt in to `normalizePeak: true`.

Today nothing in the build pipeline warns such callers.

## Decision

Adopt a **per-call-site opt-in** convention backed by a
`WavePcm16MultiSourceMixOptIn` Roslyn analyzer that detects when a caller
inside a method whose name contains Mix, Mixer, Blend, or Render calls
`WavePcm16.WriteSamplesAsync` without `normalizePeak: true`.
`PreviewRangeRenderer.cs:171` sets the precedent for the opt-in shape.
The analyzer recognizes only `WriteSamplesAsync` (which has the `normalizePeak`
parameter); `WriteMonoAsync` is excluded because it cannot resolve the finding.

**Convention:**

- Every caller buffering multi-source or post-transform mixed output that
  can exceed |1| MUST pass `normalizePeak: true`.
- Callers writing single-source PCM within a known bounded range (mono TTS,
  silence-trim pass-through) MAY pass `normalizePeak: false`.
- The default in `WavePcm16.WriteSamplesAsync` stays `false` to preserve
  historical loudness for in-range inputs.

**Deferral — explicit non-migration this lane:**

`Pcm16ReferenceClipTrimmer` (LOW-MEDIUM) and `WsolaPhonemeStretchService`
(LOW-MEDIUM) **are not migrated in this lane**. Rationale (per AGENTS.md
"Touch waves, not ripples"):

- Both are single-source transformations, not multi-source additive mixes.
- Their input loudness is bounded by upstream single-source pipelines.
- Loudness surprise for any caller downstream (final mix export, dub
  playback) is a UX decision, not a correctness fix; needs a separate
  loudness-policy ADR that covers the loudness-vs-final-mix tradeoff (see
  "Future remediation" below).
- This lane pinned the convention without flipping any defaults outside
  `PreviewRangeRenderer.cs`.

**Future remediation:** A future lane (after the §4.4 / C8 / §9.1 /
log-rotation-fix stack has merged and stabilised) should re-audit the 2
LOW-MEDIUM callers with empirical WSOLA-peak and silence-trim-passthrough
data before opting them in. The Roslyn analyzer should automatically flag
them if a new multi-source mix path appears in their input graph.

## Consequences

**Positive:**

- The bug class is caught at compile time for new code paths via the
  analyzer, replacing the current "test must exist" discovery mechanism.
- The 22/22 `PreviewRangeRenderer` suite + 9/9 `WavePcm16` suite + 155/0/3
  full `Trackdub.Media.Tests` confirm the opt-in path is safe and does not
  regress in-range sources.
- No loudness surprise for callers currently in production — the default
  stays `false`; only the 1 already-fixed production caller changed.
- The convention is small, reviewable, and survives future feature work.

**Negative:**

- The 2 LOW-MEDIUM callers remain on the un-opted default; a pathological
  WSOLA overlap or upstream hot-trim pass-through could still hard-clip
  silently.
- The Roslyn analyzer adds a new tooling surface (rule, fixtures, CI
  integration) to maintain.
- Per-call-site opt-in is review-dependent; reviewers must know the
  convention or the analyzer must catch the slip.
- This ADR fixes only the discovery mechanism. A follow-up loudness-policy
  ADR must decide whether to flip the default, and that ADR must migrate
  the 2 deferred sites if it does.

## Alternatives considered

### Flip the default (`normalizePeak: true` from overload 1) and require explicit opt-out

Rejected for this milestone:

- Every existing `WavePcm16.WriteSamplesAsync` caller would suddenly emit
  quieter output (loudness surprise), violating AGENTS.md "Touch waves, not
  ripples" until data justifies a global loudness change.
- The 4 scaler facts in `WavePcm16Tests.cs` + the hot-5.1 test pin
  specific PCM short values; the entire `Trackdub.Media.Tests` suite would
  need updating to reflect new post-scale amplitudes.
- Strongest "policy-as-default" answer but premature without empirical
  loudness impact data across all known callers.

### Introduce `WavePcm16LoudnessPolicy` enum (`Preserve` | `Normalize` | `ForceLimit`) replacing the bool

Rejected for this milestone:

- Three call-site migrations (only `PreviewRangeRenderer` opted in this
  lane; the 2 LOW-MEDIUM callers are deferred) plus tests plus contract
  docs.
- Adds runtime-introspection code paths (`Preserve` vs `ForceLimit`) the
  codebase has no immediate use for.
- Premature abstraction; defer until a second concrete policy requirement
  surfaces (peer review, dub playback loudness target, etc.).

### Keep opt-in without an analyzer

Rejected: the bug class was discovered only because a test existed.
Without the test, silent hard-clip would have shipped. Code review alone
is insufficient to catch future omissions; the analyzer is the safety net.

## References

- `src/Trackdub.Media/Waveforms/WavePcm16.cs` — the writer; overloads
  surfaced at L282–305; scaler at L370–395.
- `src/Trackdub.Media/Mixing/PreviewRangeRenderer.cs:171` — the opt-in
  precedent (`normalizePeak: true` at the `WavePcm16.WriteSamplesAsync`
  call site inside `RenderAsync`).
- `src/Trackdub.Media/Waveforms/Pcm16ReferenceClipTrimmer.cs:87` —
  LOW-MEDIUM caller, **deferred** this lane.
- `src/Trackdub.Media/Stretch/WsolaPhonemeStretchService.cs:119` —
  LOW-MEDIUM caller, **deferred** this lane.
- `src/Trackdub.Media/Tts/TtsAudioPostProcessor.cs:47` — NARROW caller,
  no change.
- `tests/Trackdub.Media.Tests/PreviewRangeRendererTests.cs` (~L1066) —
  `RenderAsync_writes_hot_five_one_pcm16_at_unit_peak_via_per_track_normalization`
  regression test that pinned the bug.
- `tests/Trackdub.Media.Tests/WavePcm16Tests.cs` (~L100–200) — four new
  scaler facts: overflowing, in-range, NaN/Inf, exact-source.
- `docs/adr/ADR-0011-contracts-domain-coupling.md` — style template
  (YAML bullet-list frontmatter + sectioned prose).
- `docs/adr/README.md` — directory purpose + ADR conventions.
- Commit `685776f3` — wave-pcm16 cherry-pick (3 files: writer + 2 test files).
- Commit `912ae905` — call-site opt-in fix (1 file: `PreviewRangeRenderer.cs`).
- PR #535 — `https://github.com/Mesililm-LLC/Trackdub/pull/535`
  (branch `chore/wave-pcm16-normalization`, base `main`).
- `AGENTS.md` — "Touch waves, not ripples" rule; preview-vs-final-mix
  loudness separation; "Encounter defects while working" remediation law.

# ADR-0015: Pipeline transient telemetry — in-process only on on-prem tiers

> Current AWS POC contract: [SnapSoft application acceptance](../operations/snapsoft-poc-acceptance.md). Historical API/Swagger/compose/telemetry examples below are not evidence of current wiring. The new CPU API uses DynamoDB/S3, contains no inference, and has no billing/OTel exporter. GPU compose runs the one-shot CLI; it does not expose HTTP.

- Status: Draft (promoted from `docs/internal/pipeline-readiness-spec.md` §9.4)
- Date: 2026-07-23

## Context

Trackdub's `PipelineTransientFaultBus` already exposes a typed, in-process pub/sub surface for stage-level transient failures (see `src/Trackdub.Application/Transcripts/Pipeline/PipelineTransientFaultBus.cs` and the upstream spec `docs/internal/pipeline-readiness-spec.md` §4.3, §9.1). The bus ring-buffers the last 50 faults and surfaces them through `IObservable<PipelineTransientFault>` plus a snapshot accessor; readers today are the diagnostics-bundle exporter (`§9.3`), the per-run aggregation reader (`SnapshotPerRun`, §9.1), the `DubbingPipelineEngine`'s `IAsyncEnumerable<PipelineTransientFault> TransientFaults` accessor (§4.5), and the on-prem tier-specific consumers (Worker metrics, Avalonia `PipelineRunViewModel`, headless SDK).

Earlier drafts assumed a cloud OpenTelemetry/billing surface. Those paths were not current implementation evidence; the new AWS CPU job API does not implement that surface.

The question this ADR promotes is whether the on-prem `Observable` should also bridge to OpenTelemetry, Sentry, or another upstream telemetry sink.

## Decision

**Recommendation (a):** `PipelineTransientFaultBus.Stream` stays in-process only on the on-prem tiers. No new upstream sink on App, SDK, Worker, or CLI. Per-tier consumers (Worker metrics, Avalonia VM, headless SDK) read directly from the bus or its snapshot.

The OTel bridge, when it becomes a customer ask, belongs to the cloud tier (option (c) below) and to a separate ADR. It does NOT belong on the on-prem tiers.

## Alternatives considered

### (a) None on the on-prem tiers — CHOSEN

`Observable` stays in-process; per-tier consumers read directly. No new external sink on `Trackdub.App.Avalonia`, `Trackdub.Sdk`, `Trackdub.Worker`, or `Trackdub.Cli`.

- **Trade-offs.** Cheapest path. Matches the existing on-prem posture: no telemetry surveillance on the end-user runtime (per `AGENTS.md` §Model governance non-negotiables). Preserves the local-first product contract for desktop + on-prem install.
- **When chosen over (b)/(c).** When the customer ask for cross-stage transient-fault telemetry does not yet exist, or when it lives in the cloud tier via (c).

### (b) `IPipelineTransientFaultExporter` adapter + OTel implementation behind a feature flag

Introduce an adapter interface in `Trackdub.Application` and a single OpenTelemetry implementation in `Trackdub.Composition` / `Trackdub.Api`, gated by a `StudioSettings` feature flag.

- **Trade-offs.** Lets the on-prem tiers stream transient-fault records to an OTel collector the user hosts themselves. Consistency with the cloud tier's OTel wiring.
- **Why not chosen.** Invites new infrastructure on a desktop product. Contradicts `AGENTS.md` §Model governance ("no new external telemetry surveillance on the end-user runtime path") — even an opt-in flag can leak through fault-onboarding flows. Adds a new dependency graph edge (`Application` → `OpenTelemetry`) that the on-prem tiers currently avoid. The same surface can be achieved via (c) without forcing the desktop binary to grow.

### (c) No integration on on-prem; centralize on the cloud tier by serializing transient-fault shipments into the existing trackdub-telemetry pipeline

On-prem tiers do nothing; the Cloud API / hosted Trackdub variant consumes `PipelineTransientFault` records end-to-end and reuses the existing `dubbing_metrics` counter surface. Fault shipments into the cloud tier are opt-in per project + per tier via Tier / project settings.

- **Trade-offs.** Preserves the on-prem posture and re-uses what already exists. Hosts transient-fault telemetry where it can be paired with project SLIs/SLOs and per-customer alerts.
- **Promotion** when one of the criteria below holds.

## Consequences

### Positive

- Preserves the no-new-runtime-telemetry posture on desktop + on-prem installs.
- Zero new binary footprint on `Trackdub.App.Avalonia`, `Trackdub.Sdk`, `Trackdub.Worker`, `Trackdub.Cli`.
- Honors the `IObservable<T>` semantics callers already rely on — no opt-in subscription layer to misuse.

### Negative

- A support engineer triaging a desktop crash cannot correlate transient-fault frequency across machines unless the customer opts into cloud-tier tracking.
- The cloud-tier upgrade path is unbounded — when the customer ask lands, a separate ADR will own the OTel adapter surface plus its consumer tests.

### Neutral

- No tests added in this ADR (per spec §11.7 — the surface is held documented but no `Exporter_publishes_to_existing_dubbing_metrics_counter` / `Exporter_does_not_initialize_when_feature_flag_disabled` facts ship until the cloud-tier ADR is opened).
- Cross-link: `docs/internal/pipeline-readiness-spec.md` §9.4 (source) + §11.7 (test surface + promotion gate).

## Promotion criteria

Promote (i.e. open a new ADR that supersedes this one for the cloud-tier OTel bridge) when one or more of the following is true:

1. `Trackdub.Cloud` or a hosted customer SLI/SLO asks for per-stage transient-fault telemetry.
2. A repeat user-facing incident traces to a failure pattern that surface logging alone cannot correlate.
3. The Cloud tier upgrades to consume `PipelineTransientFault` directly and warrants a dedicated bridge.

Per spec §11.7, that future ADR will require two tests (or analogs) at the time of promotion:

- `tests/Trackdub.Api.Tests/Observability/TransientFaultExporterTests.cs::Exporter_publishes_to_existing_dubbing_metrics_counter`
- `tests/Trackdub.Api.Tests/Observability/TransientFaultExporterTests.cs::Exporter_does_not_initialize_when_feature_flag_disabled`

This ADR holds the surface documented so future agents do not re-derive the recommendation from §9.4 verbatim.

# ADR-0016: Scoped end-user Python exception for the supervised inference worker

- Status: Accepted
- Date: 2026-10-06
- Decided by: Tony, 2026-10-06 (recorded in PR #401, following pitch PR #394; `docs/archive/rust-inference-sidecar-pitch.md`)

## Context

AGENTS.md forbids end-user runtime dependencies on Python, Conda, Docker, and
the CUDA Toolkit. That rule exists to protect the single-installer,
local-first distribution story — and it stands for the application itself.

But frontier voice models (Chatterbox, CosyVoice and successors) publish
PyTorch `.pt` weights first, sometimes only. The ONNX export tax — fragile
graph conversions, `If`/`Squeeze` graphs that won't lower, reshape profile
fights, weeks waiting on community conversions — is a recurring cost
documented in the TTS stall investigations (`.freebuff/c13-*`). Decision
2026-10-06: `.pt`-only models are inevitable, so Python returns in exactly one
place, under supervision, per the design in pitch Section 5b.

Related: ADR-0006 (Chatterbox commercial-use verification — license posture
for the likeliest first worker model), pitch PR #394.

## Decision

A single scoped exception to the no-end-user-Python rule:

1. **Exactly one Python runtime may ship**, and only as a worker behind the
   sidecar IPC contract — never as an application dependency, build
   requirement, or ad-hoc subprocess (no `espeak-ng`-style one-offs through
   this door; Decision 3 of ADR-0005 stays as-is).
2. **Rust supervises, Python computes, C# orchestrates.** The supervisor owns
   the bundled pinned interpreter (`uv python` or equivalent — never system
   Python), the dependency lockfile, process lifecycle, readiness gates,
   crash-backoff restarts, and version-stamp refusal. Python code loads
   weights and runs inference, nothing else.
3. **Same contract or nothing.** The worker speaks the typed sidecar protocol
   (`{dtype, shape, data}` envelopes, planner-approved load plans mirroring
   `StageRuntimePlan`) verbatim with the Rust ONNX path. Executors stay
   interchangeable; no caller changes per executor.
4. **No silent runtime.** Worker presence, interpreter version, model-loaded
   state, and settled accelerator are stage evidence, surfaced truthfully in
   the UI — the honest-readiness rule extends to the worker, it is not waived
   for it.
5. **Proven in the harness first.** No worker model ships until
   `controlled-matrix --executor sidecar` carries first-use latency,
   warm-call overhead, and VRAM-held evidence for it, per repo benchmark
   policy. Claims without evidence are rejected the same as in-process ones.

## Consequences

### Positive

- `.pt`-only frontier voice models become shippable in days, not
  after-export-if-ever.
- All Python pain (interpreter pinning, CUDA matching, dependency drift,
  crash containment) has exactly one owner: the supervisor.
- The application keeps its no-Python posture; the exception is bounded,
  versioned, and revocable.

### Negative

- Installer grows: pinned interpreter + torch + CUDA-matched libs + provider
  shared libs alongside the C# app's ORT copy. Distribution size and the
  cross-process ORT pin (pitch Q6) are open spike measurements.
- Second binary widens installer signing/notarization surface (pitch Q3).
- A torch/CUDA security advisory now has a response path we must own —
  lockfile bump plus version-stamp invalidation.

### Neutral

- No `src/` changes ship under this ADR. It authorizes the spike
  (`docs/archive/rust-sidecar-spike-plan.md`); promotion to `src/` needs its
  own review with harness evidence attached.

## Alternatives considered

### Continued ONNX-only

Rejected 2026-10-06: export coverage for frontier voice models is not
converging fast enough, and each export is bespoke validation work.

### Unsupervised Python daemon / system Python

Rejected: reintroduces every failure mode the AGENTS.md rule was written to
prevent (version drift, CUDA mismatch, silent breakage). Supervision is the
condition of the exception, not an implementation detail.

### Full Rust rewrite of inference

Rejected: see PR #394 discussion — the working pipeline, session pool, and
harness stay in C#; Rust owns supervised processes behind the IPC contract,
never pipeline state or UI.

## Revocation criteria

Revisit (narrow or revoke this exception) when one or more is true:

1. ONNX export coverage for shippable voice models converges (no `.pt`-only
   gap for two consecutive model evaluations).
2. The worker's measured distribution or maintenance cost exceeds its model
   coverage value for two consecutive releases.
3. A Python-supply-chain incident makes the lockfile posture untenable and no
   mitigation lands within one release.
