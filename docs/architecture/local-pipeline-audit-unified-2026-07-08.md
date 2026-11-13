# Local Dubbing Pipeline — Architectural Audit & Proposed Fixes (Unified)

> **Audit date:** 2026-07-08  
> **Verified against source:** 2026-07-08 (re-audit pass; see §8 Correction Log)  
> **Scope:** media ingest → playback → full local inference pipeline → export dubbed video  
> **Layers audited:** App.Avalonia, Media/Media.Playback, Application (pipeline orchestration), Inference/Inference.Onnx, Sdk  
> **Layers excluded:** Cloud (Api, Worker, WebhookDelivery), Benchmarks, Tools  
> **Fix posture:** proposed-only — no code changes in this audit  
> **Sources:** Two independent audits merged, then corrected against current tree. Surviving source walk: `docs/archive/pipeline-full-walk-audit-2026-07-08.md`. Cited `docs/architecture/local-pipeline-audit-2026-07-08.md` is **not present** in the tree (agent-1 content survived only via this merge). Principles supplement: `docs/architecture/pipeline-principles-review-grok-2026-07-08.md`.

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
docs/archive/pipeline-full-walk-audit-2026-07-08.md
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
