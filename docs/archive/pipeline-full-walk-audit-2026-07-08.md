# Full Pipeline Walk + Audit

**Date:** 2026-07-08  
**Scope:** End-to-end dubbing pipeline in Trackdub (SDK + Application + Composition + Domain)  
**Focus:** Structure, execution order, mechanisms (readiness, snapshots, artifacts, resume, failure), weak points, bottlenecks, breakage risks, bad configurations, and recommended fixes.  

All paths respect core invariants:
- Never fake readiness (distinct provider/model/artifact/stage states).
- Preserve original + prior usable artifacts on skip/failure.
- Strict layering (Domain → Contracts → Application → ...).

## Canonical Stage Inventory

From `src/Trackdub.Domain/StageRuns/StageNames.cs`:

- `separation`, `speech-enhancement`, `audio-preparation`
- `vad`, `diarization`, `asr`, `text-refinement-asr`, `text-refinement-translation`, `text-refinement`
- `speaker-assignment`
- `translation`
- `tts`, `voice-cloning`
- `overlap-rescue`
- `preview-mix`
- `lip-sync`, `lip-synthesis`
- `export`

**High-level SDK "DefaultStageOrder"** (`src/Trackdub.Sdk/TrackdubDubbingEngine.cs`):
```csharp
[ Separation, Vad, Diarization, Asr, Translation, Tts, Export ]
```
(Note explicit comment: Diarization before ASR so speaker labels are available when `SpeakerAssignmentAndPersistenceStage` persists the transcript.)

Many stages are granular and only appear inside the transcript sub-pipeline or as optional/side paths.

## Execution Layers

1. **Outer orchestrator** (SDK/CLI/unattended):
   - `TrackdubDubbingEngine.ExecuteAsync` (`src/Trackdub.Sdk/TrackdubDubbingEngine.cs`)
   - Source media validation (with fallback to stored path from project context)
   - Snapshot capture
   - Preflight checks
   - Loop over resolved stage order
   - Resume skip + prerequisite blocking + dispatch to `RunStageWorkflowAsync`

2. **Workspace / Workflow layer**:
   - `TranscriptWorkspace` (holds all workflows + `_pipelineGuard` SemaphoreSlim for serialization)
   - `ProjectWorkflow`, `TranslationWorkflow`, `TtsWorkflow`, `TtsOrchestrationService`, `ExportWorkflow`, `PreviewMixWorkflow`, `LipSyncWorkflow?`, `LipSynthesisWorkflow?` (latter two optional/nullable)
   - `RunInitialTranscriptionAsync`, `RunTranscriptStageAsync`, `GenerateTranslationAsync`, etc.

3. **Transcript sub-pipeline** (core ASR path):
   - `TranscriptGenerationService` (`src/Trackdub.Application/Transcripts/TranscriptGenerationService.cs`)
   - Hard-coded `TranscriptPipelineBuilder` producing `TranscriptGenerationPipeline`
   - Always executes in this order for full runs:
     1. SpeechEnhancementGenerationStage
     2. VadGenerationStage
     3. SpeakerDiarizationStage
     4. AsrGenerationStage
     5. TextRefinementGenerationStage
     6. SpeakerAssignmentAndPersistenceStage
   - `ITranscriptGenerationStage.ExecuteAsync` + resume/hydration logic

4. **Leaf implementations**:
   - `*GenerationStage` (implement `ITranscriptGenerationStage`)
   - `*StageHandler` (e.g. `VadStageHandler`, `AsrStageHandler`, `StemSeparationStageHandler`, `ExportStageHandler`, `LipSyncStageHandler`)
   - Use `StageRunHelper` for consistent `StageRunRecord` lifecycle

5. **Cross-cutting**:
   - `IArtifactStore` + `TranscriptArtifactWriter`
   - `IProjectStageRunStore` + `StageRunHelper` + `StageRunHygiene`
   - `PipelineDegradationWriter`
   - `MixPlanBuilder` / `MixPlanStore`
   - Readiness: `PipelineReadinessService`, `IPipelinePreFlightChecker`, runtime planner (per `RuntimeStage`)
   - Progress: `PipelineProgressEvent`

## Detailed Flow (Happy Path)

**Prep / Ingestion** (`ProjectWorkflow`):
- Create project / media spine
- `EnsureNormalizedProjectAudioAsync` (FFmpeg)
- Optional stem separation (`RunStemSeparationAsync` → `StemSeparationStageHandler`): produces vocals/ambiance stems + speech-enhancement as side effect (with fallback to unenhanced)
- Speech audio routing plan (prefers vocal stem for downstream VAD/ASR)

**Transcript Sub-Pipeline** (sequential):
- Speech enhancement
- VAD → `SpeechRegions` artifact (via `VadStageHandler`)
- Diarization (if enabled)
- ASR (on regions) → raw segments. Special case: VAD finds zero regions → explicit skip + `PipelineDegradationRecord("VAD_NO_REGIONS")`
- Text refinement (ASR)
- Speaker assignment + persistence → `TranscriptRevision` saved

**Post-ASR**:
- Translation (segment level)
- TTS (per-speaker, multiple candidate takes, timing)
- Optional overlap rescue

**Mix / Lip / Export**:
- `MixPlanBuilder`: selects takes (newest + user-selected candidate first), incorporates lip-sync clips when present, emits `MixPlanWarning`s for gaps/missing/stale/LipSyncArtifactMissing
- Preview mix (separate workflow for UI scrubbing)
- Lip-sync / Lip-synthesis (optional, video-frame based, heavy models, not part of default SDK order)
- Export (`ExportStageHandler`): builds final mix plan, loudness normalization, render, optional video recompose, subtitles, mux. Writes export artifacts.

All prior usable artifacts are left in place on skips/failures.

## Key Mechanisms

**Resume / Skip**:
- `DubbingEngine` + `TranscriptGenerationPipeline.ShouldResumeSkipStage`
- Checks: `!ForceRerun`, project state, artifacts exist, `StageArtifactResumeEvaluator.CanResumeStage(...)` using execution snapshot
- Special skips for disabled diarization / text-refinement
- On skip: writes `StageRunRecord` as Skipped + hydrates context from prior artifacts

**Snapshots**:
- Captured early (`CaptureExecutionSnapshot`)
- Includes model aliases, prefs, languages, `EnableAsrTextRefinement`, export format, fingerprints
- Used for both resume decisions and provenance

**Stage Run Records**:
- `StageRunHelper.StartAsync` + `RunStageAsync` / `SkipAsync` / `CompleteAsync`
- Every known stage name has an explicit case (prevents typos)
- Statuses are distinct (Running, Succeeded, Skipped, Failed, Canceled)

**Failure / Degradation**:
- Prerequisite stages (currently only Vad + Asr) block subsequent stages in outer loop
- Unhandled exceptions → degradation record (best-effort) + rethrow
- Speech enhancement failure inside separation is specially extracted and reported
- Empty-region ASR case explicitly degrades instead of pretending success

**Preflight / Readiness / Hardware**:
- Per-stage runtime planning (execution provider selection)
- Model provisioning can be deferred into stage in some cases
- Separate from "stage ran" state

## Identified Weakpoints, Bottlenecks, Breakage Risks, Bad Configs

1. **Lips not executed in main SDK pipeline** (`src/Trackdub.Sdk/TrackdubDubbingEngine.cs:665` switch + `RunStageWorkflowAsync`)
   - `DefaultStageOrder` and dispatch only cover Separation/Vad/Diarization/Asr/Translation/Tts/Export.
   - Lips are prepared in `CreateRuntimeSelectionsAsync` and snapshot (lines ~874, 942) but never dispatched.
   - `LipSyncWorkflow` / `LipSynthesisWorkflow` are nullable in `TranscriptWorkspace`.
   - **Breakage**: "Full pipeline" via SDK/CLI produces no lip-synced video. UI has exposure (options, segment provenance) but inconsistent surface.
   - Related: `docs/internal/pipeline-arch-audit.md` and architecture audits hint at this gap historically.

2. **Prerequisite blocking is too narrow** (`src/Trackdub.Sdk/TrackdubDubbingEngine.cs:40`)
   - Only `Vad` + `Asr` block later stages.
   - Diarization failure allows Translation/TTS/Export to proceed (bad or missing speaker labels).
   - **Risk**: Silent degradation of output quality.

3. **Resume logic is duplicated and fragile**
   - Spread across `DubbingEngine.HasValidExistingArtifactsAsync`, `TranscriptGenerationPipeline.ShouldResumeSkipStage`, `StageArtifactResumeEvaluator`, snapshot equality checks.
   - Snapshot includes model prefs; changing a model alias can cause wrong skip or unnecessary rerun.
   - Speaker assignment hydrate depends on prior ASR + diarization artifacts being present.
   - **Bottleneck + correctness risk** on iterative work.

4. **Purely sequential execution**
   - `TranscriptGenerationPipeline` is a simple `foreach`.
   - TTS (per speaker), ASR (per region) are serial.
   - **Major bottleneck** for long-form content or many speakers.
   - No use of parallelism for independent post-diarization work.

5. **Speech enhancement ownership is confused**
   - Runs as internal side-effect inside `StemSeparationStageHandler`.
   - Also appears as first `SpeechEnhancementGenerationStage` in the transcript builder.
   - Degradation extraction is special-cased only for the separation path.
   - Enablement and timing relative to VAD are split across flags and call sites.

6. **Mixing is buried inside Export** (`ExportStageHandler`, `MixPlanBuilder`)
   - No top-level `preview-mix` or `mix` stage in `DefaultStageOrder`.
   - `MixPlanBuilder` has complex take selection + many warning codes (MissingTake, StaleTake, LipSyncArtifactMissing, etc.).
   - Silent gaps are produced; warnings are collected but not strongly enforced before export.
   - Preview mix is a completely separate named stage/workflow.

7. **Stage graph is fragmented and stringly-typed**
   - Hard-coded array in SDK, hard-coded builder list in `TranscriptGenerationService`, switch in `ResolveTranscriptStages`, another switch in `StageRunHelper.StartKnownStageRun`, dispatch switch in DubbingEngine.
   - Coarse stage names in SDK vs. fine-grained internal stages.
   - Enable flags (`enableStemSeparation`, `enableSpeakerDiarization`, `EnableAsrTextRefinement`) not applied uniformly between SDK and shell.

8. **Lip stages are second-class / optional**
   - Depend on video extractors/recomposers + specific commercial models.
   - Not automatically included even when video source + model prefs are present.
   - Test fixtures are env-var gated.

9. **Cancellation and resource hygiene**
   - Cancellation works between outer stages but long-running inner ONNX/FFmpeg work may not be promptly interruptible.
   - Stem temp directories cleaned only at handler start.
   - Single semaphore serializes all pipeline activity per workspace.

10. **Model provisioning / preflight is scattered**
    - Multiple entry points (`EnsureImportModelsForTranscriptionAsync`, per-stage ensure, `RuntimeModelSetupWorkflow`, deferred download inside some handlers).
    - Risk of inconsistent readiness state.

11. **Edge cases around empty / failed early stages**
    - ASR handles VAD-empty gracefully.
    - Downstream stages (assignment, TTS, mix plan, export) have varying levels of defensive handling.
    - Export can be reached with incomplete prior artifacts.

12. **Other observations**
    - `StageNames` is the single source of truth (good), but many call sites still risk typos before compile-time enforcement.
    - Degradation writes are best-effort.
    - FFmpeg and ONNX session creation are repeated hot paths without obvious high-level pooling visible from Application layer.
    - Config defaults live in various `*Defaults.cs` files; no single pipeline policy object.

## Recommended Fixes (Prioritized)

**High**
- Add lipsync/lipsynthesis (conditionally) to `DefaultStageOrder` and the dispatch switch in `TrackdubDubbingEngine`. Make execution of `Lip*Workflow` part of a full run when applicable.
- Expand `PrerequisiteStages` to include `Diarization` (and consider `Translation`).
- Centralize the stage graph definition (one list + metadata for prerequisites, artifact contracts, resume eligibility). Drive builder, dispatcher, CLI filters, and UI from it.

**Medium**
- Unify and harden resume logic. Add a versioned snapshot hash + explicit "invalidate stage" API. Add comprehensive tests for snapshot drift.
- Promote an explicit `Mix` stage (or make mix plan generation a named stage that export consumes).
- Make speech enhancement a first-class, independently configurable pre-VAD stage with uniform degradation reporting.
- Introduce bounded parallelism for TTS generation across speakers and (where safe) segment batches inside ASR/translation.

**Lower / Hygiene**
- Strengthen guards in assignment/TTS/mix/export for missing prior artifacts (fail fast or produce clear degradation).
- Improve cooperative cancellation inside long-running engines and add hard timeouts with cleanup.
- Centralize model provisioning gate so every stage path goes through the same readiness check.
- Add integration-level test that runs full SDK order + resume + injected failures + lips (when fixtures available).
- Audit every stage handler for "write artifact before marking success" and "preserve prior artifacts on exception".

## References (key files walked)

- `src/Trackdub.Sdk/TrackdubDubbingEngine.cs`
- `src/Trackdub.Sdk/TrackdubPipelineStages.cs`
- `src/Trackdub.Domain/StageRuns/StageNames.cs`
- `src/Trackdub.Application/Transcripts/Transcript{Workspace,GenerationService,ProjectWorkflow}.cs`
- `src/Trackdub.Application/Transcripts/Pipeline/TranscriptGenerationPipeline.cs` + `TranscriptPipelineBuilder.cs`
- `src/Trackdub.Application/Transcripts/Stages/*.cs` (all 6)
- `src/Trackdub.Application/Transcripts/*StageHandler.cs` (StemSeparation, Export, etc.)
- `src/Trackdub.Application/Mixing/MixPlanBuilder.cs`
- `src/Trackdub.Application/Transcripts/StageRunHelper.cs`
- `src/Trackdub.Composition/CompositionRoot.cs` (registrations)
- `src/Trackdub.Application/Pipeline/README.md` + related preflight

Related prior audits:
- `docs/architecture/P0-pipeline-audit-2026-06-01.md`
- `docs/internal/pipeline-arch-audit.md`

---

**End of audit.** This document captures the pipeline as exercised through SDK entry points, the internal transcript sub-pipeline, mixing, export, and optional lip paths. Further deep dives can be performed on any specific stage or mechanism. 

To regenerate or extend: run the same exploration against current sources and update this file.