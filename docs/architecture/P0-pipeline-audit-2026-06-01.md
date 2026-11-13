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
