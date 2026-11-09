# Desktop cold-start & wait-time audit — Trackdub core + gated app

Verified against live code (not just docs). Ranked by user-visible impact. Nothing was changed — this is flag-only.

## P1 — Launch → first window (every launch, UI thread)

**1. Serial blocking chain before the window exists.** [App.axaml.cs:186-307](file:///D:/Dev/Trackdub_Workspace/Trackdub-gated/src/Trackdub.App.Avalonia/App.axaml.cs#L186) runs, in order, on the UI thread: eager DI build (`ValidateOnBuild=true` instantiates all singletons) → language apply → theme apply → GPU-admission wait → license init → MainWindow resolve. Each step waits for the previous; none overlap. Fix direction: show the window first, defer/parallelize everything that isn't needed for first paint.

**2. ~1.3s PDH GPU-counter stall inside that chain.** The GPU admission binding constructs [WindowsProcessGpuMemoryReader.cs:55](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Composition/Runtime/WindowsProcessGpuMemoryReader.cs#L55), whose constructor forces a performance-counter warmup the authors measured at **~1300ms on a cold Windows host** (:61-66). It's already on `Task.Run` via [SessionPoolBootstrap.cs:35](file:///D:/Dev/Trackdub_Workspace/Trackdub-gated/src/Trackdub.App.Avalonia/Services/SessionPoolBootstrap.cs#L35), but [App.axaml.cs:244](file:///D:/Dev/Trackdub_Workspace/Trackdub-gated/src/Trackdub.App.Avalonia/App.axaml.cs#L244) blocks on it *before window construction* — it only needs to finish before first inference, not before first paint.

**3. Settings JSON loaded twice, synchronously, pre-window.** [AvaloniaAppLanguageService.cs:50-54](file:///D:/Dev/Trackdub_Workspace/Trackdub-gated/src/Trackdub.App.Avalonia/Services/AvaloniaAppLanguageService.cs#L50) blocks on `Task.Run(LoadAsync).GetResult()`; the theme service repeats the identical load with no shared cache. The language ctor also scans satellite assemblies during the eager DI build (:34). One cached settings read would serve both.

**4. License init blocks the UI thread.** [App.axaml.cs:278](file:///D:/Dev/Trackdub_Workspace/Trackdub-gated/src/Trackdub.App.Avalonia/App.axaml.cs#L278) — `InitializeAsync(...).GetAwaiter().GetResult()` (token file + hardware fingerprint) before any ViewModel exists. Fail-open by design, so it can be deferred or time-boxed.

**5. First-paint contention.** Resolving `MainWindow` builds a 30+ dependency ViewModel whose ctor fires three background jobs — model inventory + EP capability probe, ffmpeg health check, and a **network update-feed check** — all competing with first paint (agent-verified: `AvaloniaMainWindowViewModel.cs:206-209`). First launch additionally does a full model-inventory scan before showing the modal onboarding dialog (`StarterPackOnboardingCoordinator.cs:49-74`).

## P1 — First inference per process (core)

**6. Every model graph is fully SHA-256 hashed on first use, per process.** [SessionPoolKey.cs:471-573](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Inference.Onnx/Pool/SessionPoolKey.cs#L471) streams the entire `.onnx` file to build the pool key; the digest cache is in-process only ((path, length, mtime) keyed). CosyVoice's 9 graphs, Chatterbox/LatentSync's 4 each → multi-GB of disk reads on cold start before sessions even load. Fix direction: persist the digest cache, or reuse the download-verification hash (see #8).

**7. First TRT session pays engine compilation + native-DLL hashing.** [OnnxExecutionSessionFactory.cs:1029-1047](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Inference.Onnx/OnnxExecutionSessionFactory.cs#L1029) preloads the TensorRT dependency and hashes it for the cache fingerprint. Engine cache persists, but the fingerprint includes DLL hashes — driver/ORT updates invalidate it and re-trigger a full engine build. Largely inherent; flagging so it's a known, measured cost rather than a mystery stall.

## P2 — Every run / every stage

**8. Three independent hashing subsystems, no shared cache.** The download orchestrator **re-verifies all required file hashes even when the model is Installed/Ready** on every ensure call ([ModelDownloadOrchestrator.cs:107-126](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Composition/Runtime/ModelDownloadOrchestrator.cs#L107)); diarization **re-hashes the full SortFormer model on every run** ([DiarizationStageHandler.cs:265](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Application/Transcripts/DiarizationStageHandler.cs#L265), called from :157 — the cache-record fast path exists only in preflight); then the pool key hashes the graph again (#6). One (path, length, mtime)-keyed digest service would eliminate most of this I/O.

**9. ffprobe spawns per call with no snapshot cache, and the resolver re-scans the filesystem every time.** [FfmpegMediaProbe.cs:35-54](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Media/Probe/FfmpegMediaProbe.cs#L35) spawns ffprobe for every probe (ingest, project open, export, per timeline placement). Worse, [FfmpegToolResolver.cs:35-88](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Media/Process/FfmpegToolResolver.cs#L35) has zero memoization: every resolve re-scans PATH and candidate directories, ffprobe resolution first re-resolves ffmpeg (double scan), and the fallback can **recursively search Program Files** (:73-85). Cache the resolved tool paths and probe snapshots keyed by (path, size, mtime).

**10. Project opened 2+ times per stage.** The stage gate opens the project for artifact-resume checks ([DubbingPipelineEngine.StageLoop.cs:180-184](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Application/Dubbing/DubbingPipelineEngine.StageLoop.cs#L180)), then each stage workflow opens it again — each open issues ~15+ SQLite queries (agent-verified: `StageWorkflows.cs:146,179,250…`, `TranscriptProjectStateService.cs:252`). Share one open snapshot per run.

**11. Ingest reads the source media three times.** [ProjectMediaIngestService.cs:49-102](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Application/Projects/ProjectMediaIngestService.cs#L49) (agent-verified): full-source SHA-256 + **two sequential full-source ffmpeg extractions** (48kHz and 44.1kHz). One extraction + resample, or parallel passes, halves the wall time. First run per project.

**12. Model downloads are strictly sequential across files.** [ModelDownloadOrchestrator.cs:159](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Composition/Runtime/ModelDownloadOrchestrator.cs#L159) — intra-file range parallelism exists, but a 9-file CosyVoice bundle downloads one file at a time. Bounded cross-file parallelism (2-3) is the obvious win.

**13. Startup warmup is sequential across stages and providers.** [ResidentWarmupService.cs:67](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Composition/Pipeline/ResidentWarmupService.cs#L67) and :131 — total warmup = sum of all session builds. Independent stages/providers could warm concurrently (VRAM admission already guards memory).

**14. Runtime selections computed twice per run** — preflight and post-preflight ([DubbingPipelineEngine.cs:261,507](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Application/Dubbing/DubbingPipelineEngine.cs#L261), agent-verified). Mostly cached, but the post-provision path invalidates and re-evaluates readiness.

## P3 — Targeted serialization

**15. Translation: tokenizer reloaded per call, segments strictly sequential.** [MadladTranslationEngine.cs:56](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Inference.Onnx/Madlad/MadladTranslationEngine.cs#L56) reloads the tokenizer every `TranslateAsync` (and again on CPU fallback, :88); :95 translates segment-by-segment with no batching. Same tokenizer pattern in OpusMt (:59, agent-verified). Kokoro/Whisper/Qwen3 already cache theirs statically — extend that here.

**16. CosyVoice serializes all synthesis behind one gate.** [CosyVoiceTtsEngine.cs:75-118](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Inference.Onnx/CosyVoice/CosyVoiceTtsEngine.cs#L75) holds `sessionGate` across the entire synchronous `Synthesize` — concurrent segment renders queue even though residency pins ≠ exclusivity (the comment at :85 says as much).

**17. VRAM-pressure admission: 50ms poll + warm-session eviction churn.** [InferenceSessionPool.Admission.cs:343-378](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Inference.Onnx/Pool/InferenceSessionPool.Admission.cs#L343) re-reads PDH and re-takes `creationLock` every 50ms while blocked (fail-fast ~5s empty / ~60s busy). Under pressure it evicts warm sessions, which later paid full re-creation cost. The pin-retry path adds 50-400ms backoffs with create/dispose churn ([OnnxExecutionSessionFactory.Pooled.cs:244-277](file:///D:/Dev/Trackdub_Workspace/Trackdub/src/Trackdub.Inference.Onnx/OnnxExecutionSessionFactory.Pooled.cs#L244)). Bounded and logged by design — flagging as the wait-time source on small-VRAM machines, not a bug.

## Already well-optimized — deliberately not flagged

Next-stage warmup lookahead and TTS prefetch overlap; readiness probing is cached + DOP-3; file fingerprints use a size+mtime LRU cache; process stdout/stderr drain fully async with `WaitForExitAsync`; downloads resume with intra-file range parallelism; settings init, model warmup, and libmpv prewarm are already deferred past first paint.

---

**Biggest levers, in order:** (a) break the pre-window serial chain and move the PDH wait to just-before-first-inference (#1-4) — likely 1.5-3s off every launch; (b) unify model hashing into one persisted cache (#6, #8) — seconds off every cold run; (c) memoize ffmpeg tool resolution + probe snapshots (#9) — removes filesystem scans and process spawns from every stage.
