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

