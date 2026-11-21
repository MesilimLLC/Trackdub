using System.Collections.Concurrent;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Trackdub.Application.Artifacts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

public sealed record StartTtsStageRequest(
    Guid ProjectId,
    MediaAsset MediaAsset,
    Guid SpeakerId,
    string TargetLanguage,
    VoiceAssignment VoiceAssignment,
    IReadOnlyList<TranscriptSegment> TranscriptSegments,
    IReadOnlyList<TranslatedSegment> TranslatedSegments,
    IReadOnlySet<int>? SegmentIndices = null,
    string? PreferredModelAlias = null,
    IReadOnlyList<ProjectArtifact>? ProjectArtifacts = null,
    bool UseReferenceClipForVoiceCloning = false,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record StartTtsStageResult(
    StageRunRecord StageRun,
    IReadOnlyList<TtsTake> Takes);

public sealed class StartTtsStageHandler(
    ITtsEngine ttsEngine,
    IVoiceCatalog voiceCatalog,
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository,
    ITtsTakeRepository ttsTakeRepository,
    IProjectStageRunStore stageRunStore,
    DurationAnalysisService? durationAnalysisService = null,
    IAudioTimeStretchService? audioTimeStretchService = null,
    TtsTimingOptions? timingOptions = null,
    ITtsAudioPostProcessor? ttsAudioPostProcessor = null,
    IConsentService? consentService = null,
    ISpeakerConsentService? speakerConsentService = null,
    IVoiceCloneAuditLog? voiceCloneAuditLog = null,
    IReferenceClipAnalyzer? referenceClipAnalyzer = null,
    IApplicationLogger? logger = null,
    PipelineDegradationWriter? degradationWriter = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IAtomicRevisionCommitBoundary? commitBoundary = null,
    TtsExecutionOptions? executionOptions = null,
    ITtsAcceleratorPlacementResolver? placementResolver = null)
    : IDisposable
{
    private const string TtsAudioPostProcessVersion = "tts-audio-trim-v1";

    // Appended to render identities only when AutoStretchToLimit is on, so identities for the default
    // policy stay what they were and existing takes keep matching.
    private const string AutoStretchToLimitIdentity = "auto-stretch-to-limit";

    private readonly DurationAnalysisService durationAnalysisService = durationAnalysisService ?? new DurationAnalysisService();
    private readonly TtsTimingOptions timingOptions = (timingOptions ?? TtsTimingOptions.Default).Normalize();
    private readonly TtsExecutionOptions executionOptions = executionOptions ?? TtsExecutionOptions.Default;
    private readonly SemaphoreSlim persistenceGate = new(1, 1);

    // Placement only tunes concurrency, so a planning failure falls back to the host-wide bound
    // rather than failing the stage; synthesis surfaces real planning errors itself.
    private async Task<TtsAcceleratorPlacement?> TryResolvePlacementAsync(
        InferenceRequestOptions options,
        string? languageCode,
        CancellationToken cancellationToken)
    {
        if (placementResolver is null)
        {
            return null;
        }

        try
        {
            return await placementResolver
                .ResolvePlacementAsync(options, languageCode, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or IOException)
        {
            logger?.LogWarning("TTS placement could not be resolved; using the host-wide VRAM bound.", ex);
            return null;
        }
    }

    public Task<StartTtsStageResult> HandleAsync(
        StartTtsStageRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null) =>
        HandleAsync(request, cancellationToken, progress, prefetch: null);

    internal async Task<StartTtsStageResult> HandleAsync(
        StartTtsStageRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress,
        TtsStreamingPrefetch? prefetch)
    {
        ArgumentNullException.ThrowIfNull(request);

        // Create the stage run FIRST so that any prerequisite-resolution failure
        // (voice catalog lookup, reference clip analysis, consent check) is recorded
        // as a Failed StageRun rather than escaping unobserved.
        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, request.ProjectId, StageNames.Tts, cancellationToken)
            .ConfigureAwait(false);

        bool isVoiceCloning = request.UseReferenceClipForVoiceCloning &&
            request.VoiceAssignment.ReferenceClipArtifactId is not null;
        VoiceCatalogEntry voice;
        VoiceCloneReference? voiceCloneReference;
        try
        {
            voice = ResolveVoice(request, isVoiceCloning);

            // Inside the guarded block so cancellation here still records a Canceled stage run.
            await WriteCloneModelSubstitutedDegradationAsync(request, voice, isVoiceCloning, stageRun.Id, cancellationToken)
                .ConfigureAwait(false);

            // Resolve and validate the reference clip once for the entire batch so that the
            // audio analysis is not repeated for every synthesized segment.
            voiceCloneReference = isVoiceCloning
                ? await ResolveVoiceCloneReferenceAsync(request, cancellationToken).ConfigureAwait(false)
                : null;
        }
        catch (OperationCanceledException)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, ttsEngine, "TTS canceled.", CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex)
        {
            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, ttsEngine, ex.Message, cancellationToken, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }

        var takes = new ConcurrentBag<TtsTake>();
        try
        {
            ConcurrentDictionary<string, byte> reservedArtifactRelativePaths
                = await BuildReservedArtifactRelativePathsAsync(request, cancellationToken).ConfigureAwait(false);
            Dictionary<int, TranscriptSegment> transcriptSegmentsByIndex = request.TranscriptSegments
                .Where(segment => segment.SpeakerId == request.SpeakerId)
                .ToDictionary(segment => segment.SegmentIndex);
            IEnumerable<TranslatedSegment> targetSegmentsQuery = request.TranslatedSegments
                .OrderBy(segment => segment.SegmentIndex)
                .Where(segment => transcriptSegmentsByIndex.ContainsKey(segment.SegmentIndex));

            if (request.SegmentIndices is { Count: > 0 } requestedIndices)
            {
                targetSegmentsQuery = targetSegmentsQuery.Where(segment => requestedIndices.Contains(segment.SegmentIndex));
            }

            // Materialize before the loop so we know whether any segments were actually
            // targeted for this speaker before deciding to fail on language incompatibility.
            TranslatedSegment[] targetSegments = targetSegmentsQuery.ToArray();
            bool voiceCompatible = IsVoiceLanguageCompatible(voice.LanguageCode, request.TargetLanguage);
            PipelineProgressReporter.Phase(
                progress,
                StageNames.Tts,
                "Preparing segments",
                $"{targetSegments.Length} segment(s) queued.");
            if (!voiceCompatible && targetSegments.Length > 0 && degradationWriter is not null)
            {
                try
                {
                    await degradationWriter.WriteAsync(
                        new PipelineDegradationRecord(
                            StageNames.Tts,
                            "TTS_LANGUAGE_UNSUPPORTED",
                            $"Voice '{voice.DisplayName}' supports '{voice.LanguageCode.Split('-')[0].ToLowerInvariant()}' base language; '{request.TargetLanguage}' segments skipped.",
                            Detail: null,
                            SelectedFallback: null,
                            RecommendedAction: "Assign a voice that supports the target language.",
                            DateTimeOffset.UtcNow,
                            stageRun.Id),
                        request.ProjectId,
                        request.MediaAsset.Id,
                        cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    // Degradation write is best-effort; failure must not abort the language-skip fallback.
                }
            }

            if (voiceCompatible)
            {
                // Resolve the effective TTS degree of parallelism once per run: the configured
                // max, tightened by a VRAM-aware bound for the preferred model class. The alias
                // comes from the same CreateTtsRequestOptions used for synthesis, but when the
                // alias is not required the planner may fall back to another class, so this
                // bound is an estimate for that path, not a guarantee.
                int effectiveMaxConcurrency = await ResolveEffectiveConcurrencyAsync(
                    request,
                    CreateTtsRequestOptions(request, voice, isVoiceCloning),
                    cancellationToken).ConfigureAwait(false);

                var ctx = new SegmentProcessingContext(takes, targetSegments.Length);
                PipelineProgressReporter.Determinate(
                    progress,
                    StageNames.Tts,
                    0,
                    targetSegments.Length,
                    "Synthesizing segments",
                    $"{targetSegments.Length} segment(s) queued.");
                await Parallel.ForEachAsync(
                    targetSegments,
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = effectiveMaxConcurrency,
                        CancellationToken = cancellationToken
                    },
                    (translatedSegment, ct) => ProcessTranslatedSegmentAsync(
                        translatedSegment,
                        request,
                        stageRun.Id,
                        transcriptSegmentsByIndex,
                        reservedArtifactRelativePaths,
                        voice,
                        voiceCloneReference,
                        ctx,
                        progress,
                        prefetch,
                        ct)).ConfigureAwait(false);
            }

            // Only fail when segments were actually targeted but none synthesized due to
            // language mismatch. A speaker with no segments is a legitimate no-op.
            if (!voiceCompatible && targetSegments.Length > 0 && takes.IsEmpty)
            {
                throw new InvalidOperationException(
                    $"Voice '{voice.DisplayName}' does not support target language '{request.TargetLanguage}'. No audio was synthesized.");
            }

            stageRun = await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, ttsEngine, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, ttsEngine, "TTS canceled.", CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex)
        {
            if (!takes.IsEmpty)
            {
                await StageRunHelper
                    .PartiallyCompleteAsync(
                        stageRunStore,
                        stageRun,
                        ttsEngine,
                        $"TTS generated {takes.Count} take(s) before failing: {ex.Message}",
                        CancellationToken.None,
                        runtimePlanningPreferences,
                        logger)
                    .ConfigureAwait(false);
            }
            else
            {
                await StageRunHelper
                    .FailAsync(stageRunStore, stageRun, ttsEngine, ex.Message, cancellationToken, runtimePlanningPreferences, logger)
                    .ConfigureAwait(false);
            }

            throw;
        }

        return new StartTtsStageResult(stageRun, [.. takes.OrderBy(static take => take.SegmentIndex)]);
    }

    private async Task<int> ResolveEffectiveConcurrencyAsync(
        StartTtsStageRequest request,
        InferenceRequestOptions representativeOptions,
        CancellationToken cancellationToken)
    {
        string? modelAlias = representativeOptions.NormalizedPreferredModelAlias;
        // Prefer the planned placement: the provider synthesis will use and that device's
        // memory. Without one, only a required CPU/DNNL pin keeps the run off the
        // accelerator; a non-required pin can still fall back to a GPU provider.
        TtsAcceleratorPlacement? placement = await TryResolvePlacementAsync(
            representativeOptions,
            request.TargetLanguage,
            cancellationToken).ConfigureAwait(false);
        bool acceleratorRouted = placement?.AcceleratorRouted ??
            !(request.RequirePreferredExecutionProvider &&
              request.PreferredExecutionProvider is ExecutionProviderKind.Cpu or ExecutionProviderKind.Dnnl);
        long? deviceVramMb = placement?.DeviceVramMb;
        int effectiveMaxConcurrency = executionOptions.ResolveEffectiveConcurrency(
            modelAlias,
            acceleratorRouted,
            deviceVramMb);
        logger?.LogInformation(
            $"TTS parallelism: {effectiveMaxConcurrency} (configured {executionOptions.ConfiguredMaxConcurrency?.ToString() ?? "default"}, device VRAM {deviceVramMb?.ToString(CultureInfo.InvariantCulture) ?? "unknown"} MB, host max VRAM {executionOptions.MaxAcceleratorVramMb} MB, accelerator {acceleratorRouted}, model '{modelAlias ?? "default stock"}').");
        return effectiveMaxConcurrency;
    }

    /// <summary>
    /// Resolves the stock-voice render inputs for one speaker exactly as <see cref="HandleAsync(StartTtsStageRequest, CancellationToken, IProgress{PipelineProgressEvent}?)"/>
    /// will, so prefetched clips match the stage's render keys. Returns null for voice-cloned or
    /// language-incompatible speakers, which are never prefetched.
    /// </summary>
    internal async Task<(TtsPrefetchSpeaker Speaker, int MaxConcurrency)?> PreparePrefetchSpeakerAsync(
        StartTtsStageRequest request,
        CancellationToken cancellationToken)
    {
        if (request.UseReferenceClipForVoiceCloning && request.VoiceAssignment.ReferenceClipArtifactId is not null)
        {
            return null;
        }

        VoiceCatalogEntry voice = ResolveVoice(request, isVoiceCloning: false);
        if (!IsVoiceLanguageCompatible(voice.LanguageCode, request.TargetLanguage))
        {
            return null;
        }

        InferenceRequestOptions options = CreateTtsRequestOptions(request, voice, isVoiceCloning: false);
        int maxConcurrency = await ResolveEffectiveConcurrencyAsync(request, options, cancellationToken).ConfigureAwait(false);
        Dictionary<int, TranscriptSegment> sourceSegments = request.TranscriptSegments
            .Where(segment => segment.SpeakerId == request.SpeakerId)
            .ToDictionary(segment => segment.SegmentIndex);
        return (new TtsPrefetchSpeaker(request.TargetLanguage, voice, options, sourceSegments), maxConcurrency);
    }

    /// <summary>Shared mutable state threaded through <see cref="ProcessTranslatedSegmentAsync"/>.</summary>
    private sealed class SegmentProcessingContext(ConcurrentBag<TtsTake> takes, int totalSegments)
    {
        public readonly ConcurrentBag<TtsTake> Takes = takes;
        public readonly int TotalSegments = totalSegments;
        public int CompletedSegments; // written exclusively via Interlocked
    }

    private async ValueTask ProcessTranslatedSegmentAsync(
        TranslatedSegment translatedSegment,
        StartTtsStageRequest request,
        Guid stageRunId,
        Dictionary<int, TranscriptSegment> transcriptSegmentsByIndex,
        ConcurrentDictionary<string, byte> reservedArtifactRelativePaths,
        VoiceCatalogEntry voice,
        VoiceCloneReference? voiceCloneReference,
        SegmentProcessingContext ctx,
        IProgress<PipelineProgressEvent>? progress,
        TtsStreamingPrefetch? prefetch,
        CancellationToken cancellationToken)
    {
        void ReportProgress(int completed, string detail) =>
            PipelineProgressReporter.Determinate(
                progress,
                StageNames.Tts,
                completed,
                ctx.TotalSegments,
                "Synthesizing segments",
                detail,
                currentItemLabel: $"Segment {translatedSegment.SegmentIndex}");

        TranscriptSegment sourceSegment = transcriptSegmentsByIndex[translatedSegment.SegmentIndex];
        TtsTake take;
        try
        {
            take = await SynthesizeSegmentAsync(
                request,
                stageRunId,
                translatedSegment,
                sourceSegment,
                voice,
                voiceCloneReference,
                reservedArtifactRelativePaths,
                prefetch,
                cancellationToken).ConfigureAwait(false);
        }
        catch (TtsReferenceTextRequiredException)
        {
            await WriteMissingReferenceTextDegradationAsync(
                translatedSegment.SegmentIndex,
                request.ProjectId,
                request.MediaAsset.Id,
                stageRunId,
                cancellationToken).ConfigureAwait(false);

            int skipped = Interlocked.Increment(ref ctx.CompletedSegments);
            ReportProgress(skipped, $"Segment {skipped} of {ctx.TotalSegments} (skipped: missing reference text)");
            return;
        }

        ctx.Takes.Add(take);
        if (take.ArtifactId is Guid persistedArtifactId)
        {
            PipelineProgressReporter.Output(
                progress,
                StageNames.Tts,
                PipelineOutputKind.PlayableAudioPersisted,
                translatedSegment.SegmentIndex,
                segmentId: translatedSegment.Id,
                artifactId: persistedArtifactId,
                message: $"Playable audio persisted for segment {translatedSegment.SegmentIndex}.");
        }

        int completed = Interlocked.Increment(ref ctx.CompletedSegments);
        ReportProgress(completed, $"Segment {completed} of {ctx.TotalSegments}");
    }

    /// <summary>
    /// A clone-only TTS model (Chatterbox, CosyVoice, ...) requested on a run without voice
    /// cloning is swapped for a stock voice so the run still produces audio. Say so: the request
    /// named a model the run did not use.
    /// </summary>
    private async Task WriteCloneModelSubstitutedDegradationAsync(
        StartTtsStageRequest request,
        VoiceCatalogEntry voice,
        bool isVoiceCloning,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        string? requestedAlias = request.PreferredModelAlias?.Trim();
        if (isVoiceCloning ||
            string.IsNullOrWhiteSpace(requestedAlias) ||
            !ShouldForceStockTtsAlias(requestedAlias))
        {
            return;
        }

        string message =
            $"TTS model '{requestedAlias}' needs voice cloning, which is off for this run; " +
            $"used stock voice '{voice.DisplayName}' instead.";
        logger?.LogWarning(message);
        if (degradationWriter is null)
        {
            return;
        }

        try
        {
            await degradationWriter.WriteAsync(
                new PipelineDegradationRecord(
                    StageNames.Tts,
                    "TTS_CLONE_MODEL_SUBSTITUTED",
                    message,
                    Detail: null,
                    SelectedFallback: voice.VoiceId,
                    RecommendedAction: "Enable voice cloning (--voice-clone) to use this model, or choose a stock TTS model.",
                    DateTimeOffset.UtcNow,
                    stageRunId),
                request.ProjectId,
                request.MediaAsset.Id,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Degradation write is best-effort; failure must not abort the stock-voice fallback.
            logger?.LogWarning(
                $"Failed to record clone-model substitution degradation for project {request.ProjectId}.", ex);
        }
    }

    private async Task WriteMissingReferenceTextDegradationAsync(
        int segmentIndex,
        Guid projectId,
        Guid mediaAssetId,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        if (degradationWriter is null)
        {
            return;
        }

        try
        {
            await degradationWriter.WriteAsync(
                new PipelineDegradationRecord(
                    StageNames.Tts,
                    "TTS_REFERENCE_TEXT_MISSING",
                    $"Segment {segmentIndex} skipped: Qwen3-TTS Base cloning requires source transcript text for the reference clip.",
                    Detail: null,
                    SelectedFallback: null,
                    RecommendedAction: "Transcribe the source segment or choose a preset Qwen3 CustomVoice model.",
                    DateTimeOffset.UtcNow,
                    stageRunId),
                projectId,
                mediaAssetId,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Degradation write is best-effort; failure must not abort the skip fallback.
        }
    }

    public void Dispose() => persistenceGate.Dispose();

    private async Task<TtsTake> SynthesizeSegmentAsync(
        StartTtsStageRequest request,
        Guid stageRunId,
        TranslatedSegment translatedSegment,
        TranscriptSegment sourceSegment,
        VoiceCatalogEntry voice,
        VoiceCloneReference? voiceCloneReference,
        ConcurrentDictionary<string, byte> reservedArtifactRelativePaths,
        TtsStreamingPrefetch? prefetch,
        CancellationToken cancellationToken)
    {
        IAtomicRevisionCommitBoundary boundary = commitBoundary
            ?? throw new InvalidOperationException("Atomic take commit boundary is not configured.");
        var takeInput = new AtomicTakeInput(request.ProjectId, request.TargetLanguage,
            sourceSegment.TranscriptRevisionId, translatedSegment.TranslationRevisionId, translatedSegment);
        await boundary.ValidateTakeInputAsync(takeInput, cancellationToken).ConfigureAwait(false);
        InferenceRequestOptions options = CreateTtsRequestOptions(request, voice, voiceCloneReference is not null);
        string inputFingerprint = ComputeInputFingerprint(
            translatedSegment.Id,
            translatedSegment.Text,
            voice.VoiceId,
            request.TargetLanguage,
            voiceCloneReference?.ReferenceClipArtifactId,
            sourceSegment.EndSeconds - sourceSegment.StartSeconds,
            options.NormalizedPreferredModelAlias,
            options.NormalizedPreferredModelVariantAlias,
            timingOptions,
            request.VoiceAssignment.Id);
        TtsTake? cachedTake = await RunSerializedPersistenceAsync(
            ct => ttsTakeRepository.GetByFingerprintAsync(request.ProjectId, inputFingerprint, ct),
            cancellationToken)
            .ConfigureAwait(false);
        if (cachedTake is null)
        {
            // Backward compatibility: before the timing suffix was introduced, fingerprints were
            // the bare SHA-256 hash, which already encodes AutoStretchMaxOverrun. A match on that
            // legacy hash therefore proves the same timing limit produced the take, so reuse it
            // instead of re-synthesizing every segment after upgrade.
            string legacyFingerprint = TtsTimingResumeIdentity.WithoutSuffix(inputFingerprint);
            if (!string.Equals(legacyFingerprint, inputFingerprint, StringComparison.Ordinal))
            {
                cachedTake = await RunSerializedPersistenceAsync(
                    ct => ttsTakeRepository.GetByFingerprintAsync(request.ProjectId, legacyFingerprint, ct),
                    cancellationToken)
                    .ConfigureAwait(false);
            }
        }
        if (cachedTake is not null)
        {
            // Voice-cloning consent and audit must be enforced even for cache hits —
            // the policy gate and audit trail cannot be skipped because synthesis is skipped.
            // Both are gated behind persistenceGate so the consent check and its matching audit
            // entry are written atomically relative to other parallel synthesis tasks.
            if (voiceCloneReference is not null)
            {
                await RunSerializedPersistenceAsync(
                    async ct =>
                    {
                        await EnsureVoiceCloningConsentAsync(request.SpeakerId, request.ProjectId, ct).ConfigureAwait(false);
                        if (voiceCloneAuditLog is not null)
                        {
                            await voiceCloneAuditLog.AppendAsync(
                                new VoiceCloneAuditEntry(
                                    DateTimeOffset.UtcNow,
                                    consentService!.SessionId,
                                    voiceCloneReference.SpeakerId,
                                    voiceCloneReference.ReferenceClipArtifactId),
                                ct).ConfigureAwait(false);
                        }
                    },
                    cancellationToken).ConfigureAwait(false);
            }

            LogTtsTakeProvenance(
                translatedSegment.SegmentIndex,
                cachedTake,
                "TTS cache hit",
                options);
            return cachedTake;
        }

        IReadOnlyList<TtsTake> existingTakes = await RunSerializedPersistenceAsync(
            ct => ttsTakeRepository.GetBySegmentAsync(translatedSegment.Id, ct),
            cancellationToken)
            .ConfigureAwait(false);
        string relativePath = ReserveNextTtsTakeRelativePath(
            request.SpeakerId,
            translatedSegment.Id,
            existingTakes,
            reservedArtifactRelativePaths);
        Guid takeId = Guid.NewGuid();
        relativePath = relativePath[..^4] + $"-{takeId:N}.wav";

        if (voiceCloneReference is not null)
        {
            await RunSerializedPersistenceAsync(
                async ct =>
                {
                    await EnsureVoiceCloningConsentAsync(request.SpeakerId, request.ProjectId, ct).ConfigureAwait(false);
                    if (voiceCloneAuditLog is not null)
                    {
                        await voiceCloneAuditLog.AppendAsync(
                            new VoiceCloneAuditEntry(
                                DateTimeOffset.UtcNow,
                                consentService!.SessionId,
                                voiceCloneReference.SpeakerId,
                                voiceCloneReference.ReferenceClipArtifactId),
                            ct).ConfigureAwait(false);
                    }
                },
                cancellationToken).ConfigureAwait(false);
        }
        else if (request.VoiceAssignment.ReferenceClipArtifactId is null)
        {
            logger?.LogInformation(
                $"No reference clip assigned for speaker {request.SpeakerId:D}; using stock Kokoro TTS.");
        }

        VoiceCloneReference? effectiveCloneReference = voiceCloneReference;
        if (voiceCloneReference is not null &&
            (Qwen3TtsDefaults.IsBaseAlias(options.NormalizedPreferredModelAlias) ||
             CosyVoiceDefaults.IsCosyVoiceAlias(options.NormalizedPreferredModelAlias)))
        {
            string? referenceTranscript = sourceSegment.Text?.Trim();
            if (string.IsNullOrWhiteSpace(referenceTranscript))
            {
                throw new TtsReferenceTextRequiredException();
            }

            effectiveCloneReference = voiceCloneReference with { ReferenceTranscript = referenceTranscript };
        }

        await using var tx = new ArtifactWriteTransaction(artifactStore.CreateWriteHandle(relativePath));
        Directory.CreateDirectory(Path.GetDirectoryName(tx.TemporaryPath)!);
        TtsRenderOutput render;
        TtsPrefetchedClip? prefetched = effectiveCloneReference is null && prefetch is not null
            ? await prefetch.TryClaimAsync(
                ComputeRenderKey(translatedSegment.Text, request.TargetLanguage, voice, options, sourceSegment),
                cancellationToken).ConfigureAwait(false)
            : null;
        if (prefetched is not null)
        {
            using (prefetched)
            {
                File.Copy(prefetched.AudioPath, tx.TemporaryPath, overwrite: true);
            }

            render = prefetched.Output;
        }
        else
        {
            render = await RenderToPathAsync(
                translatedSegment.Text,
                request.TargetLanguage,
                voice,
                options,
                effectiveCloneReference,
                sourceSegment,
                tx.TemporaryPath,
                cancellationToken).ConfigureAwait(false);
        }

        FileFingerprint fingerprint = await fileFingerprintService
            .ComputeAsync(tx.TemporaryPath, cancellationToken)
            .ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            request.ProjectId,
            request.MediaAsset.Id,
            ArtifactKind.TtsTake,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            render.ArtifactDurationSeconds,
            render.ArtifactSampleRate,
            ChannelCount: 1,
            DateTimeOffset.UtcNow,
            stageRunId,
            $"tts:{render.ModelId}:{render.VoiceId}");
        string translatedTextHash = TtsTextHash.Compute(translatedSegment.SegmentIndex, translatedSegment.Text);
        TtsTake take = (voiceCloneReference is null
                ? TtsTake.CreateStock(
                    request.ProjectId,
                    request.VoiceAssignment.Id,
                    translatedSegment.Id,
                    translatedSegment.SegmentIndex,
                    translatedTextHash,
                    inputFingerprint)
                : TtsTake.CreateVoiceCloned(
                    request.ProjectId,
                    request.VoiceAssignment.Id,
                    voiceCloneReference.ReferenceClipArtifactId,
                    translatedSegment.Id,
                    translatedSegment.SegmentIndex,
                    translatedTextHash,
                    inputFingerprint))
            .Complete(
                artifact.Id,
                stageRunId,
                render.ArtifactDurationSamples,
                render.ArtifactSampleRate,
                render.Provider,
                render.ModelId,
                render.VoiceId,
                render.Analysis.OverrunRatio,
                render.PreStretchDurationSeconds,
                render.StretchRatioApplied,
                render.StretchMode,
                render.StretchEngine);
        take = take with { Id = takeId };
        await RunSerializedPersistenceAsync(
            ct => boundary.CommitTakeAsync(new AtomicTakeCommitRequest(takeInput, take),
                new PreparedCommitArtifact(artifact, tx.Handle), ct), cancellationToken).ConfigureAwait(false);
        LogTtsTakeProvenance(
            translatedSegment.SegmentIndex,
            take,
            "TTS synthesized",
            options);
        return take;
    }

    /// <summary>
    /// Produces the finished take audio for one segment at <paramref name="outputPath"/>: synthesis,
    /// post-processing, duration analysis and automatic stretching. Shared by the TTS stage and
    /// <see cref="TtsStreamingPrefetch"/>, so a prefetched clip is exactly what the stage would
    /// have rendered for the same inputs.
    /// </summary>
    internal async Task<TtsRenderOutput> RenderToPathAsync(
        string text,
        string targetLanguage,
        VoiceCatalogEntry voice,
        InferenceRequestOptions options,
        VoiceCloneReference? cloneReference,
        TranscriptSegment sourceSegment,
        string outputPath,
        CancellationToken cancellationToken)
    {
        TtsSynthesisResult result = await ttsEngine.SynthesizeAsync(
            new TtsSynthesisRequest(
                text,
                targetLanguage,
                voice,
                Options: options,
                VoiceCloneReference: cloneReference,
                TargetDurationSeconds: sourceSegment.EndSeconds - sourceSegment.StartSeconds),
            cancellationToken).ConfigureAwait(false);

        double? preStretchDurationSeconds = null;
        double? stretchRatioApplied = null;
        TtsStretchMode stretchMode = TtsStretchMode.None;
        TtsStretchEngine stretchEngine = TtsStretchEngine.None;
        await File.WriteAllBytesAsync(outputPath, result.WavBytes, cancellationToken)
            .ConfigureAwait(false);
        TtsAudioPostProcessResult postProcessResult = await PostProcessTemporaryTakeAsync(
            outputPath,
            result,
            cancellationToken).ConfigureAwait(false);
        int artifactDurationSamples = postProcessResult.DurationSamples;
        int artifactSampleRate = postProcessResult.SampleRate > 0 ? postProcessResult.SampleRate : result.SampleRate;
        double? rawDurationSeconds = postProcessResult.DurationSeconds ??
                                     (artifactSampleRate > 0
                                         ? (double)artifactDurationSamples / artifactSampleRate
                                         : null);
        double? artifactDurationSeconds = rawDurationSeconds;
        DurationAnalysisResult analysis = durationAnalysisService.Analyze(sourceSegment, rawDurationSeconds, timingOptions);

        if (analysis.AutoStretchTempoRatio is double tempoRatio)
        {
            if (audioTimeStretchService is null)
            {
                throw new InvalidOperationException("Audio time stretching is not configured.");
            }

            AudioTimeStretchResult stretchResult = await StretchTemporaryTakeAsync(
                outputPath,
                tempoRatio,
                cancellationToken).ConfigureAwait(false);
            preStretchDurationSeconds = rawDurationSeconds;
            stretchRatioApplied = tempoRatio;
            stretchMode = TtsStretchMode.Automatic;
            stretchEngine = stretchResult.Engine;
            artifactDurationSeconds = analysis.AutoStretchEligible
                ? analysis.OriginalDurationSeconds
                : rawDurationSeconds!.Value / tempoRatio;
            if (artifactSampleRate > 0)
            {
                artifactDurationSamples = Math.Max(1, (int)Math.Round(artifactDurationSeconds.Value * artifactSampleRate));
            }
        }

        return new TtsRenderOutput(
            result.ModelId,
            result.VoiceId,
            result.Provider,
            artifactDurationSamples,
            artifactSampleRate,
            artifactDurationSeconds,
            analysis,
            preStretchDurationSeconds,
            stretchRatioApplied,
            stretchMode,
            stretchEngine);
    }

    /// <summary>
    /// Identifies rendered audio by everything that shapes it (text, voice, model selection, source
    /// timing as the stretch target, post-processing) and nothing about which revision or segment
    /// row will own it, so audio rendered before the translation commits can be matched to the
    /// committed segment.
    /// </summary>
    internal string ComputeRenderKey(
        string text,
        string targetLanguage,
        VoiceCatalogEntry voice,
        InferenceRequestOptions options,
        TranscriptSegment sourceSegment)
    {
        string input = string.Concat(
            text, "\0",
            targetLanguage, "\0",
            voice.VoiceId, "\0",
            options.NormalizedPreferredModelAlias ?? string.Empty, "\0",
            options.RequirePreferredModelAlias ? "1" : "0", "\0",
            options.NormalizedPreferredModelVariantAlias ?? string.Empty, "\0",
            options.PreferredExecutionProvider ?? string.Empty, "\0",
            options.RequirePreferredExecutionProvider ? "1" : "0", "\0",
            sourceSegment.StartSeconds.ToString("G17", CultureInfo.InvariantCulture), "\0",
            sourceSegment.EndSeconds.ToString("G17", CultureInfo.InvariantCulture), "\0",
            timingOptions.AutoStretchMaxOverrun.ToString("G17", CultureInfo.InvariantCulture), "\0",
            timingOptions.MinimumStretchableDurationSeconds.ToString("G17", CultureInfo.InvariantCulture), "\0",
            timingOptions.EnableRubberbandStretch ? "1" : "0", "\0",
            timingOptions.RubberbandStretchThreshold.ToString("G17", CultureInfo.InvariantCulture), "\0",
            TtsAudioPostProcessVersion,
            timingOptions.AutoStretchToLimit ? "\0" + AutoStretchToLimitIdentity : string.Empty);
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(input))).ToLowerInvariant();
    }

    private void LogTtsTakeProvenance(
        int segmentIndex,
        TtsTake take,
        string source,
        InferenceRequestOptions options)
    {
        logger?.LogInformation(
            $"{source}: {PipelineRuntimeProvenanceFormatter.FormatTtsSegmentLogLine(
                segmentIndex,
                take.Provider,
                options.NormalizedPreferredModelAlias,
                take.ModelId,
                options.NormalizedPreferredModelVariantAlias,
                take.VoiceId)}");
    }

    private async Task<T> RunSerializedPersistenceAsync<T>(
        Func<CancellationToken, Task<T>> action,
        CancellationToken cancellationToken)
    {
        await persistenceGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await action(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            persistenceGate.Release();
        }
    }

    private async Task RunSerializedPersistenceAsync(
        Func<CancellationToken, Task> action,
        CancellationToken cancellationToken)
    {
        await persistenceGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await action(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            persistenceGate.Release();
        }
    }

    private async Task<TtsAudioPostProcessResult> PostProcessTemporaryTakeAsync(
        string temporaryPath,
        TtsSynthesisResult result,
        CancellationToken cancellationToken)
    {
        if (ttsAudioPostProcessor is null ||
            result.SampleRate <= 0 ||
            result.DurationSamples < 0)
        {
            return new TtsAudioPostProcessResult(result.DurationSamples, result.SampleRate);
        }

        TtsAudioPostProcessResult postProcessResult = await ttsAudioPostProcessor
            .ProcessAsync(
                new TtsAudioPostProcessRequest(
                    temporaryPath,
                    result.SampleRate,
                    result.DurationSamples),
                cancellationToken)
            .ConfigureAwait(false);
        return postProcessResult.DurationSamples >= 0 && postProcessResult.SampleRate > 0
            ? postProcessResult
            : new TtsAudioPostProcessResult(result.DurationSamples, result.SampleRate);
    }

    private async Task<ConcurrentDictionary<string, byte>> BuildReservedArtifactRelativePathsAsync(
        StartTtsStageRequest request,
        CancellationToken cancellationToken)
    {
        var reservedPaths = new ConcurrentDictionary<string, byte>(StringComparer.OrdinalIgnoreCase);
        foreach (ProjectArtifact artifact in request.ProjectArtifacts ?? [])
        {
            reservedPaths.TryAdd(artifact.RelativePath, 0);
        }

        IReadOnlyList<ProjectArtifact> currentArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(request.ProjectId, cancellationToken)
            .ConfigureAwait(false);
        foreach (ProjectArtifact artifact in currentArtifacts)
        {
            reservedPaths.TryAdd(artifact.RelativePath, 0);
        }

        return reservedPaths;
    }

    private static string ReserveNextTtsTakeRelativePath(
        Guid speakerId,
        Guid translatedSegmentId,
        IReadOnlyList<TtsTake> existingTakes,
        ConcurrentDictionary<string, byte> reservedArtifactRelativePaths)
    {
        int takeNumber = existingTakes.Count + 1;
        while (true)
        {
            string relativePath = ProjectArtifactPaths.GetTtsTakeRelativePath(
                speakerId,
                translatedSegmentId,
                takeNumber);
            if (reservedArtifactRelativePaths.TryAdd(relativePath, 0))
            {
                return relativePath;
            }

            takeNumber++;
        }
    }

    private async Task<AudioTimeStretchResult> StretchTemporaryTakeAsync(
        string temporaryPath,
        double tempoRatio,
        CancellationToken cancellationToken)
    {
        string stretchedPath = $"{temporaryPath}.stretch.wav";
        try
        {
            AudioTimeStretchResult result = await audioTimeStretchService!.StretchAsync(
                new AudioTimeStretchRequest(
                    temporaryPath,
                    stretchedPath,
                    tempoRatio,
                    timingOptions.EnableRubberbandStretch,
                    timingOptions.RubberbandStretchThreshold),
                cancellationToken).ConfigureAwait(false);
            File.Move(stretchedPath, temporaryPath, overwrite: true);
            return result;
        }
        finally
        {
            if (File.Exists(stretchedPath))
            {
                File.Delete(stretchedPath);
            }
        }
    }

    private VoiceCatalogEntry ResolveVoice(StartTtsStageRequest request, bool isVoiceCloning)
    {
        if (isVoiceCloning)
        {
            return new VoiceCatalogEntry(
                $"voice-clone:{request.SpeakerId:D}",
                request.TargetLanguage,
                "synthetic",
                "Voice clone");
        }

        string voiceId = ResolveVoiceId(request.VoiceAssignment);
        if (Qwen3TtsDefaults.IsPresetVoiceId(voiceId) &&
            !Qwen3TtsDefaults.SupportsLanguage(request.TargetLanguage))
        {
            throw new InvalidOperationException(
                $"Qwen3 does not speak target language '{request.TargetLanguage}'.");
        }

        // Qwen3 CustomVoice serves stock voices for languages Kokoro does not cover, and any run
        // that names a CustomVoice model. Keep an explicitly assigned preset (qwen3:<speaker>);
        // otherwise use the preset whose native language matches the target.
        bool usesQwen3StockVoices =
            Qwen3TtsDefaults.SupportsLanguage(request.TargetLanguage) &&
            ((ShouldForceStockTtsAlias(request.PreferredModelAlias) &&
              IsNonEnglishSpanishLanguage(request.TargetLanguage)) ||
             Qwen3TtsDefaults.IsCustomVoiceAlias(request.PreferredModelAlias?.Trim()));
        if (usesQwen3StockVoices)
        {
            string presetVoiceId = Qwen3TtsDefaults.IsPresetVoiceId(voiceId)
                ? voiceId.Trim()
                : Qwen3TtsDefaults.ResolveDefaultPresetVoiceId(request.TargetLanguage);
            if (voiceCatalog.TryGetVoice(presetVoiceId, out VoiceCatalogEntry? preset))
            {
                return preset;
            }

            string speaker = presetVoiceId[Qwen3TtsDefaults.PresetVoicePrefix.Length..];
            return new VoiceCatalogEntry(
                presetVoiceId,
                request.TargetLanguage,
                "synthetic",
                char.ToUpperInvariant(speaker[0]) + speaker[1..]);
        }

        if (voiceCatalog.TryGetVoice(voiceId, out VoiceCatalogEntry? voice))
        {
            return voice;
        }

        throw new InvalidOperationException($"Voicepack '{voiceId}' is not available.");
    }

    private async Task<VoiceCloneReference?> ResolveVoiceCloneReferenceAsync(
        StartTtsStageRequest request,
        CancellationToken cancellationToken)
    {
        if (request.VoiceAssignment.ReferenceClipArtifactId is not Guid referenceClipArtifactId)
        {
            return null;
        }

        ProjectArtifact referenceArtifact = (request.ProjectArtifacts ?? [])
            .FirstOrDefault(artifact => artifact.Id == referenceClipArtifactId && artifact.Kind == ArtifactKind.ReferenceClip)
            ?? throw new InvalidOperationException("The selected reference clip artifact was not found.");
        string referenceClipPath = artifactStore.GetPath(referenceArtifact.RelativePath);
        if (!File.Exists(referenceClipPath))
        {
            throw new FileNotFoundException("The selected reference clip file was not found.", referenceClipPath);
        }

        if (referenceClipAnalyzer is null)
        {
            throw new InvalidOperationException("Reference clip active-speech validation is not configured.");
        }

        ReferenceClipAnalysis analysis = await referenceClipAnalyzer
            .AnalyzeAsync(referenceClipPath, cancellationToken)
            .ConfigureAwait(false);
        if (analysis.ActiveSpeechSeconds < ReferenceClipPolicy.MinimumActiveSpeechSeconds)
        {
            throw new InvalidOperationException(
                $"Reference clip needs at least {ReferenceClipPolicy.MinimumActiveSpeechSeconds:F1} seconds of active speech; detected {analysis.ActiveSpeechSeconds:F2} seconds.");
        }

        if (analysis.HasRecommendedMaximumWarning)
        {
            logger?.LogWarning(
                $"Reference clip for speaker {request.SpeakerId:D} has {analysis.ActiveSpeechSeconds:F2}s of active speech; F5/Chatterbox voice cloning is best with 3-10s.");
        }

        return new VoiceCloneReference(
            request.SpeakerId,
            referenceClipArtifactId,
            referenceClipPath,
            analysis.TotalDurationSeconds,
            analysis.ActiveSpeechSeconds);
    }

    private async Task EnsureVoiceCloningConsentAsync(
        Guid speakerId,
        Guid projectId,
        CancellationToken cancellationToken)
    {
        if (speakerConsentService is not null)
        {
            if (await speakerConsentService.IsConsentGrantedAsync(speakerId, cancellationToken).ConfigureAwait(false))
            {
                consentService?.GrantVoiceCloningConsent();
                return;
            }

            if (consentService?.IsVoiceCloningConsentGranted == true)
            {
                await speakerConsentService.RecordConsentAsync(
                    projectId,
                    speakerId,
                    isThirdPartyConsent: false,
                    notes: null,
                    cancellationToken).ConfigureAwait(false);
                return;
            }
        }
        else if (consentService?.IsVoiceCloningConsentGranted == true)
        {
            return;
        }

        throw new ConsentRequiredException();
    }

    private static InferenceRequestOptions CreateTtsRequestOptions(
        StartTtsStageRequest request,
        VoiceCatalogEntry voice,
        bool isVoiceCloning)
    {
        if (isVoiceCloning)
        {
            return new InferenceRequestOptions(
                string.IsNullOrWhiteSpace(request.PreferredModelAlias)
                    ? VoiceCloningDefaults.ResolveDefaultCloneModelAlias(request.TargetLanguage)
                    : request.PreferredModelAlias.Trim(),
                RequirePreferredModelAlias: true,
                PreferredExecutionProvider: request.PreferredExecutionProvider?.ToString(),
                RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: request.PreferredModelVariantAlias);
        }

        string? trimmedAlias = request.PreferredModelAlias?.Trim();
        bool isCosyVoiceAlias = TtsModelOverrideSettings.IsCosyVoiceAlias(trimmedAlias);
        bool isQwen3Alias = Qwen3TtsDefaults.IsAnyQwen3Alias(trimmedAlias);
        bool shouldForceStockTtsAlias = ShouldForceStockTtsAlias(trimmedAlias);
        bool shouldRequireExplicitAlias =
            shouldForceStockTtsAlias ||
            isCosyVoiceAlias ||
            isQwen3Alias;
        // Forced-stock path has no reference/speaker audio (isVoiceCloning is false here),
        // so it must stay on a stock text-only model. CosyVoice is a voice-cloning model
        // that requires a reference clip (see CosyVoiceTtsEngine), so it cannot serve this
        // path; routing a non-en/es language to it would fail at inference.
        // A Qwen3 preset voice (qwen3:<speaker>) can only be spoken by a Qwen3 CustomVoice model,
        // so route to one even on English/Spanish targets that would otherwise default to Kokoro.
        bool isQwen3PresetVoice = Qwen3TtsDefaults.IsPresetVoiceId(voice.VoiceId);
        string? preferredAlias = isQwen3PresetVoice && !Qwen3TtsDefaults.IsCustomVoiceAlias(trimmedAlias)
            ? Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null)
            : shouldForceStockTtsAlias
                ? IsNonEnglishSpanishLanguage(request.TargetLanguage)
                    ? Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null)
                    : StockTtsDefaults.KokoroPrimaryAlias
                : trimmedAlias;
        return new InferenceRequestOptions(
            preferredAlias,
            RequirePreferredModelAlias: shouldRequireExplicitAlias || isQwen3PresetVoice,
            PreferredExecutionProvider: request.PreferredExecutionProvider?.ToString(),
            RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
            PreferredModelVariantAlias: request.PreferredModelVariantAlias);
    }

    // Recognizes exactly the same clone-only alias set (Chatterbox + CosyVoice + Qwen3-base + F5)
    // as TtsOrchestrationService's non-clone substitution guard, via the shared predicate, so the
    // two notions of "clone-only alias" cannot drift.
    private static bool IsVoiceCloningAlias(string? alias) =>
        NormalizeAlias(alias) is string normalizedAlias &&
        VoiceCloningDefaults.IsCloneOnlyModelAlias(normalizedAlias);

    private static bool ShouldForceStockTtsAlias(string? alias)
    {
        string? trimmedAlias = alias?.Trim();
        return string.IsNullOrWhiteSpace(trimmedAlias) ||
               IsVoiceCloningAlias(trimmedAlias);
    }

    /// <summary>
    /// True when target language needs CosyVoice (non-en/es) — Kokoro only supports en/es.
    /// </summary>
    private static bool IsNonEnglishSpanishLanguage(string? languageCode)
    {
        if (string.IsNullOrWhiteSpace(languageCode)) return false;
        string lang = languageCode.Trim().Split('-')[0].ToLowerInvariant();
        return lang != "en" && lang != "es";
    }

    private static string? NormalizeAlias(string? alias) =>
        string.IsNullOrWhiteSpace(alias)
            ? null
            : alias.Trim();

    private static string ResolveVoiceId(VoiceAssignment assignment) =>
        string.IsNullOrWhiteSpace(assignment.VoiceVariant)
            ? assignment.VoiceModelId
            : assignment.VoiceVariant;

    public static bool HasDurationWarning(TtsTake take) =>
        take.DurationOverrunRatio is > 0.10d;

    private static bool IsVoiceLanguageCompatible(string voiceLanguageCode, string targetLanguage)
    {
        string trimmedVoiceCode = voiceLanguageCode.Trim();
        if (string.IsNullOrEmpty(trimmedVoiceCode) ||
            trimmedVoiceCode.Equals("mul", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        static string Normalize(string lang) => lang.Trim().Split('-')[0].ToLowerInvariant();
        return Normalize(trimmedVoiceCode) == Normalize(targetLanguage);
    }

    private static string ComputeInputFingerprint(
        Guid segmentId,
        string text,
        string voiceId,
        string targetLanguage,
        Guid? referenceClipArtifactId,
        double targetDurationSeconds,
        string? modelAlias,
        string? modelVariantAlias,
        TtsTimingOptions timingOptions,
        Guid voiceAssignmentId)
    {
        string input = string.Concat(
            segmentId.ToString("D"), "\0",
            text, "\0",
            voiceId, "\0",
            targetLanguage, "\0",
            referenceClipArtifactId?.ToString("D") ?? string.Empty, "\0",
            targetDurationSeconds.ToString("G17", CultureInfo.InvariantCulture), "\0",
            modelAlias ?? string.Empty, "\0",
            modelVariantAlias ?? string.Empty, "\0",
            timingOptions.AutoStretchMaxOverrun.ToString("G17", CultureInfo.InvariantCulture), "\0",
            timingOptions.MinimumStretchableDurationSeconds.ToString("G17", CultureInfo.InvariantCulture), "\0",
            timingOptions.EnableRubberbandStretch ? "1" : "0", "\0",
            timingOptions.RubberbandStretchThreshold.ToString("G17", CultureInfo.InvariantCulture), "\0",
            TtsAudioPostProcessVersion, "\0",
            voiceAssignmentId.ToString("D"),
            timingOptions.AutoStretchToLimit ? "\0" + AutoStretchToLimitIdentity : string.Empty);
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(input));
        return TtsTimingResumeIdentity.Append(Convert.ToHexString(hash).ToLowerInvariant(), timingOptions.AutoStretchMaxOverrun);
    }
}
