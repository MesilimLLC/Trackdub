using Trackdub.Application.Dubbing;
using Trackdub.Application.Projects;
using Trackdub.Application.Mixing;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Contracts.Projects;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

public static class StageArtifactResumeEvaluator
{
    public static StageRunRecord? GetLatestSuccessfulRun(
        IReadOnlyList<StageRunRecord> runs,
        string stageName) =>
        runs
            .Where(r => string.Equals(r.StageName, stageName, StringComparison.OrdinalIgnoreCase))
            .Where(IsSuccessfulRun)
            .OrderByDescending(static r => r.CompletedAtUtc ?? r.StartedAtUtc)
            .FirstOrDefault();

    public static bool RuntimeMatchesSnapshot(
        StageRunRecord run,
        string stageName,
        IReadOnlyDictionary<string, string> snapshot)
    {
        if (snapshot.TryGetValue($"Model:{stageName}", out string? expectedModel) &&
            !string.IsNullOrWhiteSpace(expectedModel))
        {
            string? actualAlias = run.RuntimeInfo?.ModelAlias;
            if (string.IsNullOrWhiteSpace(actualAlias) ||
                !string.Equals(actualAlias, expectedModel, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        if (snapshot.TryGetValue($"ModelVariant:{stageName}", out string? expectedVariant) &&
            !string.IsNullOrWhiteSpace(expectedVariant))
        {
            string? actualVariant = run.RuntimeInfo?.ModelVariant;
            if (string.IsNullOrWhiteSpace(actualVariant) ||
                !string.Equals(actualVariant, expectedVariant, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        if (snapshot.TryGetValue($"ModelId:{stageName}", out string? expectedModelId) &&
            !string.IsNullOrWhiteSpace(expectedModelId))
        {
            string? actualModelId = run.RuntimeInfo?.ModelId;
            if (string.IsNullOrWhiteSpace(actualModelId) ||
                !string.Equals(actualModelId, expectedModelId, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        if (snapshot.TryGetValue($"Provider:{stageName}", out string? expectedProvider) &&
            !string.IsNullOrWhiteSpace(expectedProvider))
        {
            string? actualProvider = run.RuntimeInfo?.RequestedProvider;
            if (string.IsNullOrWhiteSpace(actualProvider))
            {
                // The snapshot requests a specific provider but the prior run recorded none,
                // so its hardware decisions cannot be proven to match.
                return false;
            }

            // Cloud engines report "cloud" regardless of local hardware overrides; the
            // comparison only constrains runs that actually honored a provider selection.
            if (!string.Equals(actualProvider, "cloud", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(actualProvider.Trim(), expectedProvider.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        return true;
    }

    public static bool OutputsPresent(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        string stageName,
        string projectRootPath,
        string? targetLanguageCode = null,
        string? exportPath = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(artifactStore);
        ArgumentException.ThrowIfNullOrWhiteSpace(stageName);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectRootPath);

        IReadOnlyList<ProjectArtifact> artifacts = state.ProjectState.Artifacts;

        return stageName switch
        {
            _ when string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase) =>
                ArtifactExists(artifactStore, TranscriptWorkflowUtilities.GetLatestArtifactByKind(artifacts, ArtifactKind.SpeechRegions)),

            _ when string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase) =>
                AsrOutputsPresent(state, artifactStore, artifacts),

            _ when string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase) =>
                GetLatestSuccessfulRun(state.StageRuns, StageNames.TextRefinementAsr) is not null &&
                TextRefinementOutputsPresent(state, artifactStore),

            _ when string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase) =>
                DiarizationOutputsPresent(state, artifactStore, artifacts),

            _ when string.Equals(stageName, StageNames.SpeakerAssignment, StringComparison.OrdinalIgnoreCase) =>
                SpeakerAssignmentOutputsPresent(state, artifactStore, artifacts),

            _ when string.Equals(stageName, StageNames.Separation, StringComparison.OrdinalIgnoreCase) =>
                ArtifactExists(artifactStore, TranscriptWorkflowUtilities.GetLatestAcceptedVocalStem(artifacts, state.StageRuns)),

            _ when string.Equals(stageName, StageNames.Translation, StringComparison.OrdinalIgnoreCase) =>
                state.CurrentTranslationRevision is not null &&
                !state.IsTranslationStale &&
                (targetLanguageCode is null ||
                 string.Equals(state.SelectedTranslationTargetLanguage, targetLanguageCode, StringComparison.OrdinalIgnoreCase)),

            _ when string.Equals(stageName, StageNames.Tts, StringComparison.OrdinalIgnoreCase) =>
                TtsOutputsPresent(state, artifactStore, artifacts),

            _ when string.Equals(stageName, StageNames.LipSync, StringComparison.OrdinalIgnoreCase) =>
                LipSyncOutputsPresent(state.StageRuns, artifacts, artifactStore),

            _ when string.Equals(stageName, StageNames.AudioPreparation, StringComparison.OrdinalIgnoreCase) =>
                AudioPreparationOutputsPresent(state.StageRuns, artifacts, artifactStore),

            _ when string.Equals(stageName, StageNames.SpeechEnhancement, StringComparison.OrdinalIgnoreCase) =>
                StageRunScopedArtifactsPresent(
                    state.StageRuns,
                    artifacts,
                    artifactStore,
                    StageNames.SpeechEnhancement,
                    ArtifactKind.SpeechEnhancedAudio),

            _ when string.Equals(stageName, StageNames.Export, StringComparison.OrdinalIgnoreCase) =>
                ExportOutputsPresent(state.StageRuns, projectRootPath, exportPath),

            _ when string.Equals(stageName, StageNames.OverlapRescue, StringComparison.OrdinalIgnoreCase) =>
                OverlapRescueOutputsPresent(state.StageRuns, artifacts, artifactStore),

            _ => false
        };
    }

    public static bool CanResumeStage(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        string stageName,
        IReadOnlyDictionary<string, string> snapshot,
        string projectRootPath,
        string? targetLanguageCode = null,
        string? exportPath = null)
    {
        StageRunRecord? latestRun = GetLatestSuccessfulRun(state.StageRuns, stageName);
        if (latestRun is null)
        {
            return false;
        }

        if (DubbingPipelineStages.RequiresSourceMedia(stageName) &&
            !SourceMediaMatchesSnapshot(state, snapshot))
        {
            return false;
        }

        if (!RuntimeMatchesSnapshot(latestRun, stageName, snapshot))
        {
            return false;
        }

        if (!SourceLanguageMatchesSnapshot(stageName, state, snapshot))
        {
            return false;
        }

        if (!AsrUpstreamMatchesSnapshot(stageName, state, snapshot))
        {
            return false;
        }

        if (!TtsDecisionsMatchSnapshot(stageName, state, snapshot))
        {
            return false;
        }

        return OutputsPresent(
            state,
            artifactStore,
            stageName,
            projectRootPath,
            targetLanguageCode,
            exportPath);
    }

    /// <summary>
    /// Resume gate that also honors the export-gating flag comparison for the Export stage.
    /// For non-Export stages this is equivalent to <see cref="CanResumeStage"/>. For the Export
    /// stage it additionally loads the prior successful run's persisted ExportManifest and
    /// refuses to resume when any export-gating flag (ExportFormat, ApplyTimbrePolish,
    /// RestoreOriginalPan, MatchOriginalLoudness, BurnInSubtitles, SubtitleSource,
    /// SubtitleFormats, VideoEncoder) differs from the current snapshot.
    /// </summary>
    public static async Task<bool> CanResumeStageAsync(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        string stageName,
        IReadOnlyDictionary<string, string> snapshot,
        string projectRootPath,
        string? targetLanguageCode = null,
        string? exportPath = null,
        CancellationToken cancellationToken = default)
    {
        if (!CanResumeStage(
                state,
                artifactStore,
                stageName,
                snapshot,
                projectRootPath,
                targetLanguageCode,
                exportPath))
        {
            return false;
        }

        if (string.Equals(stageName, StageNames.AudioPreparation, StringComparison.OrdinalIgnoreCase))
        {
            return await AudioPreparationAuditMatchesAsync(
                state,
                artifactStore,
                cancellationToken).ConfigureAwait(false);
        }

        if (!string.Equals(stageName, StageNames.Export, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // External timeline files can change without an execution snapshot change.
        // Re-render while placements exist until their content is part of resume identity.
        if (TimelineMediaMixPreparer.GetPlacements(state).Count > 0) return false;

        return await ExportGatingMatchesSnapshotAsync(
            state,
            artifactStore,
            snapshot,
            cancellationToken).ConfigureAwait(false);
    }

    private static bool AudioPreparationOutputsPresent(
        IReadOnlyList<StageRunRecord> runs,
        IReadOnlyList<ProjectArtifact> artifacts,
        IArtifactStore artifactStore)
    {
        // GetLatestSuccessfulRun only returns Completed runs, so a partially completed
        // producer can never satisfy this gate.
        StageRunRecord? latestRun = GetLatestSuccessfulRun(runs, StageNames.AudioPreparation);
        if (latestRun is null)
        {
            return false;
        }

        ProjectArtifact[] analysisArtifacts = artifacts
            .Where(a => a.StageRunId == latestRun.Id && a.Kind == ArtifactKind.AudioQualityAnalysis)
            .ToArray();
        if (analysisArtifacts.Length == 0 || !analysisArtifacts.All(a => ArtifactExists(artifactStore, a)))
        {
            return false;
        }

        // Every processed artifact the producer registered must still be on disk; a
        // partially deleted set cannot satisfy resume even though the analysis survives.
        return artifacts
            .Where(a => a.StageRunId == latestRun.Id && a.Kind == ArtifactKind.SpeechProcessedAudio)
            .All(a => ArtifactExists(artifactStore, a));
    }

    /// <summary>
    /// The persisted preparation audit is the producer's own record of which processed
    /// artifacts it created. Resume requires every non-fallback processed output in that
    /// audit to still exist as a row from the same completed run with its file present,
    /// so a row deleted outright (invisible to the synchronous check) also fails closed.
    /// </summary>
    private static async Task<bool> AudioPreparationAuditMatchesAsync(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        CancellationToken cancellationToken)
    {
        StageRunRecord? latestRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.AudioPreparation);
        if (latestRun is null)
        {
            return false;
        }

        IReadOnlyList<ProjectArtifact> artifacts = state.ProjectState.Artifacts;
        ProjectArtifact? analysisArtifact = artifacts
            .Where(a => a.StageRunId == latestRun.Id && a.Kind == ArtifactKind.AudioQualityAnalysis)
            .OrderByDescending(static a => a.CreatedAtUtc)
            .FirstOrDefault();
        if (analysisArtifact is null)
        {
            return false;
        }

        SpeechAudioPreparationAudit? audit;
        try
        {
            audit = await artifactStore
                .ReadJsonAsync<SpeechAudioPreparationAudit>(analysisArtifact.RelativePath, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            // An audit that cannot be read cannot prove the producer's output set.
            return false;
        }

        if (audit is null || audit.ProjectId != state.ProjectState.Project.Id)
        {
            return false;
        }

        foreach (SpeechAudioDecisionAudit decision in audit.Decisions)
        {
            if (!decision.RequiresProcessing || decision.FallbackReason is not null)
            {
                continue;
            }

            if (decision.OutputArtifactId is not Guid outputArtifactId)
            {
                return false;
            }

            ProjectArtifact? output = artifacts.FirstOrDefault(a => a.Id == outputArtifactId);
            if (output is null ||
                output.StageRunId != latestRun.Id ||
                output.Kind != ArtifactKind.SpeechProcessedAudio ||
                !ArtifactExists(artifactStore, output))
            {
                return false;
            }
        }

        return true;
    }

    private static async Task<bool> ExportGatingMatchesSnapshotAsync(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        IReadOnlyDictionary<string, string> snapshot,
        CancellationToken cancellationToken)
    {
        StageRunRecord? latestExportRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.Export);
        if (latestExportRun is null)
        {
            return false;
        }

        ExportManifest? manifest;
        try
        {
            manifest = await artifactStore
                .ReadJsonAsync<ExportManifest>(
                    ProjectArtifactPaths.GetExportManifestRelativePath(latestExportRun.Id),
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            // A manifest that cannot be read tells us nothing about the prior run's flags.
            // Preserve existing behavior (resume) rather than force a spurious rerun.
            return true;
        }

        if (manifest?.HasTimelineMedia == true) return false;
        IReadOnlyDictionary<string, string>? persistedFlags = manifest?.Gating?.Flags;
        if (persistedFlags is null || persistedFlags.Count == 0)
        {
            // Older projects have no persisted gating flags. There is nothing to compare against,
            // so preserve the pre-existing resume behavior rather than force a spurious rerun.
            return true;
        }

        foreach (string key in ExportResumeGating.GatingKeys)
        {
            if (!persistedFlags.TryGetValue(key, out string? persistedValue))
            {
                // Flag absent from the persisted set: the prior run did not record it, so we
                // cannot prove a change. Skip it rather than force a rerun.
                continue;
            }

            snapshot.TryGetValue(key, out string? currentValue);
            if (!string.Equals(persistedValue, currentValue ?? string.Empty, StringComparison.Ordinal))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Resolves the effective export destination for existence checks. The caller supplies
    /// the path the next run would actually write: absolute paths are used verbatim (delivery
    /// destinations legitimately live outside the project root) and relative paths resolve
    /// against the project root.
    /// </summary>
    private static bool TryResolveExportPath(
        string projectRootPath,
        string? exportPath,
        out string? absolutePath)
    {
        absolutePath = null;

        if (string.IsNullOrWhiteSpace(exportPath))
        {
            return false;
        }

        try
        {
            absolutePath = Path.GetFullPath(exportPath, Path.GetFullPath(projectRootPath));
            return true;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }
    }

    private static bool ExportOutputsPresent(
        IReadOnlyList<StageRunRecord> runs, string projectRootPath, string? exportPath)
    {
        if (!TryResolveExportPath(projectRootPath, exportPath, out string? resolvedPath) || resolvedPath is null)
            return false;
        try
        {
            StageRunRecord? producer = GetLatestSuccessfulRun(runs, StageNames.Export);
            return producer is not null && ExportDeliveryTransaction.IsAcceptedDelivery(resolvedPath, producer.Id) &&
                File.Exists(resolvedPath);
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            // An unreadable or unrecoverable journal cannot establish delivery acceptance.
            return false;
        }
    }

    private static bool OverlapRescueOutputsPresent(
        IReadOnlyList<StageRunRecord> runs,
        IReadOnlyList<ProjectArtifact> artifacts,
        IArtifactStore artifactStore)
    {
        StageRunRecord? latestRun = GetLatestSuccessfulRun(runs, StageNames.OverlapRescue);
        if (latestRun is null)
        {
            return false;
        }

        ProjectArtifact[] metadataArtifacts = artifacts
            .Where(a => a.StageRunId == latestRun.Id && a.Kind == ArtifactKind.OverlapRescueMetadata)
            .ToArray();

        return metadataArtifacts.Length > 0 &&
               metadataArtifacts.All(a => ArtifactExists(artifactStore, a));
    }

    private static bool LipSyncOutputsPresent(
        IReadOnlyList<StageRunRecord> runs,
        IReadOnlyList<ProjectArtifact> artifacts,
        IArtifactStore artifactStore)
    {
        StageRunRecord? latestRun = GetLatestSuccessfulRun(runs, StageNames.LipSync);
        if (latestRun is null || latestRun.Status == StageRunStatus.PartiallyCompleted)
        {
            return false;
        }

        ProjectArtifact[] lipSyncArtifacts = artifacts
            .Where(a => a.StageRunId == latestRun.Id && a.Kind == ArtifactKind.LipSyncTake)
            .ToArray();

        return lipSyncArtifacts.Length > 0 &&
               lipSyncArtifacts.All(a => ArtifactExists(artifactStore, a));
    }

    private static bool TextRefinementOutputsPresent(
        TranscriptProjectState state,
        IArtifactStore artifactStore)
    {
        if (state.CurrentTranscriptRevision is null)
        {
            return false;
        }

        string provenanceRelativePath = ProjectArtifactPaths.GetTextRefinementProvenanceRelativePath(
            state.CurrentTranscriptRevision.Id);
        return artifactStore.Exists(provenanceRelativePath);
    }

    private static bool TtsOutputsPresent(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        IReadOnlyList<ProjectArtifact> artifacts)
    {
        if (state.CurrentTranslationRevision is null || state.TranslatedSegments.Count == 0)
        {
            return false;
        }

        Dictionary<Guid, ProjectArtifact> artifactsById = artifacts
            .Where(artifact => artifact.Kind == ArtifactKind.TtsTake)
            .ToDictionary(artifact => artifact.Id);

        Dictionary<int, TtsTake> latestTakesBySegmentIndex = state.TtsTakes
            .GroupBy(take => take.SegmentIndex)
            .ToDictionary(
                group => group.Key,
                group => group
                    .OrderByDescending(take => take.CreatedAtUtc)
                    .First());

        foreach (TranslatedSegment segment in state.TranslatedSegments)
        {
            if (!latestTakesBySegmentIndex.TryGetValue(segment.SegmentIndex, out TtsTake? take))
            {
                return false;
            }

            if (take.Status != TtsTakeStatus.Completed || take.IsStale)
            {
                return false;
            }

            // A take whose persisted hash no longer matches the current translated text was
            // synthesized for different words and must not be resumed.
            if (!string.Equals(
                    take.TranslatedTextHash,
                    TtsTextHash.Compute(segment.SegmentIndex, segment.Text),
                    StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (take.ArtifactId is not Guid artifactId ||
                !artifactsById.TryGetValue(artifactId, out ProjectArtifact? artifact))
            {
                return false;
            }

            if (!ArtifactExists(artifactStore, artifact))
            {
                return false;
            }
        }

        return true;
    }

    private static bool StageRunScopedArtifactsPresent(
        IReadOnlyList<StageRunRecord> runs,
        IReadOnlyList<ProjectArtifact> artifacts,
        IArtifactStore artifactStore,
        string stageName,
        ArtifactKind kind)
    {
        StageRunRecord? latestRun = GetLatestSuccessfulRun(runs, stageName);
        if (latestRun is null || latestRun.Status == StageRunStatus.PartiallyCompleted)
        {
            return false;
        }

        ProjectArtifact[] stageArtifacts = artifacts
            .Where(a => a.StageRunId == latestRun.Id && a.Kind == kind)
            .ToArray();

        return stageArtifacts.Length > 0 &&
               stageArtifacts.All(a => ArtifactExists(artifactStore, a));
    }

    private static bool AsrUpstreamMatchesSnapshot(
        string stageName,
        TranscriptProjectState state,
        IReadOnlyDictionary<string, string> snapshot)
    {
        if (!string.Equals(stageName, StageNames.SpeakerAssignment, StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase))
        {
            StageRunRecord? asrRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.Asr);
            return asrRun is not null && RuntimeMatchesSnapshot(asrRun, StageNames.Asr, snapshot);
        }

        if (state.CurrentTranscriptRevision?.StageRunId is not Guid revisionRunId)
        {
            return false;
        }

        StageRunRecord? refinementRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.TextRefinementAsr);
        if (refinementRun is not null && revisionRunId == refinementRun.Id)
        {
            return RuntimeMatchesSnapshot(refinementRun, StageNames.TextRefinementAsr, snapshot);
        }

        StageRunRecord? latestAsrRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.Asr);
        return latestAsrRun is not null &&
               revisionRunId == latestAsrRun.Id &&
               RuntimeMatchesSnapshot(latestAsrRun, StageNames.Asr, snapshot);
    }

    /// <summary>
    /// Every pipeline stage derives from the project's source media, so a missing or
    /// in-place-changed source invalidates all prior artifacts. When the caller targets a
    /// different file than the one the project ingested, the snapshot's SourceMediaPath no
    /// longer matches the persisted reference and resume is refused. Unknown/Available
    /// statuses and absent keys keep the prior behavior.
    /// </summary>
    private static bool SourceMediaMatchesSnapshot(
        TranscriptProjectState state,
        IReadOnlyDictionary<string, string> snapshot)
    {
        if (state.ProjectState.SourceStatus is SourceMediaStatus.Changed or SourceMediaStatus.Missing)
        {
            return false;
        }

        if (!snapshot.TryGetValue("SourceMediaPath", out string? requestedPath) ||
            string.IsNullOrWhiteSpace(requestedPath))
        {
            return true;
        }

        string? persistedPath = state.ProjectState.SourceReference?.OriginalPath;
        if (string.IsNullOrWhiteSpace(persistedPath))
        {
            return true;
        }
        if (!TryNormalizePath(requestedPath, out string? normalizedRequested))
        {
            return false;
        }

        if (!TryNormalizePath(persistedPath, out string? normalizedPersisted))
        {
            return false;
        }

        StringComparison comparison = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;
        return string.Equals(normalizedRequested, normalizedPersisted, comparison);
    }

    private static bool TryNormalizePath(string path, out string? normalized)
    {
        try
        {
            string fullPath = Path.GetFullPath(path.Trim());

            // Normalize path separators for cross-platform consistency
            if (Path.DirectorySeparatorChar != Path.AltDirectorySeparatorChar)
            {
                fullPath = fullPath.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);
            }

            normalized = fullPath;
            return true;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            normalized = null;
            return false;
        }
    }

    /// <summary>
    /// TTS takes persist the voice/clone decisions they were synthesized under. Resume is only
    /// honest when those decisions still match the current run's intent: the snapshot's
    /// Voice:/VoiceClone:/UseVoiceCloning keys are compared per take against its Kind, VoiceId,
    /// and reference clip (resolved through the take's persisted voice assignment, which also
    /// exposes fallback rows).
    /// </summary>
    private static bool TtsDecisionsMatchSnapshot(
        string stageName,
        TranscriptProjectState state,
        IReadOnlyDictionary<string, string> snapshot)
    {
        if (!string.Equals(stageName, StageNames.Tts, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        bool useCloning = snapshot.TryGetValue("UseVoiceCloning", out string? cloningFlag) &&
                          bool.TryParse(cloningFlag, out bool cloningRequested) &&
                          cloningRequested;

        var cloneIntentBySpeakerId = new Dictionary<Guid, bool>();
        var voiceOverrideBySpeakerId = new Dictionary<Guid, string>();
        foreach ((string key, string value) in snapshot)
        {
            if (key.StartsWith("VoiceClone:", StringComparison.Ordinal) &&
                Guid.TryParse(key.AsSpan("VoiceClone:".Length), out Guid cloneSpeakerId) &&
                bool.TryParse(value, out bool clone))
            {
                cloneIntentBySpeakerId[cloneSpeakerId] = clone;
            }
            else if (key.StartsWith("Voice:", StringComparison.Ordinal) &&
                     !string.IsNullOrWhiteSpace(value) &&
                     UnattendedVoiceSelection.TryMatchSpeaker(
                         state.Speakers,
                         key["Voice:".Length..],
                         out ProjectSpeaker? speaker) &&
                     speaker is not null)
            {
                voiceOverrideBySpeakerId[speaker.Id] = value;
            }
        }

        bool hasCloneMap = cloneIntentBySpeakerId.Count > 0;

        var assignmentsById = new Dictionary<Guid, VoiceAssignment>();
        foreach (VoiceAssignment assignment in state.VoiceAssignments)
        {
            assignmentsById[assignment.Id] = assignment;
        }

        Dictionary<int, Guid?> speakerIdBySegmentIndex = state.TranscriptSegments
            .GroupBy(segment => segment.SegmentIndex)
            .ToDictionary(group => group.Key, group => group.First().SpeakerId);

        Dictionary<int, TtsTake> latestTakesBySegmentIndex = state.TtsTakes
            .Where(take => !take.IsStale)
            .GroupBy(take => take.SegmentIndex)
            .ToDictionary(
                group => group.Key,
                group => group.OrderByDescending(take => take.CreatedAtUtc).First());

        foreach (TranslatedSegment segment in state.TranslatedSegments)
        {
            if (!latestTakesBySegmentIndex.TryGetValue(segment.SegmentIndex, out TtsTake? take))
            {
                continue;
            }

            if (snapshot.TryGetValue(TtsTimingResumeIdentity.SnapshotKey, out string? expectedOverrun) &&
                !TtsTimingResumeIdentity.Matches(take.InputFingerprint, expectedOverrun))
            {
                return false;
            }

            speakerIdBySegmentIndex.TryGetValue(segment.SegmentIndex, out Guid? speakerId);
            // Mirrors GenerateTtsForAllSpeakersAsync/BuildUnattendedTtsRequest: an explicit
            // clone-map entry wins; with no map, global cloning applies to speakers that have
            // no explicit voice override.
            bool wantsClone = speakerId is Guid resolvedSpeakerId &&
                (cloneIntentBySpeakerId.TryGetValue(resolvedSpeakerId, out bool cloneIntent)
                    ? cloneIntent
                    : !hasCloneMap && useCloning && !voiceOverrideBySpeakerId.ContainsKey(resolvedSpeakerId));

            VoiceAssignment? assignment =
                assignmentsById.TryGetValue(take.VoiceAssignmentId, out VoiceAssignment? persisted)
                    ? persisted
                    : null;

            if (take.Kind == TtsTakeKind.VoiceCloned)
            {
                if (!wantsClone)
                {
                    return false;
                }

                if (!string.Equals(
                        take.ReferenceClipArtifactId?.ToString(),
                        assignment?.ReferenceClipArtifactId?.ToString(),
                        StringComparison.Ordinal))
                {
                    return false;
                }
            }
            else if (take.Kind == TtsTakeKind.Stock)
            {
                if (wantsClone)
                {
                    // A stock take under clone intent is only honest when the prior run
                    // persisted an explicit fallback (e.g. insufficient speech) with no clip;
                    // anything else must rerun so the clone actually happens.
                    if (assignment?.IsFallback != true || assignment.ReferenceClipArtifactId is not null)
                    {
                        return false;
                    }
                }
                else
                {
                    string? expectedVoiceId =
                        speakerId is Guid stockSpeakerId &&
                        voiceOverrideBySpeakerId.TryGetValue(stockSpeakerId, out string? overrideVoiceId)
                            ? overrideVoiceId
                            : assignment?.VoiceVariant;

                    if (!string.IsNullOrWhiteSpace(expectedVoiceId) &&
                        !string.Equals(take.VoiceId, expectedVoiceId, StringComparison.OrdinalIgnoreCase))
                    {
                        return false;
                    }
                }
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    private static bool SourceLanguageMatchesSnapshot(
        string stageName,
        TranscriptProjectState state,
        IReadOnlyDictionary<string, string> snapshot)
    {
        if (!string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(stageName, StageNames.SpeakerAssignment, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (!snapshot.TryGetValue("SourceLanguage", out string? requestedLanguage) ||
            string.IsNullOrWhiteSpace(requestedLanguage))
        {
            // Snapshots captured before the key alignment wrote "SourceLanguageCode";
            // keep honoring those rows.
            if (!snapshot.TryGetValue("SourceLanguageCode", out requestedLanguage) ||
                string.IsNullOrWhiteSpace(requestedLanguage))
            {
                return true;
            }
        }

        string? normalizedRequested = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(requestedLanguage);
        if (normalizedRequested is null)
        {
            return true;
        }

        string? persistedLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(state.TranscriptLanguage);
        if (persistedLanguage is null && state.TranscriptSegments.Count > 0)
        {
            persistedLanguage = TranscriptWorkflowUtilities.ResolveDetectedTranscriptLanguage(state.TranscriptSegments);
        }

        if (persistedLanguage is null)
        {
            return true;
        }

        return string.Equals(normalizedRequested, persistedLanguage, StringComparison.OrdinalIgnoreCase);
    }

    private static bool AsrOutputsPresent(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        IReadOnlyList<ProjectArtifact> artifacts)
    {
        if (state.CurrentTranscriptRevision is null || state.TranscriptSegments.Count == 0)
        {
            return false;
        }

        StageRunRecord? latestRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.Asr);
        if (latestRun is null)
        {
            return false;
        }

        ProjectArtifact? rawArtifact = artifacts
            .FirstOrDefault(artifact =>
                artifact.StageRunId == latestRun.Id &&
                artifact.Kind == ArtifactKind.TranscriptRevision &&
                artifact.Provenance?.StartsWith("generated-asr-raw", StringComparison.OrdinalIgnoreCase) == true);

        return ArtifactExists(artifactStore, rawArtifact);
    }

    private static bool SpeakerAssignmentOutputsPresent(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        IReadOnlyList<ProjectArtifact> artifacts)
    {
        if (state.CurrentTranscriptRevision is null || state.TranscriptSegments.Count == 0)
        {
            return false;
        }

        StageRunRecord? latestRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.SpeakerAssignment);
        if (latestRun is null)
        {
            return false;
        }

        string transcriptPath = ProjectArtifactPaths.GetTranscriptRevisionRelativePath(
            state.CurrentTranscriptRevision.RevisionNumber);
        ProjectArtifact? persistedTranscriptArtifact = artifacts
            .FirstOrDefault(artifact =>
                artifact.Kind == ArtifactKind.TranscriptRevision &&
                string.Equals(artifact.RelativePath, transcriptPath, StringComparison.OrdinalIgnoreCase) &&
                (string.Equals(artifact.Provenance, "generated-asr", StringComparison.OrdinalIgnoreCase) ||
                 string.Equals(artifact.Provenance, "generated-asr-polished", StringComparison.OrdinalIgnoreCase)));

        return ArtifactExists(artifactStore, persistedTranscriptArtifact);
    }

    private static bool DiarizationOutputsPresent(
        TranscriptProjectState state,
        IArtifactStore artifactStore,
        IReadOnlyList<ProjectArtifact> artifacts)
    {
        if (state.SpeakerTurns.Count == 0)
        {
            return false;
        }

        StageRunRecord? latestRun = GetLatestSuccessfulRun(state.StageRuns, StageNames.Diarization);
        if (latestRun is null)
        {
            return false;
        }

        ProjectArtifact? diarizationArtifact = artifacts
            .FirstOrDefault(artifact =>
                artifact.StageRunId == latestRun.Id &&
                artifact.Kind == ArtifactKind.DiarizationResult);

        return ArtifactExists(artifactStore, diarizationArtifact);
    }

    private static bool IsSuccessfulRun(StageRunRecord run) =>
        run.Status is StageRunStatus.Completed;

    private static bool ArtifactExists(IArtifactStore artifactStore, ProjectArtifact? artifact) =>
        artifact is not null && artifactStore.Exists(artifact.RelativePath);
}
