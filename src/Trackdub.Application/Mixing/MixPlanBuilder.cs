using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.Contracts.Transcripts;

namespace Trackdub.Application.Mixing;

public sealed record MixPlanBuildRequest(
    Guid ProjectId,
    Guid? MediaAssetId,
    IReadOnlyList<ProjectArtifact> Artifacts,
    IReadOnlyList<TranscriptSegment> TranscriptSegments,
    IReadOnlyList<TranslatedSegment> TranslatedSegments,
    IReadOnlyList<TtsTake> TtsTakes,
    IReadOnlyList<StageRunRecord>? StageRuns,
    double SourceGainDb = 0d,
    double DubbedSpeechGainDb = 0d,
    double? DuckingGainDb = null,
    double DuckingLeadSeconds = 0.05d,
    double DuckingTailSeconds = 0.18d,
    bool RestoreOriginalPan = false,
    bool ApplyTimbrePolish = true,
    IReadOnlyList<TtsCandidateGroup>? CandidateGroups = null,
    double? MediaDurationSeconds = null,
    IReadOnlyList<MixMediaClip>? MediaClips = null,
    double BackgroundMediaGainDb = 0d);

public sealed class MixPlanBuilder(IArtifactStore? artifactStore = null)
{
    private const double OverrunToleranceSeconds = 0.010d;

    public const double CleanAmbianceDefaultDuckingGainDb = 0d;
    public const double OriginalMixDefaultDuckingGainDb = -13d;

    public MixPlan Build(MixPlanBuildRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.ProjectId == Guid.Empty)
        {
            throw new ArgumentException("Project id is required.", nameof(request));
        }

        IReadOnlyList<ProjectArtifact> stemCandidates = request.MediaAssetId is Guid assetId
            ? request.Artifacts.Where(a => a.MediaAssetId == assetId).ToList()
            : request.Artifacts;
        ProjectArtifact sourceArtifact = SelectSourceAudioArtifact(stemCandidates, request.StageRuns)
            ?? throw new InvalidOperationException("The project does not contain source audio for preview mixing.");
        ProjectArtifact? originalMixArtifact = SelectOriginalMixAudioArtifact(request.Artifacts, request.MediaAssetId);
        ProjectArtifact? vocalStemArtifact = TranscriptWorkflowUtilities.GetLatestAcceptedVocalStem(stemCandidates, request.StageRuns);
        Dictionary<Guid, ProjectArtifact> ttsArtifactsById = request.Artifacts
            .Where(static artifact => artifact.Kind == ArtifactKind.TtsTake)
            .GroupBy(static artifact => artifact.Id)
            .ToDictionary(static group => group.Key, static group => group.OrderByDescending(artifact => artifact.CreatedAtUtc).First());
        Dictionary<int, TranslatedSegment> translatedSegmentsByIndex = request.TranslatedSegments
            .GroupBy(static segment => segment.SegmentIndex)
            .ToDictionary(static group => group.Key, static group => group.Last());
        ILookup<int, TtsTake> takesBySegmentIndex = request.TtsTakes.ToLookup(static take => take.SegmentIndex);
        Dictionary<Guid, Guid>? selectedCandidateBySegmentId = request.CandidateGroups is { Count: > 0 } groups
            ? groups.ToDictionary(static g => g.TranslatedSegmentId, static g => g.SelectedCandidateId)
            : null;

        Dictionary<Guid, ProjectArtifact> lipSyncByTakeId = BuildLipSyncByTakeId(request.Artifacts);
        var clips = new List<MixSpeechClip>();
        var duckingRegions = new List<MixDuckRegion>();
        var warnings = new List<MixPlanWarning>();
        double duckingGainDb = ResolveDuckingGainDb(request.DuckingGainDb, sourceArtifact.Kind);
        // The mix renderer clamps output to the source audio duration, so a take that extends
        // past it would be silently truncated at export. Detect that here and fail loudly
        // instead of dropping speech without a trace.
        double? sourceAudioEndSeconds = sourceArtifact.DurationSeconds is double sourceDuration &&
                                        double.IsFinite(sourceDuration) &&
                                        sourceDuration > 0d
            ? sourceDuration
            : null;
        if (request.MediaDurationSeconds is double mediaDuration &&
            double.IsFinite(mediaDuration) &&
            mediaDuration > 0d)
        {
            sourceAudioEndSeconds = sourceAudioEndSeconds is double sourceEnd
                ? Math.Min(sourceEnd, mediaDuration)
                : mediaDuration;
        }
        foreach (TranscriptSegment segment in request.TranscriptSegments.OrderBy(static segment => segment.SegmentIndex))
        {
            MixSpeechClip clip = BuildSpeechClip(
                segment,
                translatedSegmentsByIndex,
                takesBySegmentIndex,
                ttsArtifactsById,
                warnings,
                lipSyncByTakeId,
                artifactStore,
                selectedCandidateBySegmentId);
            clips.Add(clip);

            if (!clip.IsSilentGap)
            {
                double duckingLeadSeconds = NormalizeSeconds(request.DuckingLeadSeconds, 0.05d);
                double duckingTailSeconds = NormalizeSeconds(request.DuckingTailSeconds, 0.18d);
                duckingRegions.Add(new MixDuckRegion(
                    segment.SegmentIndex,
                    segment.Id,
                    Math.Max(0d, clip.StartSeconds - duckingLeadSeconds),
                    ResolveClipDuckingEndSeconds(clip, segment.EndSeconds) + duckingTailSeconds,
                    duckingGainDb));

                if (sourceAudioEndSeconds is double sourceAudioEndSecondsValue &&
                    clip.TakeDurationSeconds is double takeDuration &&
                    double.IsFinite(takeDuration) &&
                    clip.StartSeconds + takeDuration - sourceAudioEndSecondsValue > OverrunToleranceSeconds)
                {
                    double overhangSeconds = clip.StartSeconds + takeDuration - sourceAudioEndSecondsValue;
                    warnings.Add(new MixPlanWarning(
                        segment.SegmentIndex,
                        segment.Id,
                        string.Create(
                            System.Globalization.CultureInfo.InvariantCulture,
                            $"Dubbed take extends {overhangSeconds:0.###} s past the end of the source audio ({sourceAudioEndSecondsValue:0.###} s) and would be cut off at export. Stretch the take to fit, trim it, or raise the auto-stretch limit."),
                        MixPlanWarningCode.TakeExceedsSourceAudio));
                }
            }
        }

        return new MixPlan(
            request.ProjectId,
            request.MediaAssetId,
            sourceArtifact.Kind,
            sourceArtifact.RelativePath,
            NormalizeDb(request.SourceGainDb, 0d),
            NormalizeDb(request.DubbedSpeechGainDb, 0d),
            duckingGainDb,
            NormalizeSeconds(request.DuckingLeadSeconds, 0.05d),
            NormalizeSeconds(request.DuckingTailSeconds, 0.18d),
            DateTimeOffset.UtcNow,
            clips,
            duckingRegions,
            warnings,
            originalMixArtifact?.RelativePath ?? sourceArtifact.RelativePath,
            ResolveOutputChannelCount(originalMixArtifact, sourceArtifact),
            request.RestoreOriginalPan,
            request.ApplyTimbrePolish,
            vocalStemArtifact?.RelativePath,
            BuildMediaClips(request));
    }

    private static IReadOnlyList<MixMediaClip>? BuildMediaClips(MixPlanBuildRequest request)
    {
        if (request.MediaClips is null)
        {
            return null;
        }

        double gainDb = NormalizeDb(request.BackgroundMediaGainDb, 0d);
        return request.MediaClips.Select(clip => clip with { GainDb = gainDb }).ToArray();
    }

    private static ProjectArtifact? SelectSourceAudioArtifact(IReadOnlyList<ProjectArtifact> artifacts, IReadOnlyList<StageRunRecord>? stageRuns) =>
        TranscriptWorkflowUtilities.GetLatestAcceptedAmbianceStem(artifacts, stageRuns)
        ?? artifacts
            .Where(static artifact => artifact.Kind == ArtifactKind.NormalizedAudio)
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();

    private static ProjectArtifact? SelectOriginalMixAudioArtifact(IReadOnlyList<ProjectArtifact> artifacts, Guid? mediaAssetId) =>
        artifacts
            .Where(artifact => artifact.Kind == ArtifactKind.NormalizedAudio &&
                               (!mediaAssetId.HasValue || artifact.MediaAssetId == mediaAssetId.Value))
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();

    private static MixSpeechClip BuildSpeechClip(
        TranscriptSegment segment,
        IReadOnlyDictionary<int, TranslatedSegment> translatedSegmentsByIndex,
        ILookup<int, TtsTake> takesBySegmentIndex,
        IReadOnlyDictionary<Guid, ProjectArtifact> ttsArtifactsById,
        List<MixPlanWarning> warnings,
        IReadOnlyDictionary<Guid, ProjectArtifact> lipSyncByTakeId,
        IArtifactStore? artifactStore,
        IReadOnlyDictionary<Guid, Guid>? selectedCandidateBySegmentId = null)
    {
        // Build initial takes ordered by newest first, then reorder so the selected candidate is first.
        TtsTake[] takes = takesBySegmentIndex[segment.SegmentIndex]
            .OrderByDescending(take => take.CreatedAtUtc)
            .ToArray();

        if (selectedCandidateBySegmentId is not null &&
            translatedSegmentsByIndex.TryGetValue(segment.SegmentIndex, out TranslatedSegment? translatedSegment) &&
            selectedCandidateBySegmentId.TryGetValue(translatedSegment.Id, out Guid selectedTakeId))
        {
            takes = [.. takes.OrderByDescending(t => t.Id == selectedTakeId).ThenByDescending(t => t.CreatedAtUtc)];
        }
        if (takes.Length == 0)
        {
            return BuildSilentGap(segment, "Missing dubbed speech take.", warnings, MixPlanWarningCode.MissingTake);
        }

        MixSpeechClip? newestRejectedClip = null;
        MixPlanWarningCode? newestRejectedWarningCode = null;
        foreach (TtsTake take in takes)
        {
            if (take.Status is not TtsTakeStatus.Completed)
            {
                if (newestRejectedClip is null)
                {
                    newestRejectedClip = BuildRejectedSilentGap(
                        segment,
                        $"Dubbed speech take is {take.Status.ToString().ToLowerInvariant()}.",
                        take.Id);
                    newestRejectedWarningCode = MixPlanWarningCode.InvalidTake;
                }
                continue;
            }

            if (take.IsStale || IsTakeStaleForTranslation(segment.SegmentIndex, take, translatedSegmentsByIndex))
            {
                if (newestRejectedClip is null)
                {
                    newestRejectedClip = BuildRejectedSilentGap(
                        segment,
                        "Dubbed speech take is stale.",
                        take.Id);
                    newestRejectedWarningCode = MixPlanWarningCode.StaleTake;
                }
                continue;
            }

            if (take.ArtifactId is not Guid artifactId ||
                !ttsArtifactsById.TryGetValue(artifactId, out ProjectArtifact? artifact))
            {
                if (newestRejectedClip is null)
                {
                    newestRejectedClip = BuildRejectedSilentGap(
                        segment,
                        "Dubbed speech artifact is missing.",
                        take.Id,
                        take.ArtifactId);
                    newestRejectedWarningCode = MixPlanWarningCode.MissingTakeArtifact;
                }
                continue;
            }

            double? durationSeconds = artifact.DurationSeconds ??
                                      (take.DurationSamples is int durationSamples && take.SampleRate is int sampleRate && sampleRate > 0
                                          ? (double)durationSamples / sampleRate
                                          : null);

            if (lipSyncByTakeId.TryGetValue(take.Id, out ProjectArtifact? lipSyncArtifact) &&
                artifactStore is not null)
            {
                if (artifactStore.Exists(lipSyncArtifact.RelativePath))
                {
                    return new MixSpeechClip(
                        segment.SegmentIndex,
                        segment.Id,
                        segment.StartSeconds,
                        segment.EndSeconds,
                        take.Id,
                        lipSyncArtifact.Id,
                        lipSyncArtifact.RelativePath,
                        lipSyncArtifact.DurationSeconds ?? durationSeconds,
                        IsSilentGap: false,
                        WarningMessage: null);
                }

                // Registered in DB but not on disk; degrade and fall through to TTS take.
                warnings.Add(new MixPlanWarning(
                    segment.SegmentIndex,
                    segment.Id,
                    "Lip-sync artifact file is missing on disk; falling back to TTS take.",
                    MixPlanWarningCode.LipSyncArtifactMissing));
            }

            return new MixSpeechClip(
                segment.SegmentIndex,
                segment.Id,
                segment.StartSeconds,
                segment.EndSeconds,
                take.Id,
                artifact.Id,
                artifact.RelativePath,
                durationSeconds,
                IsSilentGap: false,
                WarningMessage: null);
        }

        MixSpeechClip rejectedClip = newestRejectedClip ??
            BuildRejectedSilentGap(segment, "Missing dubbed speech take.");
        warnings.Add(new MixPlanWarning(
            segment.SegmentIndex,
            segment.Id,
            rejectedClip.WarningMessage ?? "Missing dubbed speech take.",
            newestRejectedWarningCode ?? MixPlanWarningCode.MissingTake));
        return rejectedClip;
    }

    private static bool IsTakeStaleForTranslation(
        int segmentIndex,
        TtsTake take,
        IReadOnlyDictionary<int, TranslatedSegment> translatedSegmentsByIndex)
    {
        if (!translatedSegmentsByIndex.TryGetValue(segmentIndex, out TranslatedSegment? translatedSegment))
        {
            return take.TranslatedSegmentId.HasValue ||
                   !string.IsNullOrWhiteSpace(take.TranslatedTextHash);
        }

        if (string.IsNullOrWhiteSpace(take.TranslatedTextHash))
        {
            return false;
        }

        return !string.Equals(
            take.TranslatedTextHash,
            TtsTextHash.Compute(segmentIndex, translatedSegment.Text),
            StringComparison.Ordinal);
    }

    private static MixSpeechClip BuildSilentGap(
        TranscriptSegment segment,
        string message,
        List<MixPlanWarning> warnings,
        MixPlanWarningCode warningCode,
        Guid? takeId = null,
        Guid? artifactId = null)
    {
        warnings.Add(new MixPlanWarning(segment.SegmentIndex, segment.Id, message, warningCode));
        return new MixSpeechClip(
            segment.SegmentIndex,
            segment.Id,
            segment.StartSeconds,
            segment.EndSeconds,
            takeId,
            artifactId,
            TakeRelativePath: null,
            TakeDurationSeconds: segment.EndSeconds - segment.StartSeconds,
            IsSilentGap: true,
            message);
    }

    private static MixSpeechClip BuildRejectedSilentGap(
        TranscriptSegment segment,
        string message,
        Guid? takeId = null,
        Guid? artifactId = null) =>
        new(
            segment.SegmentIndex,
            segment.Id,
            segment.StartSeconds,
            segment.EndSeconds,
            takeId,
            artifactId,
            TakeRelativePath: null,
            TakeDurationSeconds: segment.EndSeconds - segment.StartSeconds,
            IsSilentGap: true,
            message);

    private static double ResolveClipDuckingEndSeconds(MixSpeechClip clip, double fallbackEndSeconds)
    {
        if (clip.TakeDurationSeconds is double takeDurationSeconds &&
            double.IsFinite(takeDurationSeconds) &&
            takeDurationSeconds > 0d)
        {
            return Math.Max(fallbackEndSeconds, clip.StartSeconds + takeDurationSeconds);
        }

        return fallbackEndSeconds;
    }

    public static double ResolveAutomaticDuckingGainDb(ArtifactKind sourceAudioKind) =>
        sourceAudioKind is ArtifactKind.Ambiance
            ? CleanAmbianceDefaultDuckingGainDb
            : OriginalMixDefaultDuckingGainDb;

    private static double ResolveDuckingGainDb(double? requestedGainDb, ArtifactKind sourceAudioKind) =>
        requestedGainDb is double gainDb
            ? NormalizeDb(gainDb, ResolveAutomaticDuckingGainDb(sourceAudioKind))
            : ResolveAutomaticDuckingGainDb(sourceAudioKind);

    private static int ResolveOutputChannelCount(ProjectArtifact? originalMixArtifact, ProjectArtifact sourceArtifact) =>
        NormalizeChannelCount(originalMixArtifact?.ChannelCount ?? sourceArtifact.ChannelCount);

    private static int NormalizeChannelCount(int? channelCount) =>
        channelCount is >= 2 ? 2 : 1;

    private static double NormalizeDb(double value, double fallback) =>
        double.IsFinite(value) ? Math.Clamp(value, -96d, 24d) : fallback;

    private static double NormalizeSeconds(double value, double fallback) =>
        double.IsFinite(value) && value >= 0d ? Math.Min(value, 5d) : fallback;

    private static Dictionary<Guid, ProjectArtifact> BuildLipSyncByTakeId(
        IReadOnlyList<ProjectArtifact> artifacts)
    {
        const string provenancePrefix = "lipsync:take:";
        var result = new Dictionary<Guid, ProjectArtifact>();
        foreach (ProjectArtifact artifact in artifacts)
        {
            if (artifact.Kind != ArtifactKind.LipSyncTake ||
                artifact.Provenance is not { Length: > 0 } provenance ||
                !provenance.StartsWith(provenancePrefix, StringComparison.Ordinal))
            {
                continue;
            }

            if (!Guid.TryParseExact(provenance[provenancePrefix.Length..], "N", out Guid takeId))
            {
                continue;
            }

            if (!result.TryGetValue(takeId, out ProjectArtifact? existing) ||
                artifact.CreatedAtUtc > existing.CreatedAtUtc)
            {
                result[takeId] = artifact;
            }
        }

        return result;
    }
}
