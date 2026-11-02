using System.Security.Cryptography;
using System.Text;
using Trackdub.Application.LipSynthesis;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.LipSynthesis;
using Trackdub.Domain.Media;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

public static class TranscriptWorkflowUtilities
{
    private const string AsrStageName = "asr";
    private const string SelectedSourceProvenanceKey = "selectedSource";
    private const string EnglishLanguageCode = "en";
    private const string SpanishLanguageCode = "es";
    private const double DiarizedTranscriptMergeGapSeconds = 1.5d;
    private const double MaxDiarizedTranscriptRegionSeconds = 28d;
    private const double DiarizedTranscriptRegionPaddingSeconds = 0.5d;
    private const string DemucsV4StemProvenance = "generated-demucs-v4";
    private const string HushDialogueStemProvenance = "generated-hush-dialogue";
    private const string SpleeterStemProvenance = "generated-spleeter";
    private const string DialogueIsolationUnavailableCode = "DIALOGUE_ISOLATION_UNAVAILABLE";
    private const string DialogueIsolationUnavailableMessage =
        "Dialogue isolation model unavailable; no clean ambiance track was generated.";
    private const string LegacyStemWarning =
        "Existing Demucs or Hush stems were created by an older/non-current separator and should be regenerated with the current separation engine.";

    public static MediaAsset GetRequiredMediaAsset(TranscriptProjectState state) =>
        state.ProjectState.MediaAsset
        ?? throw new InvalidOperationException("The project does not contain a primary media asset.");

    public static TranscriptRevision GetRequiredTranscriptRevision(TranscriptProjectState state) =>
        state.CurrentTranscriptRevision
        ?? throw new InvalidOperationException("The project does not contain a transcript revision.");

    public static void EnsureRevisionMatches(TranscriptRevision revision, Guid expectedRevisionId, string message)
    {
        if (revision.Id != expectedRevisionId)
        {
            throw new InvalidOperationException(message);
        }
    }

    public static ProjectArtifact? GetLatestArtifactByKind(
        IReadOnlyList<ProjectArtifact> artifacts,
        ArtifactKind kind) =>
        artifacts
            .Where(artifact => artifact.Kind == kind)
            .OrderByDescending(artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();

    /// <summary>
    /// True when at least one TTS take exists or a segment state references a generated take artifact.
    /// Placeholder segment states (translation without dubbing) do not count.
    /// </summary>
    public static bool HasGeneratedTtsTakes(TranscriptProjectState state) =>
        state.TtsTakes.Count > 0 ||
        state.TtsSegmentStates.Any(static segment => segment.TakeId is not null);

    /// <summary>
    /// Current exported audio fallback for video lip synthesis. A fresh speech-only driver is
    /// normally rendered directly from the current mix plan before delivery; this artifact is
    /// used only when that internal render path is unavailable.
    /// </summary>
    public static ProjectArtifact? ResolveLipSynthesisDriverAudioArtifact(TranscriptProjectState state)
    {
        ProjectArtifact? exportAudio = GetLatestArtifactByKind(state.ProjectState.Artifacts, ArtifactKind.ExportAudio);
        return exportAudio is not null && IsExportAudioCurrentForLipSynthesis(state, exportAudio)
            ? exportAudio
            : null;
    }

    /// <summary>
    /// True when exported dubbed audio exists and still matches the current non-stale TTS takes.
    /// </summary>
    public static bool IsExportAudioCurrentForLipSynthesis(
        TranscriptProjectState state,
        ProjectArtifact exportAudio)
    {
        ArgumentNullException.ThrowIfNull(exportAudio);
        if (exportAudio.Kind is not ArtifactKind.ExportAudio)
        {
            return false;
        }

        if (state.TtsTakes.Any(static take => take.IsStale || take.Status is TtsTakeStatus.Stale))
        {
            return false;
        }

        DateTimeOffset latestCompletedTakeUtc = state.TtsTakes
            .Where(static take => take.Status is TtsTakeStatus.Completed && !take.IsStale)
            .Select(static take => take.CreatedAtUtc)
            .DefaultIfEmpty(DateTimeOffset.MinValue)
            .Max();

        if (latestCompletedTakeUtc == DateTimeOffset.MinValue)
        {
            return false;
        }

        return exportAudio.CreatedAtUtc >= latestCompletedTakeUtc;
    }

    /// <summary>
    /// Warn when lip-synthesis repaired clips exist but cannot be composited into export video.
    /// </summary>
    public static string? BuildLipSynthesisExportCompositingWarning(
        TranscriptProjectState state,
        IArtifactStore? artifactStore = null)
    {
        if (artifactStore is not null)
        {
            return LipSynthesisExportRecomposition.BuildExportCompositingWarning(state, artifactStore);
        }

        if (!state.ProjectState.Artifacts.Any(static artifact => artifact.Kind is ArtifactKind.LipSynthesisTake))
        {
            return null;
        }

        if (state.LipSynthesisSegmentStates?.Any(static segment =>
                segment.Status is LipSynthesisSegmentStatus.Synthesized) == true &&
            state.SpeakerTurns.Count > 0)
        {
            return null;
        }

        return "Lip-synthesis repaired clips could not be composited into export video; output uses the original source footage.";
    }

    public static ProjectArtifact? ResolveAsrAudioArtifact(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns = null) =>
        ResolveLatestCompletedAudioPreparationAsrArtifact(artifacts, stageRuns)
        ?? GetAcceptedSpeechEnhancedAudio(artifacts, stageRuns)
        ?? GetLatestAcceptedVocalStem(artifacts, stageRuns)
        ?? GetLatestArtifactByKind(artifacts, ArtifactKind.NormalizedAudio);

    public static ProjectArtifact? GetLatestAcceptedVocalStem(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns) =>
        GetAcceptedStemArtifacts(artifacts, stageRuns).FirstOrDefault(static artifact => artifact.Kind == ArtifactKind.Vocals);

    public static ProjectArtifact? GetLatestAcceptedAmbianceStem(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns) =>
        GetAcceptedStemArtifacts(artifacts, stageRuns).FirstOrDefault(static artifact => artifact.Kind == ArtifactKind.Ambiance);

    // A processed artifact is accepted only when the preparation run that produced it
    // completed. Selecting by timestamp alone would reuse output registered by a
    // canceled or failed re-run, mixing producers the same way stem selection once did.
    public static ProjectArtifact? GetAcceptedSpeechProcessedAudio(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns) =>
        GetAcceptedStageAudio(artifacts, stageRuns, StageNames.AudioPreparation, ArtifactKind.SpeechProcessedAudio);

    public static ProjectArtifact? GetAcceptedSpeechEnhancedAudio(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns) =>
        GetAcceptedStageAudio(artifacts, stageRuns, StageNames.SpeechEnhancement, ArtifactKind.SpeechEnhancedAudio);

    private static ProjectArtifact? GetAcceptedStageAudio(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns,
        string stageName,
        ArtifactKind kind)
    {
        if (stageRuns is null)
        {
            return null;
        }

        // Only the latest completed producer is accepted. An earlier producer's artifact
        // must not resurface when the newest completed run produced no matching output
        // (for example, every preparation decision failed a guardrail).
        StageRunRecord? latestCompletedRun = stageRuns
            .Where(run =>
                string.Equals(run.StageName, stageName, StringComparison.OrdinalIgnoreCase) &&
                run.Status == StageRunStatus.Completed)
            .OrderByDescending(static run => run.CompletedAtUtc ?? run.StartedAtUtc)
            .ThenByDescending(static run => run.Id)
            .FirstOrDefault();
        if (latestCompletedRun is null)
        {
            return null;
        }

        return artifacts
            .Where(artifact =>
                artifact.Kind == kind &&
                artifact.StageRunId == latestCompletedRun.Id)
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();
    }

    public static ProjectArtifact? ResolveStageProcessedAudioArtifact(
        IReadOnlyList<ProjectArtifact> artifacts,
        string stageName,
        Guid? stageRunId = null) =>
        artifacts
            .Where(artifact =>
                artifact.Kind == ArtifactKind.SpeechProcessedAudio &&
                (stageRunId is null || artifact.StageRunId == stageRunId) &&
                artifact.Provenance?.Contains($"stage={stageName}", StringComparison.OrdinalIgnoreCase) == true)
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();

    public static StemAudioRoute BuildStemAudioRoute(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns)
    {
        ProjectArtifact? asrArtifact = ResolveAsrAudioArtifact(artifacts, stageRuns);
        ProjectArtifact? vocalsArtifact = GetLatestAcceptedVocalStem(artifacts, stageRuns);
        ProjectArtifact? mixSourceArtifact = GetLatestAcceptedAmbianceStem(artifacts, stageRuns)
            ?? GetLatestArtifactByKind(artifacts, ArtifactKind.NormalizedAudio);
        string asrRelativePath = asrArtifact?.RelativePath ?? ProjectArtifactPaths.NormalizedAudioRelativePath;
        string mixRelativePath = mixSourceArtifact?.RelativePath ?? ProjectArtifactPaths.NormalizedAudioRelativePath;

        bool usesStems = vocalsArtifact is not null ||
                         mixSourceArtifact?.Kind is ArtifactKind.Ambiance;
        string? warning = null;
        if (HasLegacySeparatorStem(artifacts))
        {
            warning = LegacyStemWarning;
            if (!usesStems)
            {
                warning += " Preview, export, transcription, and reference extraction will use the original mix until current separation stems exist.";
            }
        }
        else if (usesStems)
        {
            warning = null;
        }
        else if (HasDialogueIsolationUnavailableDegradation(artifacts))
        {
            warning = DialogueIsolationUnavailableMessage;
        }

        return new StemAudioRoute(
            asrRelativePath,
            mixRelativePath,
            warning);
    }

    private static bool HasLegacySeparatorStem(IReadOnlyList<ProjectArtifact> artifacts) =>
        artifacts.Any(IsLegacySeparatorStem);

    private static bool IsLegacySeparatorStem(ProjectArtifact artifact) =>
        (artifact.Kind is ArtifactKind.Vocals or ArtifactKind.Ambiance) &&
        !string.IsNullOrWhiteSpace(artifact.Provenance) &&
        (artifact.Provenance.Contains(DemucsV4StemProvenance, StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("engine_family=demucs-v4", StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("model=demucs-v4", StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains(HushDialogueStemProvenance, StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("model=hush-dialogue", StringComparison.OrdinalIgnoreCase));

    // A producer is accepted only after all required metadata was saved. Select that
    // producer as a unit, including optional stems, rather than combining timestamps.
    internal static IReadOnlyList<ProjectArtifact> GetAcceptedStemArtifacts(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns)
    {
        if (stageRuns is null)
        {
            return [];
        }

        foreach (StageRunRecord run in stageRuns
                     .Where(static run => run.StageName == StageNames.Separation && run.Status == StageRunStatus.Completed)
                     .OrderByDescending(static run => run.CompletedAtUtc ?? run.StartedAtUtc)
                     .ThenByDescending(static run => run.Id))
        {
            foreach (IGrouping<Guid, ProjectArtifact> group in artifacts
                         .Where(artifact => artifact.ProjectId == run.ProjectId && artifact.StageRunId == run.Id)
                         .GroupBy(static artifact => artifact.MediaAssetId))
            {
                ProjectArtifact[] required = group.Where(IsAcceptedCurrentStem).ToArray();
                if (!required.Any(static artifact => artifact.Kind == ArtifactKind.Vocals) ||
                    !required.Any(static artifact => artifact.Kind == ArtifactKind.Ambiance))
                {
                    continue;
                }

                return group.Where(static artifact => IsAcceptedCurrentStem(artifact) ||
                        artifact.Kind is ArtifactKind.Music or ArtifactKind.SoundEffects)
                    .OrderByDescending(static artifact => artifact.CreatedAtUtc)
                    .ThenByDescending(static artifact => artifact.Id)
                    .DistinctBy(static artifact => artifact.Kind)
                    .ToArray();
            }
        }

        return [];
    }

    private static bool IsAcceptedCurrentStem(ProjectArtifact artifact) =>
        (artifact.Kind is ArtifactKind.Vocals or ArtifactKind.Ambiance) &&
        !string.IsNullOrWhiteSpace(artifact.Provenance) &&
        (artifact.Provenance.Contains(SpleeterStemProvenance, StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("engine_family=spleeter", StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("model=spleeter", StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("model=spleeter-2stems", StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("engine_family=sepformer", StringComparison.OrdinalIgnoreCase) ||
            artifact.Provenance.Contains("model=sepformer", StringComparison.OrdinalIgnoreCase));

    private static bool HasDialogueIsolationUnavailableDegradation(IReadOnlyList<ProjectArtifact> artifacts) =>
        artifacts
            .Where(static artifact => artifact.Kind == ArtifactKind.PipelineDegradation)
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .Any(static artifact =>
                string.Equals(artifact.DegradationCode, DialogueIsolationUnavailableCode, StringComparison.OrdinalIgnoreCase));

    private static ProjectArtifact? ResolveLatestCompletedAudioPreparationAsrArtifact(
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<StageRunRecord>? stageRuns)
    {
        StageRunRecord? stageRun = GetLatestAudioPreparationRun(stageRuns);
        if (stageRun is null)
        {
            return null;
        }

        if (stageRun.Status != StageRunStatus.Completed)
        {
            return null;
        }

        SpeechAudioSourceKind? selectedSourceKind = ResolveSelectedSourceKind(artifacts, stageRun.Id);
        if (selectedSourceKind is SpeechAudioSourceKind.VocalStem &&
            GetLatestAcceptedVocalStem(artifacts, stageRuns) is null)
        {
            return GetLatestArtifactByKind(artifacts, ArtifactKind.NormalizedAudio);
        }

        ProjectArtifact? processedArtifact = ResolveStageProcessedAudioArtifact(artifacts, AsrStageName, stageRun.Id);
        if (processedArtifact is not null)
        {
            return processedArtifact;
        }

        return selectedSourceKind switch
        {
            SpeechAudioSourceKind.FullMix => GetLatestArtifactByKind(artifacts, ArtifactKind.NormalizedAudio),
            SpeechAudioSourceKind.VocalStem => GetLatestAcceptedVocalStem(artifacts, stageRuns)
                ?? GetLatestArtifactByKind(artifacts, ArtifactKind.NormalizedAudio),
            _ => null
        };
    }

    private static StageRunRecord? GetLatestAudioPreparationRun(IReadOnlyList<StageRunRecord>? stageRuns) =>
        stageRuns?
            .Where(static stageRun =>
                string.Equals(stageRun.StageName, StageNames.AudioPreparation, StringComparison.OrdinalIgnoreCase) &&
                stageRun.Status == StageRunStatus.Completed)
            .OrderByDescending(static stageRun => stageRun.CompletedAtUtc ?? stageRun.StartedAtUtc)
            .FirstOrDefault();

    private static SpeechAudioSourceKind? ResolveSelectedSourceKind(
        IReadOnlyList<ProjectArtifact> artifacts,
        Guid stageRunId)
    {
        ProjectArtifact? analysisArtifact = artifacts
            .Where(artifact =>
                artifact.Kind == ArtifactKind.AudioQualityAnalysis &&
                artifact.StageRunId == stageRunId)
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();

        if (TryGetProvenanceValue(analysisArtifact?.Provenance, SelectedSourceProvenanceKey, out string? selectedSource) &&
            Enum.TryParse(selectedSource, ignoreCase: true, out SpeechAudioSourceKind sourceKind))
        {
            return sourceKind;
        }

        return null;
    }

    private static bool TryGetProvenanceValue(string? provenance, string key, out string? value)
    {
        value = null;
        if (string.IsNullOrWhiteSpace(provenance))
        {
            return false;
        }

        foreach (string rawPart in provenance.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string part = rawPart;
            int namespaceSeparator = part.LastIndexOf(':');
            if (namespaceSeparator >= 0)
            {
                part = part[(namespaceSeparator + 1)..];
            }

            int valueSeparator = part.IndexOf('=');
            if (valueSeparator <= 0)
            {
                continue;
            }

            string candidateKey = part[..valueSeparator];
            if (!string.Equals(candidateKey, key, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            value = part[(valueSeparator + 1)..];
            return true;
        }

        return false;
    }

    public static IReadOnlyList<TranscriptSegment> ApplySingleSpeakerDefaultAssignments(
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<ProjectSpeaker> speakers)
    {
        if (segments.Count == 0 || speakers.Count != 1)
        {
            return segments;
        }

        Guid defaultSpeakerId = speakers[0].Id;
        return segments
            .OrderBy(segment => segment.SegmentIndex)
            .Select((segment, index) => segment.SpeakerId is Guid || segment.SpeakerAssignmentOrigin == SpeakerAssignmentOrigin.Manual
                ? segment
                : TranscriptSegment.Create(
                    segment.TranscriptRevisionId,
                    index,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    segment.Text,
                    defaultSpeakerId,
                    segment.DetectedLanguage,
                    CloneWords(segment.Words),
                    SpeakerAssignmentOrigin.Automatic))
            .ToArray();
    }

    public static (string Left, string Right) SplitSegmentText(string text)
    {
        string trimmed = text.Trim();
        if (trimmed.Length == 0)
        {
            return (string.Empty, string.Empty);
        }

        int midpoint = trimmed.Length / 2;
        int splitIndex = trimmed.IndexOf(' ', midpoint);
        if (splitIndex < 0)
        {
            splitIndex = trimmed.LastIndexOf(' ', midpoint);
        }

        if (splitIndex <= 0 || splitIndex >= trimmed.Length - 1)
        {
            return (trimmed, string.Empty);
        }

        return (trimmed[..splitIndex].Trim(), trimmed[(splitIndex + 1)..].Trim());
    }

    public static bool ShouldPreserveWords(TranscriptSegment segment, string newText) =>
        string.Equals(segment.Text, newText, StringComparison.Ordinal);

    public static IReadOnlyList<TranscriptWord> CreateTranscriptWords(
        IReadOnlyList<RecognizedTranscriptWord> words) =>
        words
            .OrderBy(word => word.WordIndex)
            .Select((word, index) => TranscriptWord.Create(
                index,
                word.StartSeconds,
                word.EndSeconds,
                word.Text,
                word.Confidence))
            .ToArray();

    public static IReadOnlyList<TranscriptWord> CloneWords(
        IReadOnlyList<TranscriptWord> words) =>
        ReindexWords(words);

    public static IReadOnlyList<TranscriptWord> CloneWordsInRange(
        IReadOnlyList<TranscriptWord> words,
        double startSeconds,
        double endSeconds) =>
        ReindexWords(words.Where(word => word.StartSeconds >= startSeconds && word.EndSeconds <= endSeconds));

    public static IReadOnlyList<TranscriptWord> CloneMergedWords(
        IReadOnlyList<TranscriptSegment> segments) =>
        ReindexWords(segments.SelectMany(segment => segment.Words).OrderBy(word => word.StartSeconds));

    private static IReadOnlyList<TranscriptWord> ReindexWords(IEnumerable<TranscriptWord> words) =>
        words
            .OrderBy(word => word.StartSeconds)
            .Select((word, index) => TranscriptWord.Create(
                index,
                word.StartSeconds,
                word.EndSeconds,
                word.Text,
                word.Confidence))
            .ToArray();

    public static (double StartSeconds, double EndSeconds) ResolveRecognizedTiming(
        TranscriptSegment original,
        RecognizedTranscriptSegment recognized)
    {
        double startSeconds = double.IsFinite(recognized.StartSeconds)
            ? recognized.StartSeconds
            : original.StartSeconds;
        double endSeconds = double.IsFinite(recognized.EndSeconds)
            ? recognized.EndSeconds
            : original.EndSeconds;

        if (endSeconds <= startSeconds)
        {
            return (original.StartSeconds, original.EndSeconds);
        }

        return (startSeconds, endSeconds);
    }

    public static string MergeSegmentText(string first, string second)
    {
        if (string.IsNullOrWhiteSpace(first))
        {
            return second.Trim();
        }

        if (string.IsNullOrWhiteSpace(second))
        {
            return first.Trim();
        }

        return $"{first.Trim()} {second.Trim()}";
    }

    public static string? MergeDetectedLanguage(string? first, string? second)
    {
        string? normalizedFirst = NormalizeTranscriptLanguageCode(first);
        string? normalizedSecond = NormalizeTranscriptLanguageCode(second);
        if (normalizedFirst is null)
        {
            return normalizedSecond;
        }

        if (normalizedSecond is null)
        {
            return normalizedFirst;
        }

        return string.Equals(normalizedFirst, normalizedSecond, StringComparison.Ordinal)
            ? normalizedFirst
            : null;
    }

    public static string NormalizeRequiredTranscriptLanguageCode(string languageCode)
    {
        string? normalized = NormalizeTranscriptLanguageCode(languageCode);
        if (normalized is null)
        {
            throw new InvalidOperationException("Set the transcript language before starting translation.");
        }

        return normalized;
    }

    public static string NormalizeTranslationTargetLanguageCode(string targetLanguage)
    {
        string? normalized = NormalizeTranslationTargetLanguageCodeOrNull(targetLanguage);
        if (normalized is null)
        {
            throw new InvalidOperationException("Select a translation target language before starting translation.");
        }

        return normalized;
    }

    public static string? NormalizeTranscriptLanguageCode(string? languageCode)
    {
        string? normalized = TryNormalizeTranscriptLanguageCode(languageCode);
        return normalized is null
            ? null
            : normalized;
    }

    public static string? ResolveDetectedTranscriptLanguage(IReadOnlyList<RecognizedTranscriptSegment> recognizedSegments)
    {
        return recognizedSegments
            .Select(segment => new
            {
                Language = NormalizeTranscriptLanguageCode(segment.DetectedLanguage),
                segment.Text
            })
            .Where(segment => segment.Language is not null &&
                              HasLanguageDetectionEvidence(segment.Language, segment.Text))
            .Select(segment => segment.Language)
            .GroupBy(language => language!, StringComparer.Ordinal)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key, StringComparer.Ordinal)
            .Select(static group => group.Key)
            .FirstOrDefault();
    }

    public static string? ResolveDetectedTranscriptLanguage(IReadOnlyList<TranscriptSegment> transcriptSegments)
    {
        return transcriptSegments
            .Select(segment => new
            {
                Language = NormalizeTranscriptLanguageCode(segment.DetectedLanguage),
                segment.Text
            })
            .Where(segment => segment.Language is not null &&
                              HasLanguageDetectionEvidence(segment.Language, segment.Text))
            .Select(segment => segment.Language)
            .GroupBy(language => language!, StringComparer.Ordinal)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key, StringComparer.Ordinal)
            .Select(static group => group.Key)
            .FirstOrDefault();
    }

    public static string? TryNormalizeTranscriptLanguageCode(string? languageCode) =>
        NormalizeLanguageCodeOrNull(languageCode);

    public static string? NormalizeLanguageCodeOrNull(string? languageCode)
    {
        if (string.IsNullOrWhiteSpace(languageCode))
        {
            return null;
        }

        string normalized = languageCode.Trim().Replace('_', '-').ToLowerInvariant();
        int separatorIndex = normalized.IndexOf('-');
        if (separatorIndex > 0)
        {
            normalized = normalized[..separatorIndex];
        }

        return normalized.Length == 0 ? null : normalized;
    }

    public static bool HasTranscribedSpeechText(string? text) =>
        !string.IsNullOrWhiteSpace(text);

    private static bool HasLanguageDetectionEvidence(string language, string? text)
    {
        if (!HasTranscribedSpeechText(text))
        {
            return false;
        }

        ScriptCounts counts = CountScripts(text!);
        if (counts.Letters == 0)
        {
            return false;
        }

        return language switch
        {
            "zh" => HasExpectedScriptEvidence(counts.Cjk, counts.Latin),
            "ja" => HasExpectedScriptEvidence(counts.Cjk + counts.Kana, counts.Latin),
            "ko" => HasExpectedScriptEvidence(counts.Hangul, counts.Latin),
            _ => true
        };
    }

    private static bool HasExpectedScriptEvidence(int expectedScriptLetters, int latinLetters) =>
        expectedScriptLetters > 0 && latinLetters <= expectedScriptLetters * 3;

    private static ScriptCounts CountScripts(string text)
    {
        int letters = 0;
        int latin = 0;
        int cjk = 0;
        int kana = 0;
        int hangul = 0;

        foreach (char character in text)
        {
            if (!char.IsLetter(character))
            {
                continue;
            }

            letters++;
            if (IsLatin(character))
            {
                latin++;
            }
            else if (IsCjk(character))
            {
                cjk++;
            }
            else if (IsKana(character))
            {
                kana++;
            }
            else if (IsHangul(character))
            {
                hangul++;
            }
        }

        return new ScriptCounts(letters, latin, cjk, kana, hangul);
    }

    private static bool IsLatin(char character) =>
        character is (>= 'A' and <= 'Z') or
            (>= 'a' and <= 'z') or
            (>= '\u00C0' and <= '\u024F') or
            (>= '\u1E00' and <= '\u1EFF');

    private static bool IsCjk(char character) =>
        character is (>= '\u3400' and <= '\u4DBF') or
            (>= '\u4E00' and <= '\u9FFF') or
            (>= '\uF900' and <= '\uFAFF');

    private static bool IsKana(char character) =>
        character is (>= '\u3040' and <= '\u30FF') or
            (>= '\u31F0' and <= '\u31FF');

    private static bool IsHangul(char character) =>
        character is (>= '\uAC00' and <= '\uD7AF') or
            (>= '\u1100' and <= '\u11FF') or
            (>= '\u3130' and <= '\u318F');

    private sealed record ScriptCounts(int Letters, int Latin, int Cjk, int Kana, int Hangul);

    public static string? ResolveSelectedTranslationTargetLanguage(
        string? sourceLanguage,
        string? requestedTargetLanguage,
        IReadOnlyList<TranslationTargetLanguageOption> supportedTargetLanguages)
    {
        if (supportedTargetLanguages.Count == 0)
        {
            return NormalizeTranslationTargetLanguageCodeOrNull(requestedTargetLanguage);
        }

        string? normalizedRequested = NormalizeTranslationTargetLanguageCodeOrNull(requestedTargetLanguage);
        if (normalizedRequested is not null &&
            supportedTargetLanguages.Any(option => string.Equals(option.LanguageCode, normalizedRequested, StringComparison.Ordinal)))
        {
            return normalizedRequested;
        }

        TranslationTargetLanguageOption? preferred = supportedTargetLanguages
            .FirstOrDefault(option => IsPreferredDefaultTarget(sourceLanguage, option.LanguageCode));
        return preferred?.LanguageCode ?? supportedTargetLanguages[0].LanguageCode;
    }

    private static bool IsPreferredDefaultTarget(string? sourceLanguage, string targetLanguage)
    {
        string? normalizedSource = NormalizeTranscriptLanguageCode(sourceLanguage);
        string? normalizedTarget = NormalizeTranslationTargetLanguageCodeOrNull(targetLanguage);
        return normalizedSource switch
        {
            EnglishLanguageCode => normalizedTarget == SpanishLanguageCode,
            SpanishLanguageCode => normalizedTarget == EnglishLanguageCode,
            _ => false
        };
    }

    public static string? NormalizeTranslationTargetLanguageCodeOrNull(string? targetLanguage) =>
        NormalizeLanguageCodeOrNull(targetLanguage);

    public static IReadOnlySet<int> BuildStaleTranslatedSegmentIndices(
        TranscriptRevision? currentRevision,
        IReadOnlyList<TranscriptSegment> transcriptSegments,
        TranslationRevision? currentTranslationRevision,
        IReadOnlyList<TranslatedSegment> translatedSegments)
    {
        if (currentRevision is null || currentTranslationRevision is null)
        {
            return new HashSet<int>();
        }

        if (translatedSegments.Count == 0)
        {
            return transcriptSegments.Select(segment => segment.SegmentIndex).ToHashSet();
        }

        Dictionary<int, TranscriptSegment> sourceSegmentsByIndex = transcriptSegments
            .ToDictionary(segment => segment.SegmentIndex);
        var stale = new HashSet<int>();
        var translatedSegmentIndices = new HashSet<int>();
        foreach (TranslatedSegment translatedSegment in translatedSegments)
        {
            translatedSegmentIndices.Add(translatedSegment.SegmentIndex);
            if (!sourceSegmentsByIndex.TryGetValue(translatedSegment.SegmentIndex, out TranscriptSegment? sourceSegment))
            {
                stale.Add(translatedSegment.SegmentIndex);
                continue;
            }

            string sourceHash = ComputeSourceSegmentHash(sourceSegment);
            if (!string.Equals(sourceHash, translatedSegment.SourceSegmentHash, StringComparison.Ordinal))
            {
                stale.Add(translatedSegment.SegmentIndex);
            }
        }

        foreach (TranscriptSegment sourceSegment in sourceSegmentsByIndex.Values)
        {
            if (!translatedSegmentIndices.Contains(sourceSegment.SegmentIndex))
            {
                stale.Add(sourceSegment.SegmentIndex);
            }
        }

        return stale;
    }

    public static string ComputeSourceSegmentHash(TranscriptSegment segment)
    {
        string payload = string.Join(
            '\u001f',
            segment.SegmentIndex.ToString(System.Globalization.CultureInfo.InvariantCulture),
            segment.StartSeconds.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            segment.EndSeconds.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            segment.Text);
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(payload));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    public static TranscriptRegionPlan BuildTranscriptRegionPlan(
        IReadOnlyList<SpeechRegion> vadRegions,
        DiarizationResult? diarizationResult,
        double durationSeconds)
    {
        if (diarizationResult is null || diarizationResult.Turns.Count == 0)
        {
            return new TranscriptRegionPlan(
                vadRegions.OrderBy(static region => region.Index).ToArray(),
                new Dictionary<int, Guid>());
        }

        var drafts = new List<TranscriptRegionDraft>();
        foreach (SpeakerTurn turn in diarizationResult.Turns.OrderBy(static turn => turn.StartSeconds))
        {
            double start = Math.Max(0d, turn.StartSeconds);
            double end = Math.Min(durationSeconds, turn.EndSeconds);
            if (end <= start)
            {
                continue;
            }

            if (drafts.Count > 0)
            {
                TranscriptRegionDraft previous = drafts[^1];
                double gapSeconds = start - previous.EndSeconds;
                double mergedDurationSeconds = end - previous.StartSeconds;
                if (previous.SpeakerId == turn.SpeakerId &&
                    gapSeconds <= DiarizedTranscriptMergeGapSeconds &&
                    mergedDurationSeconds <= MaxDiarizedTranscriptRegionSeconds)
                {
                    drafts[^1] = previous with { EndSeconds = Math.Max(previous.EndSeconds, end) };
                    continue;
                }
            }

            drafts.Add(new TranscriptRegionDraft(start, end, turn.SpeakerId));
        }

        IReadOnlyList<TranscriptRegionDraft> expandedDrafts = ExpandTranscriptRegionDrafts(drafts, durationSeconds);
        SpeechRegion[] regions = expandedDrafts
            .Select((draft, index) => new SpeechRegion(index, draft.StartSeconds, draft.EndSeconds))
            .ToArray();
        Dictionary<int, Guid> speakerIdsByIndex = expandedDrafts
            .Select((draft, index) => new { Index = index, draft.SpeakerId })
            .ToDictionary(entry => entry.Index, entry => entry.SpeakerId);

        return regions.Length == 0
            ? new TranscriptRegionPlan(
                vadRegions.OrderBy(static region => region.Index).ToArray(),
                new Dictionary<int, Guid>())
            : new TranscriptRegionPlan(regions, speakerIdsByIndex);
    }

    private static IReadOnlyList<TranscriptRegionDraft> ExpandTranscriptRegionDrafts(
        IReadOnlyList<TranscriptRegionDraft> drafts,
        double durationSeconds)
    {
        if (drafts.Count == 0)
        {
            return [];
        }

        TranscriptRegionDraft[] expandedDrafts = drafts
            .ToArray();
        for (int index = 0; index < expandedDrafts.Length; index++)
        {
            TranscriptRegionDraft draft = expandedDrafts[index];
            expandedDrafts[index] = draft with
            {
                StartSeconds = Math.Max(0d, draft.StartSeconds - DiarizedTranscriptRegionPaddingSeconds),
                EndSeconds = Math.Min(durationSeconds, draft.EndSeconds + DiarizedTranscriptRegionPaddingSeconds)
            };
        }

        for (int index = 1; index < expandedDrafts.Length; index++)
        {
            TranscriptRegionDraft previous = expandedDrafts[index - 1];
            TranscriptRegionDraft current = expandedDrafts[index];
            if (current.StartSeconds >= previous.EndSeconds)
            {
                continue;
            }

            double boundary = Math.Clamp(
                (drafts[index - 1].EndSeconds + drafts[index].StartSeconds) / 2d,
                previous.StartSeconds,
                current.EndSeconds);
            expandedDrafts[index - 1] = previous with { EndSeconds = Math.Min(previous.EndSeconds, boundary) };
            expandedDrafts[index] = current with { StartSeconds = Math.Max(current.StartSeconds, boundary) };
        }

        return expandedDrafts
            .Where(static draft => draft.EndSeconds > draft.StartSeconds)
            .ToArray();
    }

    private sealed record TranscriptRegionDraft(
        double StartSeconds,
        double EndSeconds,
        Guid SpeakerId);
}

public sealed record StemAudioRoute(
    string AsrAudioRelativePath,
    string MixSourceAudioRelativePath,
    string? WarningMessage);
