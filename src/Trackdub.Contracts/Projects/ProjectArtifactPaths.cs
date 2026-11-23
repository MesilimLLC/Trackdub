namespace Trackdub.Contracts.Projects;

public static class ProjectArtifactPaths
{
    public const string DatabaseFileName = "trackdub.db";
    public const string ManifestRelativePath = "manifest.json";
    public const string SequenceRelativePath = "sequence.json";
    public const string VoiceCloneAuditRelativePath = "voice_clone_audit.jsonl";
    public const string SourceReferenceRelativePath = "media/source-reference.json";
    public const string NormalizedAudioRelativePath = "media/normalized_audio.wav";
    public const string StemSeparationSourceAudioRelativePath = "media/stem_separation_source_audio.wav";
    public const string WaveformSummaryRelativePath = "artifacts/waveform/normalized_audio.waveform.json";
    public const string SpeechRegionsRelativePath = "artifacts/audio/speech-regions.json";
    public const string AudioQualityAnalysisDirectoryRelativePath = "artifacts/audio/quality";
    public const string SpeechEnhancedAudioDirectoryRelativePath = "artifacts/audio/speech-enhancement";
    public const string SpeechProcessedAudioDirectoryRelativePath = "artifacts/audio/speech-processing";
    public const string ReferenceClipDirectoryRelativePath = "artifacts/reference-clips";
    public const string StemsDirectoryRelativePath = "artifacts/stems";
    public const string MixDirectoryRelativePath = "artifacts/mix";
    public const string MixPlanRelativePath = "artifacts/mix/mix-plan.json";
    public const string PreviewMixDirectoryRelativePath = "artifacts/preview";
    public const string ExportDirectoryRelativePath = "artifacts/export";
    public const string TranslationDirectoryRelativePath = "artifacts/translation";
    public const string TtsDirectoryRelativePath = "artifacts/tts";
    public const string LipSyncDirectoryRelativePath = "artifacts/lip-sync";
    public const string LipSynthesisDirectoryRelativePath = "artifacts/lip-synthesis";
    public const string PipelineDegradationDirectoryRelativePath = "artifacts/degradation";
    public const string OverlapRescueDirectoryRelativePath = "artifacts/overlap-rescue";

    public static readonly string[] RequiredDirectories =
    [
        "media",
        "artifacts",
        "artifacts/audio",
        "artifacts/audio/quality",
        "artifacts/audio/speech-enhancement",
        "artifacts/audio/speech-processing",
        "artifacts/degradation",
        "artifacts/mix",
        "artifacts/preview",
        "artifacts/export",
        "artifacts/reference-clips",
        "artifacts/stems",
        "artifacts/translation/en",
        "artifacts/translation",
        "artifacts/translation/es",
        "artifacts/transcript",
        "artifacts/tts",
        "artifacts/waveform",
        "artifacts/overlap-rescue",
        "logs",
        "temp"
    ];

    public static string GetSpeechRegionsRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"artifacts/audio/speech-regions-{stageRunId:N}.json";
    }

    public static string GetDiarizationResultRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"pipeline/diarization-result-{stageRunId:N}.json";
    }

    public static string GetAudioQualityAnalysisRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{AudioQualityAnalysisDirectoryRelativePath}/{stageRunId:D}/analysis.json";
    }

    public static string GetSpeechEnhancedAudioRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{SpeechEnhancedAudioDirectoryRelativePath}/{stageRunId:D}/speech.wav";
    }

    public static string GetSpeechProcessedAudioRelativePath(Guid stageRunId, string stageName)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        if (string.IsNullOrWhiteSpace(stageName))
        {
            throw new ArgumentException("Stage name is required.", nameof(stageName));
        }

        string normalizedStageName = stageName.Trim().ToLowerInvariant();
        return $"{SpeechProcessedAudioDirectoryRelativePath}/{stageRunId:D}/{normalizedStageName}.wav";
    }

    public static string GetTranscriptRevisionRelativePath(int revisionNumber)
    {
        if (revisionNumber <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(revisionNumber), "Revision number must be positive.");
        }

        return $"artifacts/transcript/transcript-revision-{revisionNumber:D4}.json";
    }

    public static string GetRawAsrTranscriptRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"artifacts/transcript/raw-asr-{stageRunId:N}.json";
    }

    public static string GetTextRefinementProvenanceRelativePath(Guid transcriptRevisionId)
    {
        if (transcriptRevisionId == Guid.Empty)
        {
            throw new ArgumentException("Transcript revision id is required.", nameof(transcriptRevisionId));
        }

        return $"artifacts/transcript/text-refinement-asr-provenance-{transcriptRevisionId:N}.json";
    }

    public static string GetTranslationRevisionRelativePath(string targetLanguage, int revisionNumber)
    {
        if (string.IsNullOrWhiteSpace(targetLanguage))
        {
            throw new ArgumentException("Target language is required.", nameof(targetLanguage));
        }

        if (revisionNumber <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(revisionNumber), "Revision number must be positive.");
        }

        string normalizedTargetLanguage = targetLanguage.Trim().ToLowerInvariant();
        return $"artifacts/translation/{normalizedTargetLanguage}/translation-revision-{revisionNumber:D4}.json";
    }

    public static string GetReferenceClipDirectoryRelativePath(Guid speakerId)
    {
        if (speakerId == Guid.Empty)
        {
            throw new ArgumentException("Speaker id is required.", nameof(speakerId));
        }

        return $"{ReferenceClipDirectoryRelativePath}/{speakerId:D}";
    }

    public static string GetReferenceClipRelativePath(Guid speakerId, DateTimeOffset createdAtUtc)
    {
        DateTimeOffset utcTimestamp = createdAtUtc.ToUniversalTime();
        return $"{GetReferenceClipDirectoryRelativePath(speakerId)}/reference-clip-{utcTimestamp:yyyyMMddHHmmssfff}.wav";
    }

    public static string GetStemVocalsRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{StemsDirectoryRelativePath}/{stageRunId:D}/vocals.wav";
    }

    public static string GetStemVocalsRelativePath(Guid stageRunId, string engineFamily) =>
        $"{GetStemEngineDirectoryRelativePath(stageRunId, engineFamily)}/vocals.wav";

    public static string GetStemAmbianceRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{StemsDirectoryRelativePath}/{stageRunId:D}/ambiance.wav";
    }

    public static string GetStemAmbianceRelativePath(Guid stageRunId, string engineFamily) =>
        $"{GetStemEngineDirectoryRelativePath(stageRunId, engineFamily)}/ambiance.wav";

    public static string GetStemMusicRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{StemsDirectoryRelativePath}/{stageRunId:D}/music.wav";
    }

    public static string GetStemMusicRelativePath(Guid stageRunId, string engineFamily) =>
        $"{GetStemEngineDirectoryRelativePath(stageRunId, engineFamily)}/music.wav";

    public static string GetStemSoundEffectsRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{StemsDirectoryRelativePath}/{stageRunId:D}/sfx.wav";
    }

    public static string GetStemSoundEffectsRelativePath(Guid stageRunId, string engineFamily) =>
        $"{GetStemEngineDirectoryRelativePath(stageRunId, engineFamily)}/sfx.wav";

    public static string GetRawStemRelativePath(Guid stageRunId, string engineFamily, string stemName) =>
        $"{GetStemEngineDirectoryRelativePath(stageRunId, engineFamily)}/{ValidateStemPathSegment(stemName, nameof(stemName))}.wav";

    private static string GetStemEngineDirectoryRelativePath(Guid stageRunId, string engineFamily)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        string normalizedEngineFamily = ValidateStemPathSegment(engineFamily, nameof(engineFamily));
        return $"{StemsDirectoryRelativePath}/{stageRunId:D}/{normalizedEngineFamily}";
    }

    public static string GetTtsTakeRelativePath(Guid speakerId, Guid segmentId, int takeNumber)
    {
        if (speakerId == Guid.Empty)
        {
            throw new ArgumentException("Speaker id is required.", nameof(speakerId));
        }

        if (segmentId == Guid.Empty)
        {
            throw new ArgumentException("Segment id is required.", nameof(segmentId));
        }

        if (takeNumber <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(takeNumber), "Take number must be positive.");
        }

        return $"{TtsDirectoryRelativePath}/{speakerId:D}/{segmentId:D}-take-{takeNumber:D4}.wav";
    }

    public static string GetLipSyncTakeRelativePath(Guid segmentId, Guid stageRunId)
    {
        if (segmentId == Guid.Empty)
            throw new ArgumentException("Segment id is required.", nameof(segmentId));
        if (stageRunId == Guid.Empty)
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));

        return $"{LipSyncDirectoryRelativePath}/{stageRunId:D}/{segmentId:D}.wav";
    }

    /// <summary>
    /// Patched-video clip for one M23 speaker turn. The source video is never overwritten; each
    /// synthesized turn writes a fresh per-turn clip under the stage-run directory.
    /// </summary>
    public static string GetLipSynthesisTakeRelativePath(Guid segmentId, Guid stageRunId)
    {
        if (segmentId == Guid.Empty)
            throw new ArgumentException("Segment id is required.", nameof(segmentId));
        if (stageRunId == Guid.Empty)
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));

        return $"{LipSynthesisDirectoryRelativePath}/{stageRunId:D}/{segmentId:D}.mp4";
    }

    public static string GetTtsCandidateRelativePath(
        Guid speakerId,
        Guid segmentId,
        Guid groupId,
        int candidateIndex,
        Guid? artifactId = null)
    {
        if (speakerId == Guid.Empty)
        {
            throw new ArgumentException("Speaker id is required.", nameof(speakerId));
        }

        if (segmentId == Guid.Empty)
        {
            throw new ArgumentException("Segment id is required.", nameof(segmentId));
        }

        if (groupId == Guid.Empty)
        {
            throw new ArgumentException("Group id is required.", nameof(groupId));
        }

        if (candidateIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(candidateIndex), "Candidate index cannot be negative.");
        }

        if (artifactId == Guid.Empty)
        {
            throw new ArgumentException("Artifact id cannot be empty.", nameof(artifactId));
        }

        string artifactSuffix = artifactId is Guid id ? $"-{id:N}" : string.Empty;
        return $"{TtsDirectoryRelativePath}/{speakerId:D}/{segmentId:D}-candidate-{groupId:N}-{candidateIndex:D2}{artifactSuffix}.wav";
    }

    public static string GetPipelineDegradationRelativePath(Guid artifactId)
    {
        if (artifactId == Guid.Empty)
        {
            throw new ArgumentException("Artifact id is required.", nameof(artifactId));
        }

        return $"{PipelineDegradationDirectoryRelativePath}/{artifactId:D}.json";
    }

    public static string GetPreviewMixRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{PreviewMixDirectoryRelativePath}/{stageRunId:D}/preview.wav";
    }

    public static string GetExportManifestRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{ExportDirectoryRelativePath}/{stageRunId:D}/export-manifest.json";
    }

    public static string GetExportTailFitTakeRelativePath(Guid stageRunId, Guid segmentId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        if (segmentId == Guid.Empty)
        {
            throw new ArgumentException("Segment id is required.", nameof(segmentId));
        }

        return $"{ExportDirectoryRelativePath}/{stageRunId:D}/tail-fit/{segmentId:D}.wav";
    }

    public static string GetExportFailureReportRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{ExportDirectoryRelativePath}/{stageRunId:D}/export-failure.json";
    }

    public static string GetExportAudioRelativePath(Guid stageRunId)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        return $"{ExportDirectoryRelativePath}/{stageRunId:D}/dub.wav";
    }

    public static string GetExportVideoRelativePath(Guid stageRunId, string extension)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        string normalizedExtension = NormalizeExtension(extension);
        return $"{ExportDirectoryRelativePath}/{stageRunId:D}/dubbed{normalizedExtension}";
    }

    public static string GetExportSubtitleRelativePath(Guid stageRunId, string extension)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        string normalizedExtension = NormalizeExtension(extension);
        return $"{ExportDirectoryRelativePath}/{stageRunId:D}/subtitles{normalizedExtension}";
    }

    public static string? ResolveAbsolutePath(string? projectRootPath, string? relativePath)
    {
        if (string.IsNullOrWhiteSpace(projectRootPath) ||
            string.IsNullOrWhiteSpace(relativePath))
        {
            return null;
        }

        if (Path.IsPathRooted(relativePath))
        {
            throw new ArgumentException(
                $"Relative path must not be rooted: '{relativePath}'.",
                nameof(relativePath));
        }

        string? pathRoot = Path.GetPathRoot(relativePath);
        if (!string.IsNullOrEmpty(pathRoot))
        {
            throw new ArgumentException(
                $"Relative path must not include a drive or UNC root: '{relativePath}'.",
                nameof(relativePath));
        }

        string normalizedRoot = Path.GetFullPath(projectRootPath);
        string combined = Path.GetFullPath(Path.Join(
            normalizedRoot,
            relativePath.Replace('/', Path.DirectorySeparatorChar)));

        string rootWithSeparator = normalizedRoot.EndsWith(Path.DirectorySeparatorChar)
            ? normalizedRoot
            : normalizedRoot + Path.DirectorySeparatorChar;
        if (!string.Equals(combined, normalizedRoot, StringComparison.OrdinalIgnoreCase) &&
            !combined.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"Relative path '{relativePath}' resolves outside the project root '{normalizedRoot}'.",
                nameof(relativePath));
        }

        return combined;
    }

    private static string NormalizeExtension(string extension)
    {
        if (string.IsNullOrWhiteSpace(extension))
        {
            throw new ArgumentException("Extension is required.", nameof(extension));
        }

        string trimmed = extension.Trim().ToLowerInvariant();
        return trimmed.StartsWith('.')
            ? trimmed
            : $".{trimmed}";
    }

    private static string ValidateStemPathSegment(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException("Stem path segment is required.", parameterName);
        }

        string trimmed = value.Trim();
        if (trimmed.Any(static character => !IsStemPathSegmentCharacter(character)))
        {
            throw new ArgumentException(
                "Stem path segment must use lowercase ASCII letters, digits, or hyphens only.",
                parameterName);
        }

        return trimmed;
    }

    private static bool IsStemPathSegmentCharacter(char value) =>
        (value >= 'a' && value <= 'z') ||
        (value >= '0' && value <= '9') ||
        value == '-';

    public static string GetOverlapRescueRegionDirectoryRelativePath(Guid stageRunId, int regionIndex)
    {
        if (stageRunId == Guid.Empty)
        {
            throw new ArgumentException("Stage run id is required.", nameof(stageRunId));
        }

        if (regionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(regionIndex), "Region index cannot be negative.");
        }

        return $"{OverlapRescueDirectoryRelativePath}/{stageRunId:D}/region-{regionIndex}";
    }

    public static string GetOverlapSourceCandidateRelativePath(Guid stageRunId, int regionIndex, int candidateIndex) =>
        $"{GetOverlapRescueRegionDirectoryRelativePath(stageRunId, regionIndex)}/source-candidate-{candidateIndex}.wav";

    public static string GetOverlapRescueMetadataRelativePath(Guid stageRunId, int regionIndex) =>
        $"{GetOverlapRescueRegionDirectoryRelativePath(stageRunId, regionIndex)}/metadata.json";

    public static string GetOverlapRescueCandidateTranscriptRelativePath(Guid stageRunId, int regionIndex, int candidateIndex) =>
        $"{GetOverlapRescueRegionDirectoryRelativePath(stageRunId, regionIndex)}/candidate-{candidateIndex}-transcript.json";
}
