using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;

namespace Trackdub.Application.Transcripts;

public sealed class TranscriptArtifactWriter(
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository)
{
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IFileFingerprintService fileFingerprintService = fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));

    public async Task WriteSpeechRegionsArtifactAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        IReadOnlyList<SpeechRegion> regions,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        // Run-scoped path: each stage run writes a unique file. The artifact metadata commit
        // below is the atomic pointer-swap that makes the new run visible to readers, which
        // resolve via the DB record (kind + latest StageRun). A crash between file write and
        // metadata save leaves only an orphan file (cleanable by a sweep), never a corrupted
        // pointer at a stable path.
        string relativePath = ProjectArtifactPaths.GetSpeechRegionsRelativePath(stageRunId);
        await artifactStore.WriteJsonAsync(
            relativePath,
            new SpeechRegionsArtifactDocument(stageRunId, regions, DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAsset.Id,
            ArtifactKind.SpeechRegions,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: stageRunId,
            Provenance: "generated-vad");

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Returns the most recent committed speech-regions artifact for the project, resolved via
    /// the artifact repository so callers always see a fully-written file (file is committed
    /// before its DB row, so a non-null row guarantees the file is present and complete).
    /// </summary>
    public async Task<IReadOnlyList<SpeechRegion>?> TryReadSpeechRegionsAsync(
        Guid projectId,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ProjectArtifact> projectArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(projectId, cancellationToken)
            .ConfigureAwait(false);
        ProjectArtifact? artifact = projectArtifacts
            .Where(a => a.Kind == ArtifactKind.SpeechRegions)
            .OrderByDescending(a => a.CreatedAtUtc)
            .FirstOrDefault();
        if (artifact is null || !artifactStore.Exists(artifact.RelativePath))
        {
            return null;
        }

        SpeechRegionsArtifactDocument? document = await artifactStore
            .ReadJsonAsync<SpeechRegionsArtifactDocument>(artifact.RelativePath, cancellationToken)
            .ConfigureAwait(false);
        return document?.Regions;
    }

    public async Task<DiarizationResult?> TryReadDiarizationResultAsync(
        Guid projectId,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ProjectArtifact> projectArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(projectId, cancellationToken)
            .ConfigureAwait(false);
        ProjectArtifact? artifact = projectArtifacts
            .Where(a => a.Kind == ArtifactKind.DiarizationResult)
            .OrderByDescending(a => a.CreatedAtUtc)
            .FirstOrDefault();
        if (artifact is null || !artifactStore.Exists(artifact.RelativePath))
        {
            return null;
        }

        DiarizationResultArtifactDocument? document = await artifactStore
            .ReadJsonAsync<DiarizationResultArtifactDocument>(artifact.RelativePath, cancellationToken)
            .ConfigureAwait(false);
        if (document is null || document.Speakers.Count == 0 || document.Turns.Count == 0)
        {
            return null;
        }

        return new DiarizationResult(document.Speakers, document.Turns);
    }

    public async Task WriteDiarizationArtifactAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        IReadOnlyList<ProjectSpeaker> speakers,
        IReadOnlyList<SpeakerTurn> turns,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        // Run-scoped path; see WriteSpeechRegionsArtifactAsync for the atomic-swap rationale.
        string relativePath = ProjectArtifactPaths.GetDiarizationResultRelativePath(stageRunId);
        await artifactStore.WriteJsonAsync(
            relativePath,
            new DiarizationResultArtifactDocument(stageRunId, speakers, turns, DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAsset.Id,
            ArtifactKind.DiarizationResult,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: stageRunId,
            Provenance: "generated-diarization");

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task WriteRawAsrTranscriptArtifactAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        IReadOnlyList<RecognizedTranscriptSegment> segments,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        string relativePath = ProjectArtifactPaths.GetRawAsrTranscriptRelativePath(stageRunId);
        await artifactStore.WriteJsonAsync(
            relativePath,
            RawAsrTranscriptArtifactDocument.From(stageRunId, segments),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAsset.Id,
            ArtifactKind.TranscriptRevision,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: stageRunId,
            Provenance: "generated-asr-raw");

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task<RawAsrTranscriptArtifact?> TryReadRawAsrTranscriptAsync(
        Guid projectId,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ProjectArtifact> projectArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(projectId, cancellationToken)
            .ConfigureAwait(false);
        ProjectArtifact? artifact = projectArtifacts
            .Where(a => a.Kind == ArtifactKind.TranscriptRevision &&
                        string.Equals(a.Provenance, "generated-asr-raw", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(a => a.CreatedAtUtc)
            .FirstOrDefault();
        if (artifact is null || !artifactStore.Exists(artifact.RelativePath))
        {
            return null;
        }

        RawAsrTranscriptArtifactDocument? document = await artifactStore
            .ReadJsonAsync<RawAsrTranscriptArtifactDocument>(artifact.RelativePath, cancellationToken)
            .ConfigureAwait(false);
        if (document is null || document.Segments.Count == 0)
        {
            return null;
        }

        return new RawAsrTranscriptArtifact(
            document.StageRunId,
            document.Segments
                .OrderBy(segment => segment.SegmentIndex)
                .Select(segment => new RecognizedTranscriptSegment(
                    segment.SegmentIndex,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    segment.Text,
                    segment.DetectedLanguage,
                    segment.Words
                        .OrderBy(word => word.WordIndex)
                        .Select(word => new RecognizedTranscriptWord(
                            word.WordIndex,
                            word.StartSeconds,
                            word.EndSeconds,
                            word.Text,
                            word.Confidence))
                        .ToArray()))
                .ToArray());
    }

    public async Task WriteTextRefinementProvenanceArtifactAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        TextRefinementProvenanceArtifactDocument document,
        CancellationToken cancellationToken)
    {
        string relativePath = ProjectArtifactPaths.GetTextRefinementProvenanceRelativePath(document.TranscriptRevisionId);
        await artifactStore.WriteJsonAsync(relativePath, document, cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAsset.Id,
            ArtifactKind.TranscriptRevision,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: document.TextRefinementStageRunId,
            Provenance: "text-refinement-asr-provenance");

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task WriteTranscriptArtifactAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        TranscriptRevision revision,
        IReadOnlyList<TranscriptSegment> segments,
        Guid? stageRunId,
        string provenance,
        CancellationToken cancellationToken)
    {
        string relativePath = ProjectArtifactPaths.GetTranscriptRevisionRelativePath(revision.RevisionNumber);
        await artifactStore.WriteJsonAsync(
            relativePath,
            TranscriptRevisionArtifactDocument.From(revision, segments, provenance),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAsset.Id,
            ArtifactKind.TranscriptRevision,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: stageRunId,
            Provenance: provenance);

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task WriteTranslationArtifactAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        TranslationRevision revision,
        IReadOnlyList<TranslatedSegment> segments,
        Guid? stageRunId,
        string provenance,
        CancellationToken cancellationToken)
    {
        string relativePath = ProjectArtifactPaths.GetTranslationRevisionRelativePath(
            revision.TargetLanguage,
            revision.RevisionNumber);
        await artifactStore.WriteJsonAsync(
            relativePath,
            TranslationRevisionArtifactDocument.From(revision, segments, provenance),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAsset.Id,
            ArtifactKind.TranslationRevision,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: stageRunId,
            Provenance: provenance);

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task<Trackdub.Contracts.Transcripts.PreparedCommitArtifact> PrepareTranslationArtifactAsync(
        Guid projectId, MediaAsset mediaAsset, TranslationRevision revision,
        IReadOnlyList<TranslatedSegment> segments, Guid? stageRunId, string provenance, CancellationToken cancellationToken)
    {
        string relativePath = $"artifacts/translation/{revision.TargetLanguage}/translation-revision-{revision.RevisionNumber:D4}-{revision.Id:N}.json";
        ArtifactWriteHandle handle = artifactStore.CreateWriteHandle(relativePath);
        try
        {
            await using (var file = new FileStream(handle.TemporaryPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await System.Text.Json.JsonSerializer.SerializeAsync(file,
                    TranslationRevisionArtifactDocument.From(revision, segments, provenance),
                    new System.Text.Json.JsonSerializerOptions { PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase, WriteIndented = true },
                    cancellationToken).ConfigureAwait(false);
            }
            FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(handle.TemporaryPath, cancellationToken).ConfigureAwait(false);
            var artifact = new ProjectArtifact(Guid.NewGuid(), projectId, mediaAsset.Id, ArtifactKind.TranslationRevision,
                relativePath, fingerprint.Sha256, fingerprint.SizeBytes, null, null, null, DateTimeOffset.UtcNow,
                stageRunId, provenance);
            return new Trackdub.Contracts.Transcripts.PreparedCommitArtifact(artifact, handle);
        }
        catch
        {
            await handle.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    private sealed record SpeechRegionsArtifactDocument(
        Guid StageRunId,
        IReadOnlyList<SpeechRegion> Regions,
        DateTimeOffset GeneratedAtUtc);

    private sealed record DiarizationResultArtifactDocument(
        Guid StageRunId,
        IReadOnlyList<ProjectSpeaker> Speakers,
        IReadOnlyList<SpeakerTurn> Turns,
        DateTimeOffset CreatedAtUtc);

    private sealed record TranscriptRevisionArtifactDocument(
        Guid RevisionId,
        Guid? StageRunId,
        int RevisionNumber,
        string Provenance,
        DateTimeOffset CreatedAtUtc,
        IReadOnlyList<TranscriptSegmentArtifactDocument> Segments)
    {
        public static TranscriptRevisionArtifactDocument From(
            TranscriptRevision revision,
            IReadOnlyList<TranscriptSegment> segments,
            string provenance) =>
            new(
                revision.Id,
                revision.StageRunId,
                revision.RevisionNumber,
                provenance,
                revision.CreatedAtUtc,
                segments
                    .OrderBy(segment => segment.SegmentIndex)
                    .Select(segment => new TranscriptSegmentArtifactDocument(
                        segment.SegmentIndex,
                        segment.StartSeconds,
                        segment.EndSeconds,
                        segment.Text,
                        segment.SpeakerId,
                        segment.DetectedLanguage,
                        segment.Words
                            .OrderBy(word => word.WordIndex)
                            .Select(word => new TranscriptWordArtifactDocument(
                                word.WordIndex,
                                word.StartSeconds,
                                word.EndSeconds,
                                word.Text,
                                word.Confidence))
                            .ToArray(),
                        segment.SpeakerAssignmentOrigin))
                    .ToArray());
    }

    private sealed record TranslationRevisionArtifactDocument(
        Guid RevisionId,
        Guid? StageRunId,
        Guid SourceTranscriptRevisionId,
        string TargetLanguage,
        string? TranslationProvider,
        string? ModelId,
        string? ExecutionProvider,
        TranslationModelProvenance? ModelProvenance,
        int RevisionNumber,
        string Provenance,
        DateTimeOffset CreatedAtUtc,
        IReadOnlyList<TranslatedSegmentArtifactDocument> Segments)
    {
        public static TranslationRevisionArtifactDocument From(
            TranslationRevision revision,
            IReadOnlyList<TranslatedSegment> segments,
            string provenance) =>
            new(
                revision.Id,
                revision.StageRunId,
                revision.SourceTranscriptRevisionId,
                revision.TargetLanguage,
                revision.TranslationProvider,
                revision.ModelId,
                revision.ExecutionProvider,
                revision.ModelProvenance,
                revision.RevisionNumber,
                provenance,
                revision.CreatedAtUtc,
                segments
                    .OrderBy(segment => segment.SegmentIndex)
                    .Select(segment => new TranslatedSegmentArtifactDocument(
                        segment.SegmentIndex,
                        segment.StartSeconds,
                        segment.EndSeconds,
                        segment.Text,
                        segment.SourceSegmentHash,
                        segment.Words
                            .OrderBy(static word => word.WordIndex)
                            .Select(static word => new TranslatedWordArtifactDocument(
                                word.WordIndex,
                                word.StartSeconds,
                                word.EndSeconds,
                                word.Text))
                            .ToArray(),
                        segment.Origin))
                    .ToArray());
    }

    private sealed record TranscriptSegmentArtifactDocument(
        int SegmentIndex,
        double StartSeconds,
        double EndSeconds,
        string Text,
        Guid? SpeakerId,
        string? DetectedLanguage,
        IReadOnlyList<TranscriptWordArtifactDocument> Words,
        SpeakerAssignmentOrigin SpeakerAssignmentOrigin);

    private sealed record TranscriptWordArtifactDocument(
        int WordIndex,
        double StartSeconds,
        double EndSeconds,
        string Text,
        double? Confidence);

    private sealed record TranslatedSegmentArtifactDocument(
        int SegmentIndex,
        double StartSeconds,
        double EndSeconds,
        string Text,
        string? SourceSegmentHash,
        IReadOnlyList<TranslatedWordArtifactDocument> Words,
        TranslationSegmentOrigin Origin);

    private sealed record TranslatedWordArtifactDocument(
        int WordIndex,
        double StartSeconds,
        double EndSeconds,
        string Text);

    private sealed record RawAsrTranscriptArtifactDocument(
        Guid StageRunId,
        IReadOnlyList<RawAsrTranscriptSegmentArtifactDocument> Segments)
    {
        public static RawAsrTranscriptArtifactDocument From(
            Guid stageRunId,
            IReadOnlyList<RecognizedTranscriptSegment> segments) =>
            new(
                stageRunId,
                segments
                    .OrderBy(segment => segment.Index)
                    .Select(segment => new RawAsrTranscriptSegmentArtifactDocument(
                        segment.Index,
                        segment.StartSeconds,
                        segment.EndSeconds,
                        segment.Text,
                        segment.DetectedLanguage,
                        segment.Words
                            .OrderBy(word => word.WordIndex)
                            .Select(word => new RawAsrTranscriptWordArtifactDocument(
                                word.WordIndex,
                                word.StartSeconds,
                                word.EndSeconds,
                                word.Text,
                                word.Confidence))
                            .ToArray()))
                    .ToArray());
    }

    private sealed record RawAsrTranscriptSegmentArtifactDocument(
        int SegmentIndex,
        double StartSeconds,
        double EndSeconds,
        string Text,
        string? DetectedLanguage,
        IReadOnlyList<RawAsrTranscriptWordArtifactDocument> Words);

    private sealed record RawAsrTranscriptWordArtifactDocument(
        int WordIndex,
        double StartSeconds,
        double EndSeconds,
        string Text,
        double? Confidence);
}
