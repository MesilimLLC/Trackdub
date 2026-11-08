using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;

namespace Trackdub.Application.Transcripts;

public sealed record SpeechRegionsArtifactWriteRequest(
    TrackdubProject Project,
    MediaAsset MediaAsset,
    IReadOnlyList<SpeechRegion> Regions,
    StageRunRecord StageRun,
    ProjectArtifact InputAudioArtifact);

public sealed record SpeechRegionsArtifactSnapshot(
    IReadOnlyList<SpeechRegion> Regions,
    Guid StageRunId)
{
    public bool HasRegions => Regions.Count > 0;
}

public sealed record DiarizationArtifactWriteRequest
{
    public required Guid ProjectId { get; init; }
    public required MediaAsset MediaAsset { get; init; }
    public required IReadOnlyList<ProjectSpeaker> Speakers { get; init; }
    public required IReadOnlyList<SpeakerTurn> Turns { get; init; }
    public required Guid StageRunId { get; init; }
    public Guid? VadStageRunId { get; init; }
}

public sealed record RawAsrTranscriptArtifactLineage
{
    public required Guid AsrAudioArtifactId { get; init; }
    public required string AsrAudioSha256 { get; init; }
    public required Guid VadAudioArtifactId { get; init; }
    public required string VadAudioSha256 { get; init; }
    public Guid? VadStageRunId { get; init; }

    public static RawAsrTranscriptArtifactLineage Create(
        ProjectArtifact asrAudioArtifact,
        ProjectArtifact vadAudioArtifact,
        Guid? vadStageRunId) => new()
        {
            AsrAudioArtifactId = asrAudioArtifact.Id,
            AsrAudioSha256 = asrAudioArtifact.Sha256,
            VadAudioArtifactId = vadAudioArtifact.Id,
            VadAudioSha256 = vadAudioArtifact.Sha256,
            VadStageRunId = vadStageRunId
        };
}

public sealed record RawAsrTranscriptArtifactWriteRequest
{
    public required Guid ProjectId { get; init; }
    public required MediaAsset MediaAsset { get; init; }
    public required IReadOnlyList<RecognizedTranscriptSegment> Segments { get; init; }
    public required Guid StageRunId { get; init; }
    public required RawAsrTranscriptArtifactLineage Lineage { get; init; }
}

public sealed record RawAsrTranscriptArtifactReadRequest
{
    public required Guid ProjectId { get; init; }
    public required RawAsrTranscriptArtifactLineage Lineage { get; init; }
}

public sealed class TranscriptArtifactWriter(
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository)
{
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IFileFingerprintService fileFingerprintService = fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));

    public async Task WriteSpeechRegionsArtifactAsync(
        SpeechRegionsArtifactWriteRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        string provenance = GetSpeechRegionsProvenance(request.InputAudioArtifact);

        // Run-scoped path: each stage run writes a unique file. The artifact metadata commit
        // below is the atomic pointer-swap that makes the new run visible to readers, which
        // resolve via the DB record (kind + latest StageRun). A crash between file write and
        // metadata save leaves only an orphan file (cleanable by a sweep), never a corrupted
        // pointer at a stable path.
        string relativePath = ProjectArtifactPaths.GetSpeechRegionsRelativePath(request.StageRun.Id);
        await artifactStore.WriteJsonAsync(
            relativePath,
            new SpeechRegionsArtifactDocument(request.StageRun.Id, request.Regions, DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            request.Project.Id,
            request.MediaAsset.Id,
            ArtifactKind.SpeechRegions,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: request.StageRun.Id,
            Provenance: provenance);

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    internal static string GetSpeechRegionsProvenance(ProjectArtifact inputAudioArtifact)
    {
        ArgumentNullException.ThrowIfNull(inputAudioArtifact);
        if (inputAudioArtifact.Id == Guid.Empty)
        {
            throw new ArgumentException("Input audio artifact ID is required.", nameof(inputAudioArtifact));
        }

        return $"generated-vad:source-audio={inputAudioArtifact.Id:D}:source-sha256={inputAudioArtifact.Sha256}";
    }

    internal static string GetRawAsrProvenance(RawAsrTranscriptArtifactLineage lineage)
    {
        ArgumentNullException.ThrowIfNull(lineage);
        return $"generated-asr-raw:asr-audio={lineage.AsrAudioArtifactId:D}:asr-sha256={lineage.AsrAudioSha256}:vad-audio={lineage.VadAudioArtifactId:D}:vad-sha256={lineage.VadAudioSha256}:vad-run={lineage.VadStageRunId?.ToString("D") ?? "none"}";
    }

    /// <summary>
    /// Resolves the most recent committed speech-regions artifact, optionally matching its input audio.
    /// </summary>
    public async Task<SpeechRegionsArtifactSnapshot?> TryReadSpeechRegionsSnapshotAsync(
        Guid projectId,
        ProjectArtifact? expectedInputAudioArtifact,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ProjectArtifact> projectArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(projectId, cancellationToken)
            .ConfigureAwait(false);
        string? expectedProvenance = expectedInputAudioArtifact is not null
            ? GetSpeechRegionsProvenance(expectedInputAudioArtifact)
            : null;
        ProjectArtifact? artifact = projectArtifacts
            .Where(a => a.Kind == ArtifactKind.SpeechRegions)
            .Where(a => expectedProvenance is null || string.Equals(a.Provenance, expectedProvenance, StringComparison.Ordinal))
            .OrderByDescending(a => a.CreatedAtUtc)
            .FirstOrDefault();
        if (artifact is null || !artifactStore.Exists(artifact.RelativePath))
        {
            return null;
        }

        SpeechRegionsArtifactDocument? document = await TryReadSpeechRegionsDocumentAsync(
            artifact,
            cancellationToken).ConfigureAwait(false);

        if (document is null)
        {
            return null;
        }

        if (artifact.StageRunId is Guid artifactStageRunId && artifactStageRunId != document.StageRunId)
        {
            return null;
        }

        return new SpeechRegionsArtifactSnapshot(document.Regions, document.StageRunId);
    }

    private async Task<SpeechRegionsArtifactDocument?> TryReadSpeechRegionsDocumentAsync(
        ProjectArtifact artifact,
        CancellationToken cancellationToken)
    {
        try
        {
            SpeechRegionsArtifactDocument? document = await artifactStore
                .ReadJsonAsync<SpeechRegionsArtifactDocument>(artifact.RelativePath, cancellationToken)
                .ConfigureAwait(false);
            return document?.Regions is null ? null : document;
        }
        catch (InvalidDataException)
        {
            // A malformed cache entry cannot be resumed. Treat it as absent so VAD can
            // regenerate the snapshot; storage and cancellation failures still propagate.
            return null;
        }
    }

    public async Task<DiarizationResult?> TryReadDiarizationResultAsync(
        Guid projectId,
        Guid? expectedVadStageRunId,
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

        if (expectedVadStageRunId is Guid expectedRunId && document.VadStageRunId != expectedRunId)
        {
            return null;
        }

        return new DiarizationResult(document.Speakers, document.Turns);
    }

    public async Task WriteDiarizationArtifactAsync(
        DiarizationArtifactWriteRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        // Run-scoped path; see WriteSpeechRegionsArtifactAsync for the atomic-swap rationale.
        string relativePath = ProjectArtifactPaths.GetDiarizationResultRelativePath(request.StageRunId);
        await artifactStore.WriteJsonAsync(
            relativePath,
            new DiarizationResultArtifactDocument(
                request.StageRunId,
                request.Speakers,
                request.Turns,
                DateTimeOffset.UtcNow)
            {
                VadStageRunId = request.VadStageRunId
            },
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            request.ProjectId,
            request.MediaAsset.Id,
            ArtifactKind.DiarizationResult,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: request.StageRunId,
            Provenance: "generated-diarization");

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task WriteRawAsrTranscriptArtifactAsync(
        RawAsrTranscriptArtifactWriteRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        string relativePath = ProjectArtifactPaths.GetRawAsrTranscriptRelativePath(request.StageRunId);
        await artifactStore.WriteJsonAsync(
            relativePath,
            RawAsrTranscriptArtifactDocument.From(
                request.StageRunId,
                request.Segments,
                request.Lineage),
            cancellationToken).ConfigureAwait(false);

        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath),
            cancellationToken).ConfigureAwait(false);

        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            request.ProjectId,
            request.MediaAsset.Id,
            ArtifactKind.TranscriptRevision,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            DurationSeconds: null,
            SampleRate: null,
            ChannelCount: null,
            DateTimeOffset.UtcNow,
            StageRunId: request.StageRunId,
            Provenance: GetRawAsrProvenance(request.Lineage));

        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
    }

    public async Task<RawAsrTranscriptArtifact?> TryReadRawAsrTranscriptAsync(
        RawAsrTranscriptArtifactReadRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        IReadOnlyList<ProjectArtifact> projectArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(request.ProjectId, cancellationToken)
            .ConfigureAwait(false);
        ProjectArtifact? artifact = projectArtifacts
            .Where(a => a.Kind == ArtifactKind.TranscriptRevision &&
                        a.Provenance?.StartsWith("generated-asr-raw", StringComparison.OrdinalIgnoreCase) == true)
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

        if (document.Lineage != request.Lineage)
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
        DateTimeOffset CreatedAtUtc)
    {
        public Guid? VadStageRunId { get; init; }
    }

    private sealed record TranscriptRevisionArtifactDocument
    {
        public required Guid RevisionId { get; init; }
        public Guid? StageRunId { get; init; }
        public required int RevisionNumber { get; init; }
        public required string Provenance { get; init; }
        public required DateTimeOffset CreatedAtUtc { get; init; }
        public required IReadOnlyList<TranscriptSegmentArtifactDocument> Segments { get; init; }

        public static TranscriptRevisionArtifactDocument From(
            TranscriptRevision revision,
            IReadOnlyList<TranscriptSegment> segments,
            string provenance) =>
            new()
            {
                RevisionId = revision.Id,
                StageRunId = revision.StageRunId,
                RevisionNumber = revision.RevisionNumber,
                Provenance = provenance,
                CreatedAtUtc = revision.CreatedAtUtc,
                Segments = segments
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
                    .ToArray()
            };
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
        IReadOnlyList<RawAsrTranscriptSegmentArtifactDocument> Segments,
        RawAsrTranscriptArtifactLineage? Lineage = null)
    {
        public static RawAsrTranscriptArtifactDocument From(
            Guid stageRunId,
            IReadOnlyList<RecognizedTranscriptSegment> segments,
            RawAsrTranscriptArtifactLineage lineage) =>
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
                    .ToArray(),
                lineage);
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
