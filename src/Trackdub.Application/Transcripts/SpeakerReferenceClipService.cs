using Trackdub.Application.Artifacts;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

/// <summary>
/// Handles reference-clip extraction, import, and voice-assignment wiring for a speaker.
/// Extracted from <see cref="SpeakerAssignmentService"/> to reduce its constructor arity.
/// </summary>
public sealed class SpeakerReferenceClipService(
    IArtifactStore artifactStore,
    IAudioClipExtractor audioClipExtractor,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository,
    IVoiceAssignmentRepository voiceAssignmentRepository,
    ITtsTakeRepository ttsTakeRepository,
    IReferenceClipAnalyzer referenceClipAnalyzer,
    IReferenceClipTrimmer referenceClipTrimmer)
{
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IAudioClipExtractor audioClipExtractor = audioClipExtractor ?? throw new ArgumentNullException(nameof(audioClipExtractor));
    private readonly IFileFingerprintService fileFingerprintService = fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));
    private readonly IVoiceAssignmentRepository voiceAssignmentRepository = voiceAssignmentRepository ?? throw new ArgumentNullException(nameof(voiceAssignmentRepository));
    private readonly ITtsTakeRepository ttsTakeRepository = ttsTakeRepository ?? throw new ArgumentNullException(nameof(ttsTakeRepository));
    private readonly IReferenceClipAnalyzer referenceClipAnalyzer = referenceClipAnalyzer ?? throw new ArgumentNullException(nameof(referenceClipAnalyzer));
    private readonly IReferenceClipTrimmer referenceClipTrimmer = referenceClipTrimmer ?? throw new ArgumentNullException(nameof(referenceClipTrimmer));

    public async Task ExtractReferenceClipAsync(
        TranscriptProjectState currentState,
        ExtractReferenceClipRequest request,
        CancellationToken cancellationToken)
    {
        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        ClipRange clipRange = ResolveReferenceClipRange(currentState, request);
        ProjectArtifact sourceArtifact = ResolveReferenceClipSourceAudioArtifact(currentState);
        string sourceWavePath = artifactStore.GetPath(sourceArtifact.RelativePath);
        DateTimeOffset now = DateTimeOffset.UtcNow;
        string relativePath = ProjectArtifactPaths.GetReferenceClipRelativePath(request.SpeakerId, now);
        await using var tx = new ArtifactWriteTransaction(artifactStore.CreateWriteHandle(relativePath));
        Guid? savedArtifactId = null;

        try
        {
            _ = await audioClipExtractor.ExtractAsync(
                sourceWavePath,
                clipRange.StartSeconds,
                clipRange.EndSeconds,
                tx.TemporaryPath,
                cancellationToken).ConfigureAwait(false);
            _ = await referenceClipTrimmer
                .TrimAsync(tx.TemporaryPath, cancellationToken)
                .ConfigureAwait(false);
            ReferenceClipAnalysis analysis = await AnalyzeAndValidateReferenceClipAsync(
                tx.TemporaryPath,
                cancellationToken).ConfigureAwait(false);
            Guid artifactId = await CommitReferenceClipAsync(
                tx,
                currentState,
                mediaAsset,
                relativePath,
                analysis,
                $"speaker-reference:{request.SpeakerId:D};source:{sourceArtifact.Kind.ToString().ToLowerInvariant()};active-speech:{analysis.ActiveSpeechSeconds:F3}",
                now,
                cancellationToken).ConfigureAwait(false);
            savedArtifactId = artifactId;
            await AssignReferenceClipArtifactAsync(
                currentState.ProjectState.Project.Id,
                request.SpeakerId,
                artifactId,
                cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await DeleteSavedReferenceClipArtifactBestEffortAsync(savedArtifactId).ConfigureAwait(false);
            DeleteCommittedReferenceClipFileBestEffort(relativePath);
            throw;
        }
    }

    /// <summary>
    /// Estimates each speaker's gender from the pitch of their longest segment, cut from the same
    /// source audio reference clips use (the vocal stem when there is one). Nothing is saved to the
    /// project. A speaker whose audio cannot be read or is not voiced enough maps to null.
    /// </summary>
    public async Task<IReadOnlyDictionary<Guid, string?>> EstimateSpeakerGendersAsync(
        TranscriptProjectState currentState,
        CancellationToken cancellationToken)
    {
        var genders = new Dictionary<Guid, string?>();
        if (currentState.Speakers.Count == 0)
        {
            return genders;
        }

        string sourceWavePath = artifactStore.GetPath(ResolveReferenceClipSourceAudioArtifact(currentState).RelativePath);
        foreach (ProjectSpeaker speaker in currentState.Speakers)
        {
            genders[speaker.Id] = await TryEstimateSpeakerGenderAsync(
                currentState,
                speaker.Id,
                sourceWavePath,
                cancellationToken).ConfigureAwait(false);
        }

        return genders;
    }

    private async Task<string?> TryEstimateSpeakerGenderAsync(
        TranscriptProjectState currentState,
        Guid speakerId,
        string sourceWavePath,
        CancellationToken cancellationToken)
    {
        string clipPath = Path.Join(Path.GetTempPath(), $"trackdub-gender-{Guid.NewGuid():N}.wav");
        try
        {
            ClipRange clipRange = ResolveReferenceClipRange(currentState, new ExtractReferenceClipRequest(speakerId));
            _ = await audioClipExtractor.ExtractAsync(
                sourceWavePath,
                clipRange.StartSeconds,
                clipRange.EndSeconds,
                clipPath,
                cancellationToken).ConfigureAwait(false);
            ReferenceClipAnalysis analysis = await referenceClipAnalyzer
                .AnalyzeAsync(clipPath, cancellationToken)
                .ConfigureAwait(false);
            return analysis.EstimatedGender;
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException or InvalidDataException or UnauthorizedAccessException or ArgumentException)
        {
            return null;
        }
        finally
        {
            try
            {
                File.Delete(clipPath);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // A leftover temp clip costs only disk.
            }
        }
    }

    public async Task ImportReferenceClipAsync(
        TranscriptProjectState currentState,
        ImportReferenceClipRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(currentState);
        ArgumentNullException.ThrowIfNull(request);
        if (!currentState.Speakers.Any(speaker => speaker.Id == request.SpeakerId))
        {
            throw new InvalidOperationException("The selected speaker was not found.");
        }

        if (string.IsNullOrWhiteSpace(request.SourcePath) || !File.Exists(request.SourcePath))
        {
            throw new FileNotFoundException("Reference clip file was not found.", request.SourcePath);
        }

        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        DateTimeOffset now = DateTimeOffset.UtcNow;
        string relativePath = ProjectArtifactPaths.GetReferenceClipRelativePath(request.SpeakerId, now);
        await using var tx = new ArtifactWriteTransaction(artifactStore.CreateWriteHandle(relativePath));
        Guid? savedArtifactId = null;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(tx.TemporaryPath)!);
            File.Copy(request.SourcePath, tx.TemporaryPath, overwrite: true);
            _ = await referenceClipTrimmer
                .TrimAsync(tx.TemporaryPath, cancellationToken)
                .ConfigureAwait(false);
            ReferenceClipAnalysis analysis = await AnalyzeAndValidateReferenceClipAsync(
                tx.TemporaryPath,
                cancellationToken).ConfigureAwait(false);
            Guid artifactId = await CommitReferenceClipAsync(
                tx,
                currentState,
                mediaAsset,
                relativePath,
                analysis,
                $"manual-speaker-reference:{request.SpeakerId:D};active-speech:{analysis.ActiveSpeechSeconds:F3}",
                now,
                cancellationToken).ConfigureAwait(false);
            savedArtifactId = artifactId;
            await AssignReferenceClipArtifactAsync(
                currentState.ProjectState.Project.Id,
                request.SpeakerId,
                artifactId,
                cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await DeleteSavedReferenceClipArtifactBestEffortAsync(savedArtifactId).ConfigureAwait(false);
            DeleteCommittedReferenceClipFileBestEffort(relativePath);
            throw;
        }
    }

    private static ClipRange ResolveReferenceClipRange(
        TranscriptProjectState state,
        ExtractReferenceClipRequest request)
    {
        if (request.SpeakerTurnId is Guid speakerTurnId)
        {
            SpeakerTurn turn = state.SpeakerTurns.FirstOrDefault(candidate => candidate.Id == speakerTurnId && candidate.SpeakerId == request.SpeakerId)
                ?? throw new InvalidOperationException("The selected speaker turn is not available for reference clip extraction.");
            return new ClipRange(turn.StartSeconds, turn.EndSeconds);
        }

        TranscriptSegment? segment = state.TranscriptSegments
            .Where(candidate => candidate.SpeakerId == request.SpeakerId)
            .OrderByDescending(candidate => candidate.EndSeconds - candidate.StartSeconds)
            .FirstOrDefault();
        if (segment is not null)
        {
            return new ClipRange(
                segment.StartSeconds,
                Math.Min(segment.EndSeconds, segment.StartSeconds + ReferenceClipPolicy.RecommendedMaximumActiveSpeechSeconds));
        }

        SpeakerTurn? fallbackTurn = state.SpeakerTurns
            .Where(candidate => candidate.SpeakerId == request.SpeakerId)
            .OrderBy(candidate => candidate.HasOverlap)
            .ThenByDescending(candidate => candidate.Confidence ?? -1d)
            .ThenByDescending(candidate => candidate.EndSeconds - candidate.StartSeconds)
            .FirstOrDefault();

        if (fallbackTurn is not null)
        {
            return new ClipRange(fallbackTurn.StartSeconds, fallbackTurn.EndSeconds);
        }

        throw new InvalidOperationException("No speaker turn or transcript segment is available for reference clip extraction.");
    }

    private static ProjectArtifact ResolveReferenceClipSourceAudioArtifact(TranscriptProjectState state)
    {
        ProjectArtifact? acceptedVocalStem = TranscriptWorkflowUtilities.GetLatestAcceptedVocalStem(state.ProjectState.Artifacts, state.StageRuns);
        ProjectArtifact? routedArtifact = state.ProjectState.Artifacts
            .FirstOrDefault(artifact => string.Equals(
                artifact.RelativePath,
                state.AsrAudioRelativePath,
                StringComparison.OrdinalIgnoreCase));
        if (routedArtifact is { Kind: ArtifactKind.Vocals } &&
            acceptedVocalStem is not null &&
            string.Equals(routedArtifact.RelativePath, acceptedVocalStem.RelativePath, StringComparison.OrdinalIgnoreCase))
        {
            return routedArtifact;
        }

        if (routedArtifact is { Kind: ArtifactKind.SpeechEnhancedAudio } ||
            (routedArtifact is { Kind: ArtifactKind.SpeechProcessedAudio } && acceptedVocalStem is not null))
        {
            return routedArtifact;
        }

        if (acceptedVocalStem is not null)
        {
            return acceptedVocalStem;
        }

        return TranscriptWorkflowUtilities.GetLatestArtifactByKind(state.ProjectState.Artifacts, ArtifactKind.NormalizedAudio)
            ?? throw new InvalidOperationException("Normalized audio is required for reference clip extraction.");
    }

    private async Task<ReferenceClipAnalysis> AnalyzeAndValidateReferenceClipAsync(
        string wavePath,
        CancellationToken cancellationToken)
    {
        ReferenceClipAnalysis analysis = await referenceClipAnalyzer
            .AnalyzeAsync(wavePath, cancellationToken)
            .ConfigureAwait(false);
        if (analysis.ActiveSpeechSeconds < ReferenceClipPolicy.MinimumActiveSpeechSeconds)
        {
            throw new InvalidOperationException(
                $"Reference clip needs at least {ReferenceClipPolicy.MinimumActiveSpeechSeconds:F1} seconds of active speech; detected {analysis.ActiveSpeechSeconds:F2} seconds.");
        }

        return analysis;
    }

    private async Task<Guid> CommitReferenceClipAsync(
        ArtifactWriteTransaction tx,
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        string relativePath,
        ReferenceClipAnalysis analysis,
        string provenance,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        await tx.CommitAsync(artifactStore, cancellationToken).ConfigureAwait(false);
        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(
            artifactStore.GetPath(relativePath), cancellationToken).ConfigureAwait(false);
        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            currentState.ProjectState.Project.Id,
            mediaAsset.Id,
            ArtifactKind.ReferenceClip,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            analysis.TotalDurationSeconds,
            analysis.SampleRate,
            analysis.ChannelCount,
            now,
            StageRunId: null,
            Provenance: provenance);
        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
        return artifact.Id;
    }

    private async Task DeleteSavedReferenceClipArtifactBestEffortAsync(Guid? artifactId)
    {
        if (artifactId is not Guid savedArtifactId)
        {
            return;
        }

        try
        {
            await mediaAssetRepository.DeleteArtifactAsync(savedArtifactId, CancellationToken.None).ConfigureAwait(false);
        }
        catch
        {
            // Preserve the original reference-clip failure; cleanup errors should not replace it.
        }
    }

    private void DeleteCommittedReferenceClipFileBestEffort(string relativePath)
    {
        try
        {
            string committedPath = artifactStore.GetPath(relativePath);
            if (File.Exists(committedPath))
            {
                File.Delete(committedPath);
            }
        }
        catch
        {
            // Preserve the original reference-clip failure; cleanup errors should not replace it.
        }
    }

    private async Task AssignReferenceClipArtifactAsync(
        Guid projectId,
        Guid speakerId,
        Guid referenceClipArtifactId,
        CancellationToken cancellationToken)
    {
        VoiceAssignment? existing = await voiceAssignmentRepository
            .GetAsync(projectId, speakerId, cancellationToken)
            .ConfigureAwait(false);
        VoiceAssignment assignment = existing is null
            ? VoiceAssignment.Create(
                projectId,
                speakerId,
                VoiceCloningDefaults.ChatterboxPrimaryAlias,
                requiresConsent: true,
                referenceClipArtifactId: referenceClipArtifactId)
            : existing with
            {
                RequiresConsent = true,
                ReferenceClipArtifactId = referenceClipArtifactId
            };

        await voiceAssignmentRepository.SaveAsync(assignment, cancellationToken).ConfigureAwait(false);

        if (existing is not null &&
            existing.ReferenceClipArtifactId != referenceClipArtifactId)
        {
            await ttsTakeRepository
                .MarkByVoiceAssignmentStaleAsync(projectId, assignment.Id, cancellationToken)
                .ConfigureAwait(false);
        }
    }
}
