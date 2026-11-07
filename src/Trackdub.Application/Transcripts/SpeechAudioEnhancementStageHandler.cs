using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

public sealed class SpeechAudioEnhancementStageHandler(
    ISpeechAudioEnhancementService speechAudioEnhancementService,
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository,
    IProjectStageRunStore stageRunStore,
    IStudioSettingsService? studioSettingsService = null,
    IApplicationLogger? logger = null,
    PipelineDegradationWriter? degradationWriter = null)
{
    private readonly ISpeechAudioEnhancementService speechAudioEnhancementService = speechAudioEnhancementService ?? throw new ArgumentNullException(nameof(speechAudioEnhancementService));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IFileFingerprintService fileFingerprintService = fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));

    public async Task<SpeechAudioEnhancementStageResult> HandleAsync(
        SpeechAudioEnhancementStageRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, request.ProjectId, StageNames.SpeechEnhancement, cancellationToken)
            .ConfigureAwait(false);

        string enhancedRelativePath = ProjectArtifactPaths.GetSpeechEnhancedAudioRelativePath(stageRun.Id);
        await using ArtifactWriteHandle enhancedHandle = artifactStore.CreateWriteHandle(enhancedRelativePath);

        try
        {
            SpeechAudioEnhancementOptions options = await ResolveOptionsAsync(request, cancellationToken)
                .ConfigureAwait(false);

            SpeechAudioEnhancementResult result = await speechAudioEnhancementService
                .EnhanceAsync(
                    new SpeechAudioEnhancementRequest(
                        artifactStore.GetPath(request.SourceAudioArtifact.RelativePath),
                        enhancedHandle.TemporaryPath,
                        options),
                    cancellationToken)
                .ConfigureAwait(false);

            await artifactStore.CommitAsync(enhancedHandle, cancellationToken).ConfigureAwait(false);

            FileFingerprint fingerprint = await fileFingerprintService
                .ComputeAsync(artifactStore.GetPath(enhancedRelativePath), cancellationToken)
                .ConfigureAwait(false);

            ProjectArtifact artifact = CreateArtifact(
                request,
                stageRun,
                enhancedRelativePath,
                fingerprint,
                result);

            await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);

            stageRun = await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, null, cancellationToken, null)
                .ConfigureAwait(false);

            return new(stageRun, artifact, result.Backend, result.BackendProfile);
        }
        catch (RequiredModelNotAvailableException ex)
        {
            if (degradationWriter is not null)
            {
                try
                {
                    await degradationWriter.WriteAsync(
                        new PipelineDegradationRecord(
                            StageNames.SpeechEnhancement,
                            "SPEECH_ENHANCEMENT_MODEL_UNAVAILABLE",
                            ex.Message,
                            Detail: null,
                            SelectedFallback: "Using source audio without enhancement",
                            RecommendedAction: "Download the speech enhancement model or disable this stage.",
                            DateTimeOffset.UtcNow,
                            stageRun.Id),
                        request.ProjectId,
                        request.MediaAsset.Id,
                        cancellationToken).ConfigureAwait(false);
                }
                catch (Exception degradationEx) when (degradationEx is not OperationCanceledException)
                {
                    logger?.LogWarning("Failed to write degradation record for speech enhancement model unavailable.", degradationEx);
                }
            }
            stageRun = await StageRunHelper
                .SkipAsync(stageRunStore, stageRun, null, ex.Message, cancellationToken, null, logger)
                .ConfigureAwait(false);
            return new SpeechAudioEnhancementStageResult(stageRun, request.SourceAudioArtifact);
        }
        catch (OperationCanceledException)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, null, "Speech enhancement canceled.", CancellationToken.None, null, logger)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
        {
            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, null, ex.Message, cancellationToken, null, logger)
                .ConfigureAwait(false);
            throw;
        }
    }

    private async Task<SpeechAudioEnhancementOptions> ResolveOptionsAsync(
        SpeechAudioEnhancementStageRequest request,
        CancellationToken cancellationToken)
    {
        SpeechAudioEnhancementOptions options;
        if (request.Options is not null)
        {
            options = request.Options;
        }
        else if (studioSettingsService is not null)
        {
            try
            {
                StudioSettings settings = await studioSettingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
                options = SpeechAudioEnhancementOptions.FromStudioSettings(settings);
            }
            catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
            {
                // Settings are optional for the live DeepFilterNet path. Never fail the stage
                // because AFX preference loading broke.
                logger?.LogWarning(
                    "Failed to load studio settings for speech enhancement options; using defaults.",
                    ex);
                options = SpeechAudioEnhancementOptions.Default;
            }
        }
        else
        {
            options = SpeechAudioEnhancementOptions.Default;
        }

        if (!string.IsNullOrWhiteSpace(request.FarEndReferenceAudioPath))
        {
            options = options with { FarEndReferenceAudioPath = request.FarEndReferenceAudioPath };
        }

        return options;
    }

    private static ProjectArtifact CreateArtifact(
        SpeechAudioEnhancementStageRequest request,
        StageRunRecord stageRun,
        string relativePath,
        FileFingerprint fingerprint,
        SpeechAudioEnhancementResult result)
    {
        Guid artifactId = request.ExistingArtifacts
            .Where(artifact => artifact.Kind == ArtifactKind.SpeechEnhancedAudio)
            .OrderByDescending(artifact => artifact.CreatedAtUtc)
            .Select(artifact => artifact.Id)
            .FirstOrDefault();

        if (artifactId == Guid.Empty)
        {
            artifactId = Guid.NewGuid();
        }

        return new ProjectArtifact(
            artifactId,
            request.ProjectId,
            request.MediaAsset.Id,
            ArtifactKind.SpeechEnhancedAudio,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            result.DurationSeconds,
            result.SampleRate,
            result.ChannelCount,
            DateTimeOffset.UtcNow,
            StageRunId: stageRun.Id,
            Provenance: BuildProvenance(request, result));
    }

    private static string BuildProvenance(
        SpeechAudioEnhancementStageRequest request,
        SpeechAudioEnhancementResult result)
    {
        string backend = result.Backend switch
        {
            SpeechAudioEnhancementBackend.NvidiaAfx => "nvidia-afx",
            SpeechAudioEnhancementBackend.DeepFilterNet => "deepfilternet",
            _ => "ffmpeg"
        };
        string profile = string.IsNullOrWhiteSpace(result.BackendProfile)
            ? string.Empty
            : $":profile={result.BackendProfile}";
        return $"generated-{backend}-speech-enhancement:{request.SourceAudioArtifact.Id:D}{profile}";
    }
}

public sealed record SpeechAudioEnhancementStageRequest(
    Guid ProjectId,
    MediaAsset MediaAsset,
    ProjectArtifact SourceAudioArtifact,
    IReadOnlyList<ProjectArtifact> ExistingArtifacts,
    SpeechAudioEnhancementOptions? Options = null,
    string? FarEndReferenceAudioPath = null);

public sealed record SpeechAudioEnhancementStageResult(
    StageRunRecord StageRun,
    ProjectArtifact EnhancedAudioArtifact,
    SpeechAudioEnhancementBackend? Backend = null,
    string? BackendProfile = null);
