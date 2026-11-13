using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

public sealed class StemSeparationStageHandler(
    IStemSeparationEngine stemSeparationEngine,
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository,
    IProjectStageRunStore stageRunStore,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IApplicationLogger? logger = null,
    PipelineDegradationWriter? degradationWriter = null)
{
    private static readonly string[] RawStemNames = ["drums", "bass", "other", "vocals"];
    private const string UnknownEngineFamily = "unknown";

    private readonly IStemSeparationEngine stemSeparationEngine = stemSeparationEngine ?? throw new ArgumentNullException(nameof(stemSeparationEngine));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IFileFingerprintService fileFingerprintService = fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));

    public async Task<StemSeparationStageResult> HandleAsync(
        StemSeparationStageRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, request.ProjectId, StageNames.Separation, cancellationToken)
            .ConfigureAwait(false);

        string tempDirectory = StemSeparationTempDirectories.GetRunDirectory(stageRun.Id);
        var registeredStemArtifactIds = new List<Guid>();
        string tempVocalsPath = Path.Join(tempDirectory, "vocals.wav");
        string tempAmbiancePath = Path.Join(tempDirectory, "ambiance.wav");
        string tempMusicPath = Path.Join(tempDirectory, "music.wav");
        string tempSoundEffectsPath = Path.Join(tempDirectory, "sfx.wav");
        IReadOnlyDictionary<string, string> rawStemTempPaths = BuildRawStemTempPaths(tempDirectory);

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            StemSeparationTempDirectories.CleanupStale(DateTimeOffset.UtcNow);
            Directory.CreateDirectory(tempDirectory);
            StemSeparationResult result = await stemSeparationEngine.SeparateAsync(
                new StemSeparationRequest(
                    artifactStore.GetPath(request.SourceAudioArtifact.RelativePath),
                    tempVocalsPath,
                    tempAmbiancePath,
                    request.PreferredModelAlias,
                    MusicOutputPath: tempMusicPath,
                    SoundEffectsOutputPath: tempSoundEffectsPath,
                    RawStemOutputPaths: rawStemTempPaths,
                    PreferredExecutionProvider: request.PreferredExecutionProvider?.ToString(),
                    RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                    PreferredModelVariantAlias: request.PreferredModelVariantAlias),
                progress,
                cancellationToken).ConfigureAwait(false);

            progress?.Report(new StemSeparationProgress(
                CompletedChunks: 1,
                TotalChunks: 1,
                ChunkStartSeconds: 0d,
                ChunkEndSeconds: result.DurationSeconds,
                IsPersistingArtifacts: true));

            string engineFamily = ResolveEngineFamily(result);
            if (string.Equals(engineFamily, UnknownEngineFamily, StringComparison.Ordinal))
            {
                // Log via IApplicationLogger so the warning lands in trackdub.log rather than a
                // Trace sink that is not reliably observed in production.
                logger?.LogWarning(
                    "StemSeparationStageHandler: engine_family metadata absent from result — stem artifacts preserved with 'unknown' path segment.");

                if (degradationWriter is not null)
                {
                    try
                    {
                        await degradationWriter.WriteAsync(
                            new PipelineDegradationRecord(
                                StageNames.Separation,
                                "STEM_SEPARATION_METADATA_MISSING",
                                "Stem separation engine did not provide engine_family metadata.",
                                Detail: "Stem artifacts saved with 'unknown' path segment",
                                SelectedFallback: null,
                                RecommendedAction: null,
                                DateTimeOffset.UtcNow,
                                stageRun.Id),
                            request.ProjectId,
                            request.MediaAsset.Id,
                            cancellationToken).ConfigureAwait(false);
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException)
                    {
                        logger?.LogWarning("Failed to write degradation record for stem separation metadata missing.", ex);
                    }
                }
            }
            string vocalsRelativePath = ProjectArtifactPaths.GetStemVocalsRelativePath(stageRun.Id, engineFamily);
            string ambianceRelativePath = ProjectArtifactPaths.GetStemAmbianceRelativePath(stageRun.Id, engineFamily);
            string musicRelativePath = ProjectArtifactPaths.GetStemMusicRelativePath(stageRun.Id, engineFamily);
            string soundEffectsRelativePath = ProjectArtifactPaths.GetStemSoundEffectsRelativePath(stageRun.Id, engineFamily);

            await CommitFileAsync(tempVocalsPath, vocalsRelativePath, cancellationToken).ConfigureAwait(false);
            await CommitFileAsync(tempAmbiancePath, ambianceRelativePath, cancellationToken).ConfigureAwait(false);
            bool hasMusicOutput = HasOutput(tempMusicPath);
            bool hasSoundEffectsOutput = HasOutput(tempSoundEffectsPath);
            if (hasMusicOutput)
            {
                await CommitFileAsync(tempMusicPath, musicRelativePath, cancellationToken).ConfigureAwait(false);
            }

            if (hasSoundEffectsOutput)
            {
                await CommitFileAsync(tempSoundEffectsPath, soundEffectsRelativePath, cancellationToken).ConfigureAwait(false);
            }

            foreach ((string stemName, string rawTempPath) in ResolveRawStemOutputs(result, rawStemTempPaths))
            {
                string rawRelativePath = ProjectArtifactPaths.GetRawStemRelativePath(stageRun.Id, engineFamily, stemName);
                await CommitFileAsync(rawTempPath, rawRelativePath, cancellationToken).ConfigureAwait(false);
            }

            FileFingerprint vocalsFingerprint = await fileFingerprintService
                .ComputeAsync(artifactStore.GetPath(vocalsRelativePath), cancellationToken)
                .ConfigureAwait(false);
            FileFingerprint ambianceFingerprint = await fileFingerprintService
                .ComputeAsync(artifactStore.GetPath(ambianceRelativePath), cancellationToken)
                .ConfigureAwait(false);
            FileFingerprint? musicFingerprint = hasMusicOutput
                ? await fileFingerprintService
                    .ComputeAsync(artifactStore.GetPath(musicRelativePath), cancellationToken)
                    .ConfigureAwait(false)
                : null;
            FileFingerprint? soundEffectsFingerprint = hasSoundEffectsOutput
                ? await fileFingerprintService
                    .ComputeAsync(artifactStore.GetPath(soundEffectsRelativePath), cancellationToken)
                    .ConfigureAwait(false)
                : null;

            ProjectArtifact vocalsArtifact = CreateArtifact(
                request,
                stageRun,
                ArtifactKind.Vocals,
                vocalsRelativePath,
                vocalsFingerprint,
                result);
            ProjectArtifact ambianceArtifact = CreateArtifact(
                request,
                stageRun,
                ArtifactKind.Ambiance,
                ambianceRelativePath,
                ambianceFingerprint,
                result);
            ProjectArtifact? musicArtifact = musicFingerprint is null
                ? null
                : CreateArtifact(
                    request,
                    stageRun,
                    ArtifactKind.Music,
                    musicRelativePath,
                    musicFingerprint,
                    result);
            ProjectArtifact? soundEffectsArtifact = soundEffectsFingerprint is null
                ? null
                : CreateArtifact(
                    request,
                    stageRun,
                    ArtifactKind.SoundEffects,
                    soundEffectsRelativePath,
                    soundEffectsFingerprint,
                    result);

            await mediaAssetRepository.SaveArtifactAsync(vocalsArtifact, cancellationToken).ConfigureAwait(false);
            registeredStemArtifactIds.Add(vocalsArtifact.Id);
            await mediaAssetRepository.SaveArtifactAsync(ambianceArtifact, cancellationToken).ConfigureAwait(false);
            registeredStemArtifactIds.Add(ambianceArtifact.Id);
            if (musicArtifact is not null)
            {
                await mediaAssetRepository.SaveArtifactAsync(musicArtifact, cancellationToken).ConfigureAwait(false);
                registeredStemArtifactIds.Add(musicArtifact.Id);
            }

            if (soundEffectsArtifact is not null)
            {
                await mediaAssetRepository.SaveArtifactAsync(soundEffectsArtifact, cancellationToken).ConfigureAwait(false);
                registeredStemArtifactIds.Add(soundEffectsArtifact.Id);
            }

            // The completed producer row accepts the entire persisted set. An interrupted
            // registration leaves only ineligible metadata and preserves the prior producer.
            cancellationToken.ThrowIfCancellationRequested();
            stageRun = await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, stemSeparationEngine, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);

            return new StemSeparationStageResult(stageRun, vocalsArtifact, ambianceArtifact, musicArtifact, soundEffectsArtifact);
        }
        catch (OperationCanceledException)
        {
            await CleanupRegisteredStemArtifactsAsync(registeredStemArtifactIds).ConfigureAwait(false);
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, stemSeparationEngine, "Stem separation canceled.", CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex)
        {
            await CleanupRegisteredStemArtifactsAsync(registeredStemArtifactIds).ConfigureAwait(false);
            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, stemSeparationEngine, ex.Message, cancellationToken, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }
        finally
        {
            StemSeparationTempDirectories.DeleteIfExists(tempDirectory);
        }
    }

    private async Task CleanupRegisteredStemArtifactsAsync(IEnumerable<Guid> registeredStemArtifactIds)
    {
        foreach (Guid artifactId in registeredStemArtifactIds)
        {
            try
            {
                await mediaAssetRepository.DeleteArtifactAsync(artifactId, CancellationToken.None).ConfigureAwait(false);
            }
            catch
            {
                // Preserve the original cancellation/failure; cleanup errors should not replace it.
            }
        }
    }

    private static IReadOnlyDictionary<string, string> BuildRawStemTempPaths(string tempDirectory)
    {
        var paths = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string stemName in RawStemNames)
        {
            paths[stemName] = Path.Join(tempDirectory, $"{stemName}.raw.wav");
        }

        return paths;
    }

    private async Task CommitFileAsync(
        string sourcePath,
        string relativePath,
        CancellationToken cancellationToken)
    {
        if (!HasOutput(sourcePath))
        {
            throw new FileNotFoundException("Stem separation output was not created.", sourcePath);
        }

        await using ArtifactWriteHandle handle = artifactStore.CreateWriteHandle(relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(handle.TemporaryPath)!);
        cancellationToken.ThrowIfCancellationRequested();
        await TransferFileAsync(sourcePath, handle.TemporaryPath, cancellationToken).ConfigureAwait(false);
        await artifactStore.CommitAsync(handle, cancellationToken).ConfigureAwait(false);
    }

    // Same-volume move is metadata-only; cross-volume/unsupported moves fall back to a
    // bounded async copy. A canceled/failed transfer leaves the source for the caller's
    // temp-dir cleanup and any partial destination for the write handle's disposal.
    internal static async Task TransferFileAsync(
        string sourcePath,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            File.Move(sourcePath, destinationPath, overwrite: true);
            return;
        }
        catch (IOException)
        {
        }

        await CopyFileAsync(sourcePath, destinationPath, cancellationToken).ConfigureAwait(false);
    }

    internal static async Task CopyFileAsync(
        string sourcePath,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        const int bufferSize = 128 * 1024;
        await using FileStream source = new(
            sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await using FileStream destination = new(
            destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await source.CopyToAsync(destination, bufferSize, cancellationToken).ConfigureAwait(false);
        await destination.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    private static bool HasOutput(string path)
    {
        try
        {
            return File.Exists(path) && new FileInfo(path).Length > 0;
        }
        catch
        {
            return false;
        }
    }

    private static string ResolveEngineFamily(StemSeparationResult result)
    {
        if (TryGetMetadataValue(result, "engine_family", out string? engineFamily))
        {
            return NormalizeStemPathSegment(engineFamily, "engine_family");
        }

        if (TryGetMetadataValue(result, "model", out string? model))
        {
            if (model.Equals("hush-dialogue", StringComparison.OrdinalIgnoreCase))
            {
                return "hush-dialogue";
            }

            if (model.Equals("spleeter", StringComparison.OrdinalIgnoreCase) ||
                model.Equals("spleeter-2stems", StringComparison.OrdinalIgnoreCase) ||
                model.Equals("spleeter-non-commercial", StringComparison.OrdinalIgnoreCase))
            {
                return "spleeter";
            }
        }

        return UnknownEngineFamily;
    }

    private static IEnumerable<KeyValuePair<string, string>> ResolveRawStemOutputs(
        StemSeparationResult result,
        IReadOnlyDictionary<string, string> rawStemTempPaths)
    {
        if (!TryGetMetadataValue(result, "raw_stems", out string? rawStems))
        {
            yield break;
        }

        foreach (string stemName in rawStems.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string normalizedStemName = NormalizeStemPathSegment(stemName, "raw_stems");
            if (rawStemTempPaths.TryGetValue(normalizedStemName, out string? rawPath) && HasOutput(rawPath))
            {
                yield return new KeyValuePair<string, string>(normalizedStemName, rawPath);
            }
        }
    }

    private static string NormalizeStemPathSegment(string value, string metadataKey)
    {
        string normalized = new(value
            .Trim()
            .ToLowerInvariant()
            .Select(static character => character == '_' || char.IsWhiteSpace(character) ? '-' : character)
            .ToArray());
        if (normalized.Length == 0 || normalized.Any(static character => !IsStemPathSegmentCharacter(character)))
        {
            throw new InvalidOperationException(
                $"Stem separation result reported an invalid {metadataKey} metadata value '{value}'.");
        }

        return normalized;
    }

    private static bool IsStemPathSegmentCharacter(char value) =>
        (value >= 'a' && value <= 'z') ||
        (value >= '0' && value <= '9') ||
        value == '-';

    private static bool TryGetMetadataValue(
        StemSeparationResult result,
        string key,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? value)
    {
        value = null;
        return result.Metadata is not null &&
               result.Metadata.TryGetValue(key, out value) &&
               !string.IsNullOrWhiteSpace(value);
    }

    private ProjectArtifact CreateArtifact(
        StemSeparationStageRequest request,
        StageRunRecord stageRun,
        ArtifactKind kind,
        string relativePath,
        FileFingerprint fingerprint,
        StemSeparationResult result)
    {
        // Never overwrite an accepted producer's metadata while publishing a new run.
        return new ProjectArtifact(
            Guid.NewGuid(),
            request.ProjectId,
            request.MediaAsset.Id,
            kind,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            result.DurationSeconds,
            result.SampleRate,
            result.ChannelCount,
            DateTimeOffset.UtcNow,
            StageRunId: stageRun.Id,
            Provenance: BuildStemProvenance(ResolveBaseProvenance(result, kind), result.Metadata));
    }

    private static string BuildStemProvenance(
        string baseProvenance,
        IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || metadata.Count == 0)
        {
            return baseProvenance;
        }

        string[] entries = metadata
            .OrderBy(static pair => pair.Key, StringComparer.Ordinal)
            .Select(static pair => $"{pair.Key}={pair.Value}")
            .ToArray();
        return $"{baseProvenance};{string.Join(';', entries)}";
    }

    // Stems from any engine other than Hush are labelled Spleeter, the current separator.
    private static string ResolveBaseProvenance(StemSeparationResult result, ArtifactKind kind)
    {
        string engine = TryGetMetadataValue(result, "model", out string? model) &&
                        model.Equals("hush-dialogue", StringComparison.OrdinalIgnoreCase)
            ? "hush-dialogue"
            : "spleeter";
        return $"generated-{engine}-{StemProvenanceName(kind)}";
    }

    private static string StemProvenanceName(ArtifactKind kind) =>
        kind switch
        {
            ArtifactKind.Vocals => "vocals",
            ArtifactKind.Ambiance => "ambiance",
            ArtifactKind.Music => "music",
            ArtifactKind.SoundEffects => "sfx",
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a stem artifact kind."),
        };
}

public sealed record StemSeparationStageRequest(
    Guid ProjectId,
    MediaAsset MediaAsset,
    ProjectArtifact SourceAudioArtifact,
    IReadOnlyList<ProjectArtifact> ExistingArtifacts,
    string? PreferredModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record StemSeparationStageResult(
    StageRunRecord StageRun,
    ProjectArtifact VocalsArtifact,
    ProjectArtifact AmbianceArtifact,
    ProjectArtifact? MusicArtifact = null,
    ProjectArtifact? SoundEffectsArtifact = null)
{
    public IReadOnlyList<ProjectArtifact> Artifacts
    {
        get
        {
            var artifacts = new List<ProjectArtifact>
            {
                VocalsArtifact,
                AmbianceArtifact
            };
            if (MusicArtifact is not null)
            {
                artifacts.Add(MusicArtifact);
            }

            if (SoundEffectsArtifact is not null)
            {
                artifacts.Add(SoundEffectsArtifact);
            }

            return artifacts;
        }
    }
}
