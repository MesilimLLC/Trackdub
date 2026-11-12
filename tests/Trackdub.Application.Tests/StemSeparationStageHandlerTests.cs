using Trackdub.Contracts;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class StemSeparationStageHandlerTests
{
    [Fact]
    public async Task Cancelled_second_stem_save_preserves_prior_pair()
    {
        string directory = CreateTempDirectory();
        try
        {
            var store = new FakeArtifactStore(directory);
            (MediaAsset media, ProjectArtifact source) = SeedSource(store, directory);
            var repository = new FakeMediaAssetRepository();
            var runs = new FakeProjectStageRunStore();
            var engine = new RecordingStemSeparationEngine("spleeter", "spleeter", true, true, []);
            var handler = new StemSeparationStageHandler(engine, store, new FakeFileFingerprintService(), repository, runs);
            StemSeparationStageResult prior = await handler.HandleAsync(
                new(media.ProjectId, media, source, []), null, TestContext.Current.CancellationToken);
            var publicationStatuses = new List<StageRunStatus>();
            var interruptedRepository = new InterruptingMediaRepository(repository, artifact =>
            {
                publicationStatuses.Add(runs.All.Last().Status);
                if (artifact.Kind == ArtifactKind.Ambiance)
                {
                    throw new OperationCanceledException();
                }
            });
            handler = new StemSeparationStageHandler(engine, store, new FakeFileFingerprintService(), interruptedRepository, runs);

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handler.HandleAsync(
                new(media.ProjectId, media, source, prior.Artifacts), null, TestContext.Current.CancellationToken));

            StemAudioRoute route = TranscriptWorkflowUtilities.BuildStemAudioRoute(repository.Artifacts, runs.All);
            ProjectArtifact selectedVocals = Assert.Single(repository.Artifacts, a => a.RelativePath == route.AsrAudioRelativePath);
            ProjectArtifact selectedAmbiance = Assert.Single(repository.Artifacts, a => a.RelativePath == route.MixSourceAudioRelativePath);
            Assert.Equal(selectedVocals.StageRunId, selectedAmbiance.StageRunId);
            Assert.Equal(prior.StageRun.Id, selectedVocals.StageRunId);
            Assert.Contains(prior.VocalsArtifact, repository.Artifacts);
            Assert.Contains(prior.AmbianceArtifact, repository.Artifacts);
            Assert.True(store.Exists(prior.VocalsArtifact.RelativePath));
            Assert.True(store.Exists(prior.AmbianceArtifact.RelativePath));
            Assert.Equal(StageRunStatus.Canceled, runs.All.Last().Status);
            Assert.All(publicationStatuses, status => Assert.Equal(StageRunStatus.Running, status));
            // Interrupted registrations must not accumulate orphan rows on the canceled run.
            Assert.DoesNotContain(repository.Artifacts, artifact => artifact.StageRunId == runs.All.Last().Id);
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task Cancellation_after_run_creation_records_cancelled()
    {
        string directory = CreateTempDirectory();
        try
        {
            var store = new FakeArtifactStore(directory);
            (MediaAsset media, ProjectArtifact source) = SeedSource(store, directory);
            using var cancellation = new CancellationTokenSource();
            var runs = new CancellingStartRunStore(cancellation);
            var handler = new StemSeparationStageHandler(
                new RecordingStemSeparationEngine("spleeter", "spleeter", false, false, []),
                store, new FakeFileFingerprintService(), new FakeMediaAssetRepository(), runs);

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handler.HandleAsync(
                new(media.ProjectId, media, source, []), null, cancellation.Token));

            Assert.Equal(StageRunStatus.Canceled, Assert.Single(runs.Inner.All).Status);
            Assert.False(Directory.Exists(StemSeparationTempDirectories.GetRunDirectory(runs.Inner.All[0].Id)));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    private sealed class CancellingStartRunStore(CancellationTokenSource cancellation) : IProjectStageRunStore
    {
        public FakeProjectStageRunStore Inner { get; } = new();
        public async Task CreateAsync(StageRunRecord run, CancellationToken token)
        {
            await Inner.CreateAsync(run, token);
            cancellation.Cancel();
        }
        public Task UpdateAsync(StageRunRecord run, CancellationToken token) => Inner.UpdateAsync(run, token);
        public Task<IReadOnlyList<StageRunRecord>> ListByProjectAsync(Guid projectId, CancellationToken token) => Inner.ListByProjectAsync(projectId, token);
    }

    private sealed class InterruptingMediaRepository(FakeMediaAssetRepository inner, Action<ProjectArtifact> beforeSave) : IMediaAssetRepository
    {
        public Task SaveAsync(MediaAsset asset, CancellationToken token) => inner.SaveAsync(asset, token);
        public Task UpdateSourcePathAsync(Guid id, string path, string name, CancellationToken token) => inner.UpdateSourcePathAsync(id, path, name, token);
        [Obsolete("Single-asset compatibility member.")]
        public Task<MediaAsset?> GetPrimaryAsync(Guid projectId, CancellationToken token) => Task.FromResult(inner.Assets.FirstOrDefault(a => a.ProjectId == projectId));
        public Task<IReadOnlyList<MediaAsset>> GetAllAsync(Guid projectId, CancellationToken token) => inner.GetAllAsync(projectId, token);
        public Task SaveArtifactAsync(ProjectArtifact artifact, CancellationToken token)
        {
            beforeSave(artifact);
            return inner.SaveArtifactAsync(artifact, token);
        }
        public Task DeleteArtifactAsync(Guid id, CancellationToken token) => inner.DeleteArtifactAsync(id, token);
        public Task<IReadOnlyList<ProjectArtifact>> GetArtifactsAsync(Guid projectId, CancellationToken token) => inner.GetArtifactsAsync(projectId, token);
        public Task<ProjectArtifact?> GetArtifactByIdAsync(Guid id, CancellationToken token) => inner.GetArtifactByIdAsync(id, token);
    }

    [Fact]
    public async Task HandleAsync_CommitsMultiStemOutputsUnderEngineFamilyFolderAndRawSidecarsWithoutLatentFolders()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var mediaRepository = new FakeMediaAssetRepository();
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new RecordingStemSeparationEngine(
                engineFamily: "spleeter",
                model: "spleeter",
                writeMusic: true,
                writeSfx: false,
                rawStemNames: ["drums", "bass", "other", "vocals"]);
            var handler = new StemSeparationStageHandler(
                engine,
                artifactStore,
                new FakeFileFingerprintService(),
                mediaRepository,
                stageRunStore);

            StemSeparationStageResult result = await handler.HandleAsync(
                new StemSeparationStageRequest(
                    mediaAsset.ProjectId,
                    mediaAsset,
                    sourceArtifact,
                    ExistingArtifacts: []),
                progress: null,
                TestContext.Current.CancellationToken);

            Guid stageRunId = result.StageRun.Id;
            Assert.Equal(
                ProjectArtifactPaths.GetStemVocalsRelativePath(stageRunId, "spleeter"),
                result.VocalsArtifact.RelativePath);
            Assert.Equal(
                ProjectArtifactPaths.GetStemAmbianceRelativePath(stageRunId, "spleeter"),
                result.AmbianceArtifact.RelativePath);
            Assert.Equal(
                ProjectArtifactPaths.GetStemMusicRelativePath(stageRunId, "spleeter"),
                result.MusicArtifact?.RelativePath);
            Assert.Null(result.SoundEffectsArtifact);
            Assert.Contains("generated-spleeter-vocals", result.VocalsArtifact.Provenance, StringComparison.Ordinal);
            Assert.Contains("raw_stems=drums,bass,other,vocals", result.VocalsArtifact.Provenance, StringComparison.Ordinal);

            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter", "drums")));
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter", "bass")));
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter", "other")));
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter", "vocals")));
            Assert.DoesNotContain(
                artifactStore.Blobs.Keys,
                path => path.Contains("/hush-dialogue/", StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task HandleAsync_CommitsSpleeterOutputsUnderSpleeterFamilyFolderWithSpleeterProvenance()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var handler = new StemSeparationStageHandler(
                new RecordingStemSeparationEngine(
                    engineFamily: "spleeter",
                    model: "spleeter",
                    writeMusic: true,
                    writeSfx: false,
                    rawStemNames: ["vocals", "other"]),
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                new FakeProjectStageRunStore());

            StemSeparationStageResult result = await handler.HandleAsync(
                new StemSeparationStageRequest(
                    mediaAsset.ProjectId,
                    mediaAsset,
                    sourceArtifact,
                    ExistingArtifacts: []),
                progress: null,
                TestContext.Current.CancellationToken);

            Guid stageRunId = result.StageRun.Id;
            Assert.Equal(
                ProjectArtifactPaths.GetStemVocalsRelativePath(stageRunId, "spleeter"),
                result.VocalsArtifact.RelativePath);
            Assert.Equal(
                ProjectArtifactPaths.GetStemAmbianceRelativePath(stageRunId, "spleeter"),
                result.AmbianceArtifact.RelativePath);
            Assert.Contains("generated-spleeter-vocals", result.VocalsArtifact.Provenance, StringComparison.Ordinal);
            Assert.Contains("engine_family=spleeter", result.VocalsArtifact.Provenance, StringComparison.Ordinal);
            Assert.Contains("model=spleeter", result.VocalsArtifact.Provenance, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task HandleAsync_RemovesStaleStemTempDirectoriesAndPreservesFreshOrUnrelatedDirectories()
    {
        string directory = CreateTempDirectory();
        string staleDirectory = StemSeparationTempDirectories.GetRunDirectory(Guid.NewGuid());
        string freshDirectory = StemSeparationTempDirectories.GetRunDirectory(Guid.NewGuid());
        string unrelatedDirectory = Path.Join(Path.GetTempPath(), $"trackdub-other-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(staleDirectory);
            Directory.CreateDirectory(freshDirectory);
            Directory.CreateDirectory(unrelatedDirectory);
            File.WriteAllText(Path.Join(staleDirectory, "leftover.tmp"), "stale");
            File.WriteAllText(Path.Join(freshDirectory, "active.tmp"), "fresh");
            File.WriteAllText(Path.Join(unrelatedDirectory, "leftover.tmp"), "unrelated");
            Directory.SetLastWriteTimeUtc(staleDirectory, DateTimeOffset.UtcNow.AddHours(-25).UtcDateTime);
            Directory.SetLastWriteTimeUtc(freshDirectory, DateTimeOffset.UtcNow.AddHours(-1).UtcDateTime);
            Directory.SetLastWriteTimeUtc(unrelatedDirectory, DateTimeOffset.UtcNow.AddHours(-25).UtcDateTime);

            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var handler = new StemSeparationStageHandler(
                new RecordingStemSeparationEngine(
                    engineFamily: "spleeter",
                    model: "spleeter",
                    writeMusic: true,
                    writeSfx: false,
                    rawStemNames: ["vocals", "other"]),
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                new FakeProjectStageRunStore());

            await handler.HandleAsync(
                new StemSeparationStageRequest(
                    mediaAsset.ProjectId,
                    mediaAsset,
                    sourceArtifact,
                    ExistingArtifacts: []),
                progress: null,
                TestContext.Current.CancellationToken);

            Assert.False(Directory.Exists(staleDirectory));
            Assert.True(Directory.Exists(freshDirectory));
            Assert.True(Directory.Exists(unrelatedDirectory));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
            DeleteDirectoryIfExists(staleDirectory);
            DeleteDirectoryIfExists(freshDirectory);
            DeleteDirectoryIfExists(unrelatedDirectory);
        }
    }

    [Fact]
    public async Task HandleAsync_DeletesCurrentRunTempDirectoryOnSuccess()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var handler = new StemSeparationStageHandler(
                new RecordingStemSeparationEngine(
                    engineFamily: "spleeter",
                    model: "spleeter",
                    writeMusic: true,
                    writeSfx: false,
                    rawStemNames: ["vocals", "other"]),
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                new FakeProjectStageRunStore());

            StemSeparationStageResult result = await handler.HandleAsync(
                new StemSeparationStageRequest(
                    mediaAsset.ProjectId,
                    mediaAsset,
                    sourceArtifact,
                    ExistingArtifacts: []),
                progress: null,
                TestContext.Current.CancellationToken);

            Assert.False(Directory.Exists(StemSeparationTempDirectories.GetRunDirectory(result.StageRun.Id)));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task HandleAsync_WhenEngineFails_DeletesCurrentRunTempDirectory()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var handler = new StemSeparationStageHandler(
                new FailingStemSeparationEngine(),
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                stageRunStore);

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                handler.HandleAsync(
                    new StemSeparationStageRequest(
                        mediaAsset.ProjectId,
                        mediaAsset,
                        sourceArtifact,
                        ExistingArtifacts: []),
                    progress: null,
                    TestContext.Current.CancellationToken));

            Assert.False(Directory.Exists(StemSeparationTempDirectories.GetRunDirectory(Assert.Single(stageRunStore.All).Id)));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task HandleAsync_WhenEngineCancels_DeletesCurrentRunTempDirectory()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var handler = new StemSeparationStageHandler(
                new CancelingStemSeparationEngine(),
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                stageRunStore);

            await Assert.ThrowsAsync<OperationCanceledException>(() =>
                handler.HandleAsync(
                    new StemSeparationStageRequest(
                        mediaAsset.ProjectId,
                        mediaAsset,
                        sourceArtifact,
                        ExistingArtifacts: []),
                    progress: null,
                    TestContext.Current.CancellationToken));

            Assert.False(Directory.Exists(StemSeparationTempDirectories.GetRunDirectory(Assert.Single(stageRunStore.All).Id)));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task HandleAsync_NormalizesEngineFamilyAndRawStemMetadataBeforeBuildingStemPaths()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var engine = new RecordingStemSeparationEngine(
                engineFamily: " Spleeter_V2 ",
                model: "spleeter",
                writeMusic: true,
                writeSfx: false,
                rawStemNames: ["drums", "bass", "other", "vocals"],
                reportedRawStemNames: [" Drums ", "BASS", "Other", "Vocals"]);
            var handler = new StemSeparationStageHandler(
                engine,
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                new FakeProjectStageRunStore());

            StemSeparationStageResult result = await handler.HandleAsync(
                new StemSeparationStageRequest(
                    mediaAsset.ProjectId,
                    mediaAsset,
                    sourceArtifact,
                    ExistingArtifacts: []),
                progress: null,
                TestContext.Current.CancellationToken);

            Guid stageRunId = result.StageRun.Id;
            Assert.Equal(
                ProjectArtifactPaths.GetStemVocalsRelativePath(stageRunId, "spleeter-v2"),
                result.VocalsArtifact.RelativePath);
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter-v2", "drums")));
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter-v2", "bass")));
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter-v2", "other")));
            Assert.True(artifactStore.Exists(ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter-v2", "vocals")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task HandleAsync_WhenCurrentEngineOmitsSfx_PreservesExistingSoundEffectsArtifact()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var mediaRepository = new FakeMediaAssetRepository();
            ProjectArtifact existingSfx = CreateExistingStemArtifact(
                mediaAsset,
                ArtifactKind.SoundEffects,
                ProjectArtifactPaths.GetStemSoundEffectsRelativePath(Guid.NewGuid(), "sepformer"));
            await mediaRepository.SaveArtifactAsync(existingSfx, TestContext.Current.CancellationToken);
            var engine = new RecordingStemSeparationEngine(
                engineFamily: "spleeter",
                model: "spleeter",
                writeMusic: true,
                writeSfx: false,
                rawStemNames: ["drums", "bass", "other", "vocals"]);
            var handler = new StemSeparationStageHandler(
                engine,
                artifactStore,
                new FakeFileFingerprintService(),
                mediaRepository,
                new FakeProjectStageRunStore());

            StemSeparationStageResult result = await handler.HandleAsync(
                new StemSeparationStageRequest(
                    mediaAsset.ProjectId,
                    mediaAsset,
                    sourceArtifact,
                    ExistingArtifacts: [existingSfx]),
                progress: null,
                TestContext.Current.CancellationToken);

            Assert.Null(result.SoundEffectsArtifact);
            // Artifact preservation invariant: a separation run that does not produce SFX must not
            // delete the prior SFX artifact — the user's earlier mix must continue to render.
            Assert.Contains(mediaRepository.Artifacts, artifact => artifact.Id == existingSfx.Id);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task HandleAsync_WhenEngineFails_DoesNotCommitEngineFamilyOutputs()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            (MediaAsset mediaAsset, ProjectArtifact sourceArtifact) = SeedSource(artifactStore, directory);
            var handler = new StemSeparationStageHandler(
                new FailingStemSeparationEngine(),
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository(),
                new FakeProjectStageRunStore());

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                handler.HandleAsync(
                    new StemSeparationStageRequest(
                        mediaAsset.ProjectId,
                        mediaAsset,
                        sourceArtifact,
                        ExistingArtifacts: []),
                    progress: null,
                    TestContext.Current.CancellationToken));

            Assert.DoesNotContain(artifactStore.Blobs.Keys, path => path.StartsWith("artifacts/stems/", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static (MediaAsset MediaAsset, ProjectArtifact SourceArtifact) SeedSource(
        FakeArtifactStore artifactStore,
        string directory)
    {
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        string sourceRelativePath = ProjectArtifactPaths.StemSeparationSourceAudioRelativePath;
        string sourcePath = Path.Join(directory, "source.wav");
        File.WriteAllBytes(sourcePath, FakeWavHelper.MinimalPcm16(durationSeconds: 1d, sampleRate: 44100, channelCount: 2));
        artifactStore.SeedPath(sourceRelativePath, sourcePath, File.ReadAllBytes(sourcePath));

        var mediaAsset = new MediaAsset(
            mediaAssetId,
            projectId,
            sourcePath,
            "source.wav",
            "source-hash",
            new FileInfo(sourcePath).Length,
            DateTimeOffset.UtcNow,
            "wav",
            1d,
            HasAudio: true,
            HasVideo: false,
            DateTimeOffset.UtcNow);
        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAssetId,
            ArtifactKind.StemSeparationSourceAudio,
            sourceRelativePath,
            "source-hash",
            new FileInfo(sourcePath).Length,
            1d,
            44100,
            2,
            DateTimeOffset.UtcNow);
        return (mediaAsset, artifact);
    }

    private static ProjectArtifact CreateExistingStemArtifact(
        MediaAsset mediaAsset,
        ArtifactKind kind,
        string relativePath) =>
        new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            kind,
            relativePath,
            "old-hash",
            100,
            mediaAsset.DurationSeconds,
            44100,
            1,
            DateTimeOffset.UtcNow.AddMinutes(-1),
            StageRunId: Guid.NewGuid(),
            Provenance: "generated-sepformer-sfx;engine_family=sepformer;model=sepformer");

    [Fact]
    public async Task TransferFileAsync_SameVolumeMove_ExactBytesAndSourceRemoved()
    {
        string directory = CreateTempDirectory();
        try
        {
            byte[] payload = new byte[300 * 1024]; // spans multiple 128 KiB buffers
            new Random(42).NextBytes(payload);
            string source = Path.Join(directory, "stem.wav");
            string destination = Path.Join(directory, "committed.wav");
            await File.WriteAllBytesAsync(source, payload, TestContext.Current.CancellationToken);

            await StemSeparationStageHandler.TransferFileAsync(
                source, destination, TestContext.Current.CancellationToken);

            Assert.False(File.Exists(source));
            Assert.Equal(payload, await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task TransferFileAsync_PreCanceled_LeavesBothPathsUnchanged()
    {
        string directory = CreateTempDirectory();
        try
        {
            string source = Path.Join(directory, "stem.wav");
            string destination = Path.Join(directory, "committed.wav");
            byte[] payload = new byte[1024];
            new Random(7).NextBytes(payload);
            await File.WriteAllBytesAsync(source, payload, TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(destination, [1], TestContext.Current.CancellationToken);

            using var cts = new CancellationTokenSource();
            await cts.CancelAsync();

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                StemSeparationStageHandler.TransferFileAsync(source, destination, cts.Token));

            Assert.Equal(payload, await File.ReadAllBytesAsync(source, TestContext.Current.CancellationToken));
            Assert.Equal([1], await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task CopyFileAsync_PreservesExactBytesAcrossMultiBufferPayload()
    {
        string directory = CreateTempDirectory();
        try
        {
            byte[] payload = new byte[(128 * 1024) + 12345]; // exceeds one buffer
            new Random(99).NextBytes(payload);
            string source = Path.Join(directory, "stem.wav");
            string destination = Path.Join(directory, "committed.wav");
            await File.WriteAllBytesAsync(source, payload, TestContext.Current.CancellationToken);

            await StemSeparationStageHandler.CopyFileAsync(
                source, destination, TestContext.Current.CancellationToken);

            Assert.Equal(payload, await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
            Assert.True(File.Exists(source)); // copy fallback does not consume the source
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    [Fact]
    public async Task CopyFileAsync_Canceled_LeavesSourceIntact()
    {
        string directory = CreateTempDirectory();
        try
        {
            byte[] payload = new byte[512 * 1024];
            new Random(5).NextBytes(payload);
            string source = Path.Join(directory, "stem.wav");
            string destination = Path.Join(directory, "committed.wav");
            await File.WriteAllBytesAsync(source, payload, TestContext.Current.CancellationToken);

            using var cts = new CancellationTokenSource();
            await cts.CancelAsync();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                StemSeparationStageHandler.CopyFileAsync(source, destination, cts.Token));

            Assert.Equal(payload, await File.ReadAllBytesAsync(source, TestContext.Current.CancellationToken));
        }
        finally
        {
            DeleteDirectoryIfExists(directory);
        }
    }

    private static string CreateTempDirectory()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-stem-stage-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void DeleteDirectoryIfExists(string directory)
    {
        if (Directory.Exists(directory))
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private sealed class RecordingStemSeparationEngine(
        string engineFamily,
        string model,
        bool writeMusic,
        bool writeSfx,
        IReadOnlyList<string> rawStemNames,
        IReadOnlyList<string>? reportedRawStemNames = null)
        : IStemSeparationEngine
    {
        private readonly IReadOnlyList<string> reportedRawStemNames = reportedRawStemNames ?? rawStemNames;

        public async Task<StemSeparationResult> SeparateAsync(
            StemSeparationRequest request,
            IProgress<StemSeparationProgress>? progress,
            CancellationToken cancellationToken)
        {
            await File.WriteAllBytesAsync(request.VocalsOutputPath, [1, 2, 3], cancellationToken);
            await File.WriteAllBytesAsync(request.AmbianceOutputPath, [4, 5, 6], cancellationToken);
            if (writeMusic && !string.IsNullOrWhiteSpace(request.MusicOutputPath))
            {
                await File.WriteAllBytesAsync(request.MusicOutputPath, [7, 8, 9], cancellationToken);
            }

            if (writeSfx && !string.IsNullOrWhiteSpace(request.SoundEffectsOutputPath))
            {
                await File.WriteAllBytesAsync(request.SoundEffectsOutputPath, [10, 11, 12], cancellationToken);
            }

            foreach (string rawStemName in rawStemNames)
            {
                string rawPath = request.RawStemOutputPaths![rawStemName];
                await File.WriteAllBytesAsync(rawPath, [13, 14, 15], cancellationToken);
            }

            return new StemSeparationResult(
                DurationSeconds: 1d,
                SampleRate: 44100,
                ChannelCount: 1,
                Metadata: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["engine_family"] = engineFamily,
                    ["model"] = model,
                    ["raw_stems"] = string.Join(',', reportedRawStemNames)
                });
        }
    }

    private sealed class FailingStemSeparationEngine : IStemSeparationEngine
    {
        public Task<StemSeparationResult> SeparateAsync(
            StemSeparationRequest request,
            IProgress<StemSeparationProgress>? progress,
            CancellationToken cancellationToken)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(request.VocalsOutputPath)!);
            File.WriteAllBytes(request.VocalsOutputPath, [1, 2, 3]);
            throw new InvalidOperationException("Expected engine failure.");
        }
    }

    private sealed class CancelingStemSeparationEngine : IStemSeparationEngine
    {
        public Task<StemSeparationResult> SeparateAsync(
            StemSeparationRequest request,
            IProgress<StemSeparationProgress>? progress,
            CancellationToken cancellationToken)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(request.VocalsOutputPath)!);
            File.WriteAllBytes(request.VocalsOutputPath, [1, 2, 3]);
            throw new OperationCanceledException(cancellationToken);
        }
    }
}
