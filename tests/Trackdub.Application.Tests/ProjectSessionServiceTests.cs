using Trackdub.Contracts;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;

namespace Trackdub.Application.Tests;

public sealed class ProjectSessionServiceTests
{
    [Fact]
    public void SetCurrentSession_disposes_previous_session_when_replaced()
    {
        var factory = new FakeWorkspaceSessionFactory();
        var service = new ProjectSessionService(factory);
        var first = new FakeWorkspaceSession("C:\\Projects\\first.trackdub");
        var second = new FakeWorkspaceSession("C:\\Projects\\second.trackdub");

        service.SetCurrentSession(first);
        service.SetCurrentSession(second);
        service.ClearCurrentSession();

        Assert.Equal(1, first.DisposeCount);
        Assert.Equal(1, second.DisposeCount);
        Assert.Null(service.CurrentProjectRootPath);
    }

    [Fact]
    public void CreatePendingSessionForMedia_allocates_available_project_root_and_uses_settings()
    {
        string tempDirectory = Path.Join(Path.GetTempPath(), "Trackdub.Application.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDirectory);
        try
        {
            Directory.CreateDirectory(Path.Join(tempDirectory, "clip.trackdub"));
            string mediaPath = Path.Join(tempDirectory, "clip.mp4");
            var settings = StudioSettings.Default;
            var factory = new FakeWorkspaceSessionFactory();
            var service = new ProjectSessionService(factory);

            PendingProjectSession pending = service.CreatePendingSessionForMedia(mediaPath, "clip", settings);

            Assert.Equal("clip #2", pending.ProjectRoot.ProjectName);
            Assert.EndsWith("clip #2.trackdub", pending.ProjectRoot.ProjectRootPath, StringComparison.Ordinal);
            Assert.Equal(pending.ProjectRoot.ProjectRootPath, pending.Session.ProjectRootPath);
            Assert.Same(settings, factory.LastSettings);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void CreatePendingSessionForMedia_places_project_beside_cloud_synced_media()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        string userDataRoot = Path.Join(Path.GetTempPath(), "Trackdub.Application.Tests", Guid.NewGuid().ToString("N"));
        string mediaRoot = Path.Join(userDataRoot, "OneDrive", "Videos");
        Directory.CreateDirectory(mediaRoot);
        string mediaPath = Path.Join(mediaRoot, "clip.mp4");
        File.WriteAllText(mediaPath, "x");
        var settings = StudioSettings.Default;
        var factory = new FakeWorkspaceSessionFactory();
        var service = new ProjectSessionService(factory, new FakeStoragePaths(userDataRoot));

        try
        {
            PendingProjectSession pending = service.CreatePendingSessionForMedia(mediaPath, "clip", settings);

            Assert.Equal("clip", pending.ProjectRoot.ProjectName);
            Assert.Equal(Path.Join(mediaRoot, "clip.trackdub"), pending.ProjectRoot.ProjectRootPath);
            Assert.Equal(pending.ProjectRoot.ProjectRootPath, pending.Session.ProjectRootPath);
        }
        finally
        {
            if (Directory.Exists(userDataRoot))
            {
                Directory.Delete(userDataRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void CreatePendingSessionForMedia_uses_configured_default_project_directory()
    {
        string tempDirectory = Path.Join(Path.GetTempPath(), "Trackdub.Application.Tests", Guid.NewGuid().ToString("N"));
        string projectDirectory = Path.Join(tempDirectory, "projects");
        Directory.CreateDirectory(projectDirectory);
        try
        {
            string mediaDirectory = Path.Join(tempDirectory, "media");
            Directory.CreateDirectory(mediaDirectory);
            string mediaPath = Path.Join(mediaDirectory, "clip.mp4");
            var settings = StudioSettings.Default with { DefaultProjectDirectory = projectDirectory };
            var factory = new FakeWorkspaceSessionFactory();
            var service = new ProjectSessionService(factory);

            PendingProjectSession pending = service.CreatePendingSessionForMedia(mediaPath, "clip", settings);

            Assert.Equal(Path.Join(projectDirectory, "clip.trackdub"), pending.ProjectRoot.ProjectRootPath);
            Assert.Equal(pending.ProjectRoot.ProjectRootPath, pending.Session.ProjectRootPath);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void CreatePendingSessionForMedia_fails_when_configured_default_project_directory_is_unavailable()
    {
        string tempDirectory = Path.Join(Path.GetTempPath(), "Trackdub.Application.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDirectory);
        try
        {
            string missingProjectDirectory = Path.Join(tempDirectory, "missing-projects");
            string mediaPath = Path.Join(tempDirectory, "clip.mp4");
            var settings = StudioSettings.Default with { DefaultProjectDirectory = missingProjectDirectory };
            var service = new ProjectSessionService(new FakeWorkspaceSessionFactory());

            Assert.Throws<DirectoryNotFoundException>(
                () => service.CreatePendingSessionForMedia(mediaPath, "clip", settings));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void RecordHistoryTransition_moves_states_between_undo_and_redo_stacks()
    {
        var service = new ProjectSessionService(new FakeWorkspaceSessionFactory());
        Guid projectId = Guid.NewGuid();
        Guid speakerId = Guid.NewGuid();
        TranscriptProjectState before = CreateState(projectId, speakerId, "Speaker 1");
        TranscriptProjectState after = CreateState(projectId, speakerId, "Narrator");

        service.RecordHistoryTransition(before, after, isRestoringHistory: false);

        Assert.True(service.CanUndo);
        Assert.False(service.CanRedo);
        Assert.Same(before, service.PopUndoState());
        Assert.False(service.CanUndo);
        Assert.True(service.CanRedo);
        Assert.Same(after, service.PopRedoState());
        Assert.True(service.CanUndo);
        Assert.False(service.CanRedo);
    }

    private static TranscriptProjectState CreateState(Guid projectId, Guid speakerId, string speakerName)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(projectId, "Test Project", now, now);
        var projectState = new OpenProjectResult(
            project,
            MediaAsset: null,
            SourceReference: null,
            SourceMediaStatus.Missing,
            SourceStatusMessage: null,
            Artifacts: [],
            TranscriptLanguage: "en");
        var speaker = new ProjectSpeaker(speakerId, projectId, speakerName, now);

        return new TranscriptProjectState(
            projectState,
            CurrentTranscriptRevision: null,
            TranscriptSegments: [],
            Speakers: [speaker],
            SpeakerTurns: [],
            CurrentTranslationRevision: null,
            TranslatedSegments: [],
            IsTranslationStale: false,
            TranscriptLanguage: "en",
            StageRuns: [],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: null,
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }

    private sealed class FakeWorkspaceSessionFactory : ITranscriptWorkspaceSessionFactory
    {
        public StudioSettings? LastSettings { get; private set; }

        public ITranscriptWorkspaceSession Create(string projectRootPath, StudioSettings? settings = null)
        {
            LastSettings = settings;
            return new FakeWorkspaceSession(projectRootPath);
        }
    }

    private static string CreateCloudSyncedMediaPath(string fileName)
    {
        string mediaDirectory = Path.Join(
            Path.GetTempPath(),
            "Trackdub.Application.Tests",
            Guid.NewGuid().ToString("N"),
            "OneDrive",
            "Videos");
        Directory.CreateDirectory(mediaDirectory);
        return Path.Join(mediaDirectory, fileName);
    }

    private sealed class FakeStoragePaths(string userDataRoot) : IAppStoragePaths
    {
        public string RootDirectory { get; } = userDataRoot;
        public string UserDataRoot { get; } = userDataRoot;
        public string UserCacheRoot { get; } = Path.Join(userDataRoot, "cache");
        public string? SharedAssetRoot { get; } = null;
        public bool IsPortable { get; } = false;
        public string ModelCacheDirectory { get; } = Path.Join(userDataRoot, "cache", "models");
        public string ModelCacheIndexPath { get; } = Path.Join(userDataRoot, "cache", "models", "index.json");
        public string LogFilePath { get; } = Path.Join(userDataRoot, "trackdub.log");
        public string SettingsPath { get; } = Path.Join(userDataRoot, "settings.json");
        public string LayoutPath { get; } = Path.Join(userDataRoot, "layout.json");
        public string ToolCacheDirectory { get; } = Path.Join(userDataRoot, "tools");
        public string FfmpegToolCacheDirectory { get; } = Path.Join(userDataRoot, "tools", "ffmpeg");
        public string EngineCacheDirectory { get; } = Path.Join(userDataRoot, "cache", "engines");
        public string ComponentCacheDirectory { get; } = Path.Join(userDataRoot, "cache", "components");
    }

    private sealed class FakeWorkspaceSession(string projectRootPath) : ITranscriptWorkspaceSession
    {
        public string ProjectRootPath { get; } = projectRootPath;

        public TranscriptWorkspace Workspace => throw new NotSupportedException();

        public IServiceProvider Services => throw new NotSupportedException();

        public int DisposeCount { get; private set; }

        public void Dispose()
        {
            DisposeCount++;
        }
    }
}
