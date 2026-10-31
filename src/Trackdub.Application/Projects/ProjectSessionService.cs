using System.Diagnostics.CodeAnalysis;
using Trackdub.Contracts;
using Trackdub.Contracts.Transcripts;

namespace Trackdub.Application.Projects;

public interface ITranscriptWorkspaceSession : IDisposable
{
    string ProjectRootPath { get; }

    TranscriptWorkspace Workspace { get; }

    /// <summary>
    /// Service provider for the session's DI scope. Workspace workflows and
    /// pipeline hosts resolve per-session services (consent, model setup
    /// interaction, tier gates) from it.
    /// </summary>
    IServiceProvider Services { get; }
}

public interface ITranscriptWorkspaceSessionFactory
{
    ITranscriptWorkspaceSession Create(string projectRootPath, StudioSettings? settings = null);
}

public sealed record PendingProjectSession(
    ProjectRootNameCandidate ProjectRoot,
    ITranscriptWorkspaceSession Session);

public sealed class ProjectSessionService(
    ITranscriptWorkspaceSessionFactory sessionFactory,
    IAppStoragePaths? storagePaths = null) : IDisposable
{
    private readonly ITranscriptWorkspaceSessionFactory sessionFactory = sessionFactory ?? throw new ArgumentNullException(nameof(sessionFactory));
    private readonly IAppStoragePaths? _storagePaths = storagePaths;
    private readonly Stack<ProjectSessionHistoryEntry> undoStack = new();
    private readonly Stack<ProjectSessionHistoryEntry> redoStack = new();
    private ITranscriptWorkspaceSession? currentSession;

    public TranscriptWorkspace? CurrentWorkspace => currentSession?.Workspace;

    public string? CurrentProjectRootPath => currentSession?.ProjectRootPath;

    /// <summary>
    /// The live workspace session, when a project is open. Callers must not
    /// dispose it; the service owns session lifetime.
    /// </summary>
    public ITranscriptWorkspaceSession? CurrentSession => currentSession;

    public bool CanUndo => undoStack.Count > 0;

    public bool CanRedo => redoStack.Count > 0;

    public ITranscriptWorkspaceSession CreatePendingSession(string projectRootPath, StudioSettings settings) =>
        sessionFactory.Create(projectRootPath, settings);

    public PendingProjectSession CreatePendingSessionForMedia(
        string mediaPath,
        string projectName,
        StudioSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        string projectParentDirectory;
        if (string.IsNullOrWhiteSpace(settings.DefaultProjectDirectory))
        {
            projectParentDirectory = ProjectRootNameResolver.ResolveProjectParentDirectory(
                mediaPath,
                _storagePaths?.UserDataRoot);
        }
        else
        {
            // A relative value would resolve against the process working directory, which is not
            // a stable project location, so require an absolute folder.
            string configuredDirectory = settings.DefaultProjectDirectory.Trim();
            if (!Path.IsPathFullyQualified(configuredDirectory))
            {
                throw new InvalidOperationException(
                    $"The default project folder must be an absolute path: {configuredDirectory}");
            }

            projectParentDirectory = Path.GetFullPath(configuredDirectory);
            if (!Directory.Exists(projectParentDirectory))
            {
                throw new DirectoryNotFoundException(
                    $"The default project folder does not exist: {projectParentDirectory}");
            }
        }

        ProjectRootNameCandidate projectRoot = ProjectRootNameResolver.CreateAvailableProjectRoot(
            mediaPath,
            projectName,
            projectParentDirectory);
        return new PendingProjectSession(projectRoot, CreatePendingSession(projectRoot.ProjectRootPath, settings));
    }

    public void SetCurrentSession(ITranscriptWorkspaceSession session)
    {
        ArgumentNullException.ThrowIfNull(session);

        ClearCurrentSession();
        currentSession = session;
    }

    public void ReplaceCurrentSession(string projectRootPath, StudioSettings settings) =>
        SetCurrentSession(sessionFactory.Create(projectRootPath, settings));

    public void ClearCurrentSession()
    {
        currentSession?.Dispose();
        currentSession = null;
    }

    public bool TryGetCurrentProject(
        [NotNullWhen(true)] out TranscriptWorkspace? workspace,
        [NotNullWhen(true)] out string? projectRootPath)
    {
        workspace = CurrentWorkspace;
        projectRootPath = CurrentProjectRootPath;
        return workspace is not null && !string.IsNullOrWhiteSpace(projectRootPath);
    }

    public void RecordHistoryTransition(
        TranscriptProjectState? before,
        TranscriptProjectState after,
        bool isRestoringHistory)
    {
        ArgumentNullException.ThrowIfNull(after);

        if (isRestoringHistory ||
            before is null ||
            before.ProjectState.Project.Id != after.ProjectState.Project.Id ||
            IsSameEditingState(before, after))
        {
            return;
        }

        undoStack.Push(new ProjectSessionHistoryEntry(before, after));
        redoStack.Clear();
    }

    public void ClearHistory()
    {
        undoStack.Clear();
        redoStack.Clear();
    }

    public TranscriptProjectState? PopUndoState()
    {
        if (undoStack.Count == 0)
        {
            return null;
        }

        ProjectSessionHistoryEntry entry = undoStack.Pop();
        redoStack.Push(entry);
        return entry.Before;
    }

    public TranscriptProjectState? PopRedoState()
    {
        if (redoStack.Count == 0)
        {
            return null;
        }

        ProjectSessionHistoryEntry entry = redoStack.Pop();
        undoStack.Push(entry);
        return entry.After;
    }

    public RestoreEditingStateRequest CreateRestoreEditingStateRequest(TranscriptProjectState state)
    {
        ArgumentNullException.ThrowIfNull(state);

        return new RestoreEditingStateRequest(
            state.SelectedTranslationTargetLanguage,
            state.TranscriptSegments,
            state.CurrentTranslationRevision is null ? null : state.TranslatedSegments,
            state.Speakers.ToDictionary(speaker => speaker.Id, speaker => speaker.DisplayName),
            state.VoiceAssignments.Where(assignment => !assignment.IsFallback).ToArray());
    }

    public void Dispose() => ClearCurrentSession();

    private static bool IsSameEditingState(TranscriptProjectState before, TranscriptProjectState after)
    {
        return before.CurrentTranscriptRevision?.Id == after.CurrentTranscriptRevision?.Id &&
               before.CurrentTranslationRevision?.Id == after.CurrentTranslationRevision?.Id &&
               before.Speakers.Select(speaker => (speaker.Id, speaker.DisplayName))
                   .SequenceEqual(after.Speakers.Select(speaker => (speaker.Id, speaker.DisplayName))) &&
               before.VoiceAssignments
                   .Where(assignment => !assignment.IsFallback)
                   .Select(assignment => (assignment.SpeakerId, assignment.VoiceModelId, assignment.VoiceVariant))
                   .OrderBy(assignment => assignment.SpeakerId)
                   .SequenceEqual(after.VoiceAssignments
                       .Where(assignment => !assignment.IsFallback)
                       .Select(assignment => (assignment.SpeakerId, assignment.VoiceModelId, assignment.VoiceVariant))
                       .OrderBy(assignment => assignment.SpeakerId));
    }

    private sealed record ProjectSessionHistoryEntry(
        TranscriptProjectState Before,
        TranscriptProjectState After);
}
