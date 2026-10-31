using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Application.Mixing;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;
using System.Security.Cryptography;

#pragma warning disable CS0618

namespace Trackdub.Application.Tests;

public partial class TranscriptProjectServiceTests
{
    [Fact]
    public async Task CreateAsync_generates_transcript_revision_and_stage_runs()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        TranscriptProjectState result = await scope.Service.CreateAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);

        Assert.NotNull(result.CurrentTranscriptRevision);
        Assert.Equal(1, result.CurrentTranscriptRevision!.RevisionNumber);
        Assert.Equal(5, result.StageRuns.Count);
        Assert.DoesNotContain(
            result.StageRuns,
            stageRun => stageRun.StageName == StageNames.TextRefinementAsr);
        Assert.Contains(
            result.StageRuns,
            stageRun => stageRun.StageName == StageNames.SpeakerAssignment && stageRun.Status == StageRunStatus.Completed);
        Assert.Equal(5, result.StageRuns.Count(stageRun => stageRun.Status == StageRunStatus.Completed));
        Assert.Equal(2, result.TranscriptSegments.Count);
        Assert.Equal(2, result.Speakers.Count);
        Assert.Equal(2, result.SpeakerTurns.Count);
        Assert.All(result.TranscriptSegments, segment => Assert.NotNull(segment.SpeakerId));
        Assert.All(result.TranscriptSegments, segment => Assert.Equal("en", segment.DetectedLanguage));
        Assert.All(result.TranscriptSegments, segment => Assert.NotEmpty(segment.Words));
        Assert.Contains(result.TranscriptSegments, segment => segment.Words.Any(word => word.Confidence < 0.75d));
        Assert.Null(result.CurrentTranslationRevision);
        Assert.Equal("en", result.TranscriptLanguage);
        Assert.Contains(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.SpeechRegions);
        Assert.Contains(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.AudioQualityAnalysis);
        Assert.Contains(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.TranscriptRevision);
    }

    [Fact]
    public async Task CreateAsync_passes_requested_source_language_to_asr()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        FakeServiceScope scope = CreateScope(tempDirectory, transcriptionEngine: transcriptionEngine);

        await scope.Service.CreateAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath, SourceLanguage: "es-MX"),
            TestContext.Current.CancellationToken);

        Assert.Equal("es", transcriptionEngine.LastSourceLanguage);
    }

    [Fact]
    public async Task RunInitialTranscriptionAsync_after_media_spine_persists_detected_transcript_language()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        TranscriptProjectState spine = await scope.Workspace.CreateMediaSpineAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);

        Assert.Null(spine.TranscriptLanguage);
        Assert.Empty(spine.TranscriptSegments);

        TranscriptProjectState result = await scope.Workspace.RunInitialTranscriptionAsync(
            new TranscriptRunOptions
            {
                EnableSpeakerDiarization = true
            },
            TestContext.Current.CancellationToken);

        Assert.Equal("en", result.TranscriptLanguage);
        Assert.NotEmpty(result.TranscriptSegments);
        Assert.All(result.TranscriptSegments, segment => Assert.Equal("en", segment.DetectedLanguage));
    }

    [Fact]
    public async Task RunInitialTranscriptionAsync_persists_requested_source_when_engine_omits_detected_language()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(
            tempDirectory,
            transcriptionEngine: new FixedAudioTranscriptionEngine(
            [
                new RecognizedTranscriptSegment(0, 0.0, 3.0, "Hola.")
            ]));
        await scope.Workspace.CreateMediaSpineAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);

        TranscriptProjectState result = await scope.Workspace.RunInitialTranscriptionAsync(
            new TranscriptRunOptions
            {
                EnableSpeakerDiarization = false,
                SourceLanguage = "es-MX"
            },
            TestContext.Current.CancellationToken);

        Assert.Equal("es", result.TranscriptLanguage);
        Assert.NotEmpty(result.TranscriptSegments);
        Assert.All(result.TranscriptSegments, segment => Assert.Null(segment.DetectedLanguage));
    }

    [Fact]
    public async Task RunInitialTranscriptionAsync_does_not_separate_stems()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        await scope.Workspace.CreateMediaSpineAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);

        TranscriptProjectState result = await scope.Workspace.RunInitialTranscriptionAsync(
            new TranscriptRunOptions
            {
                EnableSpeakerDiarization = false
            },
            TestContext.Current.CancellationToken);

        Assert.Equal(0, scope.StemSeparationEngine.CallCount);
        Assert.NotNull(result.CurrentTranscriptRevision);
        Assert.DoesNotContain(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Vocals);
    }

    [Fact]
    public async Task SetTranscriptLanguageAsync_on_media_spine_primes_initial_transcription_source()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        FakeServiceScope scope = CreateScope(tempDirectory, transcriptionEngine: transcriptionEngine);
        TranscriptProjectState spine = await scope.Workspace.CreateMediaSpineAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);

        Assert.Null(spine.CurrentTranscriptRevision);
        Assert.Null(spine.TranscriptLanguage);

        TranscriptProjectState sourceSet = await scope.Workspace.SetTranscriptLanguageAsync(
            new SetTranscriptLanguageRequest("es-MX"),
            TestContext.Current.CancellationToken);
        TranscriptProjectState result = await scope.Workspace.RunInitialTranscriptionAsync(
            new TranscriptRunOptions
            {
                EnableSpeakerDiarization = false,
                SourceLanguage = sourceSet.TranscriptLanguage
            },
            TestContext.Current.CancellationToken);

        Assert.Equal("es", transcriptionEngine.LastSourceLanguage);
        Assert.Equal("en", result.TranscriptLanguage);
    }
}
