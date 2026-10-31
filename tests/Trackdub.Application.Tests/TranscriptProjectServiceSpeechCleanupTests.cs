using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.StageRuns;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public partial class TranscriptProjectServiceTests
{
    [Fact]
    public async Task RunTranscriptStageAsync_uses_normalized_source_without_cleanup_when_disabled()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var enhancementService = new FakeSpeechAudioEnhancementService();
        var processingService = new FakeSpeechAudioProcessingService();
        FakeServiceScope scope = CreateScope(
            tempDirectory,
            speechAudioEnhancementService: enhancementService,
            speechAudioProcessingService: processingService,
            enableSpeechEnhancement: true);
        TranscriptProjectState created = await scope.Workspace.Project.CreateAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath, EnableSpeakerDiarization: false),
            TestContext.Current.CancellationToken);
        int enhancementCallsBefore = enhancementService.CallCount;
        int processingRequestsBefore = processingService.Requests.Count;
        HashSet<Guid> existingRunIds = created.StageRuns.Select(static run => run.Id).ToHashSet();

        TranscriptProjectState state = await scope.Workspace.RunTranscriptStageAsync(
            StageNames.Asr,
            new TranscriptRunOptions
            {
                EnableSpeakerDiarization = false,
                PrepareSpeechAudio = false
            },
            TestContext.Current.CancellationToken);

        Assert.Equal(enhancementCallsBefore, enhancementService.CallCount);
        Assert.Equal(processingRequestsBefore, processingService.Requests.Count);
        Assert.DoesNotContain(state.StageRuns, run => !existingRunIds.Contains(run.Id) &&
            run.StageName is StageNames.AudioPreparation or StageNames.SpeechEnhancement);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, state.AsrAudioRelativePath);
    }
}
