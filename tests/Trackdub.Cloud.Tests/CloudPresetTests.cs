using Trackdub.Cloud.Jobs;
using Trackdub.TestDoubles;

namespace Trackdub.Cloud.Tests;

public sealed class CloudPresetTests
{
    private static (JobService Service, CreateJobRequest Request) Setup(string target, string preset)
    {
        var store = new FakeJobStore(); var objects = new FakeCloudArtifactStore();
        var id = Guid.NewGuid();
        store.Uploads[id] = new(id, "owner", new("upload/input.mp4", "v1", 10, new string('a', 64)), "upload/input.mp4", 10, new string('a', 64), "video/mp4", DateTimeOffset.UtcNow.AddMinutes(15));
        return (new(store, objects, TimeProvider.System), new(1, "Project", new(id), "en", target, preset));
    }

    [Theory]
    [InlineData("es", "poc-server-v1")]
    [InlineData("zh", "poc-server-zh-v1")]
    public async Task ApprovedLanguagePresetPairsAreAccepted(string target, string preset)
    {
        var s = Setup(target, preset);
        var job = await s.Service.SubmitAsync("owner", "key", s.Request, TestContext.Current.CancellationToken);
        Assert.Equal(preset, job.Request.Preset);
        Assert.Equal(target, job.Request.TargetLanguage);
    }

    [Theory]
    [InlineData("zh", "poc-server-v1")]
    [InlineData("es", "poc-server-zh-v1")]
    [InlineData("fr", "poc-server-v1")]
    [InlineData("zh", "poc-server-quality-v1")]
    public async Task UnapprovedLanguagePresetPairsAreRejected(string target, string preset)
    {
        var s = Setup(target, preset);
        var error = await Assert.ThrowsAsync<JobException>(() => s.Service.SubmitAsync("owner", "key", s.Request, TestContext.Current.CancellationToken));
        Assert.Equal(400, error.StatusCode);
    }

    [Fact]
    public void ChinesePresetAssignsTheQwen3MandarinVoiceToTheFirstSpeaker()
    {
        Assert.True(CloudPresets.TryGet(CloudPresets.ChineseName, "zh", out var preset));
        Assert.Equal("SPEAKER_00:qwen3:vivian", preset.Voice);
    }

    [Theory]
    [InlineData(1L, true)]
    [InlineData(CloudLimits.MaxUploadBytes, true)]
    [InlineData(CloudLimits.MaxUploadBytes + 1, false)]
    [InlineData(0L, false)]
    public async Task UploadSizeLimitIsOneGibibyte(long size, bool accepted)
    {
        var uploads = new UploadService(new FakeJobStore(), new FakeCloudArtifactStore(), TimeProvider.System);
        var request = new UploadRequest(size, new string('a', 64), "video/mp4");
        if (accepted)
            await uploads.CreateAsync("owner", request, TestContext.Current.CancellationToken);
        else
            Assert.Equal(400, (await Assert.ThrowsAsync<JobException>(() => uploads.CreateAsync("owner", request, TestContext.Current.CancellationToken))).StatusCode);
    }
}
