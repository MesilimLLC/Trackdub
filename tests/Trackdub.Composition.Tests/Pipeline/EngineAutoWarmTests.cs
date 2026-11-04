using Microsoft.Extensions.DependencyInjection;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Composition.Pipeline;
using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.TestDoubles;
using Xunit;

namespace Trackdub.Composition.Tests.Pipeline;

/// <summary>
/// Post-install engine warming exists so a freshly installed model builds its TensorRT engines
/// (and EP-context artifact) while the user is still setting up, instead of on the first run and
/// again on every launch after. These tests pin the two properties that matter: only a real
/// install triggers a warm, and the warmed stage is the one the installed model belongs to.
/// </summary>
public class EngineAutoWarmTests
{
    [Fact]
    public async Task Installing_a_model_warms_exactly_the_stage_that_model_serves()
    {
        var warmup = new RecordingWarmupService();
        using var coordinator = CreateCoordinator(warmup);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        IReadOnlyList<RuntimeStage> warmed = Assert.Single(warmup.Requests).Stages;
        Assert.Equal([RuntimeStage.Asr], warmed);
    }

    [Fact]
    public void Stage_resolution_maps_each_manifest_task_to_its_stage()
    {
        using var coordinator = CreateCoordinator(new RecordingWarmupService());

        Assert.Equal([RuntimeStage.Vad], coordinator.ResolveStagesForModel("onnx-community/silero-vad"));
        Assert.Equal([RuntimeStage.Diarization], coordinator.ResolveStagesForModel("cgus/diar_streaming_sortformer_4spk-v2.1-onnx"));
        Assert.Equal([RuntimeStage.Tts], coordinator.ResolveStagesForModel("ResembleAI/chatterbox-turbo-ONNX"));
        Assert.Equal([RuntimeStage.Translation], coordinator.ResolveStagesForModel("onnx-community/opus-mt-en-es"));
    }

    [Fact]
    public void A_model_no_stage_can_select_warms_nothing()
    {
        using var coordinator = CreateCoordinator(new RecordingWarmupService());

        // Lip synthesis is not a default warmup stage, and face models map to no stage at all.
        Assert.Empty(coordinator.ResolveStagesForModel("InsightFace/scrfd-500m"));
        Assert.Empty(coordinator.ResolveStagesForModel("does-not-exist/at-all"));
    }

    [Fact]
    public async Task Duplicate_install_notifications_coalesce_into_one_warm()
    {
        var warmup = new RecordingWarmupService { Gate = new TaskCompletionSource<bool>() };
        using var coordinator = CreateCoordinator(warmup);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        warmup.Gate.SetResult(true);
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Single(warmup.Requests);
    }

    [Fact]
    public async Task A_notification_arriving_mid_warm_coalesces_instead_of_queueing_a_second_pass()
    {
        // Coalescing must span the whole warm, not just the queue: a model that reports Installed
        // twice (download completion plus a state refresh) while its engines are still building
        // must not trigger a duplicate GPU-heavy build.
        var warmup = new RecordingWarmupService { Gate = new TaskCompletionSource<bool>() };
        using var coordinator = CreateCoordinator(warmup);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        await WaitUntilAsync(() => warmup.Requests.Count == 1);
        Assert.Single(warmup.Requests);

        // Second notification while the first warm is still in flight.
        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");

        warmup.Gate.SetResult(true);
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Single(warmup.Requests);
    }

    [Fact]
    public async Task Distinct_models_each_get_their_own_warm()
    {
        var warmup = new RecordingWarmupService();
        using var coordinator = CreateCoordinator(warmup);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        coordinator.RequestWarmAfterInstall("onnx-community/opus-mt-en-es");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Equal(2, warmup.Requests.Count);
        Assert.Contains(warmup.Requests, request => request.Stages.Contains(RuntimeStage.Asr));
        Assert.Contains(warmup.Requests, request => request.Stages.Contains(RuntimeStage.Translation));
    }

    [Fact]
    public async Task The_environment_switch_turns_post_install_warming_off()
    {
        var warmup = new RecordingWarmupService();
        using var coordinator = CreateCoordinator(
            warmup,
            environment: name => name == EngineAutoWarmCoordinator.DisableEnvironmentVariable ? "0" : null);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Empty(warmup.Requests);
    }

    [Fact]
    public async Task A_failing_warm_never_surfaces_to_the_install_path_and_later_warms_still_run()
    {
        var warmup = new RecordingWarmupService { ShouldThrow = true };
        using var coordinator = CreateCoordinator(warmup);

        // Must not throw: warming is an optimization, and an install cannot fail on it.
        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        // The worker survived the failure and still accepts work: queue a second, different model.
        warmup.ShouldThrow = false;
        coordinator.RequestWarmAfterInstall("onnx-community/opus-mt-en-es");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Equal(2, warmup.Requests.Count);
        Assert.Contains(warmup.Requests, request => request.Stages.Contains(RuntimeStage.Translation));
    }

    [Fact]
    public async Task Draining_waits_for_work_already_queued()
    {
        // The gate is released by the test, so a drain that does not actually wait would return
        // before the warm finished; asserting on the record (not on task completion state) keeps
        // this free of timing races on fast hardware.
        var warmup = new RecordingWarmupService { Gate = new TaskCompletionSource<bool>() };
        using var coordinator = CreateCoordinator(warmup);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        await WaitUntilAsync(() => warmup.Requests.Count == 1);

        Task drain = coordinator.DrainAsync(TestContext.Current.CancellationToken);
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Assert.False(drain.IsCompleted, "drain must not complete while a warm is gated");

        warmup.Gate!.SetResult(true);
        await drain;
        Assert.Single(warmup.Requests);
    }

    [Fact]
    public async Task Draining_does_not_seal_the_queue_and_later_installs_still_warm()
    {
        // A drain is a flush point, not a shutdown. Sealing the channel writer here would make
        // every subsequent install silently drop its warm request.
        var warmup = new RecordingWarmupService();
        using var coordinator = CreateCoordinator(warmup);

        coordinator.RequestWarmAfterInstall("onnx-community/whisper-small");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);
        Assert.Single(warmup.Requests);

        coordinator.RequestWarmAfterInstall("onnx-community/opus-mt-en-es");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Equal(2, warmup.Requests.Count);
        Assert.Contains(warmup.Requests, request => request.Stages.Contains(RuntimeStage.Translation));
    }

    [Fact]
    public void Every_warmed_stage_is_pinned_to_the_installed_model()
    {
        const string modelId = "ResembleAI/chatterbox-turbo-ONNX";
        foreach (RuntimeStage stage in ResidentWarmupRequest.DefaultStages)
        {
            RuntimeModelSelections selections = EngineAutoWarmCoordinator.BuildSelectionsForModel(modelId, [stage]);

            Assert.Equal(modelId, AliasForStage(selections, stage));
            Assert.Equal(RuntimeModelSelectionIntent.Explicit, selections.GetSelectionIntent(stage));
            foreach (RuntimeStage otherStage in ResidentWarmupRequest.DefaultStages.Where(candidate => candidate != stage))
            {
                Assert.Null(AliasForStage(selections, otherStage));
            }
        }
    }

    private static string? AliasForStage(RuntimeModelSelections selections, RuntimeStage stage) => stage switch
    {
        RuntimeStage.Vad => selections.VadModelAlias,
        RuntimeStage.Asr => selections.AsrModelAlias,
        RuntimeStage.Translation => selections.TranslationModelAlias,
        RuntimeStage.Tts => selections.TtsModelAlias,
        RuntimeStage.Diarization => selections.DiarizationModelAlias,
        RuntimeStage.Separation => selections.SeparationModelAlias,
        RuntimeStage.SpeechEnhancement => selections.SpeechEnhancementModelAlias,
        RuntimeStage.LipSync => selections.LipSyncModelAlias,
        RuntimeStage.TextRefinement => selections.TextRefinementModelAlias,
        RuntimeStage.OverlapRescue => selections.OverlapRescueModelAlias,
        RuntimeStage.LipSynthesis => selections.LipSynthesisModelAlias,
        _ => null,
    };

    [Fact]
    public void Speech_enhancement_warm_uses_its_stage_specific_installed_model_alias()
    {
        const string modelId = "Rikorose/DeepFilterNet3";
        RuntimeModelSelections selections = EngineAutoWarmCoordinator.BuildSelectionsForModel(
            modelId,
            [RuntimeStage.SpeechEnhancement]);

        Assert.Equal(modelId, selections.SpeechEnhancementModelAlias);
        Assert.Equal(RuntimeModelSelectionIntent.Explicit, selections.GetSelectionIntent(RuntimeStage.SpeechEnhancement));
        Assert.Null(selections.TtsModelAlias);
        Assert.Equal(modelId, RuntimeModelRequestFactory.CreateStageRequest(
            RuntimeModelRequestFactory.CreateOptions(selections),
            RuntimeStage.SpeechEnhancement).PreferredModelAlias);
    }

    [Fact]
    public async Task OnModelInstalled_is_the_install_path_entry_point_and_warms_that_model()
    {
        var warmup = new RecordingWarmupService();
        using var coordinator = CreateCoordinator(warmup);

        // This is exactly what ModelDownloadOrchestrator calls after a successful install.
        coordinator.OnModelInstalled("onnx-community/whisper-small");
        await coordinator.DrainAsync(TestContext.Current.CancellationToken);

        Assert.Equal([RuntimeStage.Asr], Assert.Single(warmup.Requests).Stages);
    }

    [Fact]
    public void CompositionRoot_RegistersTheAutoWarmQueue()
    {
        var services = new ServiceCollection();
        services.AddTrackdub();
        using ServiceProvider provider = services.BuildServiceProvider(
            new ServiceProviderOptions { ValidateOnBuild = true, ValidateScopes = true });

        Assert.IsType<EngineAutoWarmCoordinator>(provider.GetRequiredService<IEngineAutoWarmQueue>());
    }

    private static async Task WaitUntilAsync(Func<bool> condition, int timeoutMs = 2000)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        while (!condition())
        {
            if (stopwatch.ElapsedMilliseconds > timeoutMs)
            {
                throw new TimeoutException("Condition was not observed within the timeout.");
            }

            await Task.Delay(10, TestContext.Current.CancellationToken);
        }
    }

    private static EngineAutoWarmCoordinator CreateCoordinator(
        RecordingWarmupService warmup,
        Func<string, string?>? environment = null)
    {
        Assert.True(
            BundledModelManifestRegistry.TryLoadDefault(out BundledModelManifestRegistry? registry, out string? error),
            error);
        return new EngineAutoWarmCoordinator(
            warmup,
            registry!,
            logger: null,
            readEnvironment: environment ?? (_ => null));
    }

    private sealed record WarmCall(IReadOnlyList<RuntimeStage> Stages);

    private sealed class RecordingWarmupService : IResidentWarmupService
    {
        private readonly List<WarmCall> requests = [];
        private readonly object gate = new();

        public bool ShouldThrow { get; set; }
        public TaskCompletionSource<bool>? Gate { get; init; }

        public IReadOnlyList<WarmCall> Requests
        {
            get
            {
                lock (gate)
                {
                    return requests.ToArray();
                }
            }
        }

        public async Task<ResidentWarmupReport> WarmAsync(
            ResidentWarmupRequest request,
            IProgress<string>? progress = null,
            CancellationToken cancellationToken = default)
        {
            lock (gate)
            {
                requests.Add(new WarmCall(request.Stages));
            }

            if (Gate is not null)
            {
                await Gate.Task.WaitAsync(cancellationToken);
            }

            if (ShouldThrow)
            {
                throw new InvalidOperationException("engine build failed");
            }

            return new ResidentWarmupReport(
                Skipped: false,
                SkipReason: null,
                Providers: [],
                Stages: request.Stages
                    .Select(stage => new ResidentStageWarmupOutcome(stage, Attempted: true, Succeeded: true, TimeSpan.Zero, null))
                    .ToList());
        }
    }

    private sealed class FakeModelDownloadOrchestrator : IModelDownloadOrchestrator
    {
        private readonly List<IObserver<ModelStateChange>> observers = [];

        public IObservable<ModelStateChange> StateChanges => new Observable(observers);

        public void Emit(ModelStateChange change)
        {
            foreach (IObserver<ModelStateChange> observer in observers.ToArray())
            {
                observer.OnNext(change);
            }
        }

        public Task<ModelDownloadResult> DownloadAsync(
            string modelId,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public Task<ModelDownloadResult> DownloadAsync(
            string modelId,
            string? variantAlias,
            IProgress<ModelDownloadProgress>? progress,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public Task<ModelDownloadResult> RepairAsync(
            string modelId,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public Task<bool> UninstallAsync(string modelId, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        private sealed class Observable(List<IObserver<ModelStateChange>> observers) : IObservable<ModelStateChange>
        {
            public IDisposable Subscribe(IObserver<ModelStateChange> observer)
            {
                observers.Add(observer);
                return new Unsubscribe(() => observers.Remove(observer));
            }

            private sealed class Unsubscribe(Action dispose) : IDisposable
            {
                public void Dispose() => dispose();
            }
        }
    }
}
