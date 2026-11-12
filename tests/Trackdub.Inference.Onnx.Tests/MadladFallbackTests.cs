using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Madlad;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class MadladFallbackTests
{
    [Theory]
    [InlineData(StageRuntimePlanStatus.Ready)]
    [InlineData(StageRuntimePlanStatus.DownloadRequired)]
    public async Task Cpu_replan_drops_trt_variant_pin_and_reports_missing_prerequisite(StageRuntimePlanStatus status)
    {
        var planner = new Planner(status);
        var engine = new MadladTranslationEngine(planner, BenchmarkModelPathResolver.CreateDefault());
        var request = new TranslationRequest("en", "fr", [], PreferredModelAlias: "madlad400",
            PreferredModelVariantAlias: "trt-fp16");
        var gpuFailure = new InvalidOperationException("GPU admission budget exhausted");

        if (status == StageRuntimePlanStatus.Ready)
        {
            StageRuntimePlan plan = await engine.ReplanForCpuAsync(request, gpuFailure, CancellationToken.None);
            Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
            Assert.Equal("quantized", plan.Variant);
        }
        else
        {
            InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                engine.ReplanForCpuAsync(request, gpuFailure, CancellationToken.None));
            Assert.Contains("Download the MADLAD quantized or int4-kv variant", failure.Message);
            Assert.Same(gpuFailure, failure.InnerException);
        }

        Assert.NotNull(planner.Request);
        Assert.Null(planner.Request.PreferredModelVariantAlias);
        Assert.Equal(ExecutionProviderKind.Cpu, planner.Request.PreferredExecutionProvider);
        Assert.True(planner.Request.RequirePreferredExecutionProvider);
        Assert.True(planner.Request.RequirePreferredModelAlias);
    }

    [Fact]
    public async Task Translate_requires_the_routed_alias_so_the_planner_cannot_substitute_another_model()
    {
        var planner = new Planner(StageRuntimePlanStatus.Blocked);
        var engine = new MadladTranslationEngine(planner, BenchmarkModelPathResolver.CreateDefault());
        var request = new TranslationRequest("es", "en", [], PreferredModelAlias: "madlad400-mt",
            PreferredModelVariantAlias: "trt-fp16", PreferredExecutionProvider: "tensorrt",
            RequirePreferredExecutionProvider: true);

        await Assert.ThrowsAnyAsync<Exception>(() => engine.TranslateAsync(request, CancellationToken.None));

        Assert.NotNull(planner.Request);
        Assert.Equal("madlad400-mt", planner.Request.NormalizedPreferredModelAlias);
        Assert.Equal("trt-fp16", planner.Request.NormalizedPreferredModelVariantAlias);
        Assert.True(planner.Request.RequirePreferredModelAlias);
    }

    [Fact]
    public async Task Stream_requires_the_routed_alias_so_the_planner_cannot_substitute_another_model()
    {
        var planner = new Planner(StageRuntimePlanStatus.Blocked);
        var engine = new MadladTranslationEngine(planner, BenchmarkModelPathResolver.CreateDefault());
        var request = new TranslationRequest("es", "en", [], PreferredModelAlias: "madlad400-mt",
            PreferredModelVariantAlias: "trt-fp16");

        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await foreach (var _ in engine.TranslateStreamAsync(request, Guid.NewGuid(), "snapshot", Guid.NewGuid(), CancellationToken.None))
            {
            }
        });

        Assert.NotNull(planner.Request);
        Assert.True(planner.Request.RequirePreferredModelAlias);
    }

    [Fact]
    public async Task Translate_without_a_routed_alias_leaves_model_choice_to_the_planner()
    {
        var planner = new Planner(StageRuntimePlanStatus.Blocked);
        var engine = new MadladTranslationEngine(planner, BenchmarkModelPathResolver.CreateDefault());
        var request = new TranslationRequest("es", "en", []);

        await Assert.ThrowsAnyAsync<Exception>(() => engine.TranslateAsync(request, CancellationToken.None));

        Assert.NotNull(planner.Request);
        Assert.False(planner.Request.RequirePreferredModelAlias);
    }

    private sealed class Planner(StageRuntimePlanStatus status) : IRuntimePlanner
    {
        public StageRuntimePlanningRequest? Request { get; private set; }
        public Task<StageRuntimePlan> PlanAsync(StageRuntimePlanningRequest request, CancellationToken cancellationToken = default)
        {
            Request = request;
            return Task.FromResult(new StageRuntimePlan
            {
                Stage = RuntimeStage.Translation,
                Status = status,
                ModelAlias = "madlad400",
                Variant = "quantized",
                ExecutionProvider = ExecutionProviderKind.Cpu
            });
        }
    }
}
