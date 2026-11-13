using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

public sealed partial class RuntimePlannerTests
{
    [Fact]
    public async Task PlanAsync_WhenPreferredVariantSearchIsBlocked_UsesRunnableBaseVariantFallback()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        const string chatterboxModelId = "ResembleAI/chatterbox-turbo-ONNX";
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateKokoroSpec(),
            new ManifestSpec(
                ModelId: chatterboxModelId,
                Task: "tts",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "chatterbox",
                Aliases: ["chatterbox-turbo-onnx", "chatterbox-turbo"],
                RootFolder: "chatterbox-turbo",
                BenchmarkEntry: "onnx/model.onnx",
                Variants: [new ManifestVariantSpec("default", "onnx/model.onnx")]));

        string kokoroRoot = workspace.CreateCacheRoot("onnx-community/Kokoro-82M-v1.0-ONNX");
        workspace.WriteCacheFile(kokoroRoot, "model.onnx");
        string chatterboxRoot = workspace.CreateCacheRoot(chatterboxModelId);
        workspace.WriteCacheFile(chatterboxRoot, "onnx/model.onnx");
        RuntimePlanner planner = CreatePlanner(
            registry,
            [
                CreateCacheRecord(registry, "onnx-community/Kokoro-82M-v1.0-ONNX", kokoroRoot),
                CreateCacheRecord(registry, chatterboxModelId, chatterboxRoot)
            ],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredExecutionProvider: ExecutionProviderKind.DirectMl,
            RequirePreferredExecutionProvider: true,
            PreferredModelVariantAlias: "missing-optimized-variant"));

        Assert.True(plan.IsRunnable(), $"Expected runnable fallback, got {plan.Status}: {plan.Fallback?.Detail}");
        Assert.Equal(chatterboxModelId, plan.ModelId);
        Assert.Equal("chatterbox", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("default", plan.Variant);
        Assert.Contains(
            plan.Warnings,
            warning => warning.Code == RuntimePlanWarningCode.PreferredOptimizedVariantUnavailable &&
                       warning.Detail?.Contains("missing-optimized-variant", StringComparison.OrdinalIgnoreCase) == true);
    }
}
