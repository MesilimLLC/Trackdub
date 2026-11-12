using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

public sealed partial class RuntimePlannerTests
{
    [Fact]
    public async Task PlanAsync_SoftAliasFallthroughPlan_IsNotServedToRequiredAliasRequest()
    {
        // A soft MADLAD request falls through to the installed OPUS-MT model. That cached plan must not
        // be returned when the MADLAD engine later requires its routed alias, or MADLAD loads OPUS files.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        string cacheRoot = workspace.CreateCacheRoot("onnx-community/opus-mt-es-en");
        WriteBundledCacheFiles(workspace, registry, "opus-es-en", cacheRoot, "merged-decoder");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/opus-mt-es-en", cacheRoot)],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        var softRequest = new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            PreferredModelAlias: "madlad400-mt",
            SourceLanguage: "es",
            TargetLanguage: "en");

        StageRuntimePlan softPlan = await planner.PlanAsync(softRequest);
        StageRuntimePlan strictPlan = await planner.PlanAsync(softRequest with { RequirePreferredModelAlias = true });

        Assert.Equal("opus-mt", softPlan.EngineFamily);
        Assert.NotEqual("opus-mt", strictPlan.EngineFamily);
        Assert.False(strictPlan.IsRunnable());
    }
}
