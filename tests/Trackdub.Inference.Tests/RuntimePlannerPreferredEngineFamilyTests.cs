using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

public sealed partial class RuntimePlannerTests
{
    [Fact]
    public async Task PlanAsync_DoesNotReuseCachedPlanAcrossPreferredEngineFamilies()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        const string firstModelId = "test/vad-a";
        const string secondModelId = "test/vad-b";
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT", modelId: firstModelId) with
            {
                EngineFamily = "vad-a",
                RootFolder = "vad-a",
                Aliases = ["vad-a"]
            },
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT", modelId: secondModelId) with
            {
                EngineFamily = "vad-b",
                RootFolder = "vad-b",
                Aliases = ["vad-b"]
            });

        string firstRoot = workspace.CreateCacheRoot(firstModelId);
        workspace.WriteCacheFile(firstRoot, "onnx/model_int8.onnx");
        string secondRoot = workspace.CreateCacheRoot(secondModelId);
        workspace.WriteCacheFile(secondRoot, "onnx/model_int8.onnx");
        RuntimePlanner planner = CreatePlanner(
            registry,
            [
                new(firstModelId, firstRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new(secondModelId, secondRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan first = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredEngineFamily: "vad-a"));
        StageRuntimePlan second = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredEngineFamily: "vad-b"));

        Assert.Equal("vad-a", first.EngineFamily);
        Assert.Equal("vad-b", second.EngineFamily);
    }
}
