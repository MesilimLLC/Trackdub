using System.Text.Json;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Composition.Runtime.Planning;
using Trackdub.Infrastructure.Settings;

namespace Trackdub.Inference.Tests;

public sealed partial class RuntimePlannerTests
{
    private const string ValidSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    [Fact]
    public async Task PlanAsync_NoVerdict_ReturnsReadyDirectMlWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            request => new ExecutionProviderSmokeTestResult(
                request.ExecutionProvider is ExecutionProviderKind.DirectMl &&
                request.Variant.Equals("fp16", StringComparison.OrdinalIgnoreCase)));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("silero-vad", plan.ModelAlias);
        Assert.Equal("silero-vad", plan.EngineFamily);
        Assert.Equal("balanced", plan.ModelTier);
        Assert.Equal("fp16", plan.Variant);
        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
    }

    [Fact]
    public async Task PlanAsync_when_selected_provider_conflicts_with_expected_runtime_adds_warning()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT") with
            {
                ExpectedRuntime = "onnxruntime-cpu"
            });

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Contains(plan.Warnings, warning => warning.Code == RuntimePlanWarningCode.ExpectedRuntimeMismatch);
    }

    [Fact]
    public async Task PlanAsync_WhenTensorRTRtxAvailableForVad_RoutesAroundTensorRtRtx()
    {
        // #329: silero-vad fails the TensorRT RTX build on every attempt (fp16 and fp32), so the
        // planner must not offer TRT for VAD even if a smoke stub would pass it.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        var smokeRequests = new List<ExecutionProviderKind>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smokeRequests.Add(request.ExecutionProvider);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("silero-vad", plan.ModelAlias);
        Assert.Equal("fp16", plan.Variant);
        Assert.DoesNotContain(ExecutionProviderKind.TensorRTRtx, smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_WhenTensorRTRtxSmokeFailsForVad_FallsBackToDirectMl()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request => new ExecutionProviderSmokeTestResult(
                request.ExecutionProvider is not ExecutionProviderKind.TensorRTRtx
                    and not ExecutionProviderKind.TensorRt));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("silero-vad", plan.ModelAlias);
    }

    [Fact]
    public async Task PlanAsync_WhenTensorRtAndCudaAvailableForVad_UsesTensorRtWhenSmokePasses()
    {
        // Milestone order probes TensorRt before Cuda; both are allowed for VAD now.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request => new ExecutionProviderSmokeTestResult(
                request.ExecutionProvider != ExecutionProviderKind.TensorRTRtx));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_WhenTensorRtAndCudaAvailableForVad_UsesCudaBecauseTensorRtExcluded()
    {
        // Both TensorRT providers are excluded for VAD (Silero's If subgraph fails engine
        // building on both; 2026-10-18 retest), so an available CUDA provider wins instead.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_int8.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.TensorRt, true),
                new(ExecutionProviderKind.Cuda, true)
            ],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.Cuda, plan.ExecutionProvider);
        Assert.Equal("int8", plan.Variant);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_RequiredCuda_ReturnsReadyWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "test/whisper-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "whisper-onnx",
                Aliases: ["whisper-tiny-onnx"],
                RootFolder: "whisper-asr",
                BenchmarkEntry: "onnx/encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("fp16", "onnx/encoder_model_fp16.onnx"),
                    new ManifestVariantSpec("int8", "onnx/encoder_model_int8.onnx")
                ],
                CapabilityTags: ["language-detection"],
                SourceLanguages: ["auto"]));

        string cacheRoot = workspace.CreateCacheRoot("test/whisper-asr");
        workspace.WriteCacheFile(cacheRoot, "onnx/encoder_model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("test/whisper-asr", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.Cuda, true)],
            _ => new ExecutionProviderSmokeTestResult(false, "CUDA smoke failed."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredExecutionProvider: ExecutionProviderKind.Cuda,
            RequirePreferredExecutionProvider: true));

        // C4: planning never smokes, so a smoke failure cannot block or fall back here;
        // proving (and failure surfacing) happens at warmup/stage execution.
        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.Cuda, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_DoesNotSmokeOrFallBack_PicksFirstProvider()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_int8.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.TensorRTRtx, true), new(ExecutionProviderKind.DirectMl, true)],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(false, "DirectML smoke-test failed.");
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("fp16", plan.Variant);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_EnumeratesDevicesAndStampsPlan_WhenExclusionsAreNull()
    {
        // Preflight plans before BeginRun, when CurrentExclusions is still null; the plan
        // must still carry the device so the run's same-key cache hit can engage fallback.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 1, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceEnumerator: enumerator);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal(0, plan.DeviceIndex);
        Assert.Equal("dGPU", plan.DeviceAdapterDescription);
        Assert.Equal(1, enumerator.GetDevicesCallCount);
    }

    [Fact]
    public async Task PlanAsync_PreflightPlanBeforeBeginRun_CarriesDeviceIntoRunCacheHit()
    {
        // Preflight (no run yet, CurrentExclusions null) and the run's empty-exclusion
        // lookup share a cache key; the cached preflight plan must already carry the
        // device or DeviceFallbackSessionCreator can never engage.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusionProvider = new PipelineDeviceExclusionProvider();
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 1, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: exclusionProvider,
            deviceEnumerator: enumerator);

        StageRuntimePlan preflightPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));
        Assert.Equal(0, preflightPlan.DeviceIndex);

        exclusionProvider.BeginRun();
        StageRuntimePlan runPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.Equal(0, runPlan.DeviceIndex);
        Assert.Equal("dGPU", runPlan.DeviceAdapterDescription);
    }

    [Fact]
    public async Task PlanAsync_UsesProviderAvailabilityUnchanged_WhenDeviceEnumeratorIsNull()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusions = new DeviceExclusionSet();
        exclusions.MarkMemoryExhausted(1);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Vad,
            DeviceExclusions: exclusions));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_AppliesDeviceExclusions_WhenExclusionsAndEnumeratorArePresent()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        var exclusions = new DeviceExclusionSet();
        exclusions.MarkMemoryExhausted(0);
        var exclusionProvider = new FixedExclusionProvider(exclusions);
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 1, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: exclusionProvider,
            deviceEnumerator: enumerator);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Vad,
            PreferredExecutionProvider: ExecutionProviderKind.DirectMl,
            RequirePreferredExecutionProvider: true));

        Assert.Equal(StageRuntimePlanStatus.Blocked, plan.Status);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ProviderUnavailable, plan.Fallback!.Code);
        Assert.Equal(1, enumerator.GetDevicesCallCount);
    }

    [Fact]
    public async Task PlanAsync_StampsSelectedDeviceIndexAndAdapter_WhenDevicesAvailable()
    {
        // C10 reachability: planner-produced plans must carry DeviceIndex/DeviceAdapterDescription
        // so the DeviceFallbackSessionCreator catch filter can engage in production.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusionProvider = new PipelineDeviceExclusionProvider();
        exclusionProvider.BeginRun();
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 1, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: exclusionProvider,
            deviceEnumerator: enumerator);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal(0, plan.DeviceIndex);
        Assert.Equal("dGPU", plan.DeviceAdapterDescription);
        Assert.Equal(1, enumerator.GetDevicesCallCount);
    }

    [Fact]
    public async Task PlanAsync_SkipsExcludedDevice_WhenStampingPlanDevice()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusions = new DeviceExclusionSet();
        exclusions.MarkMemoryExhausted(0);
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU-1", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.DiscreteGpu, 1, "dGPU-2", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 2, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: new FixedExclusionProvider(exclusions),
            deviceEnumerator: enumerator);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal(1, plan.DeviceIndex);
        Assert.Equal("dGPU-2", plan.DeviceAdapterDescription);
    }

    [Fact]
    public async Task PlanAsync_CpuPlan_StampsCpuDeviceEntry()
    {
        // Production order is GPUs-first with CPU last, and on Linux an AMD GPU
        // advertises Cpu at index 0 — a Cpu plan must still stamp the CPU entry.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusionProvider = new PipelineDeviceExclusionProvider();
        exclusionProvider.BeginRun();
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "AMD GPU", "AMD", 8192, 0, [ExecutionProviderKind.Migraphx, ExecutionProviderKind.Cpu]),
                new DeviceEntry(DeviceKind.Cpu, 1, "CPU", "Generic", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: exclusionProvider,
            deviceEnumerator: enumerator);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal(1, plan.DeviceIndex);
        Assert.Equal("CPU", plan.DeviceAdapterDescription);
    }

    [Fact]
    public async Task DeviceFallback_EndToEnd_OomOnPlannedDevice_MigratesAndReportsDegradation()
    {
        // Proves the C10 guards are reachable with production plans: the initial plan carries
        // the stamped device, the OOM engages the catch filter, and the replan migrates.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusionProvider = new PipelineDeviceExclusionProvider();
        exclusionProvider.BeginRun();
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 1, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: exclusionProvider,
            deviceEnumerator: enumerator);

        StageRuntimePlan initialPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));
        Assert.Equal(0, initialPlan.DeviceIndex);

        var oom = new Exception("[ErrorCode:RuntimeException] CUDA error: out of memory");
        int createCount = 0;
        Task<string> CreateSessionAsync(StageRuntimePlan plan, CancellationToken ct)
        {
            createCount++;
            if (plan.DeviceIndex == 0)
            {
                throw oom;
            }

            return Task.FromResult("session-on-fallback-device");
        }

        Task<StageRuntimePlan> ReplanAsync(CancellationToken ct) =>
            planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad), ct);

        DeviceFallbackSessionCreator.Result<string> result = await DeviceFallbackSessionCreator
            .CreateWithDeviceFallbackAsync(
                initialPlan,
                CreateSessionAsync,
                ReplanAsync,
                exclusionProvider,
                cancellationToken: CancellationToken.None);

        Assert.Equal("session-on-fallback-device", result.Lease);
        Assert.NotNull(result.Degradation);
        Assert.Equal(DeviceDegradationKind.MemoryExhausted, result.Degradation.Kind);
        Assert.Equal(0, result.Degradation.FailedDeviceIndex);
        Assert.Equal(1, result.Degradation.FallbackDeviceIndex);
        Assert.Equal(2, createCount);
        Assert.True(exclusionProvider.CurrentExclusions!.IsExcluded(0));
    }

    [Fact]
    public async Task DeviceFallback_EndToEnd_PersistentOom_IsBoundedByAttemptCap()
    {
        // Worst case through production replans: every device OOMs, the 4-attempt cap stops
        // the loop instead of cycling onto the terminal Blocked plan forever.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        var exclusionProvider = new PipelineDeviceExclusionProvider();
        exclusionProvider.BeginRun();
        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(DeviceKind.DiscreteGpu, 0, "dGPU-1", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.DiscreteGpu, 1, "dGPU-2", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.DiscreteGpu, 2, "dGPU-3", "NVIDIA", 4096, 0, [ExecutionProviderKind.DirectMl]),
                new DeviceEntry(DeviceKind.Cpu, 3, "CPU", "Microsoft", 0, 0, [ExecutionProviderKind.Cpu])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceExclusionProvider: exclusionProvider,
            deviceEnumerator: enumerator);

        StageRuntimePlan initialPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));
        Assert.Equal(0, initialPlan.DeviceIndex);

        var oom = new Exception("[ErrorCode:RuntimeException] CUDA error: out of memory");
        int createCount = 0;
        Task<string> CreateSessionAsync(StageRuntimePlan plan, CancellationToken ct)
        {
            createCount++;
            throw oom;
        }

        Task<StageRuntimePlan> ReplanAsync(CancellationToken ct) =>
            planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad), ct);

        Exception thrown = await Assert.ThrowsAsync<Exception>(() => DeviceFallbackSessionCreator
            .CreateWithDeviceFallbackAsync(
                initialPlan,
                CreateSessionAsync,
                ReplanAsync,
                exclusionProvider,
                cancellationToken: CancellationToken.None));

        Assert.Same(oom, thrown);
        Assert.Equal(4, createCount);
        Assert.True(exclusionProvider.CurrentExclusions!.IsExcluded(0));
        Assert.True(exclusionProvider.CurrentExclusions.IsExcluded(3));
    }

    [Fact]
    public void ApplyDeviceExclusions_suppresses_OpenVino_while_any_of_its_devices_is_excluded()
    {
        var exclusions = new DeviceExclusionSet();
        exclusions.MarkMemoryExhausted(2);
        DeviceEntry[] devices =
        [
            new(DeviceKind.Npu, 2, "NPU", "Intel", 0, 0, [ExecutionProviderKind.OpenVino]),
            new(DeviceKind.IntegratedGpu, 1, "iGPU", "Intel", 0, 0, [ExecutionProviderKind.OpenVino]),
            new(DeviceKind.Cpu, 0, "CPU", "Intel", 0, 0, [ExecutionProviderKind.Cpu]),
        ];

        IReadOnlyList<ExecutionProviderAvailability> filtered = RuntimePlanner.ApplyDeviceExclusions(
            [new(ExecutionProviderKind.OpenVino, true), new(ExecutionProviderKind.Cpu, true)],
            devices,
            exclusions);

        Assert.False(filtered.Single(a => a.Provider == ExecutionProviderKind.OpenVino).IsAvailable);
        Assert.True(filtered.Single(a => a.Provider == ExecutionProviderKind.Cpu).IsAvailable);
    }

    [Fact]
    public async Task PlanAsync_UsesDnnl_WhenCpuDeviceAdvertisesIt()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        var enumerator = new RecordingDeviceEnumerator(
            [
                new DeviceEntry(
                    DeviceKind.Cpu,
                    0,
                    "CPU",
                    "Generic",
                    0,
                    0,
                    [ExecutionProviderKind.Cpu, ExecutionProviderKind.Dnnl])
            ]);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.Cpu, true), new(ExecutionProviderKind.Dnnl, true)],
            _ => new ExecutionProviderSmokeTestResult(true),
            deviceEnumerator: enumerator);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Vad,
            PreferredExecutionProvider: ExecutionProviderKind.Dnnl,
            RequirePreferredExecutionProvider: true));

        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.Dnnl, plan.ExecutionProvider);
        Assert.Null(plan.Fallback);
    }

    [Fact]
    public async Task PlanAsync_RequiredProviderUnavailable_ReturnsBlockedInsteadOfCpuFallback()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Unavailable required providers should not smoke test."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Vad,
            PreferredExecutionProvider: ExecutionProviderKind.DirectMl,
            RequirePreferredExecutionProvider: true));

        Assert.Equal(StageRuntimePlanStatus.Blocked, plan.Status);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ProviderUnavailable, plan.Fallback!.Code);
        Assert.Contains("DirectML disabled", plan.Fallback.Detail, StringComparison.Ordinal);
        Assert.Null(plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_RequiredProvider_ReturnsReadyWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_int8.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(false, "DirectML smoke failed."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Vad,
            PreferredExecutionProvider: ExecutionProviderKind.DirectMl,
            RequirePreferredExecutionProvider: true));

        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_RequiredProviderNotAllowedForEngineFamily_PlansAllowedProviderWithSkipWarning()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.TensorRTRtx, true), new(ExecutionProviderKind.Cpu, true)],
            _ => throw new InvalidOperationException("Disallowed required providers should not smoke test."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: "kokoro-onnx",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal("kokoro", plan.EngineFamily);
        Assert.Contains(
            plan.Warnings,
            warning =>
                warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine
                && warning.Detail is not null
                && warning.Detail.Contains("TensorRTRtx", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PlanAsync_RequiredTensorRTRtxForVad_FallsBackToDmlSinceBlocked()
    {
        // VAD excludes TensorRT RTX (silero-vad fails its engine build, #329), so a required
        // TRT RTX pin is demoted to DirectML with a visible skip warning, never silently.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Vad,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Contains(
            plan.Warnings,
            warning => warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine);
    }

    [Fact]
    public async Task PlanAsync_RequiredTensorRTRtxNotAllowedForOpusMt_PlansDirectMlWithoutTrtSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/opus-mt-es-en");
        WriteBundledCacheFiles(workspace, registry, "opus-es-en", cacheRoot, "merged-decoder");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/opus-mt-es-en", cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                if (request.ExecutionProvider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt)
                {
                    throw new InvalidOperationException("TensorRT families must not be smoked for opus-mt.");
                }

                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            PreferredModelAlias: "opus-es-en",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true,
            SourceLanguage: "es",
            TargetLanguage: "en"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("opus-mt", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Contains(
            plan.Warnings,
            warning =>
                warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine
                && warning.Detail is not null
                && warning.Detail.Contains("TensorRTRtx", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PlanAsync_Madlad_PicksTensorRtRtxFirstWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladTranslationSpec());
        string cacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteCacheFile(cacheRoot, "encoder_model.onnx");
        var smoked = new List<ExecutionProviderKind>();

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("google/madlad400-3b-mt", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smoked.Add(request.ExecutionProvider);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            PreferredModelAlias: "madlad400-mt",
            RequirePreferredModelAlias: true,
            SourceLanguage: "es",
            TargetLanguage: "en"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("madlad", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Empty(smoked);
    }

    [Fact]
    public async Task PlanAsync_RequiredTensorRTRtxForWhisperOnnx_UsesTensorRTRtx()
    {
        // Whisper ONNX qualified on both TensorRT providers in the 2026-10-18 isolated
        // retest (Xenova whisper-medium passed both smokes), so a required TensorRT-RTX
        // preference is honored for whisper-onnx again.
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "onnx-community/whisper-tiny",
                Task: "asr",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "whisper-onnx",
                Aliases: ["whisper-tiny-onnx", "whisper-tiny"],
                RootFolder: "whisper-tiny-onnx",
                BenchmarkEntry: "onnx/encoder_model.onnx",
                Variants: [new ManifestVariantSpec("default", "onnx/encoder_model.onnx")],
                CapabilityTags: ["language-detection"],
                SourceLanguages: ["auto"]));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/whisper-tiny");
        workspace.WriteCacheFile(cacheRoot, "onnx/encoder_model.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/whisper-tiny", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "whisper-tiny-onnx",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("whisper-onnx", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_RequiredTensorRTRtxNotAllowedForLatentSync_PlansDirectMlWithoutTrtSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "ByteDance/LatentSync-1.6",
                Task: "lip-synthesis",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "latentsync-diffusion",
                Aliases: ["latentsync-1.6", "latentsync"],
                RootFolder: "latentsync",
                BenchmarkEntry: "unet.onnx",
                Variants: [new ManifestVariantSpec("default", "unet.onnx")]));

        string cacheRoot = workspace.CreateCacheRoot("ByteDance/LatentSync-1.6");
        workspace.WriteCacheFile(cacheRoot, "unet.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("ByteDance/LatentSync-1.6", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                if (request.ExecutionProvider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt)
                {
                    throw new InvalidOperationException("TensorRT families must not be smoked for latentsync-diffusion.");
                }

                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.LipSynthesis,
            PreferredModelAlias: "latentsync-1.6",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("latentsync-diffusion", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Contains(
            plan.Warnings,
            warning =>
                warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine
                && warning.Detail is not null
                && warning.Detail.Contains("TensorRTRtx", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PlanAsync_SkipProviderSmokeTest_ReturnsReadyForGpuWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smokeRequests.Add(request);
                throw new InvalidOperationException("Listing plans must not smoke GPU providers.");
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true,
            SkipProviderSmokeTest: true));

        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("qwen3-asr", plan.EngineFamily);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_SkipProviderSmokeTest_DoesNotPoisonPlanCache()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan listingPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true,
            SkipProviderSmokeTest: true));

        StageRuntimePlan runtimePlan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.Ready, listingPlan.Status);
        // No verdict was recorded, so the runtime plan is the same honest Ready — and
        // planning performed zero smoke creates in either call (C4).
        Assert.Equal(StageRuntimePlanStatus.Ready, runtimePlan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, runtimePlan.ExecutionProvider);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_WhenTensorRTRtxSmokePassesForQwenAsr_UsesTensorRTRtx()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Equal("qwen3-asr", plan.EngineFamily);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_QwenAsr_PicksTensorRtRtxWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(
                    request.ExecutionProvider is not ExecutionProviderKind.TensorRTRtx,
                    request.ExecutionProvider is ExecutionProviderKind.TensorRTRtx
                        ? "Qwen TRT RTX three-session smoke failed."
                        : null);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Equal("qwen3-asr", plan.EngineFamily);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_UsesOlderCompleteCacheRootWhenNewerRecordOnlyHasNonPreferredVariant()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "onnx-community/whisper-tiny",
                Task: "asr",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "whisper-onnx",
                Aliases: ["whisper-tiny-onnx"],
                RootFolder: "whisper-tiny-onnx",
                BenchmarkEntry: "onnx/encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("default", "onnx/encoder_model.onnx"),
                    new ManifestVariantSpec("fp16", "onnx/encoder_model_fp16.onnx")
                ],
                CapabilityTags: ["language-detection"],
                SourceLanguages: ["auto"]));

        string olderCompleteRoot = workspace.CreateCacheRoot("bundled-whisper");
        workspace.WriteCacheFile(olderCompleteRoot, "onnx/encoder_model.onnx");
        string newerPartialRoot = workspace.CreateCacheRoot("downloaded-whisper");
        workspace.WriteCacheFile(newerPartialRoot, "onnx/encoder_model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [
                CreateCacheRecord(registry, "onnx-community/whisper-tiny", newerPartialRoot, createdAtUtc: DateTimeOffset.UtcNow),
                CreateCacheRecord(registry, "onnx-community/whisper-tiny", olderCompleteRoot, createdAtUtc: DateTimeOffset.UtcNow.AddDays(-1))
            ],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Asr));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal("default", plan.Variant);
        Assert.Equal(Path.Join(olderCompleteRoot, "onnx", "encoder_model.onnx"), plan.ModelEntryPath);
    }

    [Fact]
    public async Task PlanAsync_WhenModelIsNotCached_ReturnsDownloadRequired()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal("int8", plan.Variant);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_WhenCachedModelHashDiffersFromManifest_DoesNotReturnReady()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_int8.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", "different-sha", DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelIntegrityMismatch, plan.Fallback!.Code);
        Assert.Contains("sha256", plan.Fallback.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task PlanAsync_CommercialSafeFilteringExcludesUnsafePreferredAlias()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "unsafe/vad",
                Task: "vad",
                License: "unknown",
                CommercialAllowed: false,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "silero-vad",
                Aliases: ["silero-vad"],
                RootFolder: "unsafe-vad",
                BenchmarkEntry: "onnx/model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("fp16", "onnx/model_fp16.onnx"),
                    new ManifestVariantSpec("int8", "onnx/model_int8.onnx")
                ]),
            new ManifestSpec(
                ModelId: "safe/vad",
                Task: "vad",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "silero-vad",
                Aliases: ["silero"],
                RootFolder: "safe-vad",
                BenchmarkEntry: "onnx/model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("fp16", "onnx/model_fp16.onnx"),
                    new ManifestVariantSpec("int8", "onnx/model_int8.onnx")
                ]));

        string safeCacheRoot = workspace.CreateCacheRoot("safe/vad");
        workspace.WriteCacheFile(safeCacheRoot, "onnx/model_int8.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("safe/vad", safeCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("safe/vad", plan.ModelId);
        Assert.Equal("silero", plan.ModelAlias);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_WhenOnlyCommercialUnsafeAsrCandidates_ReturnsDownloadRequired()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "unsafe/asr",
                Task: "asr",
                License: "unknown",
                CommercialAllowed: false,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "whisper-onnx",
                Aliases: ["whisper-tiny-onnx"],
                RootFolder: "unsafe-asr",
                BenchmarkEntry: "onnx/encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("fp16", "onnx/encoder_model_fp16.onnx"),
                    new ManifestVariantSpec("int8", "onnx/encoder_model_int8.onnx")
                ],
                CapabilityTags: ["language-detection"],
                SourceLanguages: ["auto"]));

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Asr));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("unsafe/asr", plan.ModelId);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_WhenStageHasNoManifestEntries_ReturnsMissingManifestEntry()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run when no model entries exist."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.Equal(StageRuntimePlanStatus.Blocked, plan.Status);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.NoCompatibleVariant, plan.Fallback!.Code);
        Assert.Contains("No separation model is registered", plan.Fallback.Detail, StringComparison.Ordinal);
        Assert.DoesNotContain("commercial-safe filtering", plan.Fallback.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledAsrCommercialSafeMode_ReturnsDownloadRequiredForParakeetTdt()
    {
        // The cloud POC uses English source audio; Parakeet TDT is the bundled ASR model verified on TRT-RTX.
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.TensorRTRtx, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            SourceLanguage: "en"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("tonythethompson/parakeet-tdt-0.6b-v3-onnx", plan.ModelId);
        Assert.Equal("parakeet-tdt-0.6b-v3", plan.ModelAlias);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledAsrPreferredNemotronAlias_SelectsNemotronWithoutChangingAutoRanking()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "nemotron-3.5-asr",
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx", plan.ModelId);
        Assert.Equal("nemotron-3.5-asr", plan.ModelAlias);
        Assert.Equal("nemotron-asr", plan.EngineFamily);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_AsrSourceLanguageFiltersManifestCoverage()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = CreateAsrSourceLanguageRegistry(workspace);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            SourceLanguage: "es"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("test/es-asr", plan.ModelId);
    }

    [Fact]
    public async Task PlanAsync_AsrSourceLanguageNormalizesBcp47RegionTags()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = CreateAsrSourceLanguageRegistry(workspace);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            SourceLanguage: "en-US"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("test/en-asr", plan.ModelId);
    }

    [Fact]
    public async Task PlanAsync_AsrAutoSourceRequiresLanguageDetectionManifest()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "test/non-detecting-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "qwen3-asr",
                Aliases: ["aaa-non-detecting-asr"],
                BenchmarkEntry: "model.onnx",
                RootFolder: "non-detecting-asr",
                Variants: [new ManifestVariantSpec("default", "model.onnx")]),
            new ManifestSpec(
                ModelId: "test/detecting-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "qwen3-asr",
                Aliases: ["zzz-detecting-asr"],
                RootFolder: "detecting-asr",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")],
                CapabilityTags: ["language-detection"]));

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            SourceLanguage: "auto"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("test/detecting-asr", plan.ModelId);
    }

    [Fact]
    public async Task PlanAsync_AsrBlankSourceRequiresLanguageDetectionManifest()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "test/non-detecting-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "qwen3-asr",
                Aliases: ["aaa-non-detecting-asr"],
                RootFolder: "non-detecting-asr",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")]),
            new ManifestSpec(
                ModelId: "test/detecting-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "qwen3-asr",
                Aliases: ["zzz-detecting-asr"],
                RootFolder: "detecting-asr",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")],
                SourceLanguages: ["auto"]));

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Asr));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("test/detecting-asr", plan.ModelId);
    }

    [Fact]
    public void CurrentBundledSeparationSpleeterManifest_PointsToCommercialSafeSpleeterBundle()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        Assert.True(registry.TryResolve("spleeter", out BundledModelManifestResolution? resolution));
        BundledModelManifestEntry entry = resolution!.Entry;

        Assert.Equal("csukuangfj/sherpa-onnx-spleeter-2stems", entry.ModelId);
        Assert.Equal("spleeter", entry.EngineFamily);
        Assert.Equal("MIT", entry.License);
        Assert.True(entry.CommercialAllowed);
        Assert.True(entry.CommercialSafeMode);
        Assert.Contains("speech-music-sfx-separation", entry.Capabilities);
        Assert.True(entry.RequiresAttribution);
        Assert.Equal("https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems", entry.SourceUrl);
        Assert.Equal("3e5a4dd68259675ab0ddc38269fdae1d0c41f20c", entry.Revision);
        Assert.Contains("spleeter", entry.Aliases);
        Assert.DoesNotContain("spleeter-2stems", entry.Aliases);
        Assert.Equal(
            "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/3e5a4dd68259675ab0ddc38269fdae1d0c41f20c/vocals.onnx",
            entry.DownloadFileSources["vocals.onnx"]);
        Assert.Equal(
            "vocals.onnx",
            Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath).Replace('\\', '/'));
        BundledModelManifestVariant variant = Assert.Single(entry.Variants);
        Assert.Equal("default", variant.Alias);
        Assert.Equal(
            "vocals.onnx",
            Path.GetRelativePath(entry.RootDirectory, variant.EntryPath).Replace('\\', '/'));
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationCommercialSafeOff_ReturnsDownloadRequiredForSpleeter()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("csukuangfj/sherpa-onnx-spleeter-2stems", plan.ModelId);
        Assert.Equal("spleeter", plan.ModelAlias);
        Assert.Equal("spleeter", plan.EngineFamily);
        Assert.Equal("fast", plan.ModelTier);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
        Assert.Contains(plan.Warnings, warning => warning.Code == RuntimePlanWarningCode.AttributionRequired);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationLegacyCacheStillRequiresSpleeter()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string legacyCacheRoot = workspace.CreateCacheRoot("legacy/separation-model");
        workspace.WriteCacheFile(legacyCacheRoot, "legacy-separation.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("legacy/separation-model", legacyCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("csukuangfj/sherpa-onnx-spleeter-2stems", plan.ModelId);
        Assert.Equal("spleeter", plan.ModelAlias);
        Assert.Equal("spleeter", plan.EngineFamily);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationCommercialSafeMode_AllowsCommercialSafeSpleeter()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for a download-required plan."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("csukuangfj/sherpa-onnx-spleeter-2stems", plan.ModelId);
        Assert.Equal("spleeter", plan.ModelAlias);
        Assert.Equal("spleeter", plan.EngineFamily);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationSpleeterCached_UsesCpuWithoutGpuSmokeTestAsync()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("csukuangfj/sherpa-onnx-spleeter-2stems");
        workspace.WriteCacheFile(cacheRoot, "vocals.onnx");
        workspace.WriteCacheFile(cacheRoot, "accompaniment.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "csukuangfj/sherpa-onnx-spleeter-2stems", cacheRoot)],
            [new(ExecutionProviderKind.TensorRTRtx, false, "Not supported"), new(ExecutionProviderKind.DirectMl, false, "Not supported")],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("csukuangfj/sherpa-onnx-spleeter-2stems", plan.ModelId);
        Assert.Equal("spleeter", plan.ModelAlias);
        Assert.Equal("spleeter", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal("default", plan.Variant);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationSpleeterCached_UsesTensorRtRtxWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("csukuangfj/sherpa-onnx-spleeter-2stems");
        workspace.WriteCacheFile(cacheRoot, "vocals.onnx");
        workspace.WriteCacheFile(cacheRoot, "accompaniment.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "csukuangfj/sherpa-onnx-spleeter-2stems", cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("csukuangfj/sherpa-onnx-spleeter-2stems", plan.ModelId);
        Assert.Equal("spleeter", plan.ModelAlias);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Equal("default", plan.Variant);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledOverlapRescueSepformerCached_UsesTensorRtRtxWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/sepformer-whamr16k-onnx");
        workspace.WriteCacheFile(cacheRoot, "sepformer.onnx");
        workspace.WriteCacheFile(cacheRoot, "osd.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "tonythethompson/sepformer-whamr16k-onnx", cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.OverlapRescue));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("tonythethompson/sepformer-whamr16k-onnx", plan.ModelId);
        Assert.Equal("sepformer", plan.ModelAlias);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Equal("default", plan.Variant);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationPreferredHush_IsNotEligibleForDubbingAmbiance()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("weya-ai/hush");
        workspace.WriteCacheFile(cacheRoot, "onnx/advanced_dfnet16k_model_best_onnx.tar.gz");
        workspace.WriteCacheFile(cacheRoot, "deployment/lib/weya_nc.dll");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("weya-ai/hush", cacheRoot, "a55d932cbf6344d284ac985f21e7f6e5bc4d38a5", "sha", DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Hush should not be smoke-tested for default dubbing ambiance."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation,
            PreferredModelAlias: "hush-dialogue",
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.Blocked, plan.Status);
        Assert.Null(plan.ModelId);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.NoCompatibleVariant, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledSeparationPreferredLegacySpleeterAlias_IsBlocked()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Blocked plans should not smoke-test providers."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation,
            PreferredModelAlias: "spleeter-non-commercial",
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.Blocked, plan.Status);
        Assert.Null(plan.ModelId);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.NoCompatibleVariant, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_CustomManifestRequiresDialogueAmbianceCapability()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "music/stems",
                Task: "separation",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "generic-separator",
                Aliases: ["generic-separator"],
                RootFolder: "generic-separator",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")],
                CapabilityTags: ["stem-separation"]),
            new ManifestSpec(
                ModelId: "dialogue/spleeter",
                Task: "separation",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "spleeter",
                Aliases: ["spleeter"],
                RootFolder: "spleeter",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")],
                CapabilityTags: ["cinematic-dialogue-background-separation", "speech-music-sfx-separation"]));

        string musicCacheRoot = workspace.CreateCacheRoot("music/stems");
        workspace.WriteCacheFile(musicCacheRoot, "model.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("music/stems", musicCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => throw new InvalidOperationException("Music stem separator should not be planned for dialogue ambiance."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Separation));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("dialogue/spleeter", plan.ModelId);
        Assert.Equal("spleeter", plan.ModelAlias);
        Assert.Equal("spleeter", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledTtsCommercialSafeMode_ReturnsDownloadRequiredForKokoro()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("onnx-community/Kokoro-82M-v1.0-ONNX", plan.ModelId);
        Assert.Equal("kokoro-onnx", plan.ModelAlias);
        Assert.Equal("kokoro", plan.EngineFamily);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledKokoroTts_DoesNotUseDirectMlOrTensorRTRtxWhenAvailable()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/Kokoro-82M-v1.0-ONNX");
        WriteBundledCacheFiles(workspace, registry, "kokoro-onnx", cacheRoot, "q4");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/Kokoro-82M-v1.0-ONNX", cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true)
            ],
            _ => throw new InvalidOperationException("Kokoro TTS must remain CPU-only until DirectML/TensorRT compatibility is verified."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("onnx-community/Kokoro-82M-v1.0-ONNX", plan.ModelId);
        Assert.Equal("kokoro-onnx", plan.ModelAlias);
        Assert.Equal("kokoro", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal("q4", plan.Variant);
    }

    [Fact]
    public async Task PlanAsync_RequiredGpuPin_OnKokoro_PlansCpuWithExactSkipReason()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/Kokoro-82M-v1.0-ONNX");
        WriteBundledCacheFiles(workspace, registry, "kokoro-onnx", cacheRoot, "q4");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/Kokoro-82M-v1.0-ONNX", cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true),
                new(ExecutionProviderKind.Qnn, true),
                new(ExecutionProviderKind.Cpu, true)
            ],
            _ => throw new InvalidOperationException("Kokoro TTS must remain CPU-only."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: "kokoro-onnx",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Equal("kokoro", plan.EngineFamily);
        Assert.Contains(
            plan.Warnings,
            warning =>
                warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine
                && warning.Detail is not null
                && warning.Detail.Contains("TensorRTRtx", StringComparison.Ordinal)
                && warning.Detail.Contains("kokoro", StringComparison.OrdinalIgnoreCase));
    }

    [Theory]
    [InlineData("ResembleAI/chatterbox-turbo-ONNX", VoiceCloningDefaults.ChatterboxPrimaryAlias, "chatterbox", "q4f16", true)]
    [InlineData("ResembleAI/chatterbox-turbo-ONNX", VoiceCloningDefaults.ChatterboxPrimaryAlias, "chatterbox", "q4f16", false)]
    [InlineData("tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX", "qwen3-tts", "qwen3-tts", "default", true)]
    public async Task PlanAsync_TensorRTRtxNotAllowedForTtsFamilies_PlansDirectMlWithoutTrtSmoke(
        string modelId,
        string modelAlias,
        string engineFamily,
        string variant,
        bool requirePreferredExecutionProvider)
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot(modelId);
        WriteBundledCacheFiles(workspace, registry, modelAlias, cacheRoot, variant);

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, modelId, cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true),
                new(ExecutionProviderKind.Cpu, true)
            ],
            request =>
            {
                if (request.ExecutionProvider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt)
                {
                    throw new InvalidOperationException($"TensorRT families must not be smoked for {engineFamily}.");
                }

                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: modelAlias,
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: requirePreferredExecutionProvider));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(engineFamily, plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal(requirePreferredExecutionProvider, plan.RequirePreferredExecutionProvider);
        Assert.Contains(
            plan.Warnings,
            warning =>
                warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine
                && warning.Detail is not null
                && warning.Detail.Contains("TensorRTRtx", StringComparison.Ordinal)
                && warning.Detail.Contains(engineFamily, StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task PlanAsync_CosyVoiceAllowedForTensorRtRtx_PlansTrtWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        const string modelId = "tonythethompson/CosyVoice-300M-ONNX";
        string cacheRoot = workspace.CreateCacheRoot(modelId);
        WriteBundledCacheFiles(workspace, registry, "cosyvoice", cacheRoot, "default");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, modelId, cacheRoot)],
            [
                new(ExecutionProviderKind.DirectMl, true),
                new(ExecutionProviderKind.TensorRTRtx, true),
                new(ExecutionProviderKind.Cpu, true)
            ],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: "cosyvoice",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("cosyvoice", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Empty(smokeRequests);
        Assert.DoesNotContain(
            plan.Warnings,
            warning => warning.Code == RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledVoiceCloningCommercialSafeMode_AllowsChatterboxWithConsentWarning()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: VoiceCloningDefaults.ChatterboxPrimaryAlias,
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("ResembleAI/chatterbox-turbo-ONNX", plan.ModelId);
        Assert.Equal("chatterbox-turbo-onnx", plan.ModelAlias);
        Assert.Equal("chatterbox", plan.EngineFamily);
        Assert.Contains(plan.Warnings, warning => warning.Code == RuntimePlanWarningCode.UserConsentRequired);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledVoiceCloningCached_UsesDirectMlWhenAvailable()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("ResembleAI/chatterbox-turbo-ONNX");
        WriteBundledCacheFiles(workspace, registry, VoiceCloningDefaults.ChatterboxPrimaryAlias, cacheRoot, "q4f16");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "ResembleAI/chatterbox-turbo-ONNX", cacheRoot)],
            [new(ExecutionProviderKind.DirectMl, true)],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(request.ExecutionProvider is ExecutionProviderKind.DirectMl);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: VoiceCloningDefaults.ChatterboxPrimaryAlias,
            RequirePreferredModelAlias: true));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("ResembleAI/chatterbox-turbo-ONNX", plan.ModelId);
        Assert.Equal("chatterbox-turbo-onnx", plan.ModelAlias);
        Assert.Equal("chatterbox", plan.EngineFamily);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
        Assert.Equal("q4f16", plan.Variant);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledVoiceCloningCommercialSafeModeOff_SelectsChatterboxWithConsentWarning()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistry();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: VoiceCloningDefaults.ChatterboxPrimaryAlias,
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("ResembleAI/chatterbox-turbo-ONNX", plan.ModelId);
        Assert.Equal("chatterbox-turbo-onnx", plan.ModelAlias);
        Assert.Equal("chatterbox", plan.EngineFamily);
        Assert.Contains(plan.Warnings, warning => warning.Code == RuntimePlanWarningCode.UserConsentRequired);
    }

    [Fact]
    public async Task PlanAsync_CurrentBundledVadCommercialSafeMode_RemainsRunnable()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_int8.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/silero-vad", cacheRoot)],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("onnx-community/silero-vad", plan.ModelId);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_TranslationEnToEs_ReturnsReadyBundledOpusPlan()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/opus-mt-en-es");
        WriteBundledCacheFiles(workspace, registry, "opus-en-es", cacheRoot, "merged-decoder");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/opus-mt-en-es", cacheRoot)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            SourceLanguage: "en",
            TargetLanguage: "es"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("onnx-community/opus-mt-en-es", plan.ModelId);
        Assert.Equal("opus-en-es", plan.ModelAlias);
        Assert.Equal("merged-decoder", plan.Variant);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.DoesNotContain(plan.Warnings, warning => warning.Code == RuntimePlanWarningCode.ModelIntegrityNotVerified);
    }

    [Fact]
    public async Task PlanAsync_TranslationEsToEn_ReturnsReadyBundledOpusPlan()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/opus-mt-es-en");
        WriteBundledCacheFiles(workspace, registry, "opus-es-en", cacheRoot, "merged-decoder");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/opus-mt-es-en", cacheRoot)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            SourceLanguage: "es",
            TargetLanguage: "en"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("onnx-community/opus-mt-es-en", plan.ModelId);
        Assert.Equal("opus-es-en", plan.ModelAlias);
        Assert.Equal("merged-decoder", plan.Variant);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.Contains(plan.Warnings, warning => warning.Code == RuntimePlanWarningCode.AttributionRequired);
    }

    [Fact]
    public async Task PlanAsync_TranslationEsToEn_DoesNotUseInstalledOppositeDirectionModel()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = LoadBundledRegistry();

        string oppositeDirectionCacheRoot = workspace.CreateCacheRoot("onnx-community/opus-mt-en-es");
        workspace.WriteCacheFile(oppositeDirectionCacheRoot, "onnx/encoder_model.onnx");
        workspace.WriteCacheFile(oppositeDirectionCacheRoot, "onnx/decoder_model_merged.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/opus-mt-en-es", oppositeDirectionCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            SourceLanguage: "es",
            TargetLanguage: "en"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("onnx-community/opus-mt-es-en", plan.ModelId);
        Assert.Equal("opus-es-en", plan.ModelAlias);
        Assert.Equal("merged-decoder", plan.Variant);
    }

    [Fact]
    public async Task PlanAsync_TranslationWithPreferredAlias_NoLongerBlocksBroaderPairs()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladTranslationSpec());
        string cacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteCacheFile(cacheRoot, "encoder_model.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("google/madlad400-3b-mt", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            PreferredModelAlias: "madlad400-mt",
            SourceLanguage: "en",
            TargetLanguage: "fr"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("google/madlad400-3b-mt", plan.ModelId);
        Assert.Equal("madlad400-mt", plan.ModelAlias);
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_MadladTrtFp16_IsOfferedToClassicTensorRtWhenRequested()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladTranslationSpec() with
        {
            BenchmarkEntry = "encoder_model_quantized.onnx",
            Variants =
            [
                new ManifestVariantSpec("quantized", "encoder_model_quantized.onnx", ["cpu", "dml", "cuda"]),
                new ManifestVariantSpec("trt-fp16", "trt_rtx_mixed_fp16_fp32/encoder_model.onnx", ["trt-rtx", "tensorrt"])
            ]
        });
        string cacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteCacheFile(cacheRoot, "encoder_model_quantized.onnx");
        workspace.WriteCacheFile(cacheRoot, "trt_rtx_mixed_fp16_fp32/encoder_model.onnx");
        RuntimePlanner planner = CreatePlanner(registry,
            [new("google/madlad400-3b-mt", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.TensorRt, true), new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Translation,
            PreferredModelAlias: "madlad400-mt", RequirePreferredModelAlias: true,
            SourceLanguage: "es", TargetLanguage: "en",
            PreferredExecutionProvider: ExecutionProviderKind.TensorRt, RequirePreferredExecutionProvider: true,
            PreferredModelVariantAlias: "trt-fp16"));

        Assert.True(plan.IsRunnable(), $"Expected a runnable plan, got {plan.Status}: {plan.Fallback?.Detail}");
        Assert.Equal(ExecutionProviderKind.TensorRt, plan.ExecutionProvider);
        Assert.Equal("trt-fp16", plan.Variant);
        Assert.Equal("madlad", plan.EngineFamily);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("default")]
    public async Task PlanAsync_OnlyCachedMadladQuantizedFiles_NeverSmokeTensorRtRtx(string? preferredVariant)
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladTranslationSpec() with
        {
            BenchmarkEntry = "encoder_model_quantized.onnx",
            Variants =
            [
                new ManifestVariantSpec("quantized", "encoder_model_quantized.onnx", ["cpu", "dml", "cuda"]),
                new ManifestVariantSpec("trt-fp16", "trt_rtx_mixed_fp16_fp32/encoder_model.onnx", ["trt-rtx"])
            ]
        });
        string cacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteCacheFile(cacheRoot, "encoder_model_quantized.onnx");
        var probes = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(registry,
            [new("google/madlad400-3b-mt", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.TensorRTRtx, true)],
            request => { probes.Add(request); return new ExecutionProviderSmokeTestResult(true); });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Translation,
            PreferredModelAlias: "madlad400-mt", SourceLanguage: "en", TargetLanguage: "fr",
            PreferredModelVariantAlias: preferredVariant));

        Assert.True(plan.IsRunnable(), $"Expected CPU plan, got {plan.Status}: {plan.Fallback?.Detail}");
        Assert.Equal(ExecutionProviderKind.Cpu, plan.ExecutionProvider);
        Assert.DoesNotContain(probes, probe => probe.ExecutionProvider == ExecutionProviderKind.TensorRTRtx);
    }

    [Fact]
    public async Task PlanAsync_WhenMadladQuantizedExportIsMissing_RequestsQuantizedVariantBeforeDefault()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "google/madlad400-3b-mt",
                Task: "translation",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "madlad",
                Aliases: ["madlad400-mt", "madlad400"],
                RootFolder: "madlad400",
                BenchmarkEntry: "encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("quantized", "encoder_model_quantized.onnx")
                ]));
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Translation,
            PreferredModelAlias: "madlad400-mt",
            SourceLanguage: "en",
            TargetLanguage: "fr"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("quantized", plan.Variant);
        Assert.Contains("encoder_model_quantized.onnx", plan.Fallback?.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PlanAsync_IgnoresRegisteredOptimizedVariantWithoutExplicitPreferenceAsync()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateKokoroSpec());
        string baseRoot = workspace.CreateCacheRoot("kokoro-base");
        string variantRoot = Path.Join(baseRoot, "optimized", "olive-cpu-fp32");
        workspace.WriteCacheFile(variantRoot, "model.onnx");
        var localVariant = new LocalModelVariantRecord(
            "olive-cpu-fp32",
            variantRoot,
            "model.onnx",
            ["model.onnx"],
            "olive",
            ExecutionProviderKind.Cpu,
            "fp32",
            DateTimeOffset.UtcNow,
            "main",
            "sha");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/Kokoro-82M-v1.0-ONNX", baseRoot) with { Variants = [localVariant] }],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: "kokoro",
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("default", plan.Variant);
        Assert.False(plan.IsLocalOptimizedVariant);
    }

    [Fact]
    public async Task PlanAsync_SelectsRegisteredOptimizedVariantWhenExplicitlyPreferredAsync()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateKokoroSpec());
        string baseRoot = workspace.CreateCacheRoot("kokoro-base");
        string variantRoot = Path.Join(baseRoot, "optimized", "olive-cpu-fp32");
        workspace.WriteCacheFile(variantRoot, "model.onnx");
        var localVariant = new LocalModelVariantRecord(
            "olive-cpu-fp32",
            variantRoot,
            "model.onnx",
            ["model.onnx"],
            "olive",
            ExecutionProviderKind.Cpu,
            "fp32",
            DateTimeOffset.UtcNow,
            "main",
            "sha");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/Kokoro-82M-v1.0-ONNX", baseRoot) with { Variants = [localVariant] }],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: "kokoro",
            RequirePreferredModelAlias: true,
            PreferredModelVariantAlias: "olive-cpu-fp32"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("olive-cpu-fp32", plan.Variant);
        Assert.True(plan.IsLocalOptimizedVariant);
        Assert.Equal("model.onnx", plan.ModelEntryRelativePath);
        Assert.Equal(Path.Join(variantRoot, "model.onnx"), plan.ModelEntryPath);
    }

    [Fact]
    public async Task PlanAsync_FallsBackToBaseModelWhenPreferredOptimizedVariantProviderDoesNotMatchPlanProviderAsync()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateKokoroSpec());
        string baseRoot = workspace.CreateCacheRoot("kokoro-base");
        workspace.WriteCacheFile(baseRoot, "model.onnx");
        string variantRoot = Path.Join(baseRoot, "optimized", "olive-dml-fp16");
        workspace.WriteCacheFile(variantRoot, "model.onnx");
        var localVariant = new LocalModelVariantRecord(
            "olive-dml-fp16",
            variantRoot,
            "model.onnx",
            ["model.onnx"],
            "olive",
            ExecutionProviderKind.DirectMl,
            "fp16",
            DateTimeOffset.UtcNow,
            "main",
            "sha");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [CreateCacheRecord(registry, "onnx-community/Kokoro-82M-v1.0-ONNX", baseRoot) with { Variants = [localVariant] }],
            [new(ExecutionProviderKind.DirectMl, false, "DirectML disabled for this test.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for unavailable providers."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Tts,
            PreferredModelAlias: "kokoro",
            RequirePreferredModelAlias: true,
            PreferredModelVariantAlias: "olive-dml-fp16"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("default", plan.Variant);
        Assert.False(plan.IsLocalOptimizedVariant);
        Assert.Contains(
            plan.Warnings,
            warning => warning.Code == RuntimePlanWarningCode.PreferredOptimizedVariantUnavailable &&
                       warning.Detail?.Contains("olive-dml-fp16", StringComparison.OrdinalIgnoreCase) == true);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_NonCpuProviderReturnsReadyNeverVerified()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.TensorRTRtx, true), new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(false, "Smoke-test failure is expected."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        // Ready is the codebase's file-only status for unproven plans (IsRunnable covers
        // Ready|Verified); it must never be promoted to Verified without proof.
        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.NotEqual(StageRuntimePlanStatus.Verified, plan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task StageRuntimePlan_RoundTripsWithoutLeakingMachinePaths()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string marker = $"machine-marker-{Guid.NewGuid():N}";
        string cacheRoot = workspace.CreateCacheRoot(Path.Join(marker, "onnx-community-silero-vad"));
        workspace.WriteCacheFile(cacheRoot, "onnx/model_int8.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, false, "No DirectML adapter found.")],
            _ => throw new InvalidOperationException("Smoke tests should not run for CPU-only plans."));

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));
        string json = JsonSerializer.Serialize(plan);
        StageRuntimePlan? roundTripped = JsonSerializer.Deserialize<StageRuntimePlan>(json);

        Assert.Equal(Path.Join(cacheRoot, "onnx", "model_int8.onnx"), plan.ModelEntryPath);
        Assert.NotNull(roundTripped);
        Assert.Equal(plan.Stage, roundTripped!.Stage);
        Assert.Equal(plan.Status, roundTripped.Status);
        Assert.Equal(plan.ModelId, roundTripped.ModelId);
        Assert.Equal(plan.ExecutionProvider, roundTripped.ExecutionProvider);
        Assert.Null(roundTripped.ModelEntryPath);
        Assert.Equal(plan.Warnings.Select(warning => warning.Code), roundTripped.Warnings.Select(warning => warning.Code));
        Assert.DoesNotContain(marker, json, StringComparison.Ordinal);
        Assert.DoesNotContain("EntryPath", json, StringComparison.Ordinal);
        Assert.DoesNotContain("ModelRootPath", json, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public async Task LocalModelCacheInventory_ReadsMachineLocalCacheIndex()
    {
        string rootPath = Path.Join(Path.GetTempPath(), $"trackdub-cache-{Guid.NewGuid():N}");
        var storagePaths = new TrackdubStoragePaths(rootPath);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var inventory = new LocalModelCacheInventory(store);
        LocalModelCacheRecord[] records =
        [
            new("example/model", Path.Join(rootPath, "machine-cache", "example"), "main", "abc123", DateTimeOffset.UtcNow)
        ];

        try
        {
            await store.SaveAsync(records);
            IReadOnlyList<LocalModelCacheRecord> loaded = await inventory.LoadAsync();

            LocalModelCacheRecord record = Assert.Single(loaded);
            Assert.Equal(records[0].ModelId, record.ModelId);
            Assert.Equal(records[0].RootPath, record.RootPath);
        }
        finally
        {
            if (Directory.Exists(rootPath))
            {
                Directory.Delete(rootPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task PlanAsync_CarriesResolvedModelRevisionHash_IntoPlanWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ValidSha256, plan.ModelRevisionHash);
        Assert.Empty(smokeRequests);
    }

    private static RuntimePlanner CreatePlanner(
        BundledModelManifestRegistry registry,
        IReadOnlyList<LocalModelCacheRecord> cacheRecords,
        IReadOnlyList<ExecutionProviderAvailability> availabilities,
        Func<ExecutionProviderSmokeTestRequest, ExecutionProviderSmokeTestResult> smokeHandler,
        IPipelineDeviceExclusionProvider? deviceExclusionProvider = null,
        IDeviceEnumerator? deviceEnumerator = null,
        ISmokeVerdictStore? smokeVerdictStore = null,
        IReadOnlyDictionary<RuntimeStage, StageRuntimeRequirements>? stageRequirements = null)
    {
        return new RuntimePlanner(
            registry,
            new FakeHardwareProfileProvider(),
            new FakeExecutionProviderDiscovery(availabilities),
            new FakeExecutionProviderSmokeTester(smokeHandler),
            new InMemoryModelCacheInventory(cacheRecords),
            stageRequirements: stageRequirements,
            deviceExclusionProvider,
            deviceEnumerator,
            smokeVerdictStore);
    }

    private static LocalModelCacheRecord CreateCacheRecord(
        BundledModelManifestRegistry registry,
        string modelId,
        string cacheRoot,
        string? revision = null,
        DateTimeOffset? createdAtUtc = null)
    {
        BundledModelManifestEntry entry = registry.Entries.Single(candidate =>
            candidate.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));
        return new LocalModelCacheRecord(
            entry.ModelId,
            cacheRoot,
            revision ?? entry.Revision,
            entry.Sha256,
            createdAtUtc ?? DateTimeOffset.UtcNow);
    }

    private static void WriteBundledCacheFiles(
        RuntimePlannerTestWorkspace workspace,
        BundledModelManifestRegistry registry,
        string alias,
        string cacheRoot,
        string variantAlias)
    {
        Assert.True(registry.TryResolve(alias, out BundledModelManifestResolution? resolution));
        BundledModelManifestEntry entry = resolution!.Entry;

        foreach (string relativePath in entry.DownloadFiles)
        {
            workspace.WriteCacheFile(cacheRoot, relativePath);
        }

        BundledModelManifestVariant? variant = entry.Variants
            .FirstOrDefault(candidate => candidate.Alias.Equals(variantAlias, StringComparison.OrdinalIgnoreCase));
        string entryPath = variant?.EntryPath ?? entry.DefaultBenchmarkEntryPath;
        workspace.WriteCacheFile(cacheRoot, Path.GetRelativePath(entry.RootDirectory, entryPath));

        if (variant is not null)
        {
            foreach (string relativePath in variant.DownloadFiles)
            {
                workspace.WriteCacheFile(cacheRoot, relativePath);
            }
        }
    }

    private static BundledModelManifestRegistry LoadBundledRegistry()
    {
        string? configuredPath = Environment.GetEnvironmentVariable("BUNDLED_MANIFEST_PATH");
        if (!string.IsNullOrWhiteSpace(configuredPath))
        {
            return BundledModelManifestRegistry.Load(configuredPath);
        }

        try
        {
            string repoRoot = FindRepoRoot();
            string manifestPath = Path.Join(
                repoRoot,
                "src",
                "Trackdub.Inference",
                "Runtime",
                "ModelManifest",
                "bundled-models.manifest.json");

            if (File.Exists(manifestPath))
            {
                return BundledModelManifestRegistry.Load(manifestPath);
            }
        }
        catch (DirectoryNotFoundException)
        {
            // Fall through to try assembly-relative path resolution.
        }

        string assemblyRelativePath = Path.GetFullPath(Path.Join(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "..",
            "..",
            "src",
            "Trackdub.Inference",
            "Runtime",
            "ModelManifest",
            "bundled-models.manifest.json"));

        if (File.Exists(assemblyRelativePath))
        {
            return BundledModelManifestRegistry.Load(assemblyRelativePath);
        }

        throw new FileNotFoundException("Could not locate bundled-models.manifest.json for runtime planner tests.");
    }

    private sealed class RecordingDeviceEnumerator(IReadOnlyList<DeviceEntry> devices) : IDeviceEnumerator
    {
        public int GetDevicesCallCount { get; private set; }

        public Task<IReadOnlyList<DeviceEntry>> GetDevicesAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            GetDevicesCallCount++;
            return Task.FromResult(devices);
        }

        public Task<IReadOnlyList<DeviceEntry>> ReEnumerateAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(devices);
        }
    }

    private sealed class FixedExclusionProvider(DeviceExclusionSet exclusions) : IPipelineDeviceExclusionProvider
    {
        public DeviceExclusionSet? CurrentExclusions => exclusions;

        public DeviceExclusionSet BeginRun() => exclusions;

        public void EndRun()
        {
        }
    }

    private static string FindRepoRoot()
    {
        DirectoryInfo? current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Join(current.FullName, "Trackdub.slnx")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the repository root.");
    }

    private static ManifestSpec CreateKokoroSpec() =>
        new(
            ModelId: "onnx-community/Kokoro-82M-v1.0-ONNX",
            Task: "tts",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            VoiceCloning: false,
            EngineFamily: "kokoro",
            Aliases: ["kokoro-onnx", "kokoro"],
            RootFolder: "kokoro",
            BenchmarkEntry: "model.onnx",
            Variants: [new ManifestVariantSpec("default", "model.onnx")]);

    private static ManifestSpec CreateQwenAsrSpec() =>
        new(
            ModelId: "tonythethompson/qwen3-asr-0.6b-onnx",
            Task: "asr",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            VoiceCloning: false,
            EngineFamily: "qwen3-asr",
            Aliases: ["qwen3-asr-0.6b"],
            RootFolder: "qwen3-asr",
            BenchmarkEntry: "encoder.onnx",
            Variants: [new ManifestVariantSpec("default", "encoder.onnx")],
            CapabilityTags: ["language-detection"],
            SourceLanguages: ["auto"]);

    private static ManifestSpec CreateVadSpec(
        string primaryAlias,
        bool commercialAllowed,
        string license,
        string? modelId = null,
        string tier = "balanced") =>
        new(
            ModelId: modelId ?? "onnx-community/silero-vad",
            Task: "vad",
            License: license,
            CommercialAllowed: commercialAllowed,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            VoiceCloning: false,
            EngineFamily: "silero-vad",
            Tier: tier,
            Aliases: primaryAlias.Equals("silero", StringComparison.OrdinalIgnoreCase)
                ? ["silero"]
                : primaryAlias.StartsWith("silero-vad-", StringComparison.OrdinalIgnoreCase)
                    ? [primaryAlias]
                    : ["silero-vad", "silero"],
            RootFolder: primaryAlias.Replace('/', '-'),
            BenchmarkEntry: "onnx/model.onnx",
            Variants:
            [
                new ManifestVariantSpec("fp16", "onnx/model_fp16.onnx"),
                new ManifestVariantSpec("int8", "onnx/model_int8.onnx"),
                new ManifestVariantSpec("q4f16", "onnx/model_q4f16.onnx"),
                new ManifestVariantSpec("quantized", "onnx/model_quantized.onnx"),
                new ManifestVariantSpec("uint8", "onnx/model_uint8.onnx"),
                new ManifestVariantSpec("q4", "onnx/model_q4.onnx")
            ]);

    [Fact]
    public async Task PlanAsync_WhenPreferredModelTierSpecified_SelectsMatchingTier()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad-balanced", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-balanced", tier: "balanced"),
            CreateVadSpec("silero-vad-turbo", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-turbo", tier: "turbo"));

        string balancedRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-balanced");
        workspace.WriteCacheFile(balancedRoot, "onnx/model_fp16.onnx");
        string turboRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-turbo");
        workspace.WriteCacheFile(turboRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [
                new("onnx-community/silero-vad-balanced", balancedRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new("onnx-community/silero-vad-turbo", turboRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredModelTier: "turbo"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("turbo", plan.ModelTier);
        Assert.Equal("onnx-community/silero-vad-turbo", plan.ModelId);
    }

    [Fact]
    public async Task PlanAsync_WhenPreferredModelTierNotInManifest_FallsBackToAvailableTier()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad-balanced", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-balanced", tier: "balanced"));

        string balancedRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-balanced");
        workspace.WriteCacheFile(balancedRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad-balanced", balancedRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredModelTier: "turbo"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("balanced", plan.ModelTier);
        Assert.Equal("onnx-community/silero-vad-balanced", plan.ModelId);
    }

    [Fact]
    public async Task PlanAsync_WhenPreferredModelTierNotCached_FallsBackToCachedTier()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad-balanced", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-balanced", tier: "balanced"),
            CreateVadSpec("silero-vad-turbo", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-turbo", tier: "turbo"));

        string balancedRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-balanced");
        workspace.WriteCacheFile(balancedRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad-balanced", balancedRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredModelTier: "turbo"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("balanced", plan.ModelTier);
        Assert.Equal("onnx-community/silero-vad-balanced", plan.ModelId);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_PreferredTurboRequiredGpu_ReturnsReadyWithoutSmoke()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad-turbo", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-turbo", tier: "turbo"));

        string turboRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-turbo");
        workspace.WriteCacheFile(turboRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad-turbo", turboRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => new ExecutionProviderSmokeTestResult(false, "DirectML smoke failed."));

        StageRuntimePlan plan = await planner.PlanAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.Vad,
                PreferredModelTier: "turbo",
                PreferredExecutionProvider: ExecutionProviderKind.DirectMl,
                RequirePreferredExecutionProvider: true));

        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_WhenPreferredModelTierTurbo_NotCached_ReturnsDownloadRequiredForPreferredTier()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad-turbo", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-turbo", tier: "turbo"));

        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => throw new InvalidOperationException("Smoke tests should not run for download-required plans."));

        StageRuntimePlan plan = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredModelTier: "turbo"));

        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal("onnx-community/silero-vad-turbo", plan.ModelId);
        Assert.Equal("turbo", plan.ModelTier);
        Assert.NotNull(plan.Fallback);
        Assert.Equal(RuntimePlanFallbackCode.ModelNotCached, plan.Fallback!.Code);
    }

    [Fact]
    public async Task PlanAsync_WhenPreferredModelTierIsNonCommercial_RemainsOnRequestedTierAsync()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad-balanced", commercialAllowed: true, license: "MIT", modelId: "onnx-community/silero-vad-balanced", tier: "balanced"),
            CreateVadSpec("silero-vad-turbo", commercialAllowed: false, license: "CC-BY-NC-4.0", modelId: "onnx-community/silero-vad-turbo", tier: "turbo"));

        string balancedRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-balanced");
        workspace.WriteCacheFile(balancedRoot, "onnx/model_fp16.onnx");
        string turboRoot = workspace.CreateCacheRoot("onnx-community/silero-vad-turbo");
        workspace.WriteCacheFile(turboRoot, "onnx/model_fp16.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [
                new("onnx-community/silero-vad-balanced", balancedRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new("onnx-community/silero-vad-turbo", turboRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan plan = await planner.PlanAsync(
            new StageRuntimePlanningRequest(RuntimeStage.Vad, PreferredModelTier: "turbo"));

        Assert.True(plan.IsRunnable(), $"Expected runnable plan but got {plan.Status}");
        Assert.Equal("turbo", plan.ModelTier);
        Assert.Equal("onnx-community/silero-vad-turbo", plan.ModelId);
    }

    // ── Hardware profile caching ──────────────────────────────────────────────

    [Fact]
    public async Task PlanAsync_CalledTwice_LoadsHardwareProfileOnlyOnce()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));
        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        var hardwareProvider = new CountingHardwareProfileProvider();

        var planner = new RuntimePlanner(
            registry,
            hardwareProvider,
            new FakeExecutionProviderDiscovery([new(ExecutionProviderKind.Cpu, true)]),
            new FakeExecutionProviderSmokeTester(_ => new ExecutionProviderSmokeTestResult(true)),
            new InMemoryModelCacheInventory([new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)]));

        await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));
        await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        // Hardware profile is cached; provider should only be queried once regardless of call count
        Assert.Equal(1, hardwareProvider.CallCount);
    }

    // ── Plan result caching ───────────────────────────────────────────────────

    [Fact]
    public async Task PlanAsync_SameRequestTwice_ReturnsCachedPlan()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));
        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        // Track smoke calls to prove planning creates none (C4).
        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        var planningRequest = new StageRuntimePlanningRequest(RuntimeStage.Vad);
        StageRuntimePlan plan1 = await planner.PlanAsync(planningRequest);
        StageRuntimePlan plan2 = await planner.PlanAsync(planningRequest);

        // Both plans are runnable
        Assert.True(plan1.IsRunnable(), $"plan1: {plan1.Status}");
        Assert.True(plan2.IsRunnable(), $"plan2: {plan2.Status}");
        // Planning never smokes (C4); the second call is served from the plan cache.
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_DifferentStages_NotCrossContaminated()
    {
        using var workspace = new RuntimePlannerTestWorkspace();

        var asrSpec = new ManifestSpec(
            ModelId: "onnx-community/whisper-tiny",
            Task: "asr",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            VoiceCloning: false,
            EngineFamily: "whisper-onnx",
            Aliases: ["whisper-tiny-onnx"],
            RootFolder: "whisper-tiny-onnx",
            BenchmarkEntry: "onnx/encoder_model.onnx",
            Variants:
            [
                new ManifestVariantSpec("default", "onnx/encoder_model.onnx"),
            ],
            CapabilityTags: ["language-detection"],
            SourceLanguages: ["auto"]);

        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"),
            asrSpec);

        string vadRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(vadRoot, "onnx/model_fp16.onnx");
        string asrRoot = workspace.CreateCacheRoot("onnx-community/whisper-tiny");
        workspace.WriteCacheFile(asrRoot, "onnx/encoder_model.onnx");

        RuntimePlanner planner = CreatePlanner(
            registry,
            [
                new("onnx-community/silero-vad", vadRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new("onnx-community/whisper-tiny", asrRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ],
            [new(ExecutionProviderKind.Cpu, true)],
            _ => new ExecutionProviderSmokeTestResult(true));

        StageRuntimePlan vadPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));
        StageRuntimePlan asrPlan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Asr));

        Assert.True(vadPlan.IsRunnable(), $"VAD plan not runnable: {vadPlan.Status}");
        Assert.True(asrPlan.IsRunnable(), $"ASR plan not runnable: {asrPlan.Status}");
        Assert.Equal(RuntimeStage.Vad, vadPlan.Stage);
        Assert.Equal(RuntimeStage.Asr, asrPlan.Stage);
    }

    [Fact]
    public async Task InvalidatePlanCache_ClearsCache_ForcesReplanning()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));
        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        // Track smoke calls to prove planning creates none even after cache invalidation (C4).
        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            });

        var request = new StageRuntimePlanningRequest(RuntimeStage.Vad);

        await planner.PlanAsync(request);  // populates cache
        planner.InvalidatePlanCache();     // wipes cache
        await planner.PlanAsync(request);  // must re-plan, not serve stale cache

        StageRuntimePlan replanned = await planner.PlanAsync(request);
        Assert.True(replanned.IsRunnable());
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_DownloadRequiredPlan_NotCached()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        // No cache records → plan will be DownloadRequired (non-runnable), not Ready.
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));

        int smokeCallCount = 0;
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],  // empty cache → no model files present
            [new(ExecutionProviderKind.Cpu, true)],
            _ => { smokeCallCount++; return new ExecutionProviderSmokeTestResult(true); });

        var request = new StageRuntimePlanningRequest(RuntimeStage.Vad);

        StageRuntimePlan plan1 = await planner.PlanAsync(request);
        StageRuntimePlan plan2 = await planner.PlanAsync(request);

        // Both plans should be non-runnable (DownloadRequired).
        Assert.False(plan1.IsRunnable());
        Assert.False(plan2.IsRunnable());
        // Smoke was never called (no model to smoke-test) -- also verifies DownloadRequired plans
        // are re-evaluated each time rather than served from cache.
        Assert.Equal(0, smokeCallCount);
    }

    [Fact]
    public async Task PlanAsync_VerifiedVerdictInStore_SkipsSmokeOnNextLaunch()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var cacheRecords = new[]
        {
            new LocalModelCacheRecord("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
        };
        var availabilities = new[]
        {
            new ExecutionProviderAvailability(ExecutionProviderKind.DirectMl, true),
            new ExecutionProviderAvailability(ExecutionProviderKind.TensorRTRtx, true),
        };
        var verdictStore = new InMemorySmokeVerdictStore();
        var smokeRequests = new List<ExecutionProviderSmokeTestRequest>();

        // Launch 1: no verdict yet → honest Ready, zero smoke, nothing recorded (C4:
        // verdicts are recorded by the warmup coordinator's proving smoke, not planning).
        RuntimePlanner firstLaunch = CreatePlanner(
            registry, cacheRecords, availabilities,
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            },
            smokeVerdictStore: verdictStore);
        StageRuntimePlan firstPlan = await firstLaunch.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true));
        int firstLaunchSmokeCount = smokeRequests.Count;
        Assert.Equal(0, firstLaunchSmokeCount);
        // Simulate the warmup coordinator proving the plan and recording its verdict.
        verdictStore.RecordVerified(new SmokeVerdictKey(
            ValidSha256, ExecutionProviderKind.DirectMl, NvidiaGpuArchitectureBucket.Unknown,
            DriverVersion: null,
            TrtRtxEpVersion: Trackdub.Inference.Runtime.TensorRtRtx.TensorRtRtxProviderConstants.BundledFingerprintVersion,
            OrtRuntimeVersion: null));

        // Launch 2: fresh planner (in-memory plan cache empty, as in a new process).
        RuntimePlanner secondLaunch = CreatePlanner(
            registry, cacheRecords, availabilities,
            request =>
            {
                smokeRequests.Add(request);
                return new ExecutionProviderSmokeTestResult(true);
            },
            smokeVerdictStore: verdictStore);
        StageRuntimePlan secondPlan = await secondLaunch.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true));

        Assert.Equal(StageRuntimePlanStatus.Ready, firstPlan.Status);
        Assert.Equal(StageRuntimePlanStatus.Verified, secondPlan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, firstPlan.ExecutionProvider);
        Assert.Equal(ExecutionProviderKind.DirectMl, secondPlan.ExecutionProvider);
        Assert.Equal(firstLaunchSmokeCount, smokeRequests.Count);
        Assert.Empty(smokeRequests);
        Assert.Single(verdictStore.RecordedKeys);
    }

    [Fact]
    public async Task PlanAsync_VerdictForDifferentOrtVersion_IsNotServedAsVerified()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateVadSpec("silero-vad", commercialAllowed: true, license: "MIT"));
        string cacheRoot = workspace.CreateCacheRoot("onnx-community/silero-vad");
        workspace.WriteCacheFile(cacheRoot, "onnx/model_fp16.onnx");

        var verdictStore = new InMemorySmokeVerdictStore();
        // Proven under a different ORT runtime than this host reports (FakeHardwareProfileProvider
        // reports no ORT version): the verdict must invalidate, not serve as Verified.
        verdictStore.RecordVerified(new SmokeVerdictKey(
            ValidSha256,
            ExecutionProviderKind.DirectMl,
            NvidiaGpuArchitectureBucket.Unknown,
            DriverVersion: null,
            TrtRtxEpVersion: Trackdub.Inference.Runtime.TensorRtRtx.TensorRtRtxProviderConstants.BundledFingerprintVersion,
            OrtRuntimeVersion: "0.0.0-not-this-host"));

        int smokeCallCount = 0;
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("onnx-community/silero-vad", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true)],
            _ => { smokeCallCount++; return new ExecutionProviderSmokeTestResult(true); },
            smokeVerdictStore: verdictStore);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(RuntimeStage.Vad));

        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(0, smokeCallCount);
    }

    [Fact]
    public async Task PlanAsync_NoVerdict_PerformsZeroSmokeCreatesAcrossProviders()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());
        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var startedProviders = new System.Collections.Concurrent.ConcurrentBag<ExecutionProviderKind>();
        var planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.TensorRTRtx, true)],
            request =>
            {
                startedProviders.Add(request.ExecutionProvider);
                return new ExecutionProviderSmokeTestResult(true);
            },
            stageRequirements: new Dictionary<RuntimeStage, StageRuntimeRequirements>
            {
                [RuntimeStage.Asr] = new(
                    RuntimeStage.Asr,
                    ModelTask.Asr,
                    ["qwen3-asr-0.6b"],
                    [ExecutionProviderKind.DirectMl, ExecutionProviderKind.TensorRTRtx],
                    ["default"],
                    ["default"]),
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true));

        // Regression guard for 0628bc84: planning must not smoke candidates at all.
        Assert.Empty(startedProviders);
        Assert.Equal(StageRuntimePlanStatus.Ready, plan.Status);
        Assert.Equal(ExecutionProviderKind.DirectMl, plan.ExecutionProvider);
    }

    [Fact]
    public async Task PlanAsync_ExplicitTensorRtRtxForQwenAsr_StillUsesRequestedProvider()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());
        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var smokeRequests = new List<ExecutionProviderKind>();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.DirectMl, true), new(ExecutionProviderKind.TensorRTRtx, true)],
            request =>
            {
                smokeRequests.Add(request.ExecutionProvider);
                return new ExecutionProviderSmokeTestResult(true);
            });

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true,
            PreferredExecutionProvider: ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider: true));

        Assert.Equal(ExecutionProviderKind.TensorRTRtx, plan.ExecutionProvider);
        Assert.Empty(smokeRequests);
    }

    [Fact]
    public async Task PlanAsync_VerdictInStore_StillRequiresCachedFiles()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        var verdictStore = new InMemorySmokeVerdictStore();
        verdictStore.RecordVerified(new SmokeVerdictKey(
            ValidSha256,
            ExecutionProviderKind.TensorRTRtx,
            NvidiaGpuArchitectureBucket.Unknown,
            DriverVersion: null,
            TrtRtxEpVersion: Trackdub.Inference.Runtime.TensorRtRtx.TensorRtRtxProviderConstants.BundledFingerprintVersion));

        int smokeCallCount = 0;
        RuntimePlanner planner = CreatePlanner(
            registry,
            [],
            [new(ExecutionProviderKind.TensorRTRtx, true)],
            _ =>
            {
                smokeCallCount++;
                return new ExecutionProviderSmokeTestResult(true);
            },
            smokeVerdictStore: verdictStore);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true));

        // Cheap file checks run before any verdict is consulted: no model → DownloadRequired.
        Assert.Equal(StageRuntimePlanStatus.DownloadRequired, plan.Status);
        Assert.Equal(0, smokeCallCount);
    }

    [Fact]
    public async Task PlanAsync_PlanningNeverRecordsVerdict()
    {
        using var workspace = new RuntimePlannerTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateQwenAsrSpec());

        string cacheRoot = workspace.CreateCacheRoot("tonythethompson/qwen3-asr-0.6b-onnx");
        workspace.WriteCacheFile(cacheRoot, "encoder.onnx");

        var verdictStore = new InMemorySmokeVerdictStore();
        RuntimePlanner planner = CreatePlanner(
            registry,
            [new("tonythethompson/qwen3-asr-0.6b-onnx", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)],
            [new(ExecutionProviderKind.TensorRTRtx, true)],
            _ => new ExecutionProviderSmokeTestResult(false, "compile failed"),
            smokeVerdictStore: verdictStore);

        StageRuntimePlan plan = await planner.PlanAsync(new StageRuntimePlanningRequest(
            RuntimeStage.Asr,
            PreferredModelAlias: "qwen3-asr-0.6b",
            RequirePreferredModelAlias: true));

        // A failed smoke may fall back to CPU (Ready) or block; it must never become Verified
        // or leave a reusable verdict behind.
        Assert.NotEqual(StageRuntimePlanStatus.Verified, plan.Status);
        Assert.Empty(verdictStore.RecordedKeys);
    }

    private sealed class CountingHardwareProfileProvider : IHardwareProfileProvider
    {
        public int CallCount { get; private set; }

        public Task<HardwareProfile> GetCurrentAsync(CancellationToken cancellationToken = default)
        {
            CallCount++;
            return Task.FromResult(new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "Test GPU"));
        }
    }

    private sealed class RuntimePlannerTestWorkspace : IDisposable
    {
        private static readonly JsonSerializerOptions IndentedJsonOptions = new()
        {
            WriteIndented = true
        };

        public RuntimePlannerTestWorkspace()
        {
            RootPath = Path.Join(Path.GetTempPath(), $"trackdub-runtime-planner-{Guid.NewGuid():N}");
            Directory.CreateDirectory(RootPath);
        }

        public string RootPath { get; }

        public BundledModelManifestRegistry WriteManifest(params ManifestSpec[] models)
        {
            string manifestPath = Path.Join(RootPath, "bundled-models.manifest.json");
            string json = JsonSerializer.Serialize(
                new
                {
                    models = models.Select(model => new
                    {
                        model_id = model.ModelId,
                        task = model.Task,
                        engine_family = model.EngineFamily,
                        capabilities = model.Capabilities,
                        language_coverage = new
                        {
                            source_languages = model.SourceLanguages ?? [],
                            target_languages = model.TargetLanguages ?? [],
                            language_pairs = Array.Empty<object>()
                        },
                        tier = model.Tier,
                        license = model.License,
                        commercial_allowed = model.CommercialAllowed,
                        redistribution_allowed = true,
                        requires_attribution = model.RequiresAttribution,
                        requires_user_consent = model.RequiresUserConsent,
                        voice_cloning = model.VoiceCloning,
                        commercial_use_verified = model.CommercialUseVerified ??
                                                  (model.CommercialAllowed &&
                                                   IsValidSha256(model.Sha256) &&
                                                   !model.License.Equals("unknown", StringComparison.OrdinalIgnoreCase) &&
                                                   !model.License.Equals("non-commercial", StringComparison.OrdinalIgnoreCase) &&
                                                   !model.License.Equals("CC-BY-NC-4.0", StringComparison.OrdinalIgnoreCase)),
                        source_url = $"https://example.invalid/{model.ModelId.Replace('/', '-')}",
                        revision = "main",
                        sha256 = model.Sha256,
                        aliases = model.Aliases,
                        root_path = $"./manifest-models/{model.RootFolder}",
                        expected_runtime = model.ExpectedRuntime,
                        benchmark_entry = model.BenchmarkEntry,
                        variants = model.Variants.Select(variant => new
                        {
                            alias = variant.Alias,
                            entry_path = variant.EntryPath,
                            supported_providers = variant.SupportedProviders ?? []
                        })
                    })
                },
                IndentedJsonOptions);

            File.WriteAllText(manifestPath, json);
            return BundledModelManifestRegistry.Load(manifestPath);
        }

        public string CreateCacheRoot(string name)
        {
            string cacheRoot = Path.Join(RootPath, "machine-cache", name);
            Directory.CreateDirectory(cacheRoot);
            return cacheRoot;
        }

        public void WriteCacheFile(string cacheRoot, string relativePath)
        {
            string filePath = Path.Join(cacheRoot, relativePath);
            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(filePath, "placeholder");
        }

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(RootPath))
                {
                    Directory.Delete(RootPath, recursive: true);
                }
            }
            catch (IOException)
            {
                // Best-effort cleanup for temp directories created by planner tests.
            }
            catch (UnauthorizedAccessException)
            {
                // Best-effort cleanup for temp directories created by planner tests.
            }
        }
    }

    private sealed class FakeHardwareProfileProvider : IHardwareProfileProvider
    {
        public Task<HardwareProfile> GetCurrentAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "Test GPU"));
    }

    private sealed class FakeExecutionProviderDiscovery(IReadOnlyList<ExecutionProviderAvailability> availabilities)
        : IExecutionProviderDiscovery
    {
        public Task<IReadOnlyList<ExecutionProviderAvailability>> DiscoverAsync(
            HardwareProfile hardwareProfile,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(availabilities);
    }

    private sealed class FakeExecutionProviderSmokeTester(
        Func<ExecutionProviderSmokeTestRequest, ExecutionProviderSmokeTestResult> handler)
        : IExecutionProviderSmokeTester
    {
        public Task<ExecutionProviderSmokeTestResult> SmokeTestAsync(
            ExecutionProviderSmokeTestRequest request,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(handler(request));
    }

    private sealed class InMemorySmokeVerdictStore : ISmokeVerdictStore
    {
        private readonly HashSet<string> verified = new(StringComparer.Ordinal);

        public List<SmokeVerdictKey> RecordedKeys { get; } = [];

        public bool IsVerified(SmokeVerdictKey key) => verified.Contains(key.ToStableString());

        public void RecordVerified(SmokeVerdictKey key)
        {
            verified.Add(key.ToStableString());
            RecordedKeys.Add(key);
        }

        public void Clear()
        {
            verified.Clear();
            RecordedKeys.Clear();
        }
    }

    private sealed class InMemoryModelCacheInventory(IReadOnlyList<LocalModelCacheRecord> records)
        : IModelCacheInventory
    {
        public Task<IReadOnlyList<LocalModelCacheRecord>> LoadAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(records);
    }

    private static BundledModelManifestRegistry CreateAsrSourceLanguageRegistry(RuntimePlannerTestWorkspace workspace) =>
        workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "test/en-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "whisper-onnx",
                Aliases: ["aaa-en-asr"],
                RootFolder: "en-asr",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")],
                SourceLanguages: ["en"]),
            new ManifestSpec(
                ModelId: "test/es-asr",
                Task: "asr",
                License: "MIT",
                CommercialAllowed: true,
                RequiresAttribution: false,
                RequiresUserConsent: false,
                VoiceCloning: false,
                EngineFamily: "qwen3-asr",
                Aliases: ["zzz-es-asr"],
                RootFolder: "es-asr",
                BenchmarkEntry: "model.onnx",
                Variants: [new ManifestVariantSpec("default", "model.onnx")],
                SourceLanguages: ["es"]));

    private static ManifestSpec CreateMadladTranslationSpec() =>
        new(
            ModelId: "google/madlad400-3b-mt",
            Task: "translation",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: true,
            RequiresUserConsent: false,
            VoiceCloning: false,
            EngineFamily: "madlad",
            Aliases: ["madlad400-mt", "madlad400"],
            RootFolder: "madlad400",
            BenchmarkEntry: "encoder_model.onnx",
            Variants:
            [
                new ManifestVariantSpec("int8", "encoder_model_int8.onnx"),
                new ManifestVariantSpec("fp16", "encoder_model_fp16.onnx")
            ]);

    private sealed record ManifestSpec(
        string ModelId,
        string Task,
        string License,
        bool CommercialAllowed,
        bool RequiresAttribution,
        bool RequiresUserConsent,
        bool VoiceCloning,
        IReadOnlyList<string> Aliases,
        string RootFolder,
        string BenchmarkEntry,
        IReadOnlyList<ManifestVariantSpec> Variants,
        string EngineFamily = "",
        IReadOnlyList<string>? CapabilityTags = null,
        IReadOnlyList<string>? SourceLanguages = null,
        IReadOnlyList<string>? TargetLanguages = null,
        string Tier = "balanced",
        string Sha256 = ValidSha256,
        string? ExpectedRuntime = null,
        bool? CommercialUseVerified = null)
    {
        public IReadOnlyList<string> Capabilities { get; } = CapabilityTags ?? [];
    }

    private static bool IsValidSha256(string hash) =>
        hash.Length == 64 && hash.All(static c =>
            c is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');

    private sealed record ManifestVariantSpec(
        string Alias,
        string EntryPath,
        IReadOnlyList<string>? SupportedProviders = null);
}
