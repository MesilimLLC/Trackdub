using Trackdub.Composition.StarterPacks;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

/// <summary>
/// ORT GenAI's NvTensorRtRtx device terminates the host process (native stack overflow)
/// on bundled GenAI models such as qwen-instruct (Qwen2.5-1.5B). A catchable smoke
/// failure cannot gate a fatal crash, so GenAI-loaded engine families must never be
/// offered TensorRT providers, and the smoke tester must refuse before touching
/// native code.
/// </summary>
public sealed class GenAiTensorRtExclusionTests
{
    public static IEnumerable<object[]> GenAiFamilyStages
    {
        get
        {
            yield return [RuntimeStage.Asr, "whisper-genai"];
            yield return [RuntimeStage.Translation, "phi-genai"];
            yield return [RuntimeStage.TextRefinement, "qwen-instruct"];
            yield return [RuntimeStage.TextRefinement, "phi-genai"];
        }
    }

    [Theory]
    [MemberData(nameof(GenAiFamilyStages))]
    public void GenAiEngineFamily_StageOverride_ExcludesTensorRtProviders(
        RuntimeStage stage,
        string engineFamily)
    {
        StageRuntimeRequirements requirements = StageRuntimeRequirementsCatalog.All[stage];

        Assert.NotNull(requirements.AllowedProvidersByEngineFamily);
        Assert.True(
            requirements.AllowedProvidersByEngineFamily.TryGetValue(
                engineFamily, out IReadOnlyList<ExecutionProviderKind>? allowed),
            $"Stage {stage} has no provider override for engine family '{engineFamily}'.");
        Assert.NotNull(allowed);
        Assert.DoesNotContain(ExecutionProviderKind.TensorRTRtx, allowed);
        Assert.DoesNotContain(ExecutionProviderKind.TensorRt, allowed);
        // The bundled GenAI exports are not DirectML exports (Whisper crashes the host on DirectML).
        Assert.DoesNotContain(ExecutionProviderKind.DirectMl, allowed);
        Assert.Contains(ExecutionProviderKind.Cpu, allowed);
    }

    [Theory]
    [InlineData(RuntimeStage.Asr, "whisper-genai")]
    [InlineData(RuntimeStage.Translation, "phi-genai")]
    [InlineData(RuntimeStage.TextRefinement, "qwen-instruct")]
    public async Task SmokeTest_GenAiFamilyOnTensorRtRtx_FailsWithoutTouchingNativeCode(
        RuntimeStage stage,
        string engineFamily)
    {
        var request = new ExecutionProviderSmokeTestRequest(
            stage,
            "model-id",
            "alias",
            engineFamily,
            "default",
            ExecutionProviderKind.TensorRTRtx,
            ModelRootPath: "does-not-exist",
            EntryPath: "does-not-exist/model.onnx");

        ExecutionProviderSmokeTestResult result = await new OnnxExecutionProviderSmokeTester()
            .SmokeTestAsync(request, CancellationToken.None);

        Assert.False(result.Passed);
        Assert.Contains("NvTensorRtRtx", result.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("opus-mt")]
    public async Task SmokeTest_FatalEncoderDecoderFamilyOnTensorRtRtx_FailsWithoutTouchingNativeCode(
        string engineFamily)
    {
        // The smoke sweep bypasses stage allow-lists, so the guard at the translation
        // smoke entry is what prevents the documented InferenceSession ctor stack
        // overflow from killing the host process.
        var request = new ExecutionProviderSmokeTestRequest(
            RuntimeStage.Translation,
            "model-id",
            "alias",
            engineFamily,
            "merged-decoder",
            ExecutionProviderKind.TensorRTRtx,
            ModelRootPath: "does-not-exist",
            EntryPath: "does-not-exist/model.onnx");

        ExecutionProviderSmokeTestResult result = await new OnnxExecutionProviderSmokeTester()
            .SmokeTestAsync(request, CancellationToken.None);

        Assert.False(result.Passed);
        Assert.Contains("terminates the host process", result.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OpusMtStageOverride_ExcludesTensorRtRtxButKeepsClassicTensorRt()
    {
        // Isolated retest 2026-10-18 (ORT 1.31, TensorRT 10.13, RTX 5070): the merged-decoder
        // OPUS-MT export passed classic TensorRT smoke, but session construction on
        // TensorRT-RTX aborted the host process (exit 134). Classic stays, RTX stays out.
        StageRuntimeRequirements requirements = StageRuntimeRequirementsCatalog.All[RuntimeStage.Translation];

        Assert.True(
            requirements.AllowedProvidersByEngineFamily!.TryGetValue(
                "opus-mt", out IReadOnlyList<ExecutionProviderKind>? allowed));
        Assert.DoesNotContain(ExecutionProviderKind.TensorRTRtx, allowed);
        Assert.Contains(ExecutionProviderKind.TensorRt, allowed);
    }

    [Theory]
    [InlineData(ExecutionProviderKind.TensorRt, false)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, true)]
    [InlineData(ExecutionProviderKind.Cuda, false)]
    public void OpusMt_smoke_guard_refuses_only_tensorrt_rtx(ExecutionProviderKind provider, bool rejected)
    {
        var request = new ExecutionProviderSmokeTestRequest(
            RuntimeStage.Translation,
            "model-id",
            "alias",
            "opus-mt",
            "merged-decoder",
            provider,
            ModelRootPath: "root",
            EntryPath: "onnx/model.onnx");

        AssertSmokeGuard(request, rejected);
    }

    [Fact]
    public void Qwen3AsrStageOverride_ExcludesClassicTensorRtButKeepsTensorRtRtx()
    {
        // Qwen3-ASR's bundled export is not qualified on classic TensorRT: the 2026-10-18
        // isolated retest terminated the host process during import (plugin ops the
        // importer cannot resolve), and the planner smokes in-process, so classic TensorRT
        // must not be attempted until a qualified export exists. TensorRT-RTX stays
        // reachable: its Qwen smoke historically fell back catchably after a long compile,
        // and explicit provider pins rely on that path. DirectML leads, CUDA follows.
        StageRuntimeRequirements requirements = StageRuntimeRequirementsCatalog.All[RuntimeStage.Asr];

        Assert.True(
            requirements.AllowedProvidersByEngineFamily!.TryGetValue(
                "qwen3-asr", out IReadOnlyList<ExecutionProviderKind>? allowed));
        Assert.Equal(ExecutionProviderKind.DirectMl, allowed![0]);
        Assert.DoesNotContain(ExecutionProviderKind.TensorRt, allowed);
        Assert.Contains(ExecutionProviderKind.TensorRTRtx, allowed);
    }

    [Fact]
    public void WhisperOnnxStageOverride_AllowsBothTensorRtProviders()
    {
        // Isolated retest 2026-10-18: Xenova/whisper-medium passed smoke on both classic
        // TensorRT and TensorRT-RTX, disproving the old fused-contrib-op exclusion. The
        // planner smoke-gates each variant, so unqualified Whisper exports still fall back.
        StageRuntimeRequirements requirements = StageRuntimeRequirementsCatalog.All[RuntimeStage.Asr];

        Assert.True(
            requirements.AllowedProvidersByEngineFamily!.TryGetValue(
                "whisper-onnx", out IReadOnlyList<ExecutionProviderKind>? allowed));
        Assert.Contains(ExecutionProviderKind.TensorRt, allowed);
        Assert.Contains(ExecutionProviderKind.TensorRTRtx, allowed);
    }

    [Fact]
    public void MadladStageOverride_TriesTensorRtRtxThenClassicTensorRtThenCpu()
    {
        // MADLAD's trt-fp16 export runs under both TensorRT EPs: TRT-RTX first (EP-context
        // warm path, lower VRAM admission estimate), then classic TensorRT (isolated-process
        // retest passed 2026-10-18, ORT 1.31 + TensorRT 10.13). Without a TensorRT provider,
        // CPU (int4 KV-cache) beats DirectML, which pages on a 12 GB card. The smoke guard
        // still refuses non-trt-fp16 MADLAD pairings on either TensorRT provider.
        Assert.True(
            StageRuntimeRequirementsCatalog.All[RuntimeStage.Translation].AllowedProvidersByEngineFamily!
                .TryGetValue("madlad", out IReadOnlyList<ExecutionProviderKind>? allowed));
        Assert.Equal(
            [ExecutionProviderKind.TensorRTRtx, ExecutionProviderKind.TensorRt, ExecutionProviderKind.Cpu],
            allowed!.Take(3));
    }

    [Theory]
    [InlineData(ExecutionProviderKind.TensorRt, "trt-fp16", "trt_rtx_mixed_fp16_fp32/encoder_model.onnx", false)]
    [InlineData(ExecutionProviderKind.TensorRt, "quantized", "encoder_model_quantized.onnx", true)]
    [InlineData(ExecutionProviderKind.TensorRt, "trt-fp16", "encoder_model_quantized.onnx", true)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, "quantized", "encoder_model_quantized.onnx", true)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, "trt-fp16", "encoder_model_quantized.onnx", true)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, "int4-kv", "mixed_fp16_int4/encoder_model.onnx", true)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, "trt-fp16", "trt_rtx_mixed_fp16_fp32/encoder_model.onnx", false)]
    [InlineData(ExecutionProviderKind.Cpu, "quantized", "encoder_model_quantized.onnx", false)]
    public void Madlad_smoke_guard_only_permits_validated_export_on_tensorrt_providers(
        ExecutionProviderKind provider, string variant, string entryPath, bool rejected)
    {
        AssertSmokeGuard(
            new ExecutionProviderSmokeTestRequest(
                RuntimeStage.Translation,
                "model-id",
                "alias",
                "madlad",
                variant,
                provider,
                ModelRootPath: "root",
                EntryPath: entryPath),
            rejected);
    }

    [Fact]
    public void VadStage_ExcludesBothTensorRtProviders()
    {
        // #329: silero-vad fails the TensorRT RTX build (squeeze shape error in its If/else-branch
        // subgraph). The 2026-10-18 isolated retest showed classic TensorRT fails the same
        // export too, with a catchable shape-inference rejection of the If subgraph ("TensorRT
        // input ... has no shape specified"), so both TensorRT providers stay excluded until a
        // shape-inferred export ships. The model is tiny; CPU/DirectML cost nothing meaningful.
        IReadOnlyList<ExecutionProviderKind> allowed =
            StageRuntimeRequirementsCatalog.All[RuntimeStage.Vad].AllowedProvidersThisMilestone;
        Assert.DoesNotContain(ExecutionProviderKind.TensorRTRtx, allowed);
        Assert.DoesNotContain(ExecutionProviderKind.TensorRt, allowed);
    }

    [Fact]
    public async Task SmokeRunner_ReportsPerTargetProgress_ForSkippedTargets()
    {
        // A missing model must produce an incremental Skipped result even though the
        // aggregate report is only returned at the end.
        var targets = new[]
        {
            new TrtRtxSmokeCatalog.Target("not-a-real/model-ref", null, "missing-model"),
        };
        var reported = new List<TrtRtxStarterPackSmokeTargetResult>();

        TrtRtxStarterPackSmokeReport report = await TrtRtxStarterPackSmokeRunner.RunAsync(
            modelCacheDirectory: null,
            targets,
            new Progress<TrtRtxStarterPackSmokeTargetResult>(reported.Add),
            CancellationToken.None);

        TrtRtxStarterPackSmokeTargetResult result = Assert.Single(report.Targets);
        Assert.Equal(TrtRtxStarterPackSmokeTargetStatus.Skipped, result.Status);
        // Progress<T> posts callbacks asynchronously (no SynchronizationContext under
        // xUnit), so the delivery can lose a race with this assert on loaded machines.
        // Wait boundedly for the single expected delivery instead of assuming it landed.
        for (int i = 0; i < 200 && reported.Count == 0; i++)
        {
            await Task.Delay(25);
        }

        Assert.Single(reported);
        Assert.Equal("missing-model", reported[0].Label);
    }

    private static void AssertSmokeGuard(ExecutionProviderSmokeTestRequest request, bool rejected)
    {
        if (rejected)
        {
            Assert.Throws<NotSupportedException>(
                () => OnnxExecutionProviderSmokeTester.ThrowIfFatalTensorRtFamily(request));
            return;
        }

        OnnxExecutionProviderSmokeTester.ThrowIfFatalTensorRtFamily(request);
    }
}
