using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Qwen3Tts;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.TestDoubles;

namespace Trackdub.Inference.Tests;

public sealed class Qwen3TtsEngineTests
{
    private const string ReferenceTranscript = "Hello from Qwen3.";
    private const string CloneTargetText = "This is a cloned voice line.";

    [Fact]
    public void PooledProviderRequest_ForPlan_PreservesRequiredProviderConstraint()
    {
        StageRuntimePlan requiredProviderPlan = new()
        {
            ExecutionProvider = ExecutionProviderKind.TensorRTRtx,
            RequirePreferredExecutionProvider = true,
            HasExplicitProviderPreference = true
        };
        StageRuntimePlan fallbackEnabledPlan = requiredProviderPlan with
        {
            RequirePreferredExecutionProvider = false
        };

        PooledProviderRequest requiredRequest = PooledProviderRequest.ForPlan(requiredProviderPlan);
        PooledProviderRequest fallbackRequest = PooledProviderRequest.ForPlan(fallbackEnabledPlan);

        Assert.False(requiredRequest.AllowTrtInitFallback);
        Assert.True(fallbackRequest.AllowTrtInitFallback);
        Assert.NotEqual(requiredRequest, fallbackRequest);
    }

    private static VoiceCatalogEntry TestVoice => new("qwen3:ryan", "mul", "unknown", "Ryan");

    [RequiresDirectMlBundledModelFact(
        "qwen3-tts-0.6b-customvoice/talker_prefill.onnx",
        "qwen3-tts-0.6b-customvoice/tokenizer/vocab.json",
        "qwen3-tts-0.6b-customvoice/embeddings/speaker_ids.json")]
    public async Task Qwen3TtsEngine_SynthesizesCustomVoice06WithDirectMl()
    {
        var engine = CreateEngine();
        var plan = CustomVoicePlan(
            "tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX",
            Qwen3TtsDefaults.CustomVoice06Alias,
            ExecutionProviderKind.DirectMl);

        TtsSynthesisResult result = await engine.SynthesizeAsync(
            new TtsSynthesisRequest("Hello from Qwen3 on DirectML.", "en", TestVoice),
            plan,
            CancellationToken.None);

        AssertCustomVoiceResult(result);
        Assert.Equal("dml", result.Provider);
        Assert.NotNull(engine.LastExecutionSummary);
        Assert.Equal("dml", engine.LastExecutionSummary!.SelectedProvider);
    }

    [RequiresBundledModelFact(
        "qwen3-tts-0.6b-customvoice/talker_prefill.onnx",
        "qwen3-tts-0.6b-customvoice/tokenizer/vocab.json",
        "qwen3-tts-0.6b-customvoice/embeddings/speaker_ids.json")]
    public async Task Qwen3TtsEngine_SynthesizesCustomVoice06WithBundledModel()
    {
        var engine = CreateEngine();
        var plan = CustomVoicePlan(
            "tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX",
            Qwen3TtsDefaults.CustomVoice06Alias);

        TtsSynthesisResult result = await engine.SynthesizeAsync(
            new TtsSynthesisRequest("Hello from Qwen3.", "en", TestVoice),
            plan,
            CancellationToken.None);

        AssertCustomVoiceResult(result);
    }

    [RequiresBundledModelFact(
        "qwen3-tts-1.7b-customvoice/talker_prefill.onnx",
        "qwen3-tts-1.7b-customvoice/tokenizer/vocab.json",
        "qwen3-tts-1.7b-customvoice/embeddings/speaker_ids.json")]
    public async Task Qwen3TtsEngine_SynthesizesCustomVoice17WithBundledModel()
    {
        var engine = CreateEngine();
        var plan = CustomVoicePlan(
            "tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX",
            Qwen3TtsDefaults.CustomVoice17Alias);

        TtsSynthesisResult result = await engine.SynthesizeAsync(
            new TtsSynthesisRequest("Hello from the large CustomVoice bundle.", "en", TestVoice),
            plan,
            CancellationToken.None);

        AssertCustomVoiceResult(result);
    }

    [RequiresBundledModelFact(
        "qwen3-tts-0.6b-customvoice/talker_prefill.onnx",
        "qwen3-tts-0.6b-base/talker_prefill.onnx",
        "qwen3-tts-0.6b-base/speaker_encoder.onnx",
        "qwen3-tts-0.6b-base/tokenizer/vocab.json")]
    public async Task Qwen3TtsEngine_SynthesizesBase06CloneWithBundledModel()
    {
        string referenceClipPath = await CreateReferenceClipAsync(
            Qwen3TtsDefaults.CustomVoice06Alias,
            "tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX");

        try
        {
            var engine = CreateEngine();
            var plan = BasePlan(
                "tonythethompson/Qwen3-TTS-12Hz-0.6B-Base-ONNX",
                Qwen3TtsDefaults.Base06Alias);

            var request = new TtsSynthesisRequest(
                CloneTargetText,
                "en",
                new VoiceCatalogEntry("voice-clone:ref", "en", "synthetic", "Reference"),
                VoiceCloneReference: new VoiceCloneReference(
                    Guid.NewGuid(),
                    Guid.NewGuid(),
                    referenceClipPath,
                    ReferenceTranscript.Length,
                    ReferenceTranscript.Length,
                    ReferenceTranscript: ReferenceTranscript));

            TtsSynthesisResult result = await engine.SynthesizeAsync(request, plan, CancellationToken.None);

            AssertCloneResult(result);
            Assert.Equal("voice-clone:ref", result.VoiceId);
        }
        finally
        {
            TryDelete(referenceClipPath);
        }
    }

    [RequiresBundledModelFact(
        "qwen3-tts-1.7b-customvoice/talker_prefill.onnx",
        "qwen3-tts-1.7b-base/talker_prefill.onnx",
        "qwen3-tts-1.7b-base/speaker_encoder.onnx",
        "qwen3-tts-1.7b-base/tokenizer/vocab.json")]
    public async Task Qwen3TtsEngine_SynthesizesBase17CloneWithBundledModel()
    {
        string referenceClipPath = await CreateReferenceClipAsync(
            Qwen3TtsDefaults.CustomVoice17Alias,
            "tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX");

        try
        {
            var engine = CreateEngine();
            var plan = BasePlan(
                "tonythethompson/Qwen3-TTS-12Hz-1.7B-Base-ONNX",
                Qwen3TtsDefaults.Base17Alias);

            var request = new TtsSynthesisRequest(
                CloneTargetText,
                "en",
                new VoiceCatalogEntry("voice-clone:ref", "en", "synthetic", "Reference"),
                VoiceCloneReference: new VoiceCloneReference(
                    Guid.NewGuid(),
                    Guid.NewGuid(),
                    referenceClipPath,
                    ReferenceTranscript.Length,
                    ReferenceTranscript.Length,
                    ReferenceTranscript: ReferenceTranscript));

            TtsSynthesisResult result = await engine.SynthesizeAsync(request, plan, CancellationToken.None);

            AssertCloneResult(result);
        }
        finally
        {
            TryDelete(referenceClipPath);
        }
    }

    [Fact]
    public async Task Qwen3TtsEngine_BaseClone_RequiresReferenceTranscript()
    {
        var engine = CreateEngine();
        var plan = BasePlan(
            "tonythethompson/Qwen3-TTS-12Hz-0.6B-Base-ONNX",
            Qwen3TtsDefaults.Base06Alias);

        var request = new TtsSynthesisRequest(
            "Clone this.",
            "en",
            new VoiceCatalogEntry("voice-clone:test", "en", "synthetic", "Clone"),
            VoiceCloneReference: new VoiceCloneReference(
                Guid.NewGuid(),
                Guid.NewGuid(),
                "missing.wav",
                5,
                4));

        await Assert.ThrowsAsync<TtsReferenceTextRequiredException>(() =>
            engine.SynthesizeAsync(request, plan, CancellationToken.None));
    }

    private static Qwen3TtsEngine CreateEngine() =>
        new(new StubConsentService(granted: true), BenchmarkModelPathResolver.CreateDefault());

    private static StageRuntimePlan CustomVoicePlan(
        string modelId,
        string alias,
        ExecutionProviderKind executionProvider = ExecutionProviderKind.Cpu) => new()
        {
            Stage = RuntimeStage.Tts,
            Status = StageRuntimePlanStatus.Ready,
            ModelId = modelId,
            ModelAlias = alias,
            EngineFamily = Qwen3TtsEngine.EngineFamilyName,
            ExecutionProvider = executionProvider
        };

    private static StageRuntimePlan BasePlan(
        string modelId,
        string alias,
        ExecutionProviderKind executionProvider = ExecutionProviderKind.Cpu) => new()
        {
            Stage = RuntimeStage.Tts,
            Status = StageRuntimePlanStatus.Ready,
            ModelId = modelId,
            ModelAlias = alias,
            EngineFamily = Qwen3TtsEngine.EngineFamilyName,
            ExecutionProvider = executionProvider
        };

    private static async Task<string> CreateReferenceClipAsync(string customVoiceAlias, string customVoiceModelId)
    {
        var engine = CreateEngine();
        var plan = CustomVoicePlan(customVoiceModelId, customVoiceAlias);
        TtsSynthesisResult synthesized = await engine.SynthesizeAsync(
            new TtsSynthesisRequest(ReferenceTranscript, "en", TestVoice),
            plan,
            CancellationToken.None);

        string tempPath = Path.Join(Path.GetTempPath(), $"qwen3tts_ref_{Guid.NewGuid():N}.wav");
        await File.WriteAllBytesAsync(tempPath, synthesized.WavBytes);
        return tempPath;
    }

    private static void AssertCustomVoiceResult(TtsSynthesisResult result)
    {
        Assert.True(result.WavBytes.Length > 44);
        Assert.Equal(24_000, result.SampleRate);
        Assert.True(result.DurationSamples > 0);
        Assert.Equal("qwen3:ryan", result.VoiceId);
    }

    private static void AssertCloneResult(TtsSynthesisResult result)
    {
        Assert.True(result.WavBytes.Length > 44);
        Assert.Equal(24_000, result.SampleRate);
        Assert.True(result.DurationSamples > 0);
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
        }
    }

    private sealed class StubConsentService(bool granted) : IConsentService
    {
        public Guid SessionId { get; } = Guid.NewGuid();

        public bool IsVoiceCloningConsentGranted => granted;

        public event EventHandler? VoiceCloningConsentChanged;

        public void GrantVoiceCloningConsent() => VoiceCloningConsentChanged?.Invoke(this, EventArgs.Empty);

        public void ClearVoiceCloningConsent() => VoiceCloningConsentChanged?.Invoke(this, EventArgs.Empty);
    }
}
