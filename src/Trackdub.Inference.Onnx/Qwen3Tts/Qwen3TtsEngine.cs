using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Qwen3Tts.Pipeline;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Qwen3Tts;

/// <summary>
/// Trackdub <see cref="Contracts.Pipeline.ITtsEngineAdapter"/> for bundled Qwen3-TTS ONNX models.
/// </summary>
/// <remarks>
/// <para>Routes manifest aliases to either <see cref="Pipeline.TtsPipeline"/> (CustomVoice presets)
/// or <see cref="Pipeline.VoiceClonePipeline"/> (Base + consent-gated cloning). Execution provider
/// selection comes from <see cref="Runtime.Planning.StageRuntimePlan"/>; model roots from the
/// bundled manifest via <see cref="Qwen3TtsModelFiles"/>.</para>
/// <para>Reports the ONNX EP used through <see cref="IStageRuntimeExecutionReporter"/> for pipeline provenance.</para>
/// </remarks>
public sealed class Qwen3TtsEngine(
    IConsentService consentService,
    BenchmarkModelPathResolver modelPathResolver)
    : ITtsEngineAdapter, IStageRuntimeExecutionReporter, IDisposable
{
    public const string EngineFamilyName = "qwen3-tts";

    private const int SampleRate = 24_000;

    private readonly IConsentService consentService = consentService ?? throw new ArgumentNullException(nameof(consentService));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
    private readonly SemaphoreSlim sessionGate = new(1, 1);
    private PinnedPipeline? pinnedPipeline;
    private int disposeSignaled;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public Task<TtsSynthesisResult> SynthesizeAsync(
        TtsSynthesisRequest request,
        CancellationToken cancellationToken) =>
        throw new InvalidOperationException("Qwen3-TTS must be invoked through the runtime-planned overload.");

    public async Task<TtsSynthesisResult> SynthesizeAsync(
        TtsSynthesisRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        EnsurePlanReady(plan);

        bool isCloneRequest = request.VoiceCloneReference is not null;
        if (isCloneRequest)
        {
            if (!consentService.IsVoiceCloningConsentGranted)
            {
                throw new ConsentRequiredException();
            }

            if (string.IsNullOrWhiteSpace(request.VoiceCloneReference!.ReferenceTranscript))
            {
                throw new TtsReferenceTextRequiredException();
            }
        }

        Qwen3TtsModelFiles modelFiles = Qwen3TtsModelFiles.Resolve(
            PlannedRuntimeModelResolver.ResolveCandidate(plan, modelPathResolver),
            plan);

        if (isCloneRequest != modelFiles.IsBaseModel)
        {
            throw new InvalidOperationException(
                isCloneRequest
                    ? "Voice cloning requires a Qwen3-TTS Base manifest alias."
                    : "Preset voices require a Qwen3-TTS CustomVoice manifest alias.");
        }

        ThrowIfDisposed();
        await sessionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            PinnedPipeline pipeline = await GetOrCreatePinnedPipelineAsync(modelFiles, plan, cancellationToken).ConfigureAwait(false);
            // DirectML runs the language model on the GPU but the vocoder session is
            // explicitly CPU-backed, so account the full operation against CPU admission.
            ExecutionProviderKind admissionProvider = pipeline.SelectedProvider is ExecutionProviderKind.DirectMl
                ? ExecutionProviderKind.Cpu
                : pipeline.SelectedProvider;
            using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
                .AcquireAsync(admissionProvider, cancellationToken)
                .ConfigureAwait(false);

            // In-memory samples: no temp WAV round-trip between the pipeline and the
            // WAV encoding below.
            float[] samples;
            if (modelFiles.IsBaseModel)
            {
                samples = await pipeline.VoiceClonePipeline!.SynthesizeSamplesAsync(
                    request.Text,
                    request.VoiceCloneReference!.ReferenceClipPath,
                    request.VoiceCloneReference.ReferenceTranscript!.Trim(),
                    MapLanguage(request.LanguageCode),
                    progress: null).ConfigureAwait(false);
            }
            else
            {
                string speaker = ResolveSpeakerName(request.Voice.VoiceId, modelFiles.RootDirectory);
                samples = await pipeline.CustomVoicePipeline!.SynthesizeSamplesAsync(
                    request.Text,
                    speaker,
                    MapLanguage(request.LanguageCode),
                    instruct: null,
                    progress: null).ConfigureAwait(false);
            }

            byte[] wavBytes = WaveAudioWriter.EncodeMonoPcm16(samples, SampleRate);

            LastExecutionSummary = new StageRuntimeExecutionSummary(
                pipeline.RequestedProviderLabel,
                pipeline.SelectedProviderLabel,
                plan.ModelId,
                plan.ModelAlias,
                plan.Variant,
                pipeline.BootstrapDetail);

            return new TtsSynthesisResult(
                wavBytes,
                samples.Length,
                SampleRate,
                plan.ModelId ?? EngineFamilyName,
                request.Voice.VoiceId,
                pipeline.SelectedProviderLabel);
        }
        finally
        {
            sessionGate.Release();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposeSignaled, 1) == 1)
        {
            return;
        }

        sessionGate.Wait();
        try
        {
            pinnedPipeline?.Dispose();
            pinnedPipeline = null;
        }
        finally
        {
            sessionGate.Release();
            sessionGate.Dispose();
        }
    }

    private static string MapLanguage(string languageCode)
    {
        if (string.IsNullOrWhiteSpace(languageCode))
        {
            return "auto";
        }

        return languageCode.Trim().Split('-')[0].ToLowerInvariant() switch
        {
            "en" => "english",
            "zh" => "chinese",
            "ja" => "japanese",
            "ko" => "korean",
            "de" => "german",
            "fr" => "french",
            "ru" => "russian",
            "pt" => "portuguese",
            "es" => "spanish",
            "it" => "italian",
            _ => "auto",
        };
    }

    private static string ResolveSpeakerName(string voiceId, string modelRootDirectory)
    {
        if (string.IsNullOrWhiteSpace(voiceId))
        {
            return "ryan";
        }

        string normalized = voiceId.Trim();
        if (normalized.StartsWith("qwen3:", StringComparison.OrdinalIgnoreCase))
        {
            normalized = normalized["qwen3:".Length..];
        }

        string speakerIdsPath = Path.Join(modelRootDirectory, "embeddings", "speaker_ids.json");
        if (File.Exists(speakerIdsPath))
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(speakerIdsPath));
            if (document.RootElement.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty property in document.RootElement.EnumerateObject())
                {
                    if (property.Name.Equals(normalized, StringComparison.OrdinalIgnoreCase))
                    {
                        return property.Name;
                    }
                }
            }
        }

        return normalized.ToLowerInvariant();
    }

    private async Task<PinnedPipeline> GetOrCreatePinnedPipelineAsync(
        Qwen3TtsModelFiles modelFiles,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        PooledProviderRequest providerRequest = PooledProviderRequest.ForPlan(plan);
        if (pinnedPipeline is not null && pinnedPipeline.Matches(modelFiles, providerRequest))
        {
            return pinnedPipeline;
        }

        pinnedPipeline?.Dispose();
        pinnedPipeline = await CreatePinnedPipelineAsync(modelFiles, providerRequest, cancellationToken).ConfigureAwait(false);
        return pinnedPipeline;
    }

    private static async Task<PinnedPipeline> CreatePinnedPipelineAsync(
        Qwen3TtsModelFiles modelFiles,
        PooledProviderRequest providerRequest,
        CancellationToken cancellationToken)
    {
        OnnxExecutionSessionFactory.SessionOptionsFactoryBundle sessionOptions = await OnnxExecutionSessionFactory
            .CreateSessionOptionsFactoryAsync(providerRequest.Provider, cancellationToken, hasExplicitProviderPreference: providerRequest.HasExplicitProviderPreference)
            .ConfigureAwait(false);

        string requestedLabel = ProviderNames.FormatProviderLabel(sessionOptions.RequestedProvider);
        string selectedLabel = ProviderNames.FormatProviderLabel(sessionOptions.SelectedProvider);
        string bootstrapDetail = string.IsNullOrWhiteSpace(sessionOptions.BootstrapDetail)
            ? "qwen3-tts"
            : $"qwen3-tts; {sessionOptions.BootstrapDetail}";

        // Sessions now come from OnnxExecutionSessionFactory pooling (residency pins)
        // instead of per-pipeline direct InferenceSession creation, so a second
        // synthesis reuses the warm prefill/decode/code-predictor/vocoder sessions.
        ExecutionProviderKind vocoderProvider = sessionOptions.SelectedProvider is ExecutionProviderKind.DirectMl
            ? ExecutionProviderKind.Cpu
            : sessionOptions.SelectedProvider;
        Qwen3TtsPooledSessions pooledSessions = await Qwen3TtsPooledSessions.CreateAsync(
            modelFiles,
            providerRequest,
            vocoderProvider,
            cancellationToken).ConfigureAwait(false);
        try
        {
            if (modelFiles.IsBaseModel)
            {
                var clonePipeline = new VoiceClonePipeline(modelFiles.RootDirectory, sessionOptions.CreateOptions, pooledSessions);
                return new PinnedPipeline(
                    modelFiles.RootDirectory,
                    true,
                    providerRequest,
                    sessionOptions.SelectedProvider,
                    requestedLabel,
                    selectedLabel,
                    bootstrapDetail,
                    null,
                    clonePipeline);
            }

            var customPipeline = new TtsPipeline(
                modelFiles.RootDirectory,
                sessionOptions.CreateOptions,
                ResolveVocoderSessionOptionsFactory(sessionOptions),
                modelFiles.IsLargeModel ? QwenModelVariant.Qwen17B : QwenModelVariant.Qwen06B,
                pooledSessions);
            return new PinnedPipeline(
                modelFiles.RootDirectory,
                false,
                providerRequest,
                sessionOptions.SelectedProvider,
                requestedLabel,
                selectedLabel,
                bootstrapDetail,
                customPipeline,
                null);
        }
        catch
        {
            pooledSessions.Dispose();
            throw;
        }
    }

    private static Func<SessionOptions> ResolveVocoderSessionOptionsFactory(
        OnnxExecutionSessionFactory.SessionOptionsFactoryBundle sessionOptions) =>
        sessionOptions.SelectedProvider == ExecutionProviderKind.DirectMl
            ? CreateCpuSessionOptions
            : sessionOptions.CreateOptions;

    private static SessionOptions CreateCpuSessionOptions() => new()
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
    };

    private static void EnsurePlanReady(StageRuntimePlan plan)
    {
        if (plan.Status is not (StageRuntimePlanStatus.Ready or StageRuntimePlanStatus.Verified))
        {
            throw new InvalidOperationException($"Qwen3-TTS runtime plan is not ready ({plan.Status}).");
        }

        if (!string.Equals(plan.EngineFamily, EngineFamilyName, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Qwen3-TTS engine received plan for '{plan.EngineFamily ?? "unknown"}'.");
        }

        if (plan.ExecutionProvider is null)
        {
            throw new InvalidOperationException("Qwen3-TTS runtime plan is missing an execution provider.");
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposeSignaled != 0, this);
    }

    private sealed class PinnedPipeline : IDisposable
    {
        public PinnedPipeline(
            string rootDirectory,
            bool isBaseModel,
            PooledProviderRequest providerRequest,
            ExecutionProviderKind selectedProvider,
            string requestedProviderLabel,
            string selectedProviderLabel,
            string bootstrapDetail,
            TtsPipeline? customVoicePipeline,
            VoiceClonePipeline? voiceClonePipeline)
        {
            RootDirectory = rootDirectory;
            IsBaseModel = isBaseModel;
            ProviderRequest = providerRequest;
            SelectedProvider = selectedProvider;
            RequestedProviderLabel = requestedProviderLabel;
            SelectedProviderLabel = selectedProviderLabel;
            BootstrapDetail = bootstrapDetail;
            CustomVoicePipeline = customVoicePipeline;
            VoiceClonePipeline = voiceClonePipeline;
        }

        public string RootDirectory { get; }

        public bool IsBaseModel { get; }

        public PooledProviderRequest ProviderRequest { get; }

        public ExecutionProviderKind SelectedProvider { get; }

        public string RequestedProviderLabel { get; }

        public string SelectedProviderLabel { get; }

        public string BootstrapDetail { get; }

        public TtsPipeline? CustomVoicePipeline { get; }

        public VoiceClonePipeline? VoiceClonePipeline { get; }

        public bool Matches(Qwen3TtsModelFiles modelFiles, PooledProviderRequest providerRequest) =>
            ProviderRequest == providerRequest && IsSameModel(modelFiles);

        private bool IsSameModel(Qwen3TtsModelFiles modelFiles) =>
            string.Equals(RootDirectory, modelFiles.RootDirectory, StringComparison.OrdinalIgnoreCase) &&
            IsBaseModel == modelFiles.IsBaseModel;

        public void Dispose()
        {
            CustomVoicePipeline?.Dispose();
            VoiceClonePipeline?.Dispose();
        }
    }
}
