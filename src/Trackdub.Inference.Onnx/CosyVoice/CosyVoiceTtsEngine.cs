using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.CosyVoice;

/// <summary>
/// CosyVoice voice-cloning TTS adapter for the bundled
/// <c>tonythethompson/CosyVoice-300M-ONNX</c> multi-graph ONNX bundle.
/// </summary>
public sealed class CosyVoiceTtsEngine(
    IConsentService consentService,
    BenchmarkModelPathResolver modelPathResolver)
    : ITtsEngineAdapter, IStageRuntimeExecutionReporter, IDisposable
{
    public const string EngineFamilyName = "cosyvoice";

    private readonly IConsentService consentService = consentService ?? throw new ArgumentNullException(nameof(consentService));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
    private readonly SemaphoreSlim sessionGate = new(1, 1);
    private PinnedRuntime? pinnedRuntime;
    private int disposeSignaled;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public Task<TtsSynthesisResult> SynthesizeAsync(
        TtsSynthesisRequest request,
        CancellationToken cancellationToken)
    {
        throw new InvalidOperationException("CosyVoice voice cloning must be invoked through the runtime-planned overload.");
    }

    public async Task<TtsSynthesisResult> SynthesizeAsync(
        TtsSynthesisRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        cancellationToken.ThrowIfCancellationRequested();

        if (request.VoiceCloneReference is null)
        {
            throw new InvalidOperationException("CosyVoice voice cloning requires a reference clip.");
        }

        if (!consentService.IsVoiceCloningConsentGranted)
        {
            throw new ConsentRequiredException();
        }

        if (string.IsNullOrWhiteSpace(request.VoiceCloneReference.ReferenceTranscript))
        {
            throw new TtsReferenceTextRequiredException();
        }

        CosyVoiceReferenceValidator.Validate(request.VoiceCloneReference.ReferenceClipPath);
        EnsurePlanReady(plan);

        string modelRootPath = PlannedRuntimeModelResolver.ResolveModelRootPath(plan, modelPathResolver);
        CosyVoiceModelFiles modelFiles = CosyVoiceModelFiles.Resolve(modelRootPath, plan.Variant);
        IReadOnlyList<string> missingFiles = modelFiles.FindMissingFiles();
        if (missingFiles.Count > 0)
        {
            throw new InvalidOperationException(
                "CosyVoice native runtime bundle is incomplete. " +
                $"Missing: {string.Join(", ", missingFiles)}");
        }

        ThrowIfDisposed();
        await sessionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            PinnedRuntime runtime = await GetOrCreatePinnedRuntimeAsync(
                modelFiles,
                PooledProviderRequest.ForPlan(plan),
                cancellationToken).ConfigureAwait(false);

            // Residency pins keep the nine graphs warm; execution leases are taken only
            // for this synthesis (audit §3A: residency ≠ exclusivity).
            using CosyVoiceOnnxSessions sessions = await runtime.Pins
                .AcquireAllAsync(cancellationToken)
                .ConfigureAwait(false);
            var pipeline = new CosyVoiceSynthesisPipeline(
                sessions, runtime.Embeddings, runtime.Tokenizer);

            float[] audioSamples = pipeline.Synthesize(
                request.Text,
                request.VoiceCloneReference.ReferenceTranscript.Trim(),
                request.VoiceCloneReference.ReferenceClipPath,
                cancellationToken);

            byte[] wavBytes = WaveAudioWriter.EncodeMonoPcm16(audioSamples, CosyVoiceConstants.SampleRate);
            LastExecutionSummary = new StageRuntimeExecutionSummary(
                sessions.TextEncoder.RequestedProvider,
                sessions.SelectedProvider,
                plan.ModelId,
                plan.ModelAlias,
                plan.Variant,
                sessions.TextEncoder.BootstrapDetail);

            return new TtsSynthesisResult(
                wavBytes,
                audioSamples.Length,
                CosyVoiceConstants.SampleRate,
                plan.ModelId ?? "tonythethompson/CosyVoice-300M-ONNX",
                request.Voice.VoiceId,
                sessions.SelectedProvider);
        }
        finally
        {
            sessionGate.Release();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposeSignaled, 1) != 0)
        {
            return;
        }

        sessionGate.Wait();
        try
        {
            pinnedRuntime?.Dispose();
            pinnedRuntime = null;
        }
        finally
        {
            sessionGate.Release();
            sessionGate.Dispose();
        }
    }

    private async Task<PinnedRuntime> GetOrCreatePinnedRuntimeAsync(
        CosyVoiceModelFiles modelFiles,
        PooledProviderRequest providerRequest,
        CancellationToken cancellationToken)
    {
        if (pinnedRuntime is not null &&
            pinnedRuntime.Matches(modelFiles.ModelRootPath, modelFiles.Variant, providerRequest))
        {
            return pinnedRuntime;
        }

        pinnedRuntime?.Dispose();
        pinnedRuntime = null;
        CosyVoiceSessionPins pins = await CosyVoiceSessionPins.CreateAsync(
            modelFiles,
            providerRequest,
            cancellationToken).ConfigureAwait(false);
        CosyVoiceEmbeddingTables embeddings;
        CosyVoiceWhisperTokenizer tokenizer;
        try
        {
            embeddings = CosyVoiceEmbeddingTables.Load(modelFiles.ModelRootPath);
            tokenizer = CosyVoiceWhisperTokenizer.Load(modelFiles.ModelRootPath);
        }
        catch
        {
            pins.Dispose();
            throw;
        }

        pinnedRuntime = new PinnedRuntime(
            modelFiles.ModelRootPath,
            modelFiles.Variant,
            providerRequest,
            pins,
            embeddings,
            tokenizer);
        return pinnedRuntime;
    }

    private static void EnsurePlanReady(StageRuntimePlan plan)
    {
        if (plan.ExecutionProvider is null)
        {
            throw new InvalidOperationException("Execution provider not resolved in runtime plan.");
        }

        if (plan.Status is not (StageRuntimePlanStatus.Ready or StageRuntimePlanStatus.Verified))
        {
            throw new InvalidOperationException($"CosyVoice runtime plan is not ready: {plan.Status}.");
        }
    }

    private void ThrowIfDisposed()
    {
        if (disposeSignaled != 0)
        {
            throw new ObjectDisposedException(nameof(CosyVoiceTtsEngine));
        }
    }

    private sealed class PinnedRuntime : IDisposable
    {
        public PinnedRuntime(
            string modelRootPath,
            string variant,
            PooledProviderRequest providerRequest,
            CosyVoiceSessionPins pins,
            CosyVoiceEmbeddingTables embeddings,
            CosyVoiceWhisperTokenizer tokenizer)
        {
            ModelRootPath = modelRootPath;
            Variant = variant;
            ProviderRequest = providerRequest;
            Pins = pins;
            Embeddings = embeddings;
            Tokenizer = tokenizer;
        }

        public string ModelRootPath { get; }

        public string Variant { get; }

        public PooledProviderRequest ProviderRequest { get; }

        public CosyVoiceSessionPins Pins { get; }

        public CosyVoiceEmbeddingTables Embeddings { get; }

        public CosyVoiceWhisperTokenizer Tokenizer { get; }

        public bool Matches(string modelRootPath, string variant, PooledProviderRequest providerRequest) =>
            ProviderRequest == providerRequest && IsSameModel(modelRootPath, variant);

        private bool IsSameModel(string modelRootPath, string variant) =>
            string.Equals(ModelRootPath, modelRootPath, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(Variant, variant, StringComparison.OrdinalIgnoreCase);

        public void Dispose() => Pins.Dispose();
    }
}
