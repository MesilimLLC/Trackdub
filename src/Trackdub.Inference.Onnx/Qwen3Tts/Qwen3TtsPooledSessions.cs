using System.Linq;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Qwen3Tts.Models;

namespace Trackdub.Inference.Onnx.Qwen3Tts;

/// <summary>
/// Residency pins for the Qwen3-TTS graphs, created through
/// <see cref="OnnxExecutionSessionFactory"/> pooling instead of direct
/// <c>new InferenceSession</c> calls. Ownership of each holder transfers to the model
/// wrapper that runs that graph (language model, vocoder, speaker encoder); disposing a
/// holder returns its lease to the pool rather than disposing the shared session.
/// </summary>
internal sealed class Qwen3TtsPooledSessions : IDisposable
{
    private Qwen3TtsPooledSessions(
        QwenPooledSession prefill,
        QwenPooledSession decode,
        QwenPooledSession codePredictor,
        QwenPooledSession vocoder,
        QwenPooledSession? speakerEncoder)
    {
        Prefill = prefill;
        Decode = decode;
        CodePredictor = codePredictor;
        Vocoder = vocoder;
        SpeakerEncoder = speakerEncoder;
    }

    public QwenPooledSession Prefill { get; }

    public QwenPooledSession Decode { get; }

    public QwenPooledSession CodePredictor { get; }

    public QwenPooledSession Vocoder { get; }

    public QwenPooledSession? SpeakerEncoder { get; }

    /// <summary>
    /// Fails before any residency pin when the graphs <see cref="CreateAsync"/> pins cannot all
    /// fit their buckets' budgets. Pins are held until the engine is disposed and are never
    /// evictable, so an unfittable set would pin the first graphs and then wait forever for the
    /// rest. The talker prefill, decode and code predictor graphs are also leased together for
    /// each synthesis; the vocoder and speaker encoder run on their own providers.
    /// </summary>
    internal static void ValidateAdmission(
        Qwen3TtsModelFiles modelFiles,
        PooledProviderRequest providerRequest,
        ExecutionProviderKind selectedProvider,
        ExecutionProviderKind vocoderProvider)
    {
        string modelDir = modelFiles.RootDirectory;
        bool talkerMayFallBack = providerRequest.AllowTrtInitFallback
            && selectedProvider is ExecutionProviderKind.TensorRTRtx;
        bool vocoderMayFallBack = providerRequest.AllowTrtInitFallback
            && vocoderProvider is ExecutionProviderKind.TensorRTRtx;

        var graphs = new List<(string FileName, ExecutionProviderKind Provider, bool MayFallBack)>
        {
            ("talker_prefill.onnx", selectedProvider, talkerMayFallBack),
            ("talker_decode.onnx", selectedProvider, talkerMayFallBack),
            ("code_predictor.onnx", selectedProvider, talkerMayFallBack),
            ("vocoder.onnx", vocoderProvider, vocoderMayFallBack),
        };
        if (modelFiles.IsBaseModel)
        {
            graphs.Add(("speaker_encoder.onnx", selectedProvider, talkerMayFallBack));
        }

        // Pinned keys take their OpenVINO residency from the factory (WithOpenVinoResidency), so
        // the preflight must place them in the same bucket: host RAM under the CPU proxy.
        bool openVinoCpuProxy = OnnxExecutionSessionFactory.UseOpenVinoCpuProxy;
        InferenceSessionPool.Shared.ValidateConcurrentLeaseAdmission(
            graphs
                .Select(graph => new InferenceSessionPool.ConcurrentLeaseGraph(
                    graph.Provider,
                    null,
                    SessionPoolKey.EstimateVramMb(Path.Join(modelDir, graph.FileName), graph.Provider, graph.MayFallBack),
                    openVinoCpuProxy))
                .ToArray());
    }

    public static async Task<Qwen3TtsPooledSessions> CreateAsync(
        Qwen3TtsModelFiles modelFiles,
        PooledProviderRequest providerRequest,
        ExecutionProviderKind vocoderProvider,
        CancellationToken cancellationToken)
    {
        string modelDir = modelFiles.RootDirectory;
        QwenPooledSession? prefill = null;
        QwenPooledSession? decode = null;
        QwenPooledSession? codePredictor = null;
        QwenPooledSession? vocoder = null;
        QwenPooledSession? speakerEncoder = null;
        try
        {
            prefill = await PinAsync(modelDir, "talker_prefill.onnx", providerRequest, cancellationToken).ConfigureAwait(false);
            decode = await PinAsync(modelDir, "talker_decode.onnx", providerRequest, cancellationToken).ConfigureAwait(false);
            codePredictor = await PinAsync(modelDir, "code_predictor.onnx", providerRequest, cancellationToken).ConfigureAwait(false);
            vocoder = await PinAsync(modelDir, "vocoder.onnx", providerRequest with { Provider = vocoderProvider }, cancellationToken).ConfigureAwait(false);
            if (modelFiles.IsBaseModel)
            {
                speakerEncoder = await PinAsync(modelDir, "speaker_encoder.onnx", providerRequest, cancellationToken).ConfigureAwait(false);
            }

            return new Qwen3TtsPooledSessions(prefill, decode, codePredictor, vocoder, speakerEncoder);
        }
        catch
        {
            prefill?.Dispose();
            decode?.Dispose();
            codePredictor?.Dispose();
            vocoder?.Dispose();
            speakerEncoder?.Dispose();
            throw;
        }
    }

    private static async Task<QwenPooledSession> PinAsync(
        string modelDir,
        string fileName,
        PooledProviderRequest providerRequest,
        CancellationToken cancellationToken)
    {
        string modelPath = Path.Join(modelDir, fileName);
        // SEC-3: file size pre-check before the pool opens the model, mirroring the
        // direct-session guard this replaces (1.7B models are ~5.4 GB).
        var fileInfo = new FileInfo(modelPath);
        if (!fileInfo.Exists)
        {
            throw new FileNotFoundException($"Qwen3-TTS ONNX model not found: {modelPath}", modelPath);
        }

        const long maxOnnxSize = 8_000_000_000;
        if (fileInfo.Length > maxOnnxSize)
        {
            throw new InvalidOperationException(
                $"ONNX file too large ({fileInfo.Length / 1e9:F2} GB). Maximum allowed: {maxOnnxSize / 1e9:F2} GB.");
        }

        OnnxExecutionSessionFactory.PooledSingleSessionPin pin =
            await OnnxExecutionSessionFactory.PinPooledSingleAsync(providerRequest.ToSessionRequest(Qwen3TtsEngine.EngineFamilyName, cancellationToken), modelPath).ConfigureAwait(false);
        return new QwenPooledSession(pin);
    }

    /// <summary>
    /// Disposes every holder. Holders are idempotent, so calling this after ownership
    /// transferred to model wrappers (which dispose their own holder) is harmless; it
    /// exists for construction-failure cleanup before that transfer completes.
    /// </summary>
    public void Dispose()
    {
        Prefill.Dispose();
        Decode.Dispose();
        CodePredictor.Dispose();
        Vocoder.Dispose();
        SpeakerEncoder?.Dispose();
    }
}
