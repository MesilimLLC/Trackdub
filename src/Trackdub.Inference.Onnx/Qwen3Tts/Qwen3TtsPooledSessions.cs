using Trackdub.Domain;
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

    public static async Task<Qwen3TtsPooledSessions> CreateAsync(
        string modelDir,
        ExecutionProviderKind provider,
        ExecutionProviderKind vocoderProvider,
        bool includeSpeakerEncoder,
        CancellationToken cancellationToken)
    {
        QwenPooledSession? prefill = null;
        QwenPooledSession? decode = null;
        QwenPooledSession? codePredictor = null;
        QwenPooledSession? vocoder = null;
        QwenPooledSession? speakerEncoder = null;
        try
        {
            prefill = await PinAsync(modelDir, "talker_prefill.onnx", provider, cancellationToken).ConfigureAwait(false);
            decode = await PinAsync(modelDir, "talker_decode.onnx", provider, cancellationToken).ConfigureAwait(false);
            codePredictor = await PinAsync(modelDir, "code_predictor.onnx", provider, cancellationToken).ConfigureAwait(false);
            vocoder = await PinAsync(modelDir, "vocoder.onnx", vocoderProvider, cancellationToken).ConfigureAwait(false);
            if (includeSpeakerEncoder)
            {
                speakerEncoder = await PinAsync(modelDir, "speaker_encoder.onnx", provider, cancellationToken).ConfigureAwait(false);
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
        ExecutionProviderKind provider,
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
            await OnnxExecutionSessionFactory.PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = Qwen3TtsEngine.EngineFamilyName, Provider = provider, CancellationToken = cancellationToken }, modelPath).ConfigureAwait(false);
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
