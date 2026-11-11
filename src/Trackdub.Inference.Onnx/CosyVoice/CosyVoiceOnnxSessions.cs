using Microsoft.ML.OnnxRuntime;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.CosyVoice;

/// <summary>
/// Short-lived exclusive leases over the nine CosyVoice graphs (audit §3A).
/// Acquire for the duration of one synthesis; dispose to release execution gates.
/// </summary>
internal sealed class CosyVoiceOnnxSessions : IDisposable
{
    internal CosyVoiceOnnxSessions(
        OnnxExecutionSessionFactory.SingleSessionLease campplus,
        OnnxExecutionSessionFactory.SingleSessionLease speechTokenizer,
        OnnxExecutionSessionFactory.SingleSessionLease textEncoder,
        OnnxExecutionSessionFactory.SingleSessionLease tokenGenerator,
        OnnxExecutionSessionFactory.SingleSessionLease flowEncoder,
        OnnxExecutionSessionFactory.SingleSessionLease flowEstimator,
        OnnxExecutionSessionFactory.SingleSessionLease f0Predictor,
        OnnxExecutionSessionFactory.SingleSessionLease source,
        OnnxExecutionSessionFactory.SingleSessionLease vocoder,
        string selectedProvider)
    {
        Campplus = campplus;
        SpeechTokenizer = speechTokenizer;
        TextEncoder = textEncoder;
        TokenGenerator = tokenGenerator;
        FlowEncoder = flowEncoder;
        FlowEstimator = flowEstimator;
        F0Predictor = f0Predictor;
        Source = source;
        Vocoder = vocoder;
        SelectedProvider = selectedProvider;
    }

    public OnnxExecutionSessionFactory.SingleSessionLease Campplus { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease SpeechTokenizer { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease TextEncoder { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease TokenGenerator { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease FlowEncoder { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease FlowEstimator { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease F0Predictor { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease Source { get; }

    public OnnxExecutionSessionFactory.SingleSessionLease Vocoder { get; }

    public string SelectedProvider { get; }

    public void Dispose()
    {
        Campplus.Dispose();
        SpeechTokenizer.Dispose();
        TextEncoder.Dispose();
        TokenGenerator.Dispose();
        FlowEncoder.Dispose();
        FlowEstimator.Dispose();
        F0Predictor.Dispose();
        Source.Dispose();
        Vocoder.Dispose();
    }
}

/// <summary>
/// Engine-lifetime residency pins for CosyVoice's nine graphs. Pins keep sessions warm
/// without holding execution leases between synthesizes (audit §3A).
/// </summary>
internal sealed class CosyVoiceSessionPins : IDisposable
{
    // Measured slower with capture on TensorRT-RTX (median 5.3 ms off vs 6.2 ms on, RTX 5070) and its mel
    // length changes with every prompt clip, so it stays out of the CUDA graph default.
    internal static readonly IReadOnlyDictionary<string, string> SpeechTokenizerTrtOptions =
        new Dictionary<string, string>(StringComparer.Ordinal) { ["enable_cuda_graph"] = "0" };

    private readonly OnnxExecutionSessionFactory.PooledSingleSessionPin[] pins;

    private CosyVoiceSessionPins(OnnxExecutionSessionFactory.PooledSingleSessionPin[] pins)
    {
        this.pins = pins;
    }

    public static async Task<CosyVoiceSessionPins> CreateAsync(
        CosyVoiceModelFiles modelFiles,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        bool allowTrtInitFallback = true)
    {
        async Task<OnnxExecutionSessionFactory.PooledSingleSessionPin> Pin(
            string path,
            IReadOnlyDictionary<string, string>? trtOptions = null) =>
            await OnnxExecutionSessionFactory.PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "cosyvoice", Provider = provider, CancellationToken = cancellationToken, AllowTrtInitFallback = allowTrtInitFallback, AdditionalTrtOptions = trtOptions }, path).ConfigureAwait(false);

        var pins = new OnnxExecutionSessionFactory.PooledSingleSessionPin[9];
        try
        {
            pins[0] = await Pin(modelFiles.CampPlusPath).ConfigureAwait(false);
            pins[1] = await Pin(modelFiles.SpeechTokenizerPath, SpeechTokenizerTrtOptions).ConfigureAwait(false);
            pins[2] = await Pin(modelFiles.TextEncoderPath).ConfigureAwait(false);
            pins[3] = await Pin(modelFiles.TokenGeneratorPath).ConfigureAwait(false);
            pins[4] = await Pin(modelFiles.FlowEncoderPath).ConfigureAwait(false);
            pins[5] = await Pin(modelFiles.FlowDecoderEstimatorPath).ConfigureAwait(false);
            pins[6] = await Pin(modelFiles.HiftF0PredictorPath).ConfigureAwait(false);
            pins[7] = await Pin(modelFiles.HiftSourcePath).ConfigureAwait(false);
            pins[8] = await Pin(modelFiles.HiftVocoderPath).ConfigureAwait(false);
        }
        catch
        {
            foreach (OnnxExecutionSessionFactory.PooledSingleSessionPin pin in pins)
            {
                pin?.Dispose();
            }

            throw;
        }

        return new CosyVoiceSessionPins(pins);
    }

    /// <summary>Takes short execution leases on all nine graphs for one synthesis.</summary>
    public async Task<CosyVoiceOnnxSessions> AcquireAllAsync(CancellationToken cancellationToken)
    {
        OnnxExecutionSessionFactory.SingleSessionLease[] leases = new OnnxExecutionSessionFactory.SingleSessionLease[pins.Length];
        try
        {
            for (int i = 0; i < pins.Length; i++)
            {
                leases[i] = await pins[i].AcquireAsync(cancellationToken).ConfigureAwait(false);
            }

            return new CosyVoiceOnnxSessions(
                leases[0], leases[1], leases[2], leases[3], leases[4],
                leases[5], leases[6], leases[7], leases[8],
                leases[2].SelectedProvider);
        }
        catch
        {
            for (int i = 0; i < leases.Length; i++)
            {
                leases[i]?.Dispose();
            }

            throw;
        }
    }

    public void Dispose()
    {
        foreach (OnnxExecutionSessionFactory.PooledSingleSessionPin pin in pins)
        {
            pin.Dispose();
        }
    }
}
