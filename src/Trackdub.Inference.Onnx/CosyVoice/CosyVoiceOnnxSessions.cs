using Microsoft.ML.OnnxRuntime;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.CosyVoice;

/// <summary>
/// Short-lived exclusive leases over the nine CosyVoice graphs (audit §3A).
/// Acquire for the duration of one synthesis; dispose to release execution gates.
/// </summary>
internal sealed class CosyVoiceOnnxSessions : IDisposable
{
    private readonly OnnxExecutionSessionFactory.SingleSessionLease[] leases;

    /// <summary>
    /// Wires the nine execution leases from the acquire step, validating the array arity before use.
    /// </summary>
    internal CosyVoiceOnnxSessions(
        OnnxExecutionSessionFactory.SingleSessionLease[] leases,
        string selectedProvider)
    {
        ArgumentNullException.ThrowIfNull(leases);
        if (leases.Length != 9)
        {
            throw new ArgumentException("CosyVoice requires nine session leases.", nameof(leases));
        }

        this.leases = leases;
        Campplus = leases[0];
        SpeechTokenizer = leases[1];
        TextEncoder = leases[2];
        TokenGenerator = leases[3];
        FlowEncoder = leases[4];
        FlowEstimator = leases[5];
        F0Predictor = leases[6];
        Source = leases[7];
        Vocoder = leases[8];
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
        foreach (OnnxExecutionSessionFactory.SingleSessionLease lease in leases)
        {
            lease.Dispose();
        }
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
        PooledProviderRequest providerRequest,
        CancellationToken cancellationToken)
    {
        string[] paths =
        [
            modelFiles.CampPlusPath,
            modelFiles.SpeechTokenizerPath,
            modelFiles.TextEncoderPath,
            modelFiles.TokenGeneratorPath,
            modelFiles.FlowEncoderPath,
            modelFiles.FlowDecoderEstimatorPath,
            modelFiles.HiftF0PredictorPath,
            modelFiles.HiftSourcePath,
            modelFiles.HiftVocoderPath,
        ];
        IReadOnlyDictionary<string, string>?[] trtOptions =
        [
            null,
            SpeechTokenizerTrtOptions,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
        ];

        var pins = new OnnxExecutionSessionFactory.PooledSingleSessionPin[paths.Length];
        try
        {
            for (int i = 0; i < paths.Length; i++)
            {
                pins[i] = await OnnxExecutionSessionFactory.PinPooledSingleAsync(
                    providerRequest.ToSessionRequest("cosyvoice", cancellationToken) with
                    {
                        AdditionalTrtOptions = trtOptions[i],
                    },
                    paths[i]).ConfigureAwait(false);
            }
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

            return new CosyVoiceOnnxSessions(leases, leases[2].SelectedProvider);
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
