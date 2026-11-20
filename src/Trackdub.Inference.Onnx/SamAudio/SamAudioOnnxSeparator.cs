using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Leases the encoder, flow and decoder sessions for a plan and runs the windowed separation on them.</summary>
internal sealed class SamAudioOnnxSeparator : ISamAudioSeparator
{
    public async Task<SamAudioSeparation> SeparateAsync(
        SamAudioSeparatorRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        PooledProviderRequest providerRequest = PooledProviderRequest.ForPlan(request.Plan);

        using OnnxExecutionSessionFactory.SingleSessionLease encoder = await LeaseAsync(
            providerRequest, "sam-audio-encoder", ModelPath(request, SamAudioModelConstants.EncoderFileName), cancellationToken).ConfigureAwait(false);
        using OnnxExecutionSessionFactory.SingleSessionLease dit = await LeaseAsync(
            providerRequest, "sam-audio-dit", ModelPath(request, SamAudioModelConstants.DitFileName), cancellationToken).ConfigureAwait(false);
        using OnnxExecutionSessionFactory.SingleSessionLease decoder = await LeaseAsync(
            providerRequest, "sam-audio-decoder", ModelPath(request, SamAudioModelConstants.DecoderFileName), cancellationToken).ConfigureAwait(false);

        EnsureSameProvider(encoder.SelectedProvider, dit.SelectedProvider, decoder.SelectedProvider);
        OrtSamAudioGraphs graphs = new(encoder.Session, dit.Session, decoder.Session, SamAudioPrompt.Load(request.ModelRootPath));

        SamAudioSeparation separation = await Task
            .Run(() => SamAudioSeparationRunner.Run(request.Channels, graphs, progress, cancellationToken), cancellationToken)
            .ConfigureAwait(false);

        return separation with { Metadata = BuildMetadata(encoder) };
    }

    private static Task<OnnxExecutionSessionFactory.SingleSessionLease> LeaseAsync(
        PooledProviderRequest providerRequest,
        string poolKey,
        string modelPath,
        CancellationToken cancellationToken) =>
        OnnxExecutionSessionFactory.CreatePooledSingleAsync(providerRequest.ToSessionRequest(poolKey, cancellationToken), modelPath);

    private static string ModelPath(SamAudioSeparatorRequest request, string fileName) =>
        SamAudioModelConstants.ResolveModelPath(request.ModelRootPath, fileName);

    /// <summary>Honest provider check: the three graphs must agree rather than report one provider for mixed execution.</summary>
    internal static void EnsureSameProvider(string encoder, string dit, string decoder)
    {
        if (!string.Equals(encoder, dit, StringComparison.Ordinal) || !string.Equals(encoder, decoder, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"SAM-Audio sessions landed on different providers (encoder {encoder}, flow {dit}, decoder {decoder}); "
                + "refusing to report one provider for mixed execution.");
        }
    }

    private static Dictionary<string, string> BuildMetadata(OnnxExecutionSessionFactory.SingleSessionLease lease)
    {
        Dictionary<string, string> metadata = new(StringComparer.Ordinal) { ["selected_provider"] = lease.SelectedProvider };
        if (!string.IsNullOrWhiteSpace(lease.BootstrapDetail))
        {
            metadata["bootstrap_detail"] = lease.BootstrapDetail;
        }

        return metadata;
    }
}
