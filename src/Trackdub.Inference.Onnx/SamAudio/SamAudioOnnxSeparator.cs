using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Runs the windowed separation on pooled ONNX sessions. The runner works in three passes, and the sessions are leased one
/// graph at a time (never two held together), so memory admission only ever has to fit one of these graphs plus whatever
/// else is already loaded.
/// </summary>
internal sealed class SamAudioOnnxSeparator : ISamAudioSeparator
{
    public async Task<SamAudioSeparation> SeparateAsync(
        SamAudioSeparatorRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        PooledProviderRequest providerRequest = PooledProviderRequest.ForPlan(request.Plan);
        SamAudioPrompt prompt = SamAudioPrompt.Load(request.ModelRootPath);

        // The runner executes on a worker thread (Task.Run below), so waiting on the async lease there blocks nothing else.
        using SamAudioSessionLeases leases = new(graph => OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(
                providerRequest.ToSessionRequest(SamAudioGraphInfo.PoolKey(graph), cancellationToken),
                SamAudioModelConstants.ResolveModelPath(request.ModelRootPath, SamAudioGraphInfo.FileName(graph)))
            .GetAwaiter()
            .GetResult());
        OrtSamAudioGraphs graphs = new(leases.Open, prompt);

        SamAudioSeparation separation = await Task
            .Run(() => SamAudioSeparationRunner.Run(request.Channels, graphs, progress, cancellationToken), cancellationToken)
            .ConfigureAwait(false);

        return separation with { Metadata = BuildMetadata(leases) };
    }

    private static Dictionary<string, string> BuildMetadata(SamAudioSessionLeases leases)
    {
        Dictionary<string, string> metadata = new(StringComparer.Ordinal)
        {
            ["selected_provider"] = leases.SelectedProvider ?? string.Empty,
        };
        if (!string.IsNullOrWhiteSpace(leases.BootstrapDetail))
        {
            metadata["bootstrap_detail"] = leases.BootstrapDetail;
        }

        return metadata;
    }
}
