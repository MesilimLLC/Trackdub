using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Holds at most one pooled session lease at a time. Opening a different graph releases the current lease first, so the
/// session pool's memory admission is never asked to fit two of this engine's graphs while a lease is pinned (an idle
/// session can be evicted; a held one cannot). The provider must stay the same across graphs.
/// </summary>
internal sealed class SamAudioSessionLeases(Func<SamAudioGraph, OnnxExecutionSessionFactory.SingleSessionLease> lease) : IDisposable
{
    private SamAudioGraph? current;
    private OnnxExecutionSessionFactory.SingleSessionLease? held;

    public string? SelectedProvider { get; private set; }

    public string? BootstrapDetail { get; private set; }

    public InferenceSession Open(SamAudioGraph graph)
    {
        if (held is not null && current == graph)
        {
            return held.Session;
        }

        held?.Dispose();
        held = null;
        OnnxExecutionSessionFactory.SingleSessionLease next = lease(graph);
        try
        {
            EnsureSameProvider(SelectedProvider, next.SelectedProvider);
        }
        catch (InvalidOperationException)
        {
            next.Dispose();
            throw;
        }

        SelectedProvider ??= next.SelectedProvider;
        BootstrapDetail ??= next.BootstrapDetail;
        held = next;
        current = graph;
        return next.Session;
    }

    public void Dispose()
    {
        held?.Dispose();
        held = null;
    }

    /// <summary>Honest provider check: the graphs must agree rather than report one provider for mixed execution.</summary>
    internal static void EnsureSameProvider(string? established, string observed)
    {
        if (established is not null && !string.Equals(established, observed, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"SAM-Audio session provider changed mid-separation ({established} -> {observed}); "
                + "refusing to report one provider for mixed execution.");
        }
    }
}
