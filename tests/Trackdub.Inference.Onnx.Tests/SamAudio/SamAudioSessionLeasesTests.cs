using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.SamAudio;
using Trackdub.Inference.Onnx.Tests.Spleeter;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioSessionLeasesTests
{
    /// <summary>Opens real (identity) sessions so disposal is observable: running a disposed session throws.</summary>
    private sealed class StubSource(params string[] providers)
    {
        private readonly Queue<string> providers = new(providers);

        public List<OnnxExecutionSessionFactory.SingleSessionLease> Opened { get; } = [];

        public OnnxExecutionSessionFactory.SingleSessionLease Lease(SamAudioGraph graph)
        {
            string provider = providers.TryDequeue(out string? next) ? next : "cuda";
            OnnxExecutionSessionFactory.SingleSessionLease lease = new(
                new InferenceSession(SpleeterInputNameResolutionTests.MinimalOnnxIdentityModel), "cuda", provider, $"{graph}:{provider}");
            Opened.Add(lease);
            return lease;
        }
    }

    private static void Run(InferenceSession session)
    {
        using var _ = session.Run([NamedOnnxValue.CreateFromTensor("mix_wave", new DenseTensor<float>(new[] { 1f }, [1]))]);
    }

    // ONNX Runtime does not raise a typed error for a released session; any failure to run it shows it was disposed.
    private static void AssertDisposed(InferenceSession session) => Assert.ThrowsAny<Exception>(() => Run(session));

    [Fact]
    public void Opening_the_same_graph_again_reuses_the_held_lease()
    {
        StubSource source = new();
        using SamAudioSessionLeases leases = new(source.Lease);

        InferenceSession first = leases.Open(SamAudioGraph.Encoder);
        InferenceSession second = leases.Open(SamAudioGraph.Encoder);

        Assert.Same(first, second);
        Assert.Single(source.Opened);
        Run(first);
    }

    [Fact]
    public void Opening_a_different_graph_releases_the_previous_lease_first()
    {
        StubSource source = new();
        using SamAudioSessionLeases leases = new(source.Lease);
        InferenceSession encoder = leases.Open(SamAudioGraph.Encoder);

        InferenceSession flow = leases.Open(SamAudioGraph.Flow);

        Assert.NotSame(encoder, flow);
        Assert.Equal(2, source.Opened.Count);
        AssertDisposed(encoder);
        Run(flow);
        Assert.Equal("cuda", leases.SelectedProvider);
    }

    [Fact]
    public void A_provider_mismatch_disposes_the_new_lease_and_keeps_the_established_provider()
    {
        StubSource source = new("cuda", "cpu");
        using SamAudioSessionLeases leases = new(source.Lease);
        InferenceSession encoder = leases.Open(SamAudioGraph.Encoder);

        Assert.Throws<InvalidOperationException>(() => leases.Open(SamAudioGraph.Flow));

        AssertDisposed(source.Opened[1].Session);
        AssertDisposed(encoder);
        Assert.Equal("cuda", leases.SelectedProvider);
        Assert.Equal("Encoder:cuda", leases.BootstrapDetail);

        // The failed open left nothing held, so the next open creates a fresh lease on the established provider.
        InferenceSession again = leases.Open(SamAudioGraph.Flow);
        Assert.Equal(3, source.Opened.Count);
        Run(again);
    }

    [Fact]
    public void Disposing_releases_the_held_lease()
    {
        StubSource source = new();
        SamAudioSessionLeases leases = new(source.Lease);
        InferenceSession encoder = leases.Open(SamAudioGraph.Encoder);

        leases.Dispose();

        AssertDisposed(encoder);
    }

    [Fact]
    public void EnsureSameProvider_refuses_mixed_execution()
    {
        SamAudioSessionLeases.EnsureSameProvider(null, "cuda");
        SamAudioSessionLeases.EnsureSameProvider("cuda", "cuda");

        Assert.Throws<InvalidOperationException>(() => SamAudioSessionLeases.EnsureSameProvider("cuda", "cpu"));
    }
}
