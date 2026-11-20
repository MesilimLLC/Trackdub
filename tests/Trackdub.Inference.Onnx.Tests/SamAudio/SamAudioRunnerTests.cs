using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioRunnerTests
{
    /// <summary>Stand-in graphs: zero features and velocity, so the decoded latent is exactly the seeded noise.</summary>
    private sealed class FakeGraphs : ISamAudioGraphs
    {
        /// <summary>When set, the decoder returns this many samples regardless of the frame count.</summary>
        public int? DecodedSamples { get; init; }

        public int Decodes { get; private set; }

        /// <summary>Which graph each call used, in order: E(ncode), V(elocity), D(ecode).</summary>
        public List<char> Calls { get; } = [];

        public float[]? DecodedLatent { get; private set; }

        public float[]? Message { get; private set; }

        public float[] Encode(float[] paddedWave, CancellationToken cancellationToken)
        {
            Calls.Add('E');
            Assert.Equal(0, paddedWave.Length % SamAudioModelConstants.HopLength);
            return new float[paddedWave.Length / SamAudioModelConstants.HopLength * SamAudioModelConstants.FeatureChannels];
        }

        public float[] Velocity(float time, float[] noisy, float[] features, CancellationToken cancellationToken)
        {
            Calls.Add('V');
            Assert.Equal(features.Length / SamAudioModelConstants.FeatureChannels * SamAudioModelConstants.LatentChannels, noisy.Length);
            return new float[noisy.Length];
        }

        public (float[] Target, float[] Residual) Decode(float[] latent, int frames, float[] message, CancellationToken cancellationToken)
        {
            Decodes++;
            Calls.Add('D');
            DecodedLatent = latent;
            Message = message;
            int samples = DecodedSamples ?? frames * SamAudioModelConstants.HopLength;
            return ([.. Enumerable.Repeat(1f, samples)], [.. Enumerable.Repeat(0.5f, samples)]);
        }
    }

    [Fact]
    public void Window_separation_decodes_the_seeded_noise_and_crops_to_the_window()
    {
        FakeGraphs graphs = new();
        float[] window = new float[4000];

        SamAudioFlowInputs inputs = new(new SamAudioRandom(5).Gaussian(3 * SamAudioModelConstants.LatentChannels), new SamAudioRandom(6).WatermarkMessage());

        (float[] speech, float[] bed) = SamAudioWindowSeparator.Separate(window, graphs, inputs, CancellationToken.None);

        Assert.Equal(4000, speech.Length);
        Assert.Equal(4000, bed.Length);
        Assert.Equal(inputs.Noise, graphs.DecodedLatent);
        Assert.Equal(inputs.WatermarkMessage, graphs.Message);
    }

    [Fact]
    public void Window_separation_rejects_a_decoder_that_returns_too_few_samples()
    {
        SamAudioFlowInputs inputs = new(new float[3 * SamAudioModelConstants.LatentChannels], new float[32]);

        Assert.Throws<InvalidOperationException>(() => SamAudioWindowSeparator.Separate(
            new float[4000], new FakeGraphs { DecodedSamples = 10 }, inputs, CancellationToken.None));
    }

    [Fact]
    public void Runner_separates_every_window_of_every_channel_and_keeps_gain_through_the_crossfade()
    {
        FakeGraphs graphs = new();
        int length = 65 * SamAudioModelConstants.SampleRate;
        float[][] channels = [new float[length], new float[length]];
        List<StemSeparationProgress> reports = [];

        SamAudioSeparation result = SamAudioSeparationRunner.Run(channels, graphs, new Progress(reports.Add), CancellationToken.None);

        Assert.Equal(3, result.WindowCount);
        Assert.Equal(6 * 6, graphs.Decodes); // 2 channels x 3 windows x 6 decode chunks per 30 s window
        Assert.Equal(2, result.Speech.Length);
        Assert.All(result.Speech, speech => Assert.All(speech.Where((_, i) => i % 9973 == 0), v => Assert.Equal(1f, v, 1e-5f)));
        Assert.All(result.Bed, bed => Assert.All(bed.Where((_, i) => i % 9973 == 0), v => Assert.Equal(0.5f, v, 1e-5f)));
        Assert.Equal(19, reports.Count);
        Assert.Equal(18, reports[^1].CompletedChunks);
        Assert.Equal(18, reports[^1].TotalChunks);
    }

    [Fact]
    public void Runner_runs_one_graph_at_a_time_across_all_windows()
    {
        // Encode every window, then run the flow on every window, then decode every window: a caller that leases a graph
        // for a whole pass never needs two graphs loaded together.
        FakeGraphs graphs = new();
        int length = 65 * SamAudioModelConstants.SampleRate;

        SamAudioSeparationRunner.Run([new float[length]], graphs, null, CancellationToken.None);

        string order = new([.. graphs.Calls]);
        int firstFlow = order.IndexOf('V');
        int firstDecode = order.IndexOf('D');
        Assert.Equal(3 * 6, order.Count(c => c == 'E')); // 3 windows x 6 encode chunks
        Assert.Equal(3 * 6, order.Count(c => c == 'D')); // 3 windows x 6 decode chunks
        Assert.True(order.LastIndexOf('E') < firstFlow);
        Assert.True(order.LastIndexOf('V') < firstDecode);
    }

    [Fact]
    public void Runner_is_reproducible_for_the_same_input()
    {
        float[][] channels = [new float[SamAudioModelConstants.SampleRate]];
        FakeGraphs first = new();
        FakeGraphs second = new();

        SamAudioSeparationRunner.Run(channels, first, null, CancellationToken.None);
        SamAudioSeparationRunner.Run(channels, second, null, CancellationToken.None);

        Assert.Equal(first.DecodedLatent, second.DecodedLatent);
        Assert.Equal(first.Message, second.Message);
    }

    [Fact]
    public void Runner_honours_cancellation_between_windows()
    {
        using CancellationTokenSource cts = new();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() =>
            SamAudioSeparationRunner.Run([new float[SamAudioModelConstants.SampleRate]], new FakeGraphs(), null, cts.Token));
    }

    [Fact]
    public void Runner_rejects_empty_or_ragged_channels()
    {
        Assert.Throws<ArgumentException>(() => SamAudioSeparationRunner.Run([Array.Empty<float>()], new FakeGraphs(), null, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => SamAudioSeparationRunner.Run([new float[10], new float[11]], new FakeGraphs(), null, CancellationToken.None));
    }

    [Fact]
    public void Engine_can_be_built_by_a_container_that_registers_only_the_runtime_planner()
    {
        // A DI container fills a defaulted parameter when it has no registration for it, as the benchmark host does
        // for IRuntimePlanningPreferences; without the default no public constructor is resolvable there.
        System.Reflection.ConstructorInfo constructor = Assert.Single(typeof(SamAudioStemSeparationEngine).GetConstructors());

        Assert.All(constructor.GetParameters().Skip(1), parameter => Assert.True(parameter.HasDefaultValue));
    }

    [Fact]
    public void Downmix_averages_the_channels()
    {
        float[] mono = SamAudioStemOutputs.DownmixToMono([[1f, 0f], [0f, 1f]]);

        Assert.Equal([0.5f, 0.5f], mono);
    }

    [Fact]
    public void EnsureSameProvider_refuses_mixed_execution()
    {
        SamAudioSessionLeases.EnsureSameProvider(null, "cuda");
        SamAudioSessionLeases.EnsureSameProvider("cuda", "cuda");

        Assert.Throws<InvalidOperationException>(() => SamAudioSessionLeases.EnsureSameProvider("cuda", "cpu"));
    }

    private sealed class Progress(Action<StemSeparationProgress> report) : IProgress<StemSeparationProgress>
    {
        public void Report(StemSeparationProgress value) => report(value);
    }
}
