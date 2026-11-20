using Microsoft.ML.OnnxRuntime;
using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

/// <summary>
/// Runs the real exported graphs through the C# code with fixed noise and watermark and compares the stems with a
/// Python ONNX Runtime reference that matches PyTorch SAMAudio.separate to about 100 dB SNR. Skipped unless
/// TRACKDUB_SAM_AUDIO_ONNX_PATH (the model directory) and TRACKDUB_SAM_AUDIO_PARITY_PATH (the fixture directory) are set.
/// </summary>
public sealed class SamAudioParityTests
{
    private const double MinimumSnrDb = 60;

    [Fact]
    public void Real_graphs_reproduce_the_python_onnx_reference()
    {
        string? modelDir = Environment.GetEnvironmentVariable("TRACKDUB_SAM_AUDIO_ONNX_PATH");
        string? fixtureDir = Environment.GetEnvironmentVariable("TRACKDUB_SAM_AUDIO_PARITY_PATH");
        Assert.SkipWhen(
            string.IsNullOrWhiteSpace(modelDir) || string.IsNullOrWhiteSpace(fixtureDir),
            "Set TRACKDUB_SAM_AUDIO_ONNX_PATH and TRACKDUB_SAM_AUDIO_PARITY_PATH to run the SAM-Audio parity test.");

        using InferenceSession encoder = Open(modelDir!, SamAudioModelConstants.EncoderFileName);
        using InferenceSession dit = Open(modelDir!, SamAudioModelConstants.DitFileName);
        using InferenceSession decoder = Open(modelDir!, SamAudioModelConstants.DecoderFileName);
        Assert.True(SamAudioSmokeInputs.Matches(dit));
        using (dit.Run(SamAudioSmokeInputs.Create()))
        {
            // The warmup smoke inputs are accepted by the real flow graph.
        }

        OrtSamAudioGraphs graphs = new(
            graph => graph switch { SamAudioGraph.Encoder => encoder, SamAudioGraph.Flow => dit, _ => decoder },
            SamAudioPrompt.Load(modelDir!));

        (float[] speech, float[] bed) = SamAudioWindowSeparator.Separate(
            Read(fixtureDir!, "wave"),
            graphs,
            new SamAudioFlowInputs(Read(fixtureDir!, "noise"), Read(fixtureDir!, "message")),
            CancellationToken.None);

        Assert.InRange(SnrDb(Read(fixtureDir!, "target"), speech), MinimumSnrDb, double.MaxValue);
        Assert.InRange(SnrDb(Read(fixtureDir!, "residual"), bed), MinimumSnrDb, double.MaxValue);
    }

    private static InferenceSession Open(string modelDir, string fileName) =>
        new(SamAudioModelConstants.ResolveModelPath(modelDir, fileName));

    private static float[] Read(string directory, string name)
    {
        byte[] bytes = File.ReadAllBytes(Path.Join(directory, name + ".f32"));
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static double SnrDb(float[] reference, float[] estimate)
    {
        Assert.Equal(reference.Length, estimate.Length);
        double signal = 0;
        double error = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            signal += (double)reference[i] * reference[i];
            double diff = reference[i] - estimate[i];
            error += diff * diff;
        }

        return 10 * Math.Log10(signal / Math.Max(error, 1e-20));
    }
}
