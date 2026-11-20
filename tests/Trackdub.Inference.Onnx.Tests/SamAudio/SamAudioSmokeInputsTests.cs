using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioSmokeInputsTests
{
    [Fact]
    public void Smoke_inputs_supply_every_flow_graph_input_with_consistent_frame_counts()
    {
        var inputs = SamAudioSmokeInputs.Create().ToDictionary(input => input.Name, input => input.AsTensor<float>().Dimensions.ToArray());

        Assert.Equal(["features", "noisy", "text", "time"], inputs.Keys.Order().ToArray());
        Assert.Equal(inputs["noisy"][1], inputs["features"][1]);
        Assert.Equal(SamAudioModelConstants.LatentChannels, inputs["noisy"][2]);
        Assert.Equal(SamAudioModelConstants.FeatureChannels, inputs["features"][2]);
        Assert.Equal([1], inputs["time"]);
    }

    [Fact]
    public void Graph_names_and_pool_keys_are_distinct_per_graph()
    {
        SamAudioGraph[] graphs = Enum.GetValues<SamAudioGraph>();

        Assert.Equal(graphs.Length, graphs.Select(SamAudioGraphInfo.FileName).Distinct().Count());
        Assert.Equal(graphs.Length, graphs.Select(SamAudioGraphInfo.PoolKey).Distinct().Count());
        Assert.Equal("sam-audio-dit", SamAudioGraphInfo.PoolKey(SamAudioGraph.Flow));
    }
}
