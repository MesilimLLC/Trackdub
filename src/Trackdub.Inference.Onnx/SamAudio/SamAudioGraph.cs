namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>The three exported graphs a separation runs, loaded one at a time.</summary>
internal enum SamAudioGraph
{
    Encoder,
    Flow,
    Decoder,
}

internal static class SamAudioGraphInfo
{
    public static string FileName(SamAudioGraph graph) => graph switch
    {
        SamAudioGraph.Encoder => SamAudioModelConstants.EncoderFileName,
        SamAudioGraph.Flow => SamAudioModelConstants.DitFileName,
        SamAudioGraph.Decoder => SamAudioModelConstants.DecoderFileName,
        _ => throw new ArgumentOutOfRangeException(nameof(graph), graph, "Unknown SAM-Audio graph."),
    };

    /// <summary>Session-pool family key for a graph; the warmup smoke test uses the same key so it shares the warm session.</summary>
    public static string PoolKey(SamAudioGraph graph) => graph switch
    {
        SamAudioGraph.Encoder => "sam-audio-encoder",
        SamAudioGraph.Flow => "sam-audio-dit",
        SamAudioGraph.Decoder => "sam-audio-decoder",
        _ => throw new ArgumentOutOfRangeException(nameof(graph), graph, "Unknown SAM-Audio graph."),
    };
}
