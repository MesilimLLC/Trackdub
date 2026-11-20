using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Per-channel 48 kHz samples to separate, with the model directory and runtime plan to run them on.</summary>
internal sealed record SamAudioSeparatorRequest(string ModelRootPath, StageRuntimePlan Plan, float[][] Channels);

/// <summary>Speech and residual-bed samples, one array per input channel, at the model's 48 kHz.</summary>
internal sealed record SamAudioSeparation(
    float[][] Speech,
    float[][] Bed,
    int WindowCount,
    IReadOnlyDictionary<string, string>? Metadata = null);

internal interface ISamAudioSeparator
{
    Task<SamAudioSeparation> SeparateAsync(
        SamAudioSeparatorRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken);
}
