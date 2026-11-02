using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Qwen3Asr;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.OpusMt;
using Trackdub.Inference.Onnx.ParakeetTdt;
using Trackdub.Inference.Onnx.CosyVoice;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.SortFormer;
using Trackdub.Inference.Onnx.Spleeter;
using Trackdub.Inference.Onnx.Whisper;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Runtime.Planning;

public sealed partial class OnnxExecutionProviderSmokeTester : IExecutionProviderSmokeTester
{
    private static async Task SmokeTestSeparationAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string runnableModelPath = request.EntryPath;
        string poolFamily = ResolveSeparationPoolFamily(request.EngineFamily, request.ModelId, runnableModelPath);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(poolFamily, runnableModelPath, request.ExecutionProvider, cancellationToken, allowTrtInitFallback: true)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider);

        using var inputs = CreateSeparationInputs(sessionLease.Session);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    /// <summary>
    /// Maps a separation smoke target onto the pool family strings used by the production
    /// separators (Spleeter/SepFormer) so smoke and stage runs share one warm session.
    /// </summary>
    private static string ResolveSeparationPoolFamily(string? engineFamily, string modelId, string modelPath)
    {
        string name = $"{modelId} {Path.GetFileNameWithoutExtension(modelPath)} {engineFamily}";
        if (name.Contains("vocals", StringComparison.OrdinalIgnoreCase))
        {
            return "spleeter-vocals";
        }

        if (IsSpleeterAccompanimentTarget(name))
        {
            return "spleeter-acc";
        }

        if (name.Contains("osd", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("overlap", StringComparison.OrdinalIgnoreCase))
        {
            return "sepformer-osd";
        }

        if (name.Contains("sepformer", StringComparison.OrdinalIgnoreCase))
        {
            return "sepformer";
        }

        return string.IsNullOrWhiteSpace(engineFamily) ? "separation" : engineFamily;
    }

    private static bool IsSpleeterAccompanimentTarget(string name)
    {
        if (name.Contains("accompaniment", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("spleeter-acc", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return name.Contains("acc", StringComparison.OrdinalIgnoreCase)
            && name.Contains("spleeter", StringComparison.OrdinalIgnoreCase);
    }

    private static InputSet CreateSeparationInputs(InferenceSession session)
    {
        (string inputName, NodeMetadata metadata) = session.InputMetadata.First();
        int[] dimensions = ResolveSeparationSmokeInputDimensions(metadata.Dimensions);
        int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));
        return new InputSet(
        [
            NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(new float[elementCount], dimensions))
        ]);
    }

    internal static int[] ResolveSeparationSmokeInputDimensionsForTesting(IReadOnlyList<int> modelDimensions) =>
        ResolveSeparationSmokeInputDimensions(modelDimensions);

    private static int[] ResolveSeparationSmokeInputDimensions(IReadOnlyList<int> modelDimensions)
    {
        // 4-D separation graphs are Spleeter STFT blocks. Their channel/split axis
        // order depends on the export (older: [2, num_splits, 512, 1024]; pinned
        // 3e5a4dd: [num_splits, 2, 512, 1024]), so reuse the production separator's
        // layout resolution instead of guessing one order: feeding [2, 1, 512, 1024]
        // to the splits-first export fails with "index: 1 Got: 1 Expected: 2".
        // Unmappable layouts throw there and surface as an honest smoke failure.
        if (modelDimensions.Count == 4)
        {
            return SpleeterOnnxSeparator.ResolveBlockDimensions(modelDimensions);
        }

        int[] dimensions = modelDimensions.Count switch
        {
            2 => [2, 44100],
            3 => [1, 2, 44100],
            _ => modelDimensions
                .Select(static dimension => dimension > 0 ? dimension : 1)
                .ToArray()
        };
        int sampleDimensionIndex = dimensions.Length - 1;
        dimensions[sampleDimensionIndex] = Math.Max(dimensions[sampleDimensionIndex], 44100);
        return dimensions;
    }
}
