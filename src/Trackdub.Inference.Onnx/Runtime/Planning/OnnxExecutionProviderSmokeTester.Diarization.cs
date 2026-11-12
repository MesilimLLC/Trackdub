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
    private static async Task SmokeTestDiarizationAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        // Same family + TRT profiles as SortFormerDiarizationEngine so the smoke run warms
        // the session the diarization stage will lease.
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = SortFormerDiarizationEngine.EngineFamilyName, Provider = request.ExecutionProvider, HasExplicitProviderPreference = request.HasExplicitProviderPreference, CancellationToken = cancellationToken, AdditionalTrtOptions = SortFormerDiarizationEngine.TrtOptions, AllowTrtInitFallback = true }, request.EntryPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider, request.AllowPolicyProviderSelection);

        using var inputs = CreateDiarizationInputs(sessionLease.Session);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    private static InputSet CreateDiarizationInputs(InferenceSession session)
    {
        IReadOnlyDictionary<string, NodeMetadata> inputs = session.InputMetadata;
        if (IsSortFormerStreamingInputSet(inputs.Keys))
        {
            return CreateSortFormerStreamingInputs(inputs);
        }

        if (TryCreateWaveformDiarizationInputs(inputs, out InputSet? waveformInputs))
        {
            return waveformInputs!;
        }

        return CreateMetadataDrivenInputs(inputs);
    }

    private static bool IsSortFormerStreamingInputSet(IEnumerable<string> inputNames)
    {
        HashSet<string> names = inputNames.ToHashSet(StringComparer.Ordinal);
        return names.Contains("chunk") && names.Contains("chunk_lengths") &&
               names.Contains("spkcache") && names.Contains("spkcache_lengths") &&
               names.Contains("fifo") && names.Contains("fifo_lengths");
    }

    // The streaming export expects a fixed 1x3040x128 chunk. Cache and FIFO begin empty.
    private static InputSet CreateSortFormerStreamingInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        const int chunkFrames = 3040;
        const int melBins = 128;
        const int embeddingDimensions = 512;
        var values = new List<NamedOnnxValue>(6);
        AddFloatInput("chunk", [1, chunkFrames, melBins]);
        AddInt64Input("chunk_lengths", [1], [chunkFrames]);
        AddFloatInput("spkcache", [1, 0, embeddingDimensions]);
        AddInt64Input("spkcache_lengths", [1], [0]);
        AddFloatInput("fifo", [1, 0, embeddingDimensions]);
        AddInt64Input("fifo_lengths", [1], [0]);
        return new InputSet(values);

        void AddFloatInput(string name, int[] dimensions)
        {
            ValidateStreamingInput(name, dimensions, TensorElementType.Float);
            int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));
            values.Add(NamedOnnxValue.CreateFromTensor(
                name, new DenseTensor<float>(new float[elementCount], dimensions)));
        }

        void AddInt64Input(string name, int[] dimensions, long[] data)
        {
            ValidateStreamingInput(name, dimensions, TensorElementType.Int64);
            values.Add(NamedOnnxValue.CreateFromTensor(
                name, new DenseTensor<long>(data, dimensions)));
        }

        void ValidateStreamingInput(string name, int[] dimensions, TensorElementType elementType)
        {
            if (!inputMetadata.TryGetValue(name, out NodeMetadata? metadata))
            {
                throw new InvalidOperationException(
                    $"SortFormer streaming input '{name}' does not match the expected tensor contract.");
            }

            if (!MatchesStreamingTensorContract(metadata, dimensions, elementType))
            {
                throw new InvalidOperationException(
                    $"SortFormer streaming input '{name}' does not match the expected tensor contract.");
            }

            for (int axis = 0; axis < dimensions.Length; axis++)
            {
                if (metadata!.Dimensions[axis] > 0 && metadata.Dimensions[axis] != dimensions[axis])
                {
                    throw new InvalidOperationException(
                        $"SortFormer streaming input '{name}' has an incompatible fixed dimension at axis {axis}.");
                }
            }
        }
    }

    private static bool MatchesStreamingTensorContract(
        NodeMetadata metadata,
        int[] dimensions,
        TensorElementType elementType)
    {
        if (!metadata.IsTensor || metadata.ElementDataType != elementType)
        {
            return false;
        }

        return metadata.Dimensions.Length == dimensions.Length;
    }

    private static bool TryCreateWaveformDiarizationInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputs,
        out InputSet? inputSet)
    {
        inputSet = null;
        IReadOnlyDictionary<string, Type> inputElementTypes = inputs.ToDictionary(
            static kvp => kvp.Key,
            static kvp => kvp.Value.ElementType,
            StringComparer.Ordinal);

        try
        {
            (string waveformName, string? lengthName) = ResolveDiarizationInputNames(inputElementTypes);
            int[] waveformDims = ResolveDiarizationWaveformShape(inputs[waveformName].Dimensions);
            var values = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(waveformName, new DenseTensor<float>(new float[16000], waveformDims))
            };

            if (!string.IsNullOrWhiteSpace(lengthName))
            {
                values.Add(NamedOnnxValue.CreateFromTensor(lengthName, new DenseTensor<long>(new long[] { 16000 }, [1])));
            }

            inputSet = new InputSet(values);
            return true;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static (string WaveformName, string? LengthName) ResolveDiarizationInputNames(
        IReadOnlyDictionary<string, Type> inputElementTypes) =>
        (ResolveDiarizationWaveformName(inputElementTypes),
         ResolveDiarizationLengthName(inputElementTypes));

    private static string ResolveDiarizationWaveformName(
        IReadOnlyDictionary<string, Type> inputElementTypes)
    {
        if (inputElementTypes.ContainsKey("waveform"))
        {
            return "waveform";
        }

        if (inputElementTypes.ContainsKey("audio_signal"))
        {
            return "audio_signal";
        }

        string[] floatInputs = inputElementTypes
            .Where(static candidate => candidate.Value == typeof(float))
            .Select(static candidate => candidate.Key)
            .ToArray();
        return floatInputs.Length switch
        {
            1 => floatInputs[0],
            0 => throw new InvalidOperationException("Smoke test could not locate a float diarization waveform input."),
            _ => throw new InvalidOperationException($"Smoke test found {floatInputs.Length} float diarization inputs; expected exactly one waveform input.")
        };
    }

    private static string? ResolveDiarizationLengthName(
        IReadOnlyDictionary<string, Type> inputElementTypes)
    {
        if (inputElementTypes.ContainsKey("length"))
        {
            return "length";
        }

        if (inputElementTypes.ContainsKey("audio_signal_length"))
        {
            return "audio_signal_length";
        }

        string? lengthName = inputElementTypes
            .FirstOrDefault(static candidate =>
                candidate.Value == typeof(long) &&
                candidate.Key.Contains("length", StringComparison.OrdinalIgnoreCase))
            .Key;
        return string.IsNullOrWhiteSpace(lengthName) ? null : lengthName;
    }

    /// <summary>Resolves a one-second, 16 kHz probe waveform for rank-one or rank-two diarization inputs.</summary>
    /// <exception cref="InvalidOperationException">The model declares an unsupported waveform rank.</exception>
    private static int[] ResolveDiarizationWaveformShape(IReadOnlyList<int> modelDimensions) =>
        modelDimensions.Count switch
        {
            1 => [16000],
            2 => [1, 16000],
            _ => throw new InvalidOperationException("Smoke test diarization waveform input must be rank 1 or 2.")
        };
}
