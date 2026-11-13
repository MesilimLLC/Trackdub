using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Contracts.Serving;

namespace Trackdub.Inference.Onnx.Serving;

/// <summary>Maps resident-serving tensors to and from ONNX Runtime values.</summary>
internal static class ServeTensorConverter
{
    public static IReadOnlyList<ServeTensorSpec> Describe(IReadOnlyDictionary<string, NodeMetadata> metadata) =>
        metadata
            .Select(pair => new ServeTensorSpec(
                pair.Key,
                pair.Value.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR
                    ? ToDType(pair.Value.ElementDataType) ?? pair.Value.ElementDataType.ToString().ToLowerInvariant()
                    : pair.Value.OnnxValueType.ToString(),
                pair.Value.Dimensions.Select(static dimension => (long)dimension).ToList()))
            .ToList();

    /// <summary>Throws <see cref="ResidentInferenceException"/> (bad-inputs) on any mismatch.</summary>
    public static void Validate(InferenceSession session, IReadOnlyList<ServeTensor> inputs)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (ServeTensor input in inputs)
        {
            if (!seen.Add(input.Name))
            {
                throw BadInputs($"Input '{input.Name}' was supplied more than once.");
            }

            if (!session.InputMetadata.TryGetValue(input.Name, out NodeMetadata? metadata))
            {
                throw BadInputs(
                    $"Unknown input '{input.Name}'. Expected: {string.Join(", ", session.InputMetadata.Keys)}.");
            }

            string? expectedType = metadata.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR
                ? ToDType(metadata.ElementDataType)
                : null;
            if (expectedType is null)
            {
                throw BadInputs($"Input '{input.Name}' has a type resident serving does not support.");
            }

            if (!string.Equals(input.DType, expectedType, StringComparison.Ordinal))
            {
                throw BadInputs($"Input '{input.Name}' expects {expectedType}, got {input.DType}.");
            }

            if (input.Data.GetType().GetElementType() != ClrElementType(input.DType))
            {
                throw BadInputs($"Input '{input.Name}' data does not hold {input.DType} values.");
            }

            int[] dimensions = metadata.Dimensions;
            if (input.Shape.Count != dimensions.Length)
            {
                throw BadInputs(
                    $"Input '{input.Name}' expects rank {dimensions.Length}, got rank {input.Shape.Count}.");
            }

            long elementCount = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                long actual = input.Shape[i];
                if (actual < 0 || (dimensions[i] >= 0 && actual != dimensions[i]))
                {
                    throw BadInputs(
                        $"Input '{input.Name}' dimension {i} must be {FormatDimension(dimensions[i])}, got {actual}.");
                }

                if (actual > 0 && elementCount > Array.MaxLength / actual)
                {
                    throw BadInputs($"Input '{input.Name}' shape [{string.Join(",", input.Shape)}] is too large.");
                }

                elementCount *= actual;
            }

            if (input.Data.LongLength != elementCount)
            {
                throw BadInputs(
                    $"Input '{input.Name}' shape [{string.Join(",", input.Shape)}] needs {elementCount} values, got {input.Data.LongLength}.");
            }
        }

        string[] missing = session.InputMetadata.Keys.Where(name => !seen.Contains(name)).ToArray();
        if (missing.Length > 0)
        {
            throw BadInputs($"Missing input(s): {string.Join(", ", missing)}.");
        }
    }

    /// <summary>Zero tensors for every declared input, or null when one cannot be synthesized.</summary>
    public static IReadOnlyList<ServeTensor>? SynthesizeZeroInputs(IReadOnlyList<ServeTensorSpec> inputs)
    {
        var tensors = new List<ServeTensor>(inputs.Count);
        foreach (ServeTensorSpec input in inputs)
        {
            if (ClrElementType(input.DType) is not { } elementType)
            {
                return null;
            }

            long[] shape = input.Shape.Select(static dimension => dimension < 0 ? 1L : dimension).ToArray();
            long count = 1;
            foreach (long dimension in shape)
            {
                // A shape too large for one array cannot be synthesized; skip warmup, don't throw.
                if (dimension > 0 && count > Array.MaxLength / dimension)
                {
                    return null;
                }

                count *= dimension;
            }

            tensors.Add(new ServeTensor(input.Name, input.DType, shape, Array.CreateInstance(elementType, count)));
        }

        return tensors;
    }

    /// <remarks>
    /// Dispatches on <see cref="ServeTensor.DType"/>, not the array type: the CLR lets a byte[]
    /// match an sbyte[] pattern (and vice versa), so array-type patterns would mislabel uint8 as int8.
    /// </remarks>
    public static OrtValue ToOrtValue(ServeTensor tensor)
    {
        if (tensor.Data.GetType().GetElementType() != ClrElementType(tensor.DType))
        {
            throw BadInputs($"Input '{tensor.Name}' data does not hold {tensor.DType} values.");
        }

        long[] shape = tensor.Shape.ToArray();
        return tensor.DType switch
        {
            ServeTensorTypes.Float32 => OrtValue.CreateTensorValueFromMemory((float[])tensor.Data, shape),
            ServeTensorTypes.Float16 => ToFloat16OrtValue((Half[])tensor.Data, shape),
            ServeTensorTypes.Int64 => OrtValue.CreateTensorValueFromMemory((long[])tensor.Data, shape),
            ServeTensorTypes.Int32 => OrtValue.CreateTensorValueFromMemory((int[])tensor.Data, shape),
            ServeTensorTypes.Int8 => OrtValue.CreateTensorValueFromMemory((sbyte[])tensor.Data, shape),
            ServeTensorTypes.UInt8 => OrtValue.CreateTensorValueFromMemory((byte[])tensor.Data, shape),
            ServeTensorTypes.Bool => OrtValue.CreateTensorValueFromMemory((bool[])tensor.Data, shape),
            _ => throw BadInputs($"Input '{tensor.Name}' has unsupported type '{tensor.DType}'."),
        };
    }

    public static ServeTensor FromOrtValue(string name, OrtValue value)
    {
        if (value.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
        {
            throw new NotSupportedException($"Output '{name}' is a {value.OnnxType}; resident serving returns tensors only.");
        }

        OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
        string dtype = ToDType(info.ElementDataType)
            ?? throw new NotSupportedException($"Output '{name}' has unsupported element type {info.ElementDataType}.");
        Array data = info.ElementDataType switch
        {
            TensorElementType.Float => value.GetTensorDataAsSpan<float>().ToArray(),
            TensorElementType.Float16 => MemoryMarshal.Cast<Float16, Half>(
                value.GetTensorDataAsSpan<Float16>()).ToArray(),
            TensorElementType.Int64 => value.GetTensorDataAsSpan<long>().ToArray(),
            TensorElementType.Int32 => value.GetTensorDataAsSpan<int>().ToArray(),
            TensorElementType.Int8 => value.GetTensorDataAsSpan<sbyte>().ToArray(),
            TensorElementType.UInt8 => value.GetTensorDataAsSpan<byte>().ToArray(),
            TensorElementType.Bool => value.GetTensorDataAsSpan<bool>().ToArray(),
            _ => throw new NotSupportedException($"Output '{name}' has unsupported element type {info.ElementDataType}."),
        };
        return new ServeTensor(name, dtype, info.Shape, data);
    }

    public static Type? ClrElementType(string dtype) => dtype switch
    {
        ServeTensorTypes.Float32 => typeof(float),
        ServeTensorTypes.Float16 => typeof(Half),
        ServeTensorTypes.Int64 => typeof(long),
        ServeTensorTypes.Int32 => typeof(int),
        ServeTensorTypes.Int8 => typeof(sbyte),
        ServeTensorTypes.UInt8 => typeof(byte),
        ServeTensorTypes.Bool => typeof(bool),
        _ => null,
    };

    private static string? ToDType(TensorElementType type) => type switch
    {
        TensorElementType.Float => ServeTensorTypes.Float32,
        TensorElementType.Float16 => ServeTensorTypes.Float16,
        TensorElementType.Int64 => ServeTensorTypes.Int64,
        TensorElementType.Int32 => ServeTensorTypes.Int32,
        TensorElementType.Int8 => ServeTensorTypes.Int8,
        TensorElementType.UInt8 => ServeTensorTypes.UInt8,
        TensorElementType.Bool => ServeTensorTypes.Bool,
        _ => null,
    };

    private static OrtValue ToFloat16OrtValue(Half[] halves, long[] shape)
    {
        // OrtValue requires a Float16[] that stays alive for the tensor's lifetime;
        // reinterpret the Half bits (identical 16-bit layout) with a single copy.
        var float16 = new Float16[halves.Length];
        MemoryMarshal.Cast<Half, Float16>(halves).CopyTo(float16);
        return OrtValue.CreateTensorValueFromMemory(float16, shape);
    }

    private static string FormatDimension(int dimension) => dimension < 0 ? "dynamic (>= 0)" : dimension.ToString();

    private static ResidentInferenceException BadInputs(string message) =>
        new(ResidentInferenceErrorReasons.BadInputs, message);
}
