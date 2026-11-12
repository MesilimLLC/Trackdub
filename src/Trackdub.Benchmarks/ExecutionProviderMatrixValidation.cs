namespace Trackdub.Benchmarks;

internal static class ExecutionProviderMatrixValidation
{
    private static readonly string[] ValidProviders =
        ["cpu", "dml", "directml", "cuda", "tensorrt", "trt", "tensorrtrtx", "trt-rtx", "openvino", "migraphx", "rocm", "coreml", "nnapi", "xnnpack", "snpe", "qnn"];

    internal static bool TryValidateProviders(
        IReadOnlyList<string> providers, string baselineProvider, TextWriter error)
    {
        foreach (string p in providers)
        {
            if (!ValidProviders.Contains(BenchmarkComparison.NormalizeProvider(p), StringComparer.OrdinalIgnoreCase) &&
                !ValidProviders.Contains(p, StringComparer.OrdinalIgnoreCase))
            {
                error.WriteLine($"Unknown provider '{p}'. Expected one of: cpu, directml, cuda, tensorrt, openvino, etc.");
                return false;
            }
        }

        if (!providers.Any(p => BenchmarkComparison.NormalizeProvider(p).Equals(
                BenchmarkComparison.NormalizeProvider(baselineProvider), StringComparison.OrdinalIgnoreCase)))
        {
            error.WriteLine($"Baseline provider '{baselineProvider}' must be one of --providers ({string.Join(", ", providers)}).");
            return false;
        }

        return true;
    }
}
