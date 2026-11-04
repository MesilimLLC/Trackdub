using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.ExecutionProviders;

public sealed class PortableExecutionProviderBootstrapper : IExecutionProviderBootstrapper
{
    private readonly ITensorRtRtxProviderBootstrap? _tensorRtRtxBootstrap;

    public PortableExecutionProviderBootstrapper()
        : this(null)
    {
    }

    public PortableExecutionProviderBootstrapper(ITensorRtRtxProviderBootstrap? tensorRtRtxBootstrap)
    {
        _tensorRtRtxBootstrap = tensorRtRtxBootstrap;
    }

    public async Task<ExecutionProviderBootstrapResult> BootstrapAsync(
        ExecutionProviderKind provider,
        bool allowDownloads,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (provider is ExecutionProviderKind.TensorRTRtx && CanBootstrapTensorRtRtx())
        {
            TensorRtRtxBootstrapResult bootstrap = await _tensorRtRtxBootstrap!
                .EnsureRegisteredAsync(allowDownloads, cancellationToken)
                .ConfigureAwait(false);
            return CreateTensorRtRtxResult(provider, bootstrap);
        }

        return CreateResult(provider);
    }

    public async Task<ExecutionProviderBootstrapResult> CheckReadinessAsync(
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (provider is ExecutionProviderKind.TensorRTRtx && CanBootstrapTensorRtRtx())
        {
            TensorRtRtxBootstrapResult bootstrap = await _tensorRtRtxBootstrap!
                .EnsureRegisteredAsync(allowProviderDownloads: false, cancellationToken)
                .ConfigureAwait(false);
            return CreateTensorRtRtxResult(provider, bootstrap);
        }

        return CreateResult(provider);
    }

    private bool CanBootstrapTensorRtRtx() =>
        _tensorRtRtxBootstrap is not null
        && (OperatingSystem.IsWindows() || OperatingSystem.IsLinux());

    private static ExecutionProviderBootstrapResult CreateTensorRtRtxResult(
        ExecutionProviderKind provider,
        TensorRtRtxBootstrapResult bootstrap) =>
        bootstrap.Succeeded
            ? new(provider, provider, Succeeded: true, Detail: bootstrap.Detail)
            : new(provider, ExecutionProviderKind.Cpu, Succeeded: false,
                Detail: bootstrap.Detail,
                FailureReason: $"TensorRT RTX bootstrap failed: {bootstrap.Detail}");

    private static ExecutionProviderBootstrapResult CreateResult(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.Cpu =>
                new(provider, provider, Succeeded: true,
                    Detail: "CPU provider is available in the portable ONNX runtime build."),

            ExecutionProviderKind.DirectMl or
            ExecutionProviderKind.Dnnl or
            ExecutionProviderKind.TensorRTRtx or
            ExecutionProviderKind.OpenVino or
            ExecutionProviderKind.OpenVinoCatalog or
            ExecutionProviderKind.CoreMl or
            ExecutionProviderKind.Cuda or
            ExecutionProviderKind.TensorRt or
            ExecutionProviderKind.Migraphx or
            ExecutionProviderKind.Qnn or
            ExecutionProviderKind.VitisAi =>
                new(provider, ExecutionProviderKind.Cpu, Succeeded: false,
                    Detail: $"{provider} is not available in the portable net11.0 build. Falling back to CPU.",
                    FailureReason: $"{provider} unavailable in portable build; CPU fallback activated."),

            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported execution provider kind."),
        };
}
