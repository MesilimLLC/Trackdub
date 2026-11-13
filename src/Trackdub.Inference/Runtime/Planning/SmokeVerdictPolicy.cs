using Trackdub.Domain;

namespace Trackdub.Inference.Runtime.Planning;

/// <summary>Provider classification used when persisted smoke verdicts lack Windows ML policy context.</summary>
public static class SmokeVerdictPolicy
{
    public static bool IsWindowsMlPolicyDependent(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.Qnn
            or ExecutionProviderKind.VitisAi
            or ExecutionProviderKind.OpenVinoCatalog
            or ExecutionProviderKind.Migraphx;
}
