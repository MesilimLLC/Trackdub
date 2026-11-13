using Trackdub.Domain;
using System.Text.Json.Serialization;

namespace Trackdub.Inference.Runtime.Planning;

public sealed record StageRuntimePlan
{
    public RuntimeStage Stage { get; init; }

    public StageRuntimePlanStatus Status { get; init; }

    public string? ModelId { get; init; }

    public string? ModelAlias { get; init; }

    public string? EngineFamily { get; init; }

    public string? ModelTier { get; init; }

    public string? Variant { get; init; }

    public ExecutionProviderKind? ExecutionProvider { get; init; }

    /// <summary>True when the caller selected a stage provider instead of Auto.</summary>
    public bool HasExplicitProviderPreference { get; init; }

    /// <summary>
    /// When <see langword="true"/>, the caller required the preferred EP (hard pin).
    /// Session factories must not soft-retry TensorRT RTX init onto DirectML/CPU.
    /// </summary>
    public bool RequirePreferredExecutionProvider { get; init; }

    public RuntimeModelIntegrityStatus ModelIntegrityStatus { get; init; }

    [JsonIgnore]
    public string? ModelEntryPath { get; init; }

    [JsonIgnore]
    public string? ModelRootPath { get; init; }

    [JsonIgnore]
    public string? ModelEntryRelativePath { get; init; }

    [JsonIgnore]
    public IReadOnlyList<string> RequiredModelRelativePaths { get; init; } = [];

    [JsonIgnore]
    public bool IsLocalOptimizedVariant { get; init; }

    /// <summary>
    /// Resolved model revision/content digest (manifest sha256) captured during planning;
    /// feeds residency identity so a same-path weights replacement cannot reuse stale
    /// state. This is an identity value, not an integrity verdict — it does not imply the
    /// payload was verified; <see cref="ModelIntegrityStatus"/> remains the source of
    /// truth (it may be <see cref="RuntimeModelIntegrityStatus.Skipped"/>). Null on
    /// non-runnable plans.
    /// </summary>
    [JsonIgnore]
    public string? ModelRevisionHash { get; init; }

    /// <summary>
    /// Planner-ranked device for this plan's execution provider: the lowest-index
    /// non-excluded device advertising the provider (CPU plans always name the CPU
    /// entry). Drives device-fallback exclusion routing, not session placement —
    /// pooled session creation keys on <c>deviceId: null</c> and binds provider device 0
    /// (CUDA/DirectML), so on same-provider multi-GPU hosts a fallback replan can still
    /// bind the failed physical device. Null when planning ran without a device list.
    /// </summary>
    public int? DeviceIndex { get; init; }

    /// <summary>
    /// Human-readable adapter for <see cref="DeviceIndex"/>; used in degradation reports.
    /// </summary>
    public string? DeviceAdapterDescription { get; init; }

    public RuntimePlanFallback? Fallback { get; init; }

    public IReadOnlyList<RuntimePlanWarning> Warnings { get; init; } = [];
}

public sealed record RuntimePlanFallback(
    RuntimePlanFallbackCode Code,
    string? Detail = null);

public sealed record RuntimePlanWarning(
    RuntimePlanWarningCode Code,
    string? Detail = null);

public sealed record HardwareProfile(
    string OperatingSystem,
    string Architecture,
    bool HasGpu,
    string? GpuDescription = null,
    IReadOnlyList<DeviceEntry>? Devices = null,
    string? CpuName = null,
    long TotalRamMb = 0,
    long DedicatedVramMb = 0,
    NvidiaGpuArchitectureBucket NvidiaGpuArchitecture = NvidiaGpuArchitectureBucket.Unknown,
    string? GpuDriverVersion = null,
    string? OrtRuntimeVersion = null);

/// <param name="EngineFamilies">
/// When set, only these engine families may run on the provider: it is reachable only through a
/// host that runs those engines (for example the ORT 1.30 inference worker, ADR-0017).
/// </param>
public sealed record ExecutionProviderAvailability(
    ExecutionProviderKind Provider,
    bool IsAvailable,
    string? Detail = null,
    IReadOnlyList<string>? EngineFamilies = null);

public sealed record ExecutionProviderSmokeTestRequest(
    RuntimeStage Stage,
    string ModelId,
    string ModelAlias,
    string? EngineFamily,
    string Variant,
    ExecutionProviderKind ExecutionProvider,
    string ModelRootPath,
    string EntryPath,
    string? ModelRevisionHash = null)
{
    /// <summary>Use explicit catalog selection when the stage has a provider override.</summary>
    public bool HasExplicitProviderPreference { get; init; }

    /// <summary>Set by the smoke tester only when a Windows ML policy can select another provider.</summary>
    public bool AllowPolicyProviderSelection { get; init; }

    /// <summary>
    /// Set only by optimization-only warmup callers: a GenAI smoke must then never evict
    /// a resident model (or wait for eviction headroom) to make room. When loading the
    /// target model would require eviction, the smoke reports
    /// <see cref="ExecutionProviderSmokeTestResult.NotAttempted"/> instead of loading it.
    /// Planner verification leaves this false — its verdict is authoritative, so it uses
    /// the regular evicting lease.
    /// </summary>
    public bool NonEvictingWarmup { get; init; }
}

public sealed record ExecutionProviderSmokeTestResult(
    bool Passed,
    string? Detail = null)
{
    /// <summary>
    /// True when the smoke honestly did not run (a non-evicting warmup skipped because
    /// loading would have evicted a resident model). Distinct from a failed smoke:
    /// nothing was proven or disproven about the provider/model pair.
    /// </summary>
    public bool NotAttempted { get; init; }
}
