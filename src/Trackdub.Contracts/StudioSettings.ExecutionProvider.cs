using Trackdub.Domain;

namespace Trackdub.Contracts;

public sealed partial record StudioSettings
{
    /// <summary>
    /// Soft default execution-provider preference applied to runtime stages unless a stage-specific
    /// override is configured. Null lets the runtime planner choose automatically.
    /// </summary>
    public ExecutionProviderKind? PreferredExecutionProvider { get; init; }
}
