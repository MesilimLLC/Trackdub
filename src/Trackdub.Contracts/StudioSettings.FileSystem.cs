namespace Trackdub.Contracts;

public sealed partial record StudioSettings
{
    /// <summary>Optional parent directory for newly created project folders.</summary>
    public string? DefaultProjectDirectory { get; init; }

    /// <summary>Optional initial directory for media, subtitle, and transcript export pickers.</summary>
    public string? DefaultExportDirectory { get; init; }

    /// <summary>Optional default cache root for model downloads. Existing model files remain in place.</summary>
    public string? DefaultModelCacheDirectory { get; init; }
}
