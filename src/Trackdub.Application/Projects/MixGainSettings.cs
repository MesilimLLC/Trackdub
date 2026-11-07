namespace Trackdub.Application.Projects;

/// <summary>
/// UI-framework-neutral mix gain state shared by preview mix and export stages.
/// </summary>
public sealed record MixGainSettings(
    double SourceGainDb = 0d,
    double DubbedSpeechGainDb = 0d,
    double? DuckingGainDb = null,
    bool DuckingGainExplicit = false,
    bool RestoreOriginalPan = false,
    bool ApplyTimbrePolish = true)
{
    public double BackgroundMediaGainDb { get; init; }

    public ProjectMixSettings ToProjectMixSettings() =>
        new(
            SourceGainDb: NormalizeGainDb(SourceGainDb, 0d),
            DubbedSpeechGainDb: NormalizeGainDb(DubbedSpeechGainDb, 0d),
            DuckingGainDb: DuckingGainExplicit ? NormalizeGainDb(DuckingGainDb ?? 0d, 0d) : null,
            DuckingGainExplicit: DuckingGainExplicit,
            RestoreOriginalPan: RestoreOriginalPan,
            ApplyTimbrePolish: ApplyTimbrePolish)
        {
            BackgroundMediaGainDb = NormalizeGainDb(BackgroundMediaGainDb, 0d)
        };

    public static MixGainSettings FromProjectMixSettings(ProjectMixSettings? settings)
    {
        if (settings is null)
        {
            return new MixGainSettings();
        }

        ProjectMixSettings normalized = settings.Normalize();
        return new MixGainSettings(
            normalized.SourceGainDb,
            normalized.DubbedSpeechGainDb,
            normalized.DuckingGainDb,
            normalized.DuckingGainExplicit,
            normalized.RestoreOriginalPan,
            normalized.ApplyTimbrePolish)
        {
            BackgroundMediaGainDb = normalized.BackgroundMediaGainDb
        };
    }

    public static double NormalizeGainDb(double gainDb, double fallback) =>
        double.IsFinite(gainDb) ? Math.Clamp(gainDb, -96d, 24d) : fallback;
}
