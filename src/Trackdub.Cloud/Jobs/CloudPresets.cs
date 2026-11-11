namespace Trackdub.Cloud.Jobs;

/// <summary>A cloud-approved CLI preset together with the fixed target language and speaker voice it runs with.</summary>
public sealed record CloudPreset(string Name, string TargetLanguage, string Voice);

/// <summary>Approved cloud presets. Clients choose a name from this list; they never supply commands, voices or model IDs.</summary>
public static class CloudPresets
{
    public const string SpanishName = "poc-server-v1";
    public const string ChineseName = "poc-server-zh-v1";

    public static IReadOnlyList<CloudPreset> All { get; } =
    [
        new(SpanishName, "es", "SPEAKER_00:ef_dora"),
        new(ChineseName, "zh", "SPEAKER_00:qwen3:vivian"),
    ];

    public static bool TryGet(string? name, string? targetLanguage, out CloudPreset preset)
    {
        foreach (var candidate in All)
        {
            if (string.Equals(candidate.Name, name, StringComparison.Ordinal)
                && string.Equals(candidate.TargetLanguage, targetLanguage, StringComparison.Ordinal))
            {
                preset = candidate;
                return true;
            }
        }

        preset = null!;
        return false;
    }
}

/// <summary>Input limits shared by upload reservation and media validation.</summary>
public static class CloudLimits
{
    public const long MaxUploadBytes = 1024L * 1024 * 1024;
    public const double MaxDurationSeconds = 600;
}
