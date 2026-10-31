using System.Text.Json;
using System.Text.Json.Serialization;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Infrastructure.Persistence.Repositories;

namespace Trackdub.Infrastructure.Settings;

public sealed class JsonStudioSettingsService(
    IAppStoragePaths storagePaths,
    IApplicationLogger? logger = null) : IStudioSettingsService
{
    private const int RecentProjectLimit = 10;
    private readonly IApplicationLogger? logger = logger;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        Converters =
        {
            new TolerantAsrModelOverrideJsonConverter(),
            new TolerantTranslationModelOverrideJsonConverter(),
            new TolerantTtsModelOverrideJsonConverter(),
            new TolerantSeparationModelOverrideJsonConverter(),
            new TolerantWindowsMlExecutionDevicePolicyJsonConverter()
        }
    };

    public async Task<StudioSettings> LoadAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(storagePaths.SettingsPath))
        {
            return StudioSettings.Default;
        }

        try
        {
            await using FileStream stream = File.OpenRead(storagePaths.SettingsPath);
            StudioSettings? settings = await JsonSerializer.DeserializeAsync<StudioSettings>(
                stream,
                JsonOptions,
                cancellationToken).ConfigureAwait(false);
            return Normalize(settings ?? StudioSettings.Default);
        }
        catch (JsonException ex)
        {
            ArchiveCorruptSettingsFile(ex);
            return Normalize(StudioSettings.Default);
        }
    }

    public async Task SaveAsync(StudioSettings settings, CancellationToken cancellationToken)
    {
        StudioSettings normalized = Normalize(settings);
        normalized = normalized with
        {
            DefaultModelCacheDirectory = ResolveSavedModelCacheDirectory(normalized.DefaultModelCacheDirectory),
        };
        // Hold a settings lock across the migration and the publish so concurrent saves cannot
        // interleave: each publishes its own content after its matching migration.
        using FileStream saveLock = await LocalModelCacheRecordStore.AcquireFileLockAsync(
            storagePaths.SettingsPath + ".lock", cancellationToken).ConfigureAwait(false);
        Directory.CreateDirectory(storagePaths.RootDirectory);

        string tempPath = $"{storagePaths.SettingsPath}.{Guid.NewGuid():N}.tmp";
        try
        {
            await using (FileStream stream = File.Create(tempPath))
            {
                await JsonSerializer.SerializeAsync(stream, normalized, JsonOptions, cancellationToken).ConfigureAwait(false);
            }

            // Migrate before publishing settings.json so a failed migration does not leave settings
            // that point at a cache directory the records were never moved to.
            TrackdubStoragePathResolver.MigrateModelCacheIfChanged(
                new TrackdubStorageOptions(
                    storagePaths.UserDataRoot,
                    storagePaths.UserCacheRoot,
                    storagePaths.SharedAssetRoot,
                    storagePaths.IsPortable,
                    normalized.DefaultModelCacheDirectory),
                cancellationToken);
            File.Move(tempPath, storagePaths.SettingsPath, overwrite: true);
        }
        finally
        {
            // After a successful publish the temp path no longer exists, so this only removes
            // the file left behind by a failed save.
            File.Delete(tempPath);
        }
    }

    // The saved value is persisted in its absolute form, so the settings and the cache resolver
    // read the same directory. A relative value would resolve against the working directory and
    // is treated as unset.
    private static string? ResolveSavedModelCacheDirectory(string? configured)
    {
        return TrackdubStoragePathResolver.NormalizeSavedModelCacheDirectory(configured);
    }

    public async Task<StudioSettings> TouchRecentProjectAsync(
        string projectPath,
        string projectName,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectName);

        StudioSettings current = await LoadAsync(cancellationToken).ConfigureAwait(false);
        string normalizedPath = Path.GetFullPath(projectPath);
        DateTimeOffset now = DateTimeOffset.UtcNow;
        RecentProjectEntry entry = new(projectName.Trim(), normalizedPath, now);

        RecentProjectEntry[] updatedRecentProjects =
            [entry, .. current.RecentProjects
                .Where(candidate => !string.Equals(candidate.ProjectPath, normalizedPath, StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(candidate => candidate.LastOpenedAtUtc)
                .Take(RecentProjectLimit - 1)];

        StudioSettings updated = Normalize(current with { RecentProjects = updatedRecentProjects });
        await SaveAsync(updated, cancellationToken).ConfigureAwait(false);
        return updated;
    }

    private static StudioSettings Normalize(StudioSettings settings)
    {
        string? defaultSourceLanguage = NormalizeLanguageCode(settings.DefaultSourceLanguage);
        string? defaultTargetLanguage = NormalizeLanguageCode(settings.DefaultTargetLanguage);
        string modelTierPreference = string.IsNullOrWhiteSpace(settings.ModelTierPreference)
            ? "balanced"
            : settings.ModelTierPreference.Trim().ToLowerInvariant();

        RecentProjectEntry[] recentProjects = settings.RecentProjects
            .Where(entry => !string.IsNullOrWhiteSpace(entry.ProjectPath) && !string.IsNullOrWhiteSpace(entry.ProjectName))
            .Select(entry => new RecentProjectEntry(
                entry.ProjectName.Trim(),
                Path.GetFullPath(entry.ProjectPath),
                entry.LastOpenedAtUtc))
            .OrderByDescending(entry => entry.LastOpenedAtUtc)
            .DistinctBy(entry => entry.ProjectPath, StringComparer.OrdinalIgnoreCase)
            .Take(RecentProjectLimit)
            .ToArray();

        return settings with
        {
            DefaultSourceLanguage = defaultSourceLanguage ?? StudioSettings.Default.DefaultSourceLanguage,
            DefaultTargetLanguage = defaultTargetLanguage ?? StudioSettings.Default.DefaultTargetLanguage,
            ModelTierPreference = modelTierPreference,
            WindowLayout = NormalizeWindowLayout(settings.WindowLayout ?? StudioSettings.Default.WindowLayout),
            RecentProjects = recentProjects,
            TtsTiming = NormalizeTtsTiming(settings.TtsTiming),
            TtsMaxConcurrency = NormalizeTtsMaxConcurrency(settings.TtsMaxConcurrency),
            TranscriptConfidenceThreshold = NormalizeConfidenceThreshold(settings.TranscriptConfidenceThreshold),
            AsrModelOverride = NormalizeAsrModelOverride(settings.AsrModelOverride),
            TranslationModelOverride = NormalizeTranslationModelOverride(settings.TranslationModelOverride),
            TtsModelOverride = NormalizeTtsModelOverride(settings.TtsModelOverride),
            SeparationModelOverride = NormalizeSeparationModelOverride(settings.SeparationModelOverride),
            Export = NormalizeExportSettings(settings.Export),
            Playback = NormalizePlaybackSettings(settings.Playback),
            ModelVariantOverrides = NormalizeModelVariantOverrides(settings.ModelVariantOverrides),
            StageModelAliases = NormalizeStageModelAliases(settings.StageModelAliases),
            AppliedStarterPackId = NormalizeOptionalKey(settings.AppliedStarterPackId),
            AppliedStarterPackProfileId = NormalizeOptionalKey(settings.AppliedStarterPackProfileId),
            ModelOptimizationPrecisionOverrides = NormalizePrecisionOverrides(settings.ModelOptimizationPrecisionOverrides),
            AllowNativeCudaTensorRtOnWindows = settings.AllowNativeCudaTensorRtOnWindows,
            WindowsMlExecutionDevicePolicy = NormalizeWindowsMlExecutionDevicePolicy(settings.WindowsMlExecutionDevicePolicy),
            TensorRtRtxPluginDirectory = NormalizeOptionalPath(settings.TensorRtRtxPluginDirectory),
            HardwareQualityPresetOverrideKey = NormalizeOptionalKey(settings.HardwareQualityPresetOverrideKey),
            HardwareProfilerEvidenceId = NormalizeOptionalKey(settings.HardwareProfilerEvidenceId),
            HardwareProfilerFingerprint = NormalizeOptionalKey(settings.HardwareProfilerFingerprint),
            ThemeName = AppThemeNames.Normalize(settings.ThemeName),
            CloudEgressConsents = NormalizeCloudEgressConsents(settings.CloudEgressConsents)
        };
    }

    private static IReadOnlyDictionary<string, bool> NormalizeCloudEgressConsents(
        IReadOnlyDictionary<string, bool>? consents)
    {
        if (consents is null || consents.Count == 0)
        {
            return new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);
        }

        var normalized = new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);
        foreach ((string key, bool value) in consents)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                continue;
            }

            normalized[key.Trim().ToLowerInvariant()] = value;
        }

        return normalized;
    }

    private static string? NormalizeOptionalKey(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value.Trim();

    private static string? NormalizeOptionalPath(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value.Trim();

    private static WindowsMlExecutionDevicePolicy NormalizeWindowsMlExecutionDevicePolicy(
        WindowsMlExecutionDevicePolicy policy) =>
        Enum.IsDefined(policy) ? policy : WindowsMlExecutionDevicePolicy.Explicit;

    private static IReadOnlyDictionary<string, string> NormalizeStageModelAliases(
        IReadOnlyDictionary<string, string>? aliases)
    {
        if (aliases is null || aliases.Count == 0)
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        var normalized = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach ((string key, string value) in aliases)
        {
            if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            normalized[ModelVariantOverrideKeys.NormalizeToken(key)] = value.Trim();
        }

        return normalized;
    }

    private static IReadOnlyDictionary<string, string> NormalizeModelVariantOverrides(
        IReadOnlyDictionary<string, string>? overrides)
    {
        if (overrides is null || overrides.Count == 0)
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        var normalized = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach ((string key, string value) in overrides)
        {
            if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            string normalizedValue = value.Trim();
            if (ModelVariantOverrideKeys.TryParse(key, out string stageKey, out string modelAlias))
            {
                normalized[ModelVariantOverrideKeys.Build(stageKey, modelAlias)] = normalizedValue;
                continue;
            }

            // Keep legacy stage-only keys normalized for backward compatibility.
            normalized[ModelVariantOverrideKeys.NormalizeToken(key)] = normalizedValue;
        }

        return normalized;
    }

    private static IReadOnlyDictionary<string, string> NormalizePrecisionOverrides(
        IReadOnlyDictionary<string, string>? overrides)
    {
        if (overrides is null || overrides.Count == 0)
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        var normalized = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach ((string key, string value) in overrides)
        {
            if (!ModelVariantOverrideKeys.TryParse(key, out string stageKey, out string modelAlias))
            {
                continue;
            }

            if (string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            normalized[ModelVariantOverrideKeys.Build(stageKey, modelAlias)] = value.Trim().ToLowerInvariant();
        }

        return normalized;
    }

    private static WindowLayoutSettings NormalizeWindowLayout(WindowLayoutSettings windowLayout)
    {
        WindowPanelLayoutSettings? panelLayout = windowLayout.PanelLayout is { } panel
            ? new WindowPanelLayoutSettings(
                NormalizePanelLength(panel.LeftPanelWidth, allowZero: true),
                NormalizePanelLength(panel.RightPanelWidth, allowZero: true),
                NormalizePanelLength(panel.CenterTopPanelHeight),
                NormalizePanelLength(panel.CenterBottomPanelHeight))
            : null;

        return windowLayout with { PanelLayout = panelLayout };
    }

    private static double? NormalizePanelLength(double? length, bool allowZero = false)
    {
        return (allowZero
                ? length is >= 0d and < double.MaxValue
                : length is > 0d and < double.MaxValue)
            ? length
            : null;
    }

    private static TtsTimingSettings NormalizeTtsTiming(TtsTimingSettings? settings)
    {
        settings ??= TtsTimingSettings.Default;
        double threshold = double.IsFinite(settings.RubberbandStretchThreshold) &&
                           settings.RubberbandStretchThreshold is >= 0d and <= 1d
            ? settings.RubberbandStretchThreshold
            : TtsTimingSettings.Default.RubberbandStretchThreshold;
        double? autoStretchMaxOverrun = settings.AutoStretchMaxOverrun is double autoStretch &&
                                        double.IsFinite(autoStretch) &&
                                        autoStretch is >= 0d and <= 1d
            ? autoStretch
            : null;
        return settings with
        {
            RubberbandStretchThreshold = threshold,
            AutoStretchMaxOverrun = autoStretchMaxOverrun
        };
    }

    private static int? NormalizeTtsMaxConcurrency(int? configured) =>
        configured is int value && value > 0 ? Math.Min(value, 8) : null;

    private static double NormalizeConfidenceThreshold(double threshold) =>
        double.IsFinite(threshold) && threshold is >= 0d and <= 1d
            ? threshold
            : StudioSettings.DefaultTranscriptConfidenceThreshold;

    private static AsrModelOverride NormalizeAsrModelOverride(AsrModelOverride modelOverride) =>
        AsrModelOverrideSettings.IsSupported(modelOverride)
            ? modelOverride
            : StudioSettings.Default.AsrModelOverride;

    private static TranslationModelOverride NormalizeTranslationModelOverride(TranslationModelOverride modelOverride) =>
        modelOverride is TranslationModelOverride.Auto
            or TranslationModelOverride.Madlad
            or TranslationModelOverride.DeepL
            or TranslationModelOverride.OpenAiGpt
            or TranslationModelOverride.GeminiTranslation
            ? modelOverride
            : TranslationModelOverride.Auto;

    private static TtsModelOverride NormalizeTtsModelOverride(TtsModelOverride modelOverride) =>
        modelOverride is TtsModelOverride.Auto
            or TtsModelOverride.Kokoro
            or TtsModelOverride.Chatterbox
            or TtsModelOverride.CosyVoice
            or TtsModelOverride.ElevenLabs
            or TtsModelOverride.OpenAiTts
            or TtsModelOverride.GoogleTts
            ? modelOverride
            : TtsModelOverride.Auto;

    private static SeparationModelOverride NormalizeSeparationModelOverride(SeparationModelOverride modelOverride) =>
        modelOverride switch
        {
            SeparationModelOverride.Spleeter => modelOverride,
            SeparationModelOverride.Auto => SeparationModelOverride.Auto,
            _ => SeparationModelOverride.Auto
        };

    private static StudioExportSettings NormalizeExportSettings(StudioExportSettings? settings)
    {
        settings ??= StudioExportSettings.Default;
        string container = string.Equals(settings.Container, StudioExportSettings.MkvContainer, StringComparison.OrdinalIgnoreCase)
            ? StudioExportSettings.MkvContainer
            : StudioExportSettings.Mp4Container;
        string subtitleSource = settings.SubtitleSource?.Trim().ToLowerInvariant() switch
        {
            StudioExportSettings.TranscriptSubtitleSource => StudioExportSettings.TranscriptSubtitleSource,
            StudioExportSettings.BilingualSubtitleSource => StudioExportSettings.BilingualSubtitleSource,
            _ => StudioExportSettings.TranslatedSubtitleSource
        };
        string videoEncoder = VideoEncoderPreferenceSettings.FromKey(settings.VideoEncoder) switch
        {
            VideoEncoderPreference.Software => VideoEncoderPreferenceSettings.SoftwareKey,
            VideoEncoderPreference.Nvenc => VideoEncoderPreferenceSettings.NvencKey,
            VideoEncoderPreference.Qsv => VideoEncoderPreferenceSettings.QsvKey,
            VideoEncoderPreference.Amf => VideoEncoderPreferenceSettings.AmfKey,
            VideoEncoderPreference.VideoToolbox => VideoEncoderPreferenceSettings.VideoToolboxKey,
            VideoEncoderPreference.Vaapi => VideoEncoderPreferenceSettings.VaapiKey,
            _ => VideoEncoderPreferenceSettings.AutoKey
        };

        return settings with
        {
            TargetLufs = ExportLoudnessTargets.NormalizeTargetLufs(settings.TargetLufs),
            Container = container,
            SubtitleSource = subtitleSource,
            VideoEncoder = videoEncoder
        };
    }

    private static StudioPlaybackSettings NormalizePlaybackSettings(StudioPlaybackSettings? settings)
    {
        settings ??= StudioPlaybackSettings.Default;
        string mode = settings.SubtitleContentMode?.Trim().ToLowerInvariant() switch
        {
            StudioPlaybackSettings.SourceSubtitleContentMode => StudioPlaybackSettings.SourceSubtitleContentMode,
            StudioPlaybackSettings.TranslatedSubtitleContentMode => StudioPlaybackSettings.TranslatedSubtitleContentMode,
            StudioPlaybackSettings.BilingualSubtitleContentMode => StudioPlaybackSettings.BilingualSubtitleContentMode,
            _ => StudioPlaybackSettings.TranslatedSubtitleContentMode
        };

        bool validInputMode = string.Equals(mode, settings.SubtitleContentMode?.Trim().ToLowerInvariant(), StringComparison.Ordinal);
        string videoDecode = PlaybackVideoDecodePreferenceSettings.FromKey(settings.VideoDecode) switch
        {
            PlaybackVideoDecodePreference.Software => PlaybackVideoDecodePreferenceSettings.SoftwareKey,
            _ => PlaybackVideoDecodePreferenceSettings.AutoKey
        };

        return new StudioPlaybackSettings(
            SubtitlesEnabled: settings.SubtitlesEnabled && validInputMode,
            SubtitleContentMode: mode,
            VideoDecode: videoDecode,
            KaraokeEnabled: settings.KaraokeEnabled);
    }

    private void ArchiveCorruptSettingsFile(Exception exception)
    {
        string settingsPath = storagePaths.SettingsPath;

        try
        {
            if (!File.Exists(settingsPath))
            {
                logger?.LogWarning(
                    $"Studio settings at '{settingsPath}' could not be parsed; starting with defaults.",
                    exception);
                return;
            }

            string directory = Path.GetDirectoryName(settingsPath) ?? storagePaths.RootDirectory;
            Directory.CreateDirectory(directory);
            string timestamp = DateTimeOffset.UtcNow.ToString("yyyyMMddHHmmss");
            string backupPath = Path.Join(directory, $"settings.json.{timestamp}.corrupt");
            File.Move(settingsPath, backupPath, overwrite: false);
            logger?.LogWarning(
                $"Studio settings at '{settingsPath}' could not be parsed; starting with defaults. Corrupt file archived to '{backupPath}'.",
                exception);
        }
        catch (Exception archiveEx)
        {
            logger?.LogWarning(
                $"Studio settings at '{settingsPath}' could not be parsed; starting with defaults. Failed to archive corrupt file.",
                archiveEx);
        }
    }

    private static string? NormalizeLanguageCode(string? languageCode) =>
        string.IsNullOrWhiteSpace(languageCode)
            ? null
            : languageCode.Trim().ToLowerInvariant();

    private sealed class TolerantAsrModelOverrideJsonConverter : JsonConverter<AsrModelOverride>
    {
        public override AsrModelOverride Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType is JsonTokenType.String)
            {
                return AsrModelOverrideSettings.FromKey(reader.GetString());
            }

            if (reader.TokenType is JsonTokenType.Number &&
                reader.TryGetInt32(out int numericValue))
            {
                return NormalizeAsrModelOverride((AsrModelOverride)numericValue);
            }

            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
            {
                reader.Skip();
            }

            return StudioSettings.Default.AsrModelOverride;
        }

        public override void Write(
            Utf8JsonWriter writer,
            AsrModelOverride value,
            JsonSerializerOptions options) =>
            writer.WriteStringValue(AsrModelOverrideSettings.ToKey(NormalizeAsrModelOverride(value)));
    }

    private sealed class TolerantTranslationModelOverrideJsonConverter : JsonConverter<TranslationModelOverride>
    {
        public override TranslationModelOverride Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType is JsonTokenType.String)
            {
                return TranslationModelOverrideSettings.FromKey(reader.GetString());
            }

            if (reader.TokenType is JsonTokenType.Number &&
                reader.TryGetInt32(out int numericValue) &&
                Enum.IsDefined(typeof(TranslationModelOverride), numericValue))
            {
                return (TranslationModelOverride)numericValue;
            }

            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
            {
                reader.Skip();
            }

            return TranslationModelOverride.Auto;
        }

        public override void Write(
            Utf8JsonWriter writer,
            TranslationModelOverride value,
            JsonSerializerOptions options) =>
            writer.WriteStringValue(TranslationModelOverrideSettings.ToKey(value));
    }

    private sealed class TolerantTtsModelOverrideJsonConverter : JsonConverter<TtsModelOverride>
    {
        public override TtsModelOverride Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType is JsonTokenType.String)
            {
                return TtsModelOverrideSettings.FromKey(reader.GetString());
            }

            if (reader.TokenType is JsonTokenType.Number &&
                reader.TryGetInt32(out int numericValue) &&
                Enum.IsDefined(typeof(TtsModelOverride), numericValue))
            {
                return (TtsModelOverride)numericValue;
            }

            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
            {
                reader.Skip();
            }

            return TtsModelOverride.Auto;
        }

        public override void Write(
            Utf8JsonWriter writer,
            TtsModelOverride value,
            JsonSerializerOptions options) =>
            writer.WriteStringValue(TtsModelOverrideSettings.ToKey(value));
    }

    private sealed class TolerantSeparationModelOverrideJsonConverter : JsonConverter<SeparationModelOverride>
    {
        public override SeparationModelOverride Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType is JsonTokenType.String)
            {
                return SeparationModelOverrideSettings.FromKey(reader.GetString());
            }

            if (reader.TokenType is JsonTokenType.Number &&
                reader.TryGetInt32(out int numericValue))
            {
                return NormalizeSeparationModelOverride(
                    Enum.IsDefined(typeof(SeparationModelOverride), numericValue)
                        ? (SeparationModelOverride)numericValue
                        : SeparationModelOverride.Auto);
            }

            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
            {
                reader.Skip();
            }

            return SeparationModelOverride.Auto;
        }

        public override void Write(
            Utf8JsonWriter writer,
            SeparationModelOverride value,
            JsonSerializerOptions options) =>
            writer.WriteStringValue(SeparationModelOverrideSettings.ToKey(value));
    }

    private sealed class TolerantWindowsMlExecutionDevicePolicyJsonConverter : JsonConverter<WindowsMlExecutionDevicePolicy>
    {
        public override WindowsMlExecutionDevicePolicy Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType is JsonTokenType.String)
            {
                return WindowsMlExecutionDevicePolicySettings.FromKey(reader.GetString());
            }

            if (reader.TokenType is JsonTokenType.Number &&
                reader.TryGetInt32(out int numericValue) &&
                Enum.IsDefined(typeof(WindowsMlExecutionDevicePolicy), numericValue))
            {
                return (WindowsMlExecutionDevicePolicy)numericValue;
            }

            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
            {
                reader.Skip();
            }

            return WindowsMlExecutionDevicePolicy.Explicit;
        }

        public override void Write(
            Utf8JsonWriter writer,
            WindowsMlExecutionDevicePolicy value,
            JsonSerializerOptions options) =>
            writer.WriteStringValue(WindowsMlExecutionDevicePolicySettings.ToKey(value));
    }
}
