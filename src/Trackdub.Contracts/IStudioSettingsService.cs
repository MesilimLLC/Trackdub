using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;

namespace Trackdub.Contracts;

public interface IStudioSettingsService
{
    Task<StudioSettings> LoadAsync(CancellationToken cancellationToken);

    Task SaveAsync(StudioSettings settings, CancellationToken cancellationToken);

    Task<StudioSettings> TouchRecentProjectAsync(
        string projectPath,
        string projectName,
        CancellationToken cancellationToken);
}

public sealed partial record StudioSettings(
    string? DefaultSourceLanguage,
    string? DefaultTargetLanguage,
    string ModelTierPreference,
    WindowLayoutSettings WindowLayout,
    IReadOnlyList<RecentProjectEntry> RecentProjects,
    TtsTimingSettings? TtsTiming = null,
    double TranscriptConfidenceThreshold = StudioSettings.DefaultTranscriptConfidenceThreshold,
    AsrModelOverride AsrModelOverride = AsrModelOverride.Auto,
    TranslationModelOverride TranslationModelOverride = TranslationModelOverride.Madlad,
    TtsModelOverride TtsModelOverride = TtsModelOverride.Auto,
    SeparationModelOverride SeparationModelOverride = SeparationModelOverride.Auto,
    StudioExportSettings? Export = null,
    StudioPlaybackSettings? Playback = null,
    IReadOnlyDictionary<string, ExecutionProviderKind>? HardwareOverrides = null,
    /// <summary>
    /// When true, <see cref="HardwareOverrides"/> are required for stages that allow the provider
    /// (headless CLI/SDK pins). UI soft preferences leave this false.
    /// </summary>
    bool RequirePreferredExecutionProviders = false,
    IReadOnlyDictionary<string, string>? ModelVariantOverrides = null,
    IReadOnlyDictionary<string, string>? StageModelAliases = null,
    string? AppliedStarterPackId = null,
    string? AppliedStarterPackProfileId = null,
    IReadOnlyDictionary<string, string>? ModelOptimizationPrecisionOverrides = null,
    bool ShowLocalModelsAtStartup = true,
    /// <summary>
    /// When true, the first-run starter pack onboarding modal has been shown or dismissed.
    /// </summary>
    bool StarterPackOnboardingCompleted = false,
    /// <summary>
    /// Advanced native ORT CUDA/TensorRT on Windows. Default off. Not exposed in Settings UI;
    /// kept so JSON / <c>INativeCudaTensorRtWindowsPolicy</c> wiring still functions.
    /// </summary>
    bool AllowNativeCudaTensorRtOnWindows = false,
    WindowsMlExecutionDevicePolicy WindowsMlExecutionDevicePolicy = WindowsMlExecutionDevicePolicy.Explicit,
    string? HardwareQualityPresetOverrideKey = null,
    string? HardwareProfilerEvidenceId = null,
    string? HardwareProfilerFingerprint = null,
    bool AmdRyzenAiLicenseAccepted = false,
    bool NvidiaTensorRtRtxLicenseAccepted = false,
    bool IntelOpenVinoLicenseAccepted = false,
    bool QualcommQnnLicenseAccepted = false,
    /// <summary>
    /// User accepted the NVIDIA AFX / Maxine redistributable license. Required before any
    /// Trackdub-hosted AFX package download once packaging URLs are real.
    /// </summary>
    bool NvidiaAfxLicenseAccepted = false,
    /// <summary>
    /// Optional local AFX runtime root (NVIDIA Maxine installer output or manually staged
    /// package containing NvAudioEffects.dll + models). Overrides ComponentStore install path.
    /// </summary>
    string? NvidiaAfxRuntimeDirectory = null,
    string ThemeName = AppThemeNames.Dark,
    /// <summary>
    /// When true, disables breathing pulse animation on running pipeline stage accent stripes.
    /// Default false (full motion) per Small Polish Pass P2 / Phase 3.
    /// </summary>
    bool ReduceMotion = false,
    bool EnableNvidiaAfx = false,
    NvidiaAfxProfile NvidiaAfxProfile = NvidiaAfxProfile.NoiseAndReverb,
    float NvidiaAfxIntensityRatio = 1.0f,
    string? UiLanguage = null,
    string? TensorRtRtxPluginDirectory = null,
    UpdateChannel UpdateChannelPreference = UpdateChannel.Stable,
    /// <summary>Application-generated default aliases, keyed by stage. Configured stage aliases and
    /// caller preferences are explicit even when they equal these defaults. Legacy settings without
    /// this provenance treat non-Auto overrides as explicit.</summary>
    IReadOnlyDictionary<string, string>? AutomaticModelAliases = null,
    /// <summary>
    /// Configured maximum TTS degree of parallelism (before the per-model VRAM-aware cap
    /// applied in <c>TtsExecutionOptions</c>). Null means the historical default (4).
    /// Not exposed in Settings UI yet; CLI/SDK and settings.json can set it.
    /// </summary>
    int? TtsMaxConcurrency = null,
    /// <summary>
    /// When true, the desktop warms the pipeline's planner-selected models in the background
    /// after startup (IResidentWarmupService), so the first run skips most cold-start cost.
    /// </summary>
    bool WarmModelsAtStartup = true,
    /// <summary>
    /// Persisted user consent for cloud data egress keyed by "{egressType}:{providerKey}".
    /// Null or absent entry means consent has not been granted.
    /// </summary>
    IReadOnlyDictionary<string, bool>? CloudEgressConsents = null,
    bool AutoDownloadTranslationModels = true,
    bool OfflineMode = false,
    /// <summary>
    /// Local NVIDIA Maxine AR SDK root used for video lip repair (contains bin/nvARPose.dll,
    /// bin/models and features). Overrides TRACKDUB_MAXINE_ARSDK_ROOT when set.
    /// </summary>
    string? MaxineLipSyncArSdkRoot = null,
    /// <summary>
    /// Maxine LipSync host executable. Overrides TRACKDUB_MAXINE_LIPSYNC_HOST when set.
    /// </summary>
    string? MaxineLipSyncHostPath = null,
    /// <summary>
    /// LipSync model language: "generic" (default), "es", "de" or "fr". Overrides
    /// TRACKDUB_MAXINE_LIPSYNC_LANGUAGE when set.
    /// </summary>
    string? MaxineLipSyncLanguage = null)
{
    public const double DefaultTranscriptConfidenceThreshold = 0.75d;

    public static StudioSettings Default { get; } = new(
        DefaultSourceLanguage: null,
        DefaultTargetLanguage: null,
        ModelTierPreference: "balanced",
        WindowLayout: new WindowLayoutSettings(null, null, IsMaximized: false),
        RecentProjects: [],
        TtsTiming: TtsTimingSettings.Default,
        TranscriptConfidenceThreshold: DefaultTranscriptConfidenceThreshold,
        AsrModelOverride: AsrModelOverride.Auto,
        TranslationModelOverride: TranslationModelOverride.Madlad,
        TtsModelOverride: TtsModelOverride.Auto,
        SeparationModelOverride: SeparationModelOverride.Auto,
        Export: StudioExportSettings.Default,
        Playback: StudioPlaybackSettings.Default,
        HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
        RequirePreferredExecutionProviders: false,
        ModelVariantOverrides: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        StageModelAliases: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        AppliedStarterPackId: null,
        AppliedStarterPackProfileId: null,
        ModelOptimizationPrecisionOverrides: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        AutomaticModelAliases: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            [Trackdub.Domain.StageRuns.StageNames.Translation] = TranslationModelOverrideSettings.MadladModelAlias
        },
        CloudEgressConsents: new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase));
}

/// <summary>
/// Canonical persisted identifiers for the four shipping UI themes.
/// Strings (not an enum) so that serialized settings stay forward-compatible
/// if additional themes are added later — unknown values normalize to
/// <see cref="Dark"/> via <see cref="Normalize"/>.
/// </summary>
public static class AppThemeNames
{
    public const string Dark = "dark";
    public const string Light = "light";
    public const string Amber = "amber";
    public const string Green = "green";

    public static IReadOnlyList<string> All { get; } = [Dark, Light, Amber, Green];

    public static string Normalize(string? candidate)
    {
        if (string.IsNullOrWhiteSpace(candidate))
        {
            return Dark;
        }

        string trimmed = candidate.Trim().ToLowerInvariant();
        return trimmed switch
        {
            Dark or Light or Amber or Green => trimmed,
            _ => Dark
        };
    }
}

public enum AsrModelOverride
{
    Auto = 0,
    GenAi = 1,
    OnnxRuntime = 2,
    OpenAiWhisper = 3,
    GeminiAsr = 4,
    Nemotron35 = 5
}

public static class AsrModelOverrideSettings
{
    public const string AutoKey = "auto";
    public const string GenAiKey = "genai";
    public const string OnnxRuntimeKey = "onnxruntime";
    public const string Nemotron35Key = "nemotron-3.5";
    public const string OrtKey = OnnxRuntimeKey;
    public const string OrtDisplayName = "ONNX Runtime";
    public const string GenAiModelAlias = "whisper-tiny-genai";
    public const string OnnxRuntimeModelAlias = "whisper-tiny-onnx";
    public const string Nemotron35ModelAlias = "nemotron-3.5-asr";
    public const string OpenAiWhisperKey = "openai-whisper";
    public const string GeminiAsrKey = "gemini-asr";
    public const string OpenAiWhisperCloudAlias = "openai-whisper-cloud";
    public const string GeminiAsrCloudAlias = "gemini-asr-cloud";

    public static string ToKey(AsrModelOverride modelOverride) =>
        modelOverride switch
        {
            AsrModelOverride.GenAi => GenAiKey,
            AsrModelOverride.OnnxRuntime => OnnxRuntimeKey,
            AsrModelOverride.OpenAiWhisper => OpenAiWhisperKey,
            AsrModelOverride.GeminiAsr => GeminiAsrKey,
            AsrModelOverride.Nemotron35 => Nemotron35Key,
            _ => AutoKey
        };

    public static AsrModelOverride FromKey(string? key) =>
        string.IsNullOrWhiteSpace(key)
            ? AsrModelOverride.Auto
            : key.Trim().ToLowerInvariant() switch
            {
                GenAiKey => AsrModelOverride.GenAi,
                OnnxRuntimeKey => AsrModelOverride.OnnxRuntime,
                Nemotron35Key or Nemotron35ModelAlias => AsrModelOverride.Nemotron35,
                OpenAiWhisperKey or OpenAiWhisperCloudAlias => AsrModelOverride.OpenAiWhisper,
                GeminiAsrKey or GeminiAsrCloudAlias => AsrModelOverride.GeminiAsr,
                _ => AsrModelOverride.Auto
            };

    public static string? ResolveModelAlias(AsrModelOverride modelOverride) =>
        modelOverride switch
        {
            AsrModelOverride.GenAi => GenAiModelAlias,
            AsrModelOverride.OnnxRuntime => OnnxRuntimeModelAlias,
            AsrModelOverride.Nemotron35 => Nemotron35ModelAlias,
            AsrModelOverride.OpenAiWhisper => OpenAiWhisperCloudAlias,
            AsrModelOverride.GeminiAsr => GeminiAsrCloudAlias,
            _ => null
        };

    public static bool RequiresModelAlias(AsrModelOverride modelOverride) =>
        modelOverride is AsrModelOverride.GenAi
            or AsrModelOverride.OnnxRuntime
            or AsrModelOverride.Nemotron35
            or AsrModelOverride.OpenAiWhisper
            or AsrModelOverride.GeminiAsr;

    public static bool IsSupported(AsrModelOverride modelOverride) =>
        modelOverride is AsrModelOverride.Auto
            or AsrModelOverride.GenAi
            or AsrModelOverride.OnnxRuntime
            or AsrModelOverride.Nemotron35
            or AsrModelOverride.OpenAiWhisper
            or AsrModelOverride.GeminiAsr;

    public static bool IsOpenAiWhisperAlias(string? modelAlias) =>
        string.Equals(modelAlias, OpenAiWhisperCloudAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, OpenAiWhisperKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsGeminiAsrAlias(string? modelAlias) =>
        !string.IsNullOrWhiteSpace(modelAlias) &&
        (string.Equals(modelAlias, GeminiAsrCloudAlias, StringComparison.OrdinalIgnoreCase) ||
         string.Equals(modelAlias, GeminiAsrKey, StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-asr", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-3.8", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-2.5", StringComparison.OrdinalIgnoreCase));

    public static bool IsCloudAlias(string? modelAlias) =>
        IsOpenAiWhisperAlias(modelAlias) || IsGeminiAsrAlias(modelAlias);
}

public enum TranslationModelOverride
{
    Auto = 0,
    Madlad = 1,
    DeepL = 2,
    OpenAiGpt = 3,
    GeminiTranslation = 4
}

public static class TranslationModelOverrideSettings
{
    public const string AutoKey = "auto";
    public const string MadladKey = "madlad";
    public const string DeepLKey = "deepl";
    public const string MadladModelAlias = "madlad400";
    public const string DeepLModelAlias = "deepl-cloud";
    public const string OpenAiGptKey = "openai-gpt";
    public const string GeminiTranslationKey = "gemini-translation";
    public const string OpenAiGptCloudAlias = "openai-gpt-cloud";
    public const string GeminiTranslationCloudAlias = "gemini-translation-cloud";

    public static string ToKey(TranslationModelOverride modelOverride) =>
        modelOverride switch
        {
            TranslationModelOverride.Madlad => MadladKey,
            TranslationModelOverride.DeepL => DeepLKey,
            TranslationModelOverride.OpenAiGpt => OpenAiGptKey,
            TranslationModelOverride.GeminiTranslation => GeminiTranslationKey,
            _ => AutoKey
        };

    public static TranslationModelOverride FromKey(string? key) =>
        string.IsNullOrWhiteSpace(key)
            ? TranslationModelOverride.Auto
            : key.Trim().ToLowerInvariant() switch
            {
                MadladKey => TranslationModelOverride.Madlad,
                DeepLKey or DeepLModelAlias => TranslationModelOverride.DeepL,
                OpenAiGptKey or OpenAiGptCloudAlias => TranslationModelOverride.OpenAiGpt,
                GeminiTranslationKey or GeminiTranslationCloudAlias or "gemini-3.8-flash" or "gemini-3.5-flash-lite" or "gemini-2.5-flash" or "gemini-2.5-pro" => TranslationModelOverride.GeminiTranslation,
                _ => TranslationModelOverride.Auto
            };

    public static string? ResolveModelAlias(TranslationModelOverride modelOverride) =>
        modelOverride switch
        {
            TranslationModelOverride.Madlad => MadladModelAlias,
            TranslationModelOverride.DeepL => DeepLModelAlias,
            TranslationModelOverride.OpenAiGpt => OpenAiGptCloudAlias,
            TranslationModelOverride.GeminiTranslation => GeminiTranslationCloudAlias,
            _ => null
        };

    public static bool RequiresModelAlias(TranslationModelOverride modelOverride) =>
        modelOverride is TranslationModelOverride.Madlad
            or TranslationModelOverride.DeepL
            or TranslationModelOverride.OpenAiGpt
            or TranslationModelOverride.GeminiTranslation;

    public static bool IsDeepLModelAlias(string? modelAlias) =>
        string.Equals(modelAlias, DeepLModelAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, DeepLKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsOpenAiGptAlias(string? modelAlias) =>
        string.Equals(modelAlias, OpenAiGptCloudAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, OpenAiGptKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsGeminiTranslationAlias(string? modelAlias) =>
        !string.IsNullOrWhiteSpace(modelAlias) &&
        (string.Equals(modelAlias, GeminiTranslationCloudAlias, StringComparison.OrdinalIgnoreCase) ||
         string.Equals(modelAlias, GeminiTranslationKey, StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-translation", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-3.8", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-3.5", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-2.5", StringComparison.OrdinalIgnoreCase));

    public static bool IsCloudAlias(string? modelAlias) =>
        IsDeepLModelAlias(modelAlias) || IsOpenAiGptAlias(modelAlias) || IsGeminiTranslationAlias(modelAlias);
}

public enum TtsModelOverride
{
    Auto = 0,
    Kokoro = 1,
    Chatterbox = 2,
    ElevenLabs = 3,
    OpenAiTts = 4,
    GoogleTts = 5,
    CosyVoice = 6,
    Qwen3Tts = 7,
    GeminiTts = 8
}

public static class TtsModelOverrideSettings
{
    public const string AutoKey = "auto";
    public const string KokoroKey = "kokoro";
    public const string ChatterboxKey = "chatterbox";
    public const string CosyVoiceKey = "cosyvoice";
    public const string Qwen3TtsKey = "qwen3-tts";
    public const string KokoroModelAlias = "kokoro-onnx";
    public const string ChatterboxModelAlias = "chatterbox-turbo-onnx";
    public const string CosyVoiceModelAlias = "cosyvoice-300m";
    public const string ElevenLabsKey = "elevenlabs";
    public const string OpenAiTtsKey = "openai-tts";
    public const string GoogleTtsKey = "google-tts";
    public const string GeminiTtsKey = "gemini-tts";
    public const string ElevenLabsCloudAlias = "elevenlabs-tts-cloud";
    public const string OpenAiTtsCloudAlias = "openai-tts-cloud";
    public const string GoogleTtsCloudAlias = "google-tts-cloud";
    public const string GeminiTtsCloudAlias = "gemini-tts-cloud";

    public static string ToKey(TtsModelOverride modelOverride) =>
        modelOverride switch
        {
            TtsModelOverride.Kokoro => KokoroKey,
            TtsModelOverride.Chatterbox => ChatterboxKey,
            TtsModelOverride.CosyVoice => CosyVoiceKey,
            TtsModelOverride.Qwen3Tts => Qwen3TtsKey,
            TtsModelOverride.ElevenLabs => ElevenLabsKey,
            TtsModelOverride.OpenAiTts => OpenAiTtsKey,
            TtsModelOverride.GoogleTts => GoogleTtsKey,
            TtsModelOverride.GeminiTts => GeminiTtsKey,
            _ => AutoKey
        };

    public static TtsModelOverride FromKey(string? key) =>
        string.IsNullOrWhiteSpace(key)
            ? TtsModelOverride.Auto
            : key.Trim().ToLowerInvariant() switch
            {
                KokoroKey => TtsModelOverride.Kokoro,
                ChatterboxKey => TtsModelOverride.Chatterbox,
                CosyVoiceKey or CosyVoiceModelAlias => TtsModelOverride.CosyVoice,
                Qwen3TtsKey => TtsModelOverride.Qwen3Tts,
                ElevenLabsKey or ElevenLabsCloudAlias => TtsModelOverride.ElevenLabs,
                OpenAiTtsKey or OpenAiTtsCloudAlias => TtsModelOverride.OpenAiTts,
                GoogleTtsKey or GoogleTtsCloudAlias => TtsModelOverride.GoogleTts,
                GeminiTtsKey or GeminiTtsCloudAlias or "gemini-3.8-flash-tts" or "gemini-3.8-flash-lite-tts" => TtsModelOverride.GeminiTts,
                _ => TtsModelOverride.Auto
            };

    public static string? ResolveModelAlias(TtsModelOverride modelOverride) =>
        modelOverride switch
        {
            TtsModelOverride.Kokoro => KokoroModelAlias,
            TtsModelOverride.Chatterbox => ChatterboxModelAlias,
            TtsModelOverride.CosyVoice => CosyVoiceModelAlias,
            TtsModelOverride.ElevenLabs => ElevenLabsCloudAlias,
            TtsModelOverride.OpenAiTts => OpenAiTtsCloudAlias,
            TtsModelOverride.GoogleTts => GoogleTtsCloudAlias,
            TtsModelOverride.GeminiTts => GeminiTtsCloudAlias,
            _ => null
        };

    public static bool RequiresModelAlias(TtsModelOverride modelOverride) =>
        modelOverride is TtsModelOverride.Kokoro
            or TtsModelOverride.Chatterbox
            or TtsModelOverride.CosyVoice
            or TtsModelOverride.ElevenLabs
            or TtsModelOverride.OpenAiTts
            or TtsModelOverride.GoogleTts
            or TtsModelOverride.GeminiTts;

    public static bool IsElevenLabsAlias(string? modelAlias) =>
        string.Equals(modelAlias, ElevenLabsCloudAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, ElevenLabsKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsOpenAiTtsAlias(string? modelAlias) =>
        string.Equals(modelAlias, OpenAiTtsCloudAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, OpenAiTtsKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsGoogleTtsAlias(string? modelAlias) =>
        string.Equals(modelAlias, GoogleTtsCloudAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, GoogleTtsKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsGeminiTtsAlias(string? modelAlias) =>
        !string.IsNullOrWhiteSpace(modelAlias) &&
        (string.Equals(modelAlias, GeminiTtsCloudAlias, StringComparison.OrdinalIgnoreCase) ||
         string.Equals(modelAlias, GeminiTtsKey, StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-tts", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-3.8-flash-tts", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("gemini-3.8-flash-lite-tts", StringComparison.OrdinalIgnoreCase));

    public static bool IsCosyVoiceAlias(string? modelAlias) =>
        string.Equals(modelAlias, CosyVoiceModelAlias, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(modelAlias, CosyVoiceKey, StringComparison.OrdinalIgnoreCase);

    public static bool IsQwen3TtsAlias(string? modelAlias) =>
        !string.IsNullOrWhiteSpace(modelAlias) &&
        (modelAlias.Equals(Qwen3TtsKey, StringComparison.OrdinalIgnoreCase) ||
         modelAlias.StartsWith("qwen3-tts", StringComparison.OrdinalIgnoreCase) ||
         modelAlias.Equals("qwen-tts", StringComparison.OrdinalIgnoreCase));

    public static bool IsCloudAlias(string? modelAlias) =>
        IsElevenLabsAlias(modelAlias) || IsOpenAiTtsAlias(modelAlias) || IsGoogleTtsAlias(modelAlias) || IsGeminiTtsAlias(modelAlias);
}

public enum SeparationModelOverride
{
    Auto = 0,
    Spleeter = 1
}

public static class SeparationModelOverrideSettings
{
    public const string AutoKey = "auto";
    public const string SpleeterKey = "spleeter";
    public const string SpleeterModelAlias = "spleeter";

    public static string ToKey(SeparationModelOverride modelOverride) =>
        modelOverride switch
        {
            SeparationModelOverride.Spleeter => SpleeterKey,
            _ => AutoKey
        };

    public static SeparationModelOverride FromKey(string? key) =>
        string.IsNullOrWhiteSpace(key)
            ? SeparationModelOverride.Auto
            : key.Trim().ToLowerInvariant() switch
            {
                SpleeterKey or "spleeter-2stems" or "spleeter-non-commercial" => SeparationModelOverride.Spleeter,
                // Known aliases that are intentionally treated as Auto (no explicit override).
                "demucs-v4" or "demucs" or "htdemucs" => SeparationModelOverride.Auto,
                _ => SeparationModelOverride.Auto
            };

    public static string? ResolveModelAlias(SeparationModelOverride modelOverride) =>
        modelOverride switch
        {
            SeparationModelOverride.Spleeter => SpleeterModelAlias,
            _ => null
        };

    public static bool RequiresModelAlias(SeparationModelOverride modelOverride) =>
        modelOverride is SeparationModelOverride.Spleeter;
}

public static class ModelVariantOverrideKeys
{
    public static string Build(string stageKey, string modelAlias) =>
        $"{NormalizeToken(stageKey)}:{NormalizeToken(modelAlias)}";

    public static bool TryParse(string key, out string stageKey, out string modelAlias)
    {
        stageKey = string.Empty;
        modelAlias = string.Empty;
        if (string.IsNullOrWhiteSpace(key))
        {
            return false;
        }

        string trimmed = key.Trim();
        int separatorIndex = trimmed.IndexOf(':');
        if (separatorIndex <= 0 || separatorIndex >= trimmed.Length - 1)
        {
            return false;
        }

        stageKey = NormalizeToken(trimmed[..separatorIndex]);
        modelAlias = NormalizeToken(trimmed[(separatorIndex + 1)..]);
        return !string.IsNullOrWhiteSpace(stageKey) && !string.IsNullOrWhiteSpace(modelAlias);
    }

    public static string NormalizeToken(string token) =>
        string.IsNullOrWhiteSpace(token)
            ? string.Empty
            : token.Trim().ToLowerInvariant();
}

public sealed record WindowLayoutSettings(
    double? Width,
    double? Height,
    bool IsMaximized,
    WindowPanelLayoutSettings? PanelLayout = null);

public sealed record WindowPanelLayoutSettings(
    double? LeftPanelWidth,
    double? RightPanelWidth,
    double? CenterTopPanelHeight,
    double? CenterBottomPanelHeight);

public sealed record RecentProjectEntry(
    string ProjectName,
    string ProjectPath,
    DateTimeOffset LastOpenedAtUtc);

public sealed record StudioExportSettings(
    bool Srt,
    bool Vtt,
    bool Ass,
    bool BurnInSubtitles,
    double TargetLufs,
    string Container,
    string SubtitleSource,
    bool MatchOriginalLoudness = false,
    string VideoEncoder = VideoEncoderPreferenceSettings.AutoKey)
{
    public const string Mp4Container = "mp4";
    public const string MkvContainer = "mkv";
    public const string TranslatedSubtitleSource = "translated";
    public const string TranscriptSubtitleSource = "transcript";
    public const string BilingualSubtitleSource = "bilingual";

    public static StudioExportSettings Default { get; } = new(
        Srt: true,
        Vtt: false,
        Ass: false,
        BurnInSubtitles: false,
        TargetLufs: ExportLoudnessTargets.OnlineLufs,
        Container: Mp4Container,
        SubtitleSource: TranslatedSubtitleSource);
}

public sealed record StudioPlaybackSettings(
    bool SubtitlesEnabled,
    string SubtitleContentMode,
    string VideoDecode = PlaybackVideoDecodePreferenceSettings.AutoKey,
    bool KaraokeEnabled = false)
{
    public const string SourceSubtitleContentMode = "source";
    public const string TranslatedSubtitleContentMode = "translated";
    public const string BilingualSubtitleContentMode = "bilingual";

    public static StudioPlaybackSettings Default { get; } = new(
        SubtitlesEnabled: false,
        SubtitleContentMode: TranslatedSubtitleContentMode);
}

public sealed record TtsTimingSettings(
    bool EnableRubberbandStretch,
    double RubberbandStretchThreshold,
    double? AutoStretchMaxOverrun = null)
{
    public static TtsTimingSettings Default { get; } = new(
        EnableRubberbandStretch: false,
        RubberbandStretchThreshold: 0.15d);
}
