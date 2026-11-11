using Trackdub.Sdk;

namespace Trackdub.Cli;

/// <summary>Versioned application preset. Host qualification is a separate acceptance step.</summary>
internal static class PocServerPreset
{
    internal const string Name = "poc-server-v1";
    internal static PipelinePreset Create() => new()
    {
        Version = 1,
        SourceLanguage = "en",
        TargetLanguage = "es",
        ExportFormat = "mp4",
        EnableAsrTextRefinement = false,
        Models = new Dictionary<string, string>
        {
            ["vad"] = "silero-vad",
            ["asr"] = "parakeet-tdt-0.6b-v3",
            ["translation"] = "madlad400",
            ["tts"] = "kokoro",
            ["diarization"] = "sortformer-4spk",
            ["separation"] = "spleeter",
            ["speech-enhancement"] = "deepfilternet3",
        },
    };
    internal static IReadOnlyDictionary<string, string> Providers { get; } = new Dictionary<string, string>
    {
        ["vad"] = "cpu",
        ["asr"] = "cuda",
        ["translation"] = "cuda",
        ["tts"] = "cuda",
        ["diarization"] = "cuda",
        ["separation"] = "cuda",
        ["speech-enhancement"] = "cuda",
    };
    internal static IReadOnlyDictionary<string, string> Variants { get; } = new Dictionary<string, string>
    { ["translation"] = "quantized", ["tts"] = "default" };
}

/// <summary>
/// Quality-tier evaluation preset for the server profile: the same pipeline and provider pins as
/// <see cref="PocServerPreset"/>, with Qwen3-TTS-1.7B CustomVoice replacing Kokoro.
/// Host qualification and promotion are separate acceptance steps.
/// </summary>
internal static class PocServerQualityPreset
{
    internal const string Name = "poc-server-quality-v1";
    internal static PipelinePreset Create() => new()
    {
        Version = 1,
        SourceLanguage = "en",
        TargetLanguage = "es",
        ExportFormat = "mp4",
        EnableAsrTextRefinement = false,
        Models = new Dictionary<string, string>
        {
            ["vad"] = "silero-vad",
            ["asr"] = "parakeet-tdt-0.6b-v3",
            ["translation"] = "madlad400",
            ["tts"] = "qwen3-tts-1.7b-customvoice",
            ["diarization"] = "sortformer-4spk",
            ["separation"] = "spleeter",
            ["speech-enhancement"] = "deepfilternet3",
        },
    };
    internal static IReadOnlyDictionary<string, string> Providers { get; } = new Dictionary<string, string>
    {
        ["vad"] = "cpu",
        ["asr"] = "cuda",
        ["translation"] = "cuda",
        ["tts"] = "cuda",
        ["diarization"] = "cuda",
        ["separation"] = "cuda",
        ["speech-enhancement"] = "cuda",
    };
    internal static IReadOnlyDictionary<string, string> Variants { get; } = new Dictionary<string, string>
    { ["translation"] = "quantized", ["tts"] = "default" };
}

/// <summary>
/// English-to-Mandarin preset for the server profile: the quality-tier pipeline with the target language set to Chinese.
/// Qwen3-TTS CustomVoice is used because it speaks Mandarin natively; Kokoro has no verified Mandarin route in this codebase.
/// Host qualification and audio quality are separate acceptance steps and are not established by this definition.
/// </summary>
internal static class PocServerZhPreset
{
    internal const string Name = "poc-server-zh-v1";
    internal static PipelinePreset Create() => PocServerQualityPreset.Create() with { TargetLanguage = "zh" };
    internal static IReadOnlyDictionary<string, string> Providers => PocServerQualityPreset.Providers;
    internal static IReadOnlyDictionary<string, string> Variants => PocServerQualityPreset.Variants;
}

/// <summary>Built-in preset lookup shared by single-run and batch execution.</summary>
internal static class PocPresets
{
    internal static bool TryResolve(
        string? name,
        out PipelinePreset? preset,
        out IReadOnlyDictionary<string, string>? providers,
        out IReadOnlyDictionary<string, string>? variants)
    {
        if (string.Equals(name, PocServerPreset.Name, StringComparison.Ordinal))
        {
            preset = PocServerPreset.Create();
            providers = PocServerPreset.Providers;
            variants = PocServerPreset.Variants;
            return true;
        }

        if (string.Equals(name, PocServerQualityPreset.Name, StringComparison.Ordinal))
        {
            preset = PocServerQualityPreset.Create();
            providers = PocServerQualityPreset.Providers;
            variants = PocServerQualityPreset.Variants;
            return true;
        }

        if (string.Equals(name, PocServerZhPreset.Name, StringComparison.Ordinal))
        {
            preset = PocServerZhPreset.Create();
            providers = PocServerZhPreset.Providers;
            variants = PocServerZhPreset.Variants;
            return true;
        }

        preset = null;
        providers = null;
        variants = null;
        return false;
    }
}
