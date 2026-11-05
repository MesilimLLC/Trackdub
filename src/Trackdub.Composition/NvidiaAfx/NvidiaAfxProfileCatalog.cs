using Trackdub.Contracts;

namespace Trackdub.Composition.NvidiaAfx;

/// <summary>
/// One Maxine feature-folder model requirement. Resolved on disk via
/// <see cref="Infrastructure.Components.NvidiaAfx.NvidiaAfxRuntimeLayout"/>.
/// </summary>
public sealed record NvidiaAfxRequiredModel(
    string FeatureFolder,
    string ModelStem);

/// <summary>
/// Maxine ships rate-specific model payloads (for example <c>denoiser_16k</c> vs
/// <c>denoiser_48k</c>). Session create must bind the stem that matches the input rate.
/// </summary>
public sealed record NvidiaAfxSampleRateModels(
    int InputSampleRate,
    NvidiaAfxRequiredModel[] Models);

public sealed record NvidiaAfxProfileDefinition(
    NvidiaAfxProfile Profile,
    string DisplayName,
    string Selector,
    bool IsChainedEffect,
    /// <summary>Supported input sample rates for this effect.</summary>
    int[] SupportedSampleRates,
    int MaxChannels,
    /// <summary>Per-input-rate Maxine feature+stem requirements (NGC download_features layout).</summary>
    NvidiaAfxSampleRateModels[] ModelsBySampleRate,
    bool RequiresFarEndReference,
    bool SupportsIntensityRatio,
    /// <summary>
    /// Explicit output sample rate when it differs from the selected input rate
    /// (for example telephony upscale 8 kHz → 16 kHz). Null means output matches input.
    /// </summary>
    int? OutputSampleRate = null,
    /// <summary>NVIDIA Early Access effect under an evaluation license; disabled unless explicitly allowed.</summary>
    bool IsEarlyAccess = false,
    /// <summary>Fallback input frame duration when the native effect does not report its frame size.</summary>
    int InputFrameDurationMilliseconds = 10,
    /// <summary>Number of native frames to run for a readiness probe.</summary>
    int ReadinessProbeFrameCount = 100)
{
    /// <summary>
    /// Number of native frames to run for a readiness probe. A probe that processes no audio
    /// cannot prove the effect works, so non-positive values are rejected at construction.
    /// </summary>
    public int ReadinessProbeFrameCount { get; init; } = ReadinessProbeFrameCount >= 1
        ? ReadinessProbeFrameCount
        : throw new ArgumentOutOfRangeException(
            nameof(ReadinessProbeFrameCount),
            "A readiness probe must process at least one native frame.");

    public int ResolveOutputSampleRate(int inputSampleRate) =>
        OutputSampleRate ?? inputSampleRate;

    /// <summary>
    /// Preferred probe rate: 48 kHz when supported (shipping quality), else the highest listed rate.
    /// </summary>
    public int PreferredProbeSampleRate =>
        SupportedSampleRates.Contains(48000)
            ? 48000
            : SupportedSampleRates.Max();

    /// <summary>Union of every rate-specific model (full-package readiness / manifests).</summary>
    public IReadOnlyList<NvidiaAfxRequiredModel> AllRequiredModels =>
        ModelsBySampleRate
            .SelectMany(entry => entry.Models)
            .DistinctBy(model => (
                model.FeatureFolder.ToLowerInvariant(),
                model.ModelStem.ToLowerInvariant()))
            .ToArray();

    /// <summary>Distinct Maxine feature folders required by this profile.</summary>
    public IReadOnlyList<string> RequiredFeatureFolders =>
        AllRequiredModels
            .Select(model => model.FeatureFolder)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

    public NvidiaAfxRequiredModel[] ResolveRequiredModels(int inputSampleRate)
    {
        NvidiaAfxSampleRateModels? match = ModelsBySampleRate
            .FirstOrDefault(entry => entry.InputSampleRate == inputSampleRate);
        if (match is null)
        {
            throw new ArgumentOutOfRangeException(
                nameof(inputSampleRate),
                inputSampleRate,
                $"Profile '{Selector}' has no Maxine model set for {inputSampleRate} Hz. " +
                $"Supported: {string.Join(", ", ModelsBySampleRate.Select(entry => entry.InputSampleRate))}.");
        }

        return match.Models;
    }

    /// <summary>
    /// Preferred relative paths for manifests / diagnostics (Maxine 3.x features layout).
    /// Actual resolution accepts additional arch folders and extensions.
    /// </summary>
    public IReadOnlyList<string> PreferredModelRelativePaths =>
        AllRequiredModels
            .Select(model => Path.Join("features", model.FeatureFolder, "models", model.ModelStem + ".trtpkg"))
            .ToArray();
}

public static class NvidiaAfxProfileCatalog
{
    /// <summary>
    /// First-class discovery surface for AFX profiles. Profiles are listed even while the
    /// integration is stubbed so planners/settings can enumerate them without implying readiness.
    /// </summary>
    public static IReadOnlyList<NvidiaAfxProfileDefinition> Definitions { get; } =
    [
        new(
            NvidiaAfxProfile.NoiseOnly,
            "Noise Removal",
            Selector: "denoiser",
            IsChainedEffect: false,
            SupportedSampleRates: [16000, 48000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(16000, [new("nvafxdenoiser", "denoiser_16k")]),
                new(48000, [new("nvafxdenoiser", "denoiser_48k")]),
            ],
            RequiresFarEndReference: false,
            SupportsIntensityRatio: true),
        new(
            NvidiaAfxProfile.ReverbOnly,
            "Reverb Removal",
            Selector: "dereverb",
            IsChainedEffect: false,
            SupportedSampleRates: [16000, 48000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(16000, [new("nvafxdereverb", "dereverb_16k")]),
                new(48000, [new("nvafxdereverb", "dereverb_48k")]),
            ],
            RequiresFarEndReference: false,
            SupportsIntensityRatio: true),
        new(
            NvidiaAfxProfile.NoiseAndReverb,
            "Noise + Reverb Removal",
            Selector: "dereverb_denoiser",
            IsChainedEffect: false,
            SupportedSampleRates: [16000, 48000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(16000, [new("nvafxdereverbdenoiser", "dereverb_denoiser_16k")]),
                new(48000, [new("nvafxdereverbdenoiser", "dereverb_denoiser_48k")]),
            ],
            RequiresFarEndReference: false,
            SupportsIntensityRatio: true),
        new(
            NvidiaAfxProfile.TelephonyUpscale,
            "Telephony Upscale",
            // Matches NVIDIA Maxine chained selector (8 kHz → 16 kHz + denoise).
            Selector: "superres8kto16k_denoiser16k",
            IsChainedEffect: true,
            SupportedSampleRates: [8000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(
                    8000,
                    [
                        new("nvafxsuperres", "superres_8kto16k"),
                        new("nvafxdenoiser", "denoiser_16k"),
                    ]),
            ],
            RequiresFarEndReference: false,
            // SDK 2.x/3.x reject the scalar setter on a chain, and the documented NvAFX_SetFloatList
            // (per-effect ratios) is accepted but leaves the output unchanged on Windows (verified
            // live on SDK 3.0.0), so a slider here would do nothing.
            SupportsIntensityRatio: false,
            OutputSampleRate: 16000),
        new(
            NvidiaAfxProfile.SpeakerFocus,
            "Speaker Focus (Early Access)",
            Selector: "speaker_focus",
            IsChainedEffect: false,
            SupportedSampleRates: [16000, 48000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(16000, [new("nvafxspeakerfocus", "speaker_focus_16k")]),
                new(48000, [new("nvafxspeakerfocus", "speaker_focus_48k")]),
            ],
            RequiresFarEndReference: false,
            // The SDK accepts intensity_ratio on this effect but the output does not change
            // (verified live on SDK 3.0.0), so no control is offered.
            SupportsIntensityRatio: false,
            IsEarlyAccess: true),
        new(
            NvidiaAfxProfile.StudioVoice,
            "Studio Voice (High Quality)",
            Selector: NvidiaAfxIntegration.StudioVoiceHighQualitySelector,
            IsChainedEffect: false,
            SupportedSampleRates: [16000, 48000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(16000, [new("nvafxstudiovoicehighquality", "studio_voice_high_quality_16k")]),
                new(48000, [new("nvafxstudiovoicehighquality", "studio_voice_high_quality_48k")]),
            ],
            RequiresFarEndReference: false,
            SupportsIntensityRatio: false,
            InputFrameDurationMilliseconds: 6000,
            ReadinessProbeFrameCount: 1),
        new(
            NvidiaAfxProfile.AcousticEchoCancellation,
            "Acoustic Echo Cancellation",
            // Not listed in Maxine AFX 3.x public effect selectors; kept for discovery.
            // Readiness/model resolution will fail until NVIDIA ships a matching feature package.
            Selector: "aec",
            IsChainedEffect: false,
            SupportedSampleRates: [16000, 48000],
            MaxChannels: 1,
            ModelsBySampleRate:
            [
                new(16000, [new("nvafxaec", "aec_16k")]),
                new(48000, [new("nvafxaec", "aec_48k")]),
            ],
            RequiresFarEndReference: true,
            SupportsIntensityRatio: false)
    ];

    /// <summary>
    /// Profiles to offer in a UI: Early Access effects are listed only when explicitly allowed.
    /// </summary>
    public static IReadOnlyList<NvidiaAfxProfileDefinition> SelectableDefinitions =>
        [.. Definitions.Where(definition =>
            !definition.RequiresFarEndReference
            && (!definition.IsEarlyAccess || NvidiaAfxIntegration.AllowEarlyAccessEffects()))];

    public static NvidiaAfxProfileDefinition GetDefinition(NvidiaAfxProfile profile) =>
        Definitions.FirstOrDefault(definition => definition.Profile == profile)
        ?? Definitions.First(definition => definition.Profile == NvidiaAfxProfile.NoiseAndReverb);
}
