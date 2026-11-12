using Trackdub.Domain.StageRuns;
using Trackdub.Sdk;

namespace Trackdub.Cli;

internal static class CliModelOverrides
{
    internal static Dictionary<string, string>? Parse(string[] modelOverrides)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (string entry in modelOverrides)
        {
            int colonIndex = entry.IndexOf(':');
            if (colonIndex <= 0 || colonIndex >= entry.Length - 1)
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --model format: '{entry}'. Expected format: stage:alias (e.g., asr:large-v3)",
                    "--model");
                return null;
            }

            string stage = entry[..colonIndex].Trim();
            string alias = entry[(colonIndex + 1)..].Trim();

            if (string.IsNullOrEmpty(stage) || string.IsNullOrEmpty(alias))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --model format: '{entry}'. Both stage and alias must be non-empty.",
                    "--model");
                return null;
            }

            result[stage] = alias;
        }

        return result;
    }

    /// <summary>Pipeline stage names that accept a model-variant preference.</summary>
    private static readonly HashSet<string> VariantStages = new(StringComparer.OrdinalIgnoreCase)
    {
        StageNames.Vad, StageNames.Asr, StageNames.Diarization, StageNames.Translation, StageNames.Tts,
        StageNames.Separation, StageNames.AudioPreparation, StageNames.SpeechEnhancement, StageNames.OverlapRescue,
        StageNames.TextRefinementAsr, StageNames.LipSync, StageNames.LipSynthesis,
    };

    /// <summary>
    /// Parses repeatable <c>--variant stage:alias</c> tokens (for example <c>translation:trt-fp16</c>).
    /// The stage must be a pipeline stage that supports variants; the alias is checked later against the
    /// installed model's variants.
    /// </summary>
    /// <returns>The preference map (empty when no tokens), or <see langword="null"/> after an error was reported.</returns>
    internal static Dictionary<string, string>? ParseVariantOverrides(string[] variantOverrides)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string entry in variantOverrides)
        {
            int colonIndex = entry.IndexOf(':');
            string stage = colonIndex > 0 ? entry[..colonIndex].Trim() : string.Empty;
            string alias = colonIndex > 0 ? entry[(colonIndex + 1)..].Trim() : string.Empty;
            if (stage.Length == 0 || alias.Length == 0)
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --variant format: '{entry}'. Expected format: stage:variant (e.g., translation:trt-fp16)",
                    "--variant");
                return null;
            }

            if (!VariantStages.Contains(stage))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --variant stage: '{stage}'. Expected one of: {string.Join(", ", VariantStages.Order())}.",
                    "--variant");
                return null;
            }

            result[stage] = alias;
        }

        return result;
    }

    /// <summary>Overlays explicit <c>--variant</c> choices on a preset's variants; explicit choices win.</summary>
    internal static IReadOnlyDictionary<string, string>? MergeVariants(
        IReadOnlyDictionary<string, string>? presetVariants,
        IReadOnlyDictionary<string, string>? explicitVariants)
    {
        if (explicitVariants is not { Count: > 0 })
        {
            return presetVariants;
        }

        var merged = new Dictionary<string, string>(presetVariants ?? new Dictionary<string, string>(), StringComparer.OrdinalIgnoreCase);
        foreach ((string stage, string alias) in explicitVariants)
        {
            merged[stage] = alias;
        }

        return merged;
    }

    /// <summary>
    /// Accepts a bare alias (<c>madlad400</c>) or the dub-style <c>stage:alias</c> form
    /// for a single <c>run-stage</c> invocation. A stage prefix that does not match
    /// <paramref name="stageName"/> is rejected: passing it through would match no
    /// model and silently fall back to the default route.
    /// </summary>
    /// <returns>
    /// A one-entry preference map keyed by <paramref name="stageName"/>, or <see langword="null"/>
    /// after a validation error has been reported.
    /// </returns>
    internal static Dictionary<string, string>? TryBuildRunStageModelPreferences(string stageName, string modelAlias)
    {
        string normalizedAlias = modelAlias.Trim();
        int colonIndex = normalizedAlias.IndexOf(':');
        if (colonIndex >= 0)
        {
            string prefix = normalizedAlias[..colonIndex].Trim();
            if (!string.Equals(prefix, stageName, StringComparison.OrdinalIgnoreCase))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --model value: '{modelAlias}'. Expected a bare alias for stage '{stageName}' (e.g., madlad400) or '{stageName}:alias'.",
                    "--model");
                return null;
            }

            normalizedAlias = normalizedAlias[(colonIndex + 1)..].Trim();
        }

        if (string.IsNullOrEmpty(normalizedAlias))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                $"Invalid --model value: '{modelAlias}'. Model alias must be non-empty.",
                "--model");
            return null;
        }

        return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            [stageName] = normalizedAlias,
        };
    }

    internal static Dictionary<string, string>? ParseVoiceOverrides(string[] voiceOverrides)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (string entry in voiceOverrides)
        {
            int colonIndex = entry.IndexOf(':');
            if (colonIndex <= 0 || colonIndex >= entry.Length - 1)
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --voice format: '{entry}'. Expected format: SPEAKER_ID:voice_id (e.g., SPEAKER_00:af_bella)",
                    "--voice");
                return null;
            }

            string speakerId = entry[..colonIndex].Trim();
            string voiceId = entry[(colonIndex + 1)..].Trim();

            if (string.IsNullOrEmpty(speakerId) || string.IsNullOrEmpty(voiceId))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Invalid --voice format: '{entry}'. Both speaker ID and voice ID must be non-empty.",
                    "--voice");
                return null;
            }

            result[speakerId] = voiceId;
        }

        return result;
    }
}
