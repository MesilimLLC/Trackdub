using Trackdub.Contracts.Translation;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private sealed record RouteQuery
    {
        public required string SourceLanguage { get; init; }
        public required string TargetLanguage { get; init; }
        public string? DisplayName { get; init; }
        public string? PreferredModelAlias { get; init; }
        public bool IsOpusMtAuto { get; init; }

        public bool HasPreferredModel => !IsOpusMtAuto && !string.IsNullOrWhiteSpace(PreferredModelAlias);

        public bool IsSourceUnresolved => SourceLanguage is "auto" or "und";

        public string ResolveDisplayName(TranslationLanguageDefinition? targetDefinition) =>
            DisplayName ?? targetDefinition?.DisplayName ?? TargetLanguage;
    }

    private static RouteQuery CreateRouteQuery(
        string sourceLanguage,
        string targetLanguage,
        string? displayName,
        string? preferredModelAlias)
    {
        bool isOpusMtAuto = OpusMtAutoModel.IsAutoAlias(preferredModelAlias);
        string normalizedSourceLanguage = NormalizeLanguageCode(sourceLanguage)
            ?? (isOpusMtAuto ? "auto" : throw new InvalidOperationException("Source language is required."));
        string normalizedTargetLanguage = NormalizeLanguageCode(targetLanguage)
            ?? (isOpusMtAuto ? "unknown" : throw new InvalidOperationException("Target language is required."));
        return new RouteQuery
        {
            SourceLanguage = normalizedSourceLanguage,
            TargetLanguage = normalizedTargetLanguage,
            DisplayName = displayName,
            PreferredModelAlias = preferredModelAlias,
            IsOpusMtAuto = isOpusMtAuto
        };
    }
}
