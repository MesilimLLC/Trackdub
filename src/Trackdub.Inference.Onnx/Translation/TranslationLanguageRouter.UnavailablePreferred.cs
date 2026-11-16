using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static TranslationRouteSelection CreateUnavailablePreferredRoute(
        TranslationRoutingContext context,
        RouteQuery query)
    {
        if (query.IsOpusMtAuto)
        {
            return CreateOpusUnavailableRoute(context, query);
        }

        bool gemmaRuntimeUnavailable = query.PreferredModelAlias!.Equals("gemma-4-e4b-translation", StringComparison.OrdinalIgnoreCase);
        string fallbackReason = gemmaRuntimeUnavailable
            ? "Gemma 4 E4B translation is not available until its runtime and memory requirements have been validated."
            : $"Selected translation model '{query.PreferredModelAlias}' is not available for this language pair.";
        return new TranslationRouteSelection(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            ProviderName: gemmaRuntimeUnavailable ? "gemma-genai" : "none",
            RouteDetail: "Unavailable",
            UnavailableReason: fallbackReason);
    }

    private static TranslationRouteSelection CreateOpusUnavailableRoute(
        TranslationRoutingContext context,
        RouteQuery query)
    {
        TranslationLanguageCoverageMatrix.TryGetLanguage(query.SourceLanguage, out TranslationLanguageDefinition? sourceDefinition);
        TranslationLanguageCoverageMatrix.TryGetLanguage(query.TargetLanguage, out TranslationLanguageDefinition? targetDefinition);
        string sourceDisplay = sourceDefinition?.DisplayName ?? query.SourceLanguage;
        string targetDisplay = targetDefinition?.DisplayName ?? query.TargetLanguage;
        bool hasCoverage = context.DirectEntriesByPair.ContainsKey((query.SourceLanguage, query.TargetLanguage));
        return new TranslationRouteSelection(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            ProviderName: "none",
            RouteDetail: "Unavailable",
            UnavailableReason: ResolveOpusUnavailableReason(context, hasCoverage, sourceDisplay, targetDisplay));
    }

    private static string ResolveOpusUnavailableReason(
        TranslationRoutingContext context,
        bool hasCoverage,
        string sourceDisplay,
        string targetDisplay)
    {
        if (!hasCoverage)
        {
            return $"Opus MT has no model for {sourceDisplay} to {targetDisplay}. Choose a different target language or a cloud provider.";
        }

        if (!context.CanDownload)
        {
            return $"Opus MT needs the {sourceDisplay} to {targetDisplay} model, but automatic model downloads are turned off. You can download it in Settings under Models.";
        }

        return $"Opus MT is unavailable for {sourceDisplay} to {targetDisplay}.";
    }
}
