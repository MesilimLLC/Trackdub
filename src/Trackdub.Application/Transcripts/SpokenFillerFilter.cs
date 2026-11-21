using System.Text.RegularExpressions;

namespace Trackdub.Application.Transcripts;

/// <summary>
/// Removes spoken fillers ("uh", "um", ...) from English source text before machine translation.
/// The transcript itself keeps them; only the text sent to the translation engine changes.
/// </summary>
/// <remarks>
/// Fillers carry no meaning to translate, lengthen the dubbed speech, and can push a greedy decoder
/// into repeating them: MADLAD turned "thing uh out there ... uh ... uh" into hundreds of "ah".
/// </remarks>
public static partial class SpokenFillerFilter
{
    public static string StripForTranslation(string text, string? sourceLanguage)
    {
        if (string.IsNullOrWhiteSpace(text) || !IsEnglish(sourceLanguage))
        {
            return text;
        }

        string stripped = FillerPattern().Replace(text, " ");
        stripped = SpaceBeforePunctuation().Replace(stripped, "$1");
        stripped = RepeatedSpace().Replace(stripped, " ").Trim();
        stripped = LeadingPunctuation().Replace(stripped, string.Empty);
        if (stripped.Length == 0)
        {
            return text;
        }

        string original = text.TrimStart();
        if (original.Length > 0 && char.IsUpper(original[0]) && char.IsLower(stripped[0]))
        {
            stripped = char.ToUpperInvariant(stripped[0]) + stripped[1..];
        }

        return stripped;
    }

    private static bool IsEnglish(string? language) =>
        language is not null &&
        (language.Equals("en", StringComparison.OrdinalIgnoreCase) ||
         language.StartsWith("en-", StringComparison.OrdinalIgnoreCase) ||
         language.StartsWith("en_", StringComparison.OrdinalIgnoreCase));

    // A filler is a whole word, optionally followed by the comma a transcriber puts after it.
    [GeneratedRegex(@"\b(?:u+h+|u+m+|uhm+|e+r+m+|e+r+|a+h+|h+m+|m+h*m+)\b,?", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex FillerPattern();

    [GeneratedRegex(@"\s+([,.;:!?])")]
    private static partial Regex SpaceBeforePunctuation();

    [GeneratedRegex(@"\s{2,}")]
    private static partial Regex RepeatedSpace();

    [GeneratedRegex(@"^[,.;:]\s*")]
    private static partial Regex LeadingPunctuation();
}
