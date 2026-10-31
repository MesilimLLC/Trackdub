namespace Trackdub.Domain.Translation;

/// <summary>Identifies which source clips a translation run should process.</summary>
public enum TranslationRunScope
{
    /// <summary>Translate every source clip, replacing the existing translated revision.</summary>
    All,

    /// <summary>
    /// Translate only clips without translated text or with outdated generated text. Clips
    /// with manual or legacy (Unknown) text are skipped.
    /// </summary>
    MissingOrOutdated
}
