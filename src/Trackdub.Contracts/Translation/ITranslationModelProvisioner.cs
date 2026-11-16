namespace Trackdub.Contracts.Translation;

public interface ITranslationModelProvisioner
{
    Task RefreshAsync(CancellationToken cancellationToken);
    Task<string?> EnsureAsync(string sourceLanguage, string targetLanguage, string? preferredModelAlias, CancellationToken cancellationToken,
        IProgress<string>? preparationProgress = null, string? preferredModelVariantAlias = null, string? preferredExecutionProvider = null);
}

public sealed record TranslationModelEnsureRequest
{
    public required string SourceLanguage { get; init; }
    public required string TargetLanguage { get; init; }
    public string? PreferredModelAlias { get; init; }
    public IProgress<string>? PreparationProgress { get; init; }
    public string? PreferredModelVariantAlias { get; init; }
    public string? PreferredExecutionProvider { get; init; }

    public static TranslationModelEnsureRequest Create(string sourceLanguage, string targetLanguage, string? preferredModelAlias) =>
        new()
        {
            SourceLanguage = sourceLanguage,
            TargetLanguage = targetLanguage,
            PreferredModelAlias = preferredModelAlias
        };
}
