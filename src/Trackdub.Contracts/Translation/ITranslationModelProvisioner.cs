namespace Trackdub.Contracts.Translation;

public interface ITranslationModelProvisioner
{
    Task RefreshAsync(CancellationToken cancellationToken);
    Task<string?> EnsureAsync(string sourceLanguage, string targetLanguage, string? preferredModelAlias, CancellationToken cancellationToken,
        IProgress<string>? preparationProgress = null, string? preferredModelVariantAlias = null);
}
