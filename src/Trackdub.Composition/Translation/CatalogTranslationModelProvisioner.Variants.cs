using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private sealed record VariantSelection(string? VariantAlias, string? ExecutionProvider);

    private async Task<IReadOnlyList<string?>> ResolveVariantAliasesAsync(
        BundledModelManifestEntry selected,
        VariantSelection selection,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(selection.VariantAlias)) return [selection.VariantAlias];
        if (!string.IsNullOrWhiteSpace(selection.ExecutionProvider)) return [ResolveVariantForProvider(selected, selection)];
        // Auto provider mode: provision every variant this host can run so the planner can pick the accelerated one.
        HashSet<string>? available = await DiscoverAvailableProvidersAsync(cancellationToken).ConfigureAwait(false);
        if (available is null) return [null];
        string?[] aliases = selected.Variants
            .Where(variant => variant.SupportedProviders?.Any(available.Contains) == true)
            .Select(variant => variant.Alias).ToArray();
        return aliases.Length > 0 ? aliases : [null];
    }

    private async Task<HashSet<string>?> DiscoverAvailableProvidersAsync(CancellationToken cancellationToken)
    {
        if (hardwareProfileProvider is null || providerDiscovery is null) return null;
        try
        {
            var profile = await hardwareProfileProvider.GetCurrentAsync(cancellationToken).ConfigureAwait(false);
            var providers = await providerDiscovery.DiscoverAsync(profile, cancellationToken).ConfigureAwait(false);
            var available = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "cpu" };
            foreach (var provider in providers.Where(provider => provider.IsAvailable)) available.Add(provider.Provider.ToString());
            return available;
        }
        catch (Exception ex) when (ex is not OperationCanceledException) { return null; }
    }

    private static string? ResolveVariantForProvider(BundledModelManifestEntry selected, VariantSelection selection) =>
        string.IsNullOrWhiteSpace(selection.ExecutionProvider) ? null
        : selected.Variants.FirstOrDefault(variant =>
            variant.SupportedProviders?.Contains(selection.ExecutionProvider.Trim(), StringComparer.OrdinalIgnoreCase) == true)?.Alias;
}
