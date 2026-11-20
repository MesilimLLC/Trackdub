using System.Reflection;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Tests;

/// <summary>
/// Loads fixture manifests for tests of download mechanics and marks every entry commercially verified, so the commercial
/// gate on download and model-manager paths is not what those tests are about. Gate behaviour has its own tests.
/// </summary>
internal static class ClearedManifestRegistry
{
    public static BundledModelManifestRegistry Load(string manifestPath) =>
        Clear(BundledModelManifestRegistry.Load(manifestPath));

    public static BundledModelManifestRegistry Clear(BundledModelManifestRegistry registry)
    {
        BundledModelManifestEntry[] entries = [.. registry.Entries.Select(static entry => entry with { CommercialUseVerified = true })];
        Dictionary<string, BundledModelManifestEntry> aliasIndex = new(StringComparer.OrdinalIgnoreCase);
        foreach (BundledModelManifestEntry entry in entries)
        {
            foreach (string alias in entry.Aliases)
            {
                aliasIndex[alias] = entry;
            }
        }

        ConstructorInfo constructor = typeof(BundledModelManifestRegistry).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic,
            binder: null,
            [typeof(string), typeof(IReadOnlyList<BundledModelManifestEntry>), typeof(IReadOnlyDictionary<string, BundledModelManifestEntry>)],
            modifiers: null)
            ?? throw new InvalidOperationException("BundledModelManifestRegistry constructor shape changed.");
        return (BundledModelManifestRegistry)constructor.Invoke([registry.ManifestPath, entries, aliasIndex]);
    }
}
