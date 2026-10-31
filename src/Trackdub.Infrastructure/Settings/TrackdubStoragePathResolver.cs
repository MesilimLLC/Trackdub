using System.Collections;
using System.Text.Json;
using Trackdub.Infrastructure;
using Trackdub.Infrastructure.Persistence.Repositories;

namespace Trackdub.Infrastructure.Settings;

public sealed record TrackdubStorageOptions(
    string UserDataRoot,
    string UserCacheRoot,
    string? SharedAssetRoot,
    bool IsPortable,
    string? ExplicitModelCacheDirectory = null,
    string? ExplicitEngineCacheDirectory = null);

public sealed record TrackdubStoragePathResolutionContext(
    string AppBaseDirectory,
    string LocalAppDataRoot,
    string CommonAppDataRoot,
    IReadOnlyDictionary<string, string?> EnvironmentVariables);

public static class TrackdubStoragePathResolver
{
    public const string ProductDirectoryName = "Trackdub";
    public const string PortableMarkerFileName = "Trackdub.portable";
    public const string PortableDataDirectoryName = "portable-data";
    public const string InstallerConfigFileName = "storage.json";
    public const string DataRootEnvironmentVariable = "TRACKDUB_DATA_ROOT";
    public const string CacheRootEnvironmentVariable = "TRACKDUB_CACHE_ROOT";
    public const string SharedAssetRootEnvironmentVariable = "TRACKDUB_SHARED_ASSET_ROOT";
    public const string ToolCacheRootEnvironmentVariable = "TRACKDUB_TOOL_CACHE_ROOT";
    public const string EngineCacheRootEnvironmentVariable = "TRACKDUB_ENGINE_CACHE_ROOT";
    public const string PortableEnvironmentVariable = "TRACKDUB_PORTABLE";
    public const string PortableDataRootEnvironmentVariable = "TRACKDUB_PORTABLE_DATA_ROOT";

    // Shared LIFO stacks backing ApplyToCurrentProcessScoped, keyed by env var name. Lets
    // an out-of-order Dispose() correctly rebase to whatever is now the topmost surviving
    // override (or the true pre-any-scope baseline) instead of an intermediate value some
    // other scope already unwound past.
    private static readonly object EnvironmentStackSyncRoot = new();
    private static readonly Dictionary<string, LinkedList<string?>> EnvironmentValueStacks =
        new(StringComparer.OrdinalIgnoreCase);

    public static TrackdubStorageOptions Resolve()
    {
        string localAppDataRoot = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppDataRoot))
        {
            localAppDataRoot = AppContext.BaseDirectory;
        }

        string commonAppDataRoot = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
        if (string.IsNullOrWhiteSpace(commonAppDataRoot))
        {
            commonAppDataRoot = localAppDataRoot;
        }

        return Resolve(new TrackdubStoragePathResolutionContext(
            AppContext.BaseDirectory,
            localAppDataRoot,
            commonAppDataRoot,
            CaptureEnvironment()));
    }

    public static TrackdubStorageOptions Resolve(TrackdubStoragePathResolutionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        // TRACKDUB_ENGINE_CACHE_ROOT is an explicit override: honoring it here keeps the engine
        // cache directory that ApplyToCurrentProcess writes back equal to the operator's value.
        TrackdubStorageOptions roots = ResolveRoots(context);
        string? modelCacheOverride = roots.ExplicitModelCacheDirectory
            ?? ReadDefaultModelCacheDirectory(new DirectoryInfo(roots.UserDataRoot));
        string? engineCacheOverride = NormalizeOptionalPath(GetValue(context.EnvironmentVariables, EngineCacheRootEnvironmentVariable));
        return roots with
        {
            ExplicitModelCacheDirectory = modelCacheOverride,
            ExplicitEngineCacheDirectory = engineCacheOverride ?? roots.ExplicitEngineCacheDirectory
        };
    }

    public static void MigrateModelCacheIfChanged(
        TrackdubStorageOptions roots,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(roots);
        string statePath = Path.Join(roots.UserDataRoot, "model-cache-directory.txt");
        using FileStream stateLock = LocalModelCacheRecordStore.AcquireFileLockAsync(
            statePath + ".lock", cancellationToken).GetAwaiter().GetResult();
        string defaultDirectory = Path.Join(roots.UserCacheRoot, "model-cache");
        bool stateExists = File.Exists(statePath);
        string previousDirectory = stateExists ? File.ReadAllText(statePath) : defaultDirectory;
        string destinationDirectory = roots.ExplicitModelCacheDirectory ?? defaultDirectory;
        bool invalidState = string.IsNullOrWhiteSpace(previousDirectory) || !Path.IsPathFullyQualified(previousDirectory);
        if (invalidState)
        {
            previousDirectory = destinationDirectory;
        }
        if (!LocalModelCacheRecordStore.RootsEqual(previousDirectory, destinationDirectory))
        {
            var previousPaths = new TrackdubStoragePaths(roots with { ExplicitModelCacheDirectory = previousDirectory });
            var store = new LocalModelCacheRecordStore(previousPaths);
            store.CopyRecordsToDirectoryAsync(new DirectoryInfo(destinationDirectory), cancellationToken).GetAwaiter().GetResult();
        }
        else if (stateExists && !invalidState)
        {
            return;
        }

        string tempPath = statePath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllText(tempPath, destinationDirectory);
            File.Move(tempPath, statePath, overwrite: true);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    private static TrackdubStorageOptions ResolveRoots(TrackdubStoragePathResolutionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        string appBaseDirectory = NormalizeRequiredPath(context.AppBaseDirectory);
        string localAppDataRoot = NormalizeRequiredPath(context.LocalAppDataRoot);
        string commonAppDataRoot = NormalizeRequiredPath(context.CommonAppDataRoot);
        IReadOnlyDictionary<string, string?> environment = context.EnvironmentVariables;

        TrackdubStorageOptions? environmentRoots = ResolveEnvironmentRoots(environment, localAppDataRoot);
        if (environmentRoots is not null)
        {
            return environmentRoots;
        }

        TrackdubStorageOptions? portableRoots = ResolvePortableRoots(environment, appBaseDirectory);
        if (portableRoots is not null)
        {
            return portableRoots;
        }

        return ResolveConfiguredRoots(localAppDataRoot, commonAppDataRoot);
    }

    private static TrackdubStorageOptions? ResolveEnvironmentRoots(
        IReadOnlyDictionary<string, string?> environment,
        string localAppDataRoot)
    {
        string? dataRoot = GetValue(environment, DataRootEnvironmentVariable);
        string? cacheRoot = GetValue(environment, CacheRootEnvironmentVariable);
        string? sharedAssetRoot = GetValue(environment, SharedAssetRootEnvironmentVariable);
        bool hasOverride = !string.IsNullOrWhiteSpace(dataRoot)
            || !string.IsNullOrWhiteSpace(cacheRoot)
            || !string.IsNullOrWhiteSpace(sharedAssetRoot);
        if (!hasOverride)
        {
            return null;
        }

        string normalizedDataRoot = NormalizeOptionalPath(dataRoot) ?? Path.Join(localAppDataRoot, ProductDirectoryName);
        string normalizedCacheRoot = NormalizeOptionalPath(cacheRoot) ?? normalizedDataRoot;
        return new TrackdubStorageOptions(
            normalizedDataRoot,
            normalizedCacheRoot,
            NormalizeOptionalPath(sharedAssetRoot),
            IsPortable: false);
    }

    private static TrackdubStorageOptions? ResolvePortableRoots(
        IReadOnlyDictionary<string, string?> environment,
        string appBaseDirectory)
    {
        if (!IsPortableRequested(environment, appBaseDirectory))
        {
            return null;
        }

        string portableDataRoot = NormalizeOptionalPath(GetValue(environment, PortableDataRootEnvironmentVariable))
            ?? Path.Join(appBaseDirectory, PortableDataDirectoryName);
        return new TrackdubStorageOptions(portableDataRoot, portableDataRoot, SharedAssetRoot: null, IsPortable: true);
    }

    private static bool IsPortableRequested(
        IReadOnlyDictionary<string, string?> environment,
        string appBaseDirectory)
    {
        string? portableValue = GetValue(environment, PortableEnvironmentVariable);
        return string.Equals(portableValue, "1", StringComparison.OrdinalIgnoreCase)
            || string.Equals(portableValue, "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(portableValue, "yes", StringComparison.OrdinalIgnoreCase)
            || File.Exists(Path.Join(appBaseDirectory, PortableMarkerFileName))
            || Directory.Exists(Path.Join(appBaseDirectory, PortableDataDirectoryName));
    }

    private static TrackdubStorageOptions ResolveConfiguredRoots(
        string localAppDataRoot,
        string commonAppDataRoot)
    {
        StorageConfig? config = ReadStorageConfig(
            new DirectoryInfo(localAppDataRoot),
            new DirectoryInfo(commonAppDataRoot));
        if (config is null)
        {
            string defaultRoot = Path.Join(localAppDataRoot, ProductDirectoryName);
            return new TrackdubStorageOptions(defaultRoot, defaultRoot, SharedAssetRoot: null, IsPortable: false);
        }

        string configuredDataRoot = NormalizeOptionalPath(config.UserDataRoot)
            ?? Path.Join(localAppDataRoot, ProductDirectoryName);
        string configuredCacheRoot = NormalizeOptionalPath(config.UserCacheRoot)
            ?? configuredDataRoot;
        return new TrackdubStorageOptions(
            configuredDataRoot,
            configuredCacheRoot,
            NormalizeOptionalPath(config.SharedAssetRoot),
            config.IsPortable ?? config.Portable ?? false);
    }

    public static void ApplyToCurrentProcess(TrackdubStoragePaths storagePaths)
    {
        ArgumentNullException.ThrowIfNull(storagePaths);

        Environment.SetEnvironmentVariable(DataRootEnvironmentVariable, storagePaths.UserDataRoot);
        Environment.SetEnvironmentVariable(CacheRootEnvironmentVariable, storagePaths.UserCacheRoot);
        Environment.SetEnvironmentVariable(ToolCacheRootEnvironmentVariable, storagePaths.ToolCacheDirectory);
        Environment.SetEnvironmentVariable(EngineCacheRootEnvironmentVariable, storagePaths.EngineCacheDirectory);
        Environment.SetEnvironmentVariable(SharedAssetRootEnvironmentVariable, storagePaths.SharedAssetRoot);
        Environment.SetEnvironmentVariable(PortableEnvironmentVariable, storagePaths.IsPortable ? "1" : null);
    }

    /// <summary>
    /// Same as <see cref="ApplyToCurrentProcess"/>, but returns a handle that restores the
    /// prior environment variable values on <see cref="IDisposable.Dispose"/>. Intended for
    /// hosts (e.g. headless composition roots) that may be created and torn down multiple
    /// times within a single process, so overrides do not leak across instances.
    /// </summary>
    /// <remarks>
    /// Process environment variables are process-global state. Applies and disposals are
    /// tracked on a shared per-variable stack, so scopes can be disposed in any order: an
    /// older host disposing while a newer one is still active never clobbers the newer
    /// override, and once every scope has disposed the variable correctly rebases all the
    /// way back to the true pre-any-scope baseline — not an intermediate value some other
    /// scope already unwound past. A disposing scope restores that preceding value only
    /// when the process environment still contains the value that scope applied; a newer
    /// unscoped or external write is preserved. This still isn't true isolation: while two
    /// hosts with different storage overrides are alive at the same time, whichever applied
    /// last "wins" the shared variable, and both see that value regardless of which host is
    /// doing the reading. Callers that need per-host storage overrides and true concurrent
    /// isolation must run each host in its own process, or hold them one at a time.
    /// </remarks>
    public static IDisposable ApplyToCurrentProcessScoped(TrackdubStoragePaths storagePaths)
    {
        ArgumentNullException.ThrowIfNull(storagePaths);

        var newValues = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase)
        {
            [DataRootEnvironmentVariable] = storagePaths.UserDataRoot,
            [CacheRootEnvironmentVariable] = storagePaths.UserCacheRoot,
            [ToolCacheRootEnvironmentVariable] = storagePaths.ToolCacheDirectory,
            [EngineCacheRootEnvironmentVariable] = storagePaths.EngineCacheDirectory,
            [SharedAssetRootEnvironmentVariable] = storagePaths.SharedAssetRoot,
            [PortableEnvironmentVariable] = storagePaths.IsPortable ? "1" : null,
        };

        lock (EnvironmentStackSyncRoot)
        {
            var nodes = new Dictionary<string, LinkedListNode<string?>>(StringComparer.OrdinalIgnoreCase);
            foreach ((string key, string? value) in newValues)
            {
                if (!EnvironmentValueStacks.TryGetValue(key, out LinkedList<string?>? stack))
                {
                    // Seed the bottom of the stack with the true pre-any-scope baseline,
                    // captured the first time this variable is ever overridden.
                    stack = new LinkedList<string?>();
                    stack.AddFirst(Environment.GetEnvironmentVariable(key));
                    EnvironmentValueStacks[key] = stack;
                }

                nodes[key] = stack.AddLast(value);
                Environment.SetEnvironmentVariable(key, value);
            }

            return new ProcessEnvironmentRestoreScope(nodes);
        }
    }

    private sealed class ProcessEnvironmentRestoreScope(IReadOnlyDictionary<string, LinkedListNode<string?>> nodes) : IDisposable
    {
        public void Dispose()
        {
            lock (EnvironmentStackSyncRoot)
            {
                foreach (KeyValuePair<string, LinkedListNode<string?>> entry in nodes)
                {
                    RestoreEnvironmentValue(entry);
                }
            }
        }

        private static bool TryGetActiveStack(
            string key,
            LinkedListNode<string?> node,
            [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out LinkedList<string?>? stack)
        {
            if (!EnvironmentValueStacks.TryGetValue(key, out stack) || node.List != stack)
            {
                stack = null;
                return false;
            }

            return true;
        }

        private static void RestoreEnvironmentValue(KeyValuePair<string, LinkedListNode<string?>> entry)
        {
            string key = entry.Key;
            LinkedListNode<string?> node = entry.Value;
            if (!TryGetActiveStack(key, node, out LinkedList<string?>? stack))
            {
                return;
            }

            // Only rewrite the env var if this scope's entry was the topmost
            // (most recently applied) override. If a newer scope is still on top,
            // just remove ourselves from underneath it — the value it sees stays
            // untouched, and whichever scope eventually disposes as the new top
            // will correctly rebase to what's now beneath it (possibly the true
            // baseline), instead of an intermediate value some other scope had
            // already unwound.
            bool wasTop = ReferenceEquals(stack.Last, node);
            string? currentValue = Environment.GetEnvironmentVariable(key);
            stack.Remove(node);

            // A non-scoped caller may have intentionally changed the process
            // environment while this scope was active. Restore the preceding
            // scoped/baseline value only if the environment still contains the
            // exact value applied by this disposing scope; otherwise preserve the
            // newer write.
            if (wasTop)
            {
                if (string.Equals(currentValue, node.Value, StringComparison.Ordinal))
                {
                    Environment.SetEnvironmentVariable(key, stack.Last?.Value);
                }
            }

            // The bottom entry is always the true pre-any-scope baseline captured
            // the first time this variable was overridden. Once every scoped
            // override has unwound and only that baseline remains, drop the entry
            // entirely so a later ApplyToCurrentProcessScoped re-captures whatever
            // the process environment now holds as a fresh true baseline — rather
            // than reusing the stale first baseline and corrupting static consumers
            // if TRACKDUB_* changed between headless hosts.
            if (stack.Count == 1)
            {
                EnvironmentValueStacks.Remove(key);
            }
        }
    }

    private static StorageConfig? ReadStorageConfig(DirectoryInfo localAppDataRoot, DirectoryInfo commonAppDataRoot)
    {
        DirectoryInfo[] candidates =
        [
            new(Path.Join(localAppDataRoot.FullName, ProductDirectoryName)),
            new(Path.Join(commonAppDataRoot.FullName, ProductDirectoryName)),
        ];
        foreach (DirectoryInfo candidate in candidates)
        {
            StorageConfig? config = TryReadStorageConfig(
                new FileInfo(Path.Join(candidate.FullName, InstallerConfigFileName)));
            if (config is not null)
            {
                return config;
            }
        }

        return null;
    }

    private static StorageConfig? TryReadStorageConfig(FileInfo configFile)
    {
        if (!configFile.Exists)
        {
            return null;
        }

        try
        {
            using FileStream stream = configFile.OpenRead();
            StorageConfig? config = JsonSerializer.Deserialize(stream, InfrastructureSerializationContext.Default.StorageConfig);
            return config?.HasAnyRoot == true ? config : null;
        }
        catch (Exception ex) when (IsRecoverableConfigReadException(ex))
        {
            return null;
        }
    }

    private static bool IsRecoverableConfigReadException(Exception ex) =>
        ex is JsonException or IOException or UnauthorizedAccessException;

    private static string? ReadDefaultModelCacheDirectory(DirectoryInfo userDataRoot)
    {
        FileInfo settingsFile = new(Path.Join(userDataRoot.FullName, "settings.json"));
        return ReadDefaultModelCacheDirectory(settingsFile);
    }

    private static string? ReadDefaultModelCacheDirectory(FileInfo settingsFile)
    {
        if (!settingsFile.Exists)
        {
            return null;
        }

        try
        {
            using FileStream stream = settingsFile.OpenRead();
            using JsonDocument document = JsonDocument.Parse(stream);
            return ReadDefaultModelCacheDirectory(document);
        }
        catch (Exception ex) when (IsRecoverableSettingsReadException(ex))
        {
            return null;
        }
    }

    private static bool IsRecoverableSettingsReadException(Exception ex) =>
        ex is JsonException
            or IOException
            or UnauthorizedAccessException
            or ArgumentException
            or NotSupportedException;

    private static string? ReadDefaultModelCacheDirectory(JsonDocument settings)
    {
        JsonElement root = settings.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (!root.TryGetProperty("defaultModelCacheDirectory", out JsonElement value))
        {
            return null;
        }

        if (value.ValueKind != JsonValueKind.String)
        {
            return null;
        }

        return NormalizeSavedModelCacheDirectory(value.GetString());
    }

    private static IReadOnlyDictionary<string, string?> CaptureEnvironment()
    {
        var dictionary = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        IDictionary values = Environment.GetEnvironmentVariables();
        foreach (DictionaryEntry entry in values)
        {
            if (entry.Key is string key)
            {
                dictionary[key] = entry.Value as string;
            }
        }

        return dictionary;
    }

    private static string? GetValue(IReadOnlyDictionary<string, string?> environment, string name)
    {
        if (environment.TryGetValue(name, out string? value))
        {
            return value;
        }

        foreach ((string key, string? candidateValue) in environment)
        {
            if (string.Equals(key, name, StringComparison.OrdinalIgnoreCase))
            {
                return candidateValue;
            }
        }

        return null;
    }

    private static string NormalizeRequiredPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Storage path must not be empty.");
        }

        return Path.GetFullPath(Environment.ExpandEnvironmentVariables(path));
    }

    private static string? NormalizeOptionalPath(string? path) =>
        string.IsNullOrWhiteSpace(path)
            ? null
            : Path.GetFullPath(Environment.ExpandEnvironmentVariables(path));

    // A saved model-cache directory is kept only in absolute form. A relative value would
    // resolve against the working directory, so it is treated as unset on both save and read.
    internal static string? NormalizeSavedModelCacheDirectory(string? saved)
    {
        if (string.IsNullOrWhiteSpace(saved))
        {
            return null;
        }

        string expanded = Environment.ExpandEnvironmentVariables(saved.Trim());
        return Path.IsPathFullyQualified(expanded) ? Path.GetFullPath(expanded) : null;
    }

    // Kept internal (not private) so the JSON source generator in
    // InfrastructureSerializationContext can bind metadata for it at compile time.
    internal sealed class StorageConfig
    {
        public string? UserDataRoot { get; set; }

        public string? UserCacheRoot { get; set; }

        public string? SharedAssetRoot { get; set; }

        public bool? IsPortable { get; set; }

        public bool? Portable { get; set; }

        public bool HasAnyRoot =>
            HasAnyConfiguredRoot()
            || IsPortable is not null
            || Portable is not null;

        private bool HasAnyConfiguredRoot() =>
            new[] { UserDataRoot, UserCacheRoot, SharedAssetRoot }
                .Any(static value => !string.IsNullOrWhiteSpace(value));
    }
}
