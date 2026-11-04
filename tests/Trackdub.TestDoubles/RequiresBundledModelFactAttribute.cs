using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Trackdub.TestDoubles;

/// <summary>
/// Marks an xUnit test that needs one or more on-disk bundled models under
/// the repo's gitignored <c>models/</c> directory. If any required path is
/// missing the test is skipped, so <c>dotnet test</c> works cleanly on
/// machines (and CI agents) that haven't downloaded the bundle.
/// </summary>
/// <remarks>
/// The attribute evaluates at test-discovery time via <see cref="FactAttribute.Skip"/>,
/// so missing models never touch the engine code paths under test.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class RequiresBundledModelFactAttribute : FactAttribute
{
    // Single-path overload: preferred by compiler over params, passes source info to base (xunit.v3 xUnit3003).
    public RequiresBundledModelFactAttribute(
        string relativePath,
        [CallerFilePath] string sourceFilePath = "",
        [CallerLineNumber] int sourceLineNumber = 0)
    {
        Skip = BundledModelSkipResolver.Resolve([relativePath]);
    }

    // Multi-path overload: used when multiple model paths must be checked.
    public RequiresBundledModelFactAttribute(params string[] relativePaths)
    {
        Skip = BundledModelSkipResolver.Resolve(relativePaths);
    }
}

/// <summary>
/// Theory variant of <see cref="RequiresBundledModelFactAttribute"/>.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class RequiresBundledModelTheoryAttribute : TheoryAttribute
{
    // Single-path overload: preferred by compiler over params, passes source info to base (xunit.v3 xUnit3003).
    public RequiresBundledModelTheoryAttribute(
        string relativePath,
        [CallerFilePath] string sourceFilePath = "",
        [CallerLineNumber] int sourceLineNumber = 0)
    {
        Skip = BundledModelSkipResolver.Resolve([relativePath]);
    }

    // Multi-path overload: used when multiple model paths must be checked.
    public RequiresBundledModelTheoryAttribute(params string[] relativePaths)
    {
        Skip = BundledModelSkipResolver.Resolve(relativePaths);
    }
}

/// <summary>
/// Marks an xUnit test that only makes sense on Windows (for example, tests
/// that parse Windows-style path separators or assert Windows-specific I/O
/// behaviour). On non-Windows runs the test is skipped instead of failing.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class WindowsOnlyFactAttribute : FactAttribute
{
    public WindowsOnlyFactAttribute(
        [CallerFilePath] string sourceFilePath = "",
        [CallerLineNumber] int sourceLineNumber = 0)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Skip = "Windows-only test (skipped on non-Windows test runs).";
        }
    }
}

internal static class BundledModelSkipResolver
{
    internal static string? Resolve(string[] relativePaths)
    {
        ArgumentNullException.ThrowIfNull(relativePaths);

        var repoRoot = TestRepoRootResolver.TryFindRepoRoot();
        if (repoRoot is null)
        {
            return "Unable to locate Trackdub.sln from the test runner base directory; skipping bundled-model test.";
        }

        foreach (var relative in relativePaths)
        {
            var full = Path.GetFullPath(Path.Join(repoRoot, "models", relative));
            if (Directory.Exists(full) || File.Exists(full))
            {
                continue;
            }

            return $"Required bundled model not present at {Path.Join("models", relative)}. Download the model bundle (gitignored under models/) to run this test.";
        }

        return null;
    }
}

/// <summary>
/// Locates the repository root by walking parents of <see cref="AppContext.BaseDirectory"/>
/// until <c>Trackdub.sln</c> is found. Works for any test output layout depth (for example
/// <c>bin\x64\Debug\net11.0-windows10.0.19041.0\</c> vs <c>bin\Debug\net11.0\</c>).
/// </summary>
public static class TestRepoRootResolver
{
    public static string? TryFindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Join(dir.FullName, "Trackdub.slnx")))
            {
                return dir.FullName;
            }
        }

        return null;
    }

    public static string FindRepoRoot() =>
        TryFindRepoRoot()
        ?? throw new InvalidOperationException(
            "Unable to locate Trackdub.sln by walking parents from AppContext.BaseDirectory.");
}
