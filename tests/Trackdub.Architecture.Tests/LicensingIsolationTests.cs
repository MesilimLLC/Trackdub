using System.Xml.Linq;

namespace Trackdub.Architecture.Tests;

/// <summary>
/// Enforces that <c>Trackdub.Licensing</c> remains a standalone, zero-dependency library:
/// no project references to other Trackdub projects, no third-party crypto packages,
/// and targets only <c>net11.0</c>.
/// </summary>
public sealed class LicensingIsolationTests
{
    private static readonly string[] ForbiddenCryptoPackages =
    [
        "BouncyCastle",
        "BouncyCastle.Cryptography",
        "Portable.BouncyCastle",
        "jose-jwt",
        "Microsoft.IdentityModel.Tokens",
        "Microsoft.IdentityModel.JsonWebTokens",
        "System.IdentityModel.Tokens.Jwt",
        "NSec.Cryptography",
        "Sodium.Core",
        "libsodium"
    ];

    [Fact]
    public void LicensingHasNoProjectReferences()
    {
        var csproj = LoadLicensingCsproj();

        var projectReferences = csproj
            .Descendants("ProjectReference")
            .Select(el => el.Attribute("Include")?.Value)
            .Where(v => v is not null)
            .Cast<string>()
            .ToArray();

        Assert.True(
            projectReferences.Length == 0,
            "Trackdub.Licensing must have zero ProjectReference elements to remain standalone. Found: " +
            string.Join(", ", projectReferences));
    }

    [Fact]
    public void LicensingHasNoThirdPartyCryptoPackages()
    {
        var csproj = LoadLicensingCsproj();

        var cryptoReferences = csproj
            .Descendants("PackageReference")
            .Select(el => el.Attribute("Include")?.Value)
            .Where(package => package is not null &&
                              ForbiddenCryptoPackages.Any(forbidden =>
                                  package.Contains(forbidden, StringComparison.OrdinalIgnoreCase)))
            .Cast<string>()
            .ToArray();

        Assert.True(
            cryptoReferences.Length == 0,
            "Trackdub.Licensing must use BCL-only crypto (System.Security.Cryptography). Found forbidden packages: " +
            string.Join(", ", cryptoReferences));
    }

    [Fact]
    public void LicensingTargetsOnlyNet11()
    {
        var repoRoot = FindRepoRoot();
        var csproj = LoadLicensingCsproj();

        // The project must not multi-target.
        var targetFrameworks = csproj
            .Descendants("TargetFrameworks")
            .Select(el => el.Value)
            .FirstOrDefault();

        Assert.True(
            targetFrameworks is null,
            "Trackdub.Licensing must not multi-target. Found <TargetFrameworks>: " + targetFrameworks);

        // Resolve effective TargetFramework: csproj override or Directory.Build.props inherited value.
        var localTargetFramework = csproj
            .Descendants("TargetFramework")
            .Select(el => el.Value)
            .FirstOrDefault();

        var effectiveTargetFramework = localTargetFramework
            ?? ResolveDirectoryBuildPropsTargetFramework(repoRoot);

        Assert.Equal("net11.0", effectiveTargetFramework);
    }

    private static XDocument LoadLicensingCsproj()
    {
        var repoRoot = FindRepoRoot();
        var csprojPath = Path.Join(repoRoot, "src", "Trackdub.Licensing", "Trackdub.Licensing.csproj");
        Assert.True(File.Exists(csprojPath), $"Trackdub.Licensing.csproj not found at: {csprojPath}");
        return XDocument.Load(csprojPath);
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Join(dir.FullName, "Trackdub.slnx")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        throw new InvalidOperationException("Could not locate Trackdub.slnx from the test runner base directory.");
    }

    private static string? ResolveDirectoryBuildPropsTargetFramework(string repoRoot)
    {
        var propsPath = Path.Join(repoRoot, "Directory.Build.props");
        if (!File.Exists(propsPath))
        {
            return null;
        }

        var props = XDocument.Load(propsPath);
        return props
            .Descendants("TargetFramework")
            .Select(el => el.Value)
            .FirstOrDefault();
    }
}
