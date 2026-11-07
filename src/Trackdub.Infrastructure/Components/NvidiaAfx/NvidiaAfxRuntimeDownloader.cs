using System.IO.Compression;
using System.Security.Cryptography;
using Trackdub.Contracts;
using Trackdub.Infrastructure.Components;

namespace Trackdub.Infrastructure.Components.NvidiaAfx;

public sealed class NvidiaAfxRuntimeDownloader(
    ComponentStore componentStore,
    HttpClient httpClient,
    IApplicationLogger logger,
    Func<bool>? isStubbed = null)
{
    private readonly Func<bool> _isStubbed = isStubbed ?? NvidiaAfxIntegration.IsStubbed;

    public const string ComponentId = "nvidia-afx-runtime";
    private const string TempSuffix = ".downloading";
    private const string StagingSuffix = ".staging";

    public async Task<string> DownloadAndInstallAsync(
        NvidiaAfxRuntimePackage package,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(package);

        if (_isStubbed())
        {
            throw new InvalidOperationException(NvidiaAfxIntegration.StubReason);
        }

        if (!NvidiaAfxRuntimePackageGates.IsDownloadable(package))
        {
            throw new InvalidOperationException(NvidiaAfxRuntimePackageGates.DescribeBlocker(package));
        }

        string componentDirectory = componentStore.GetComponentDirectory(ComponentId);
        string archivePath = Path.Join(componentDirectory, $"{package.Architecture}{TempSuffix}");
        Directory.CreateDirectory(componentDirectory);

        try
        {
            await DownloadArchiveAsync(package, archivePath, progress, cancellationToken).ConfigureAwait(false);
            await VerifyArchiveAsync(archivePath, package.Sha256, package.SizeBytes, cancellationToken).ConfigureAwait(false);

            string installPath = await ExtractAtomicallyAsync(componentDirectory, archivePath, cancellationToken).ConfigureAwait(false);
            componentStore.MarkInstalled(ComponentId);
            progress?.Report(1.0);
            logger.LogInformation($"NVIDIA AFX runtime package installed at '{installPath}' ({package.Architecture}).");
            return installPath;
        }
        catch
        {
            TryDelete(archivePath);
            throw;
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Preserve the original failure; a leftover partial archive is retried and truncated
            // by the next download.
        }
    }

    private async Task DownloadArchiveAsync(
        NvidiaAfxRuntimePackage package,
        string destinationPath,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        await using FileStream output = File.Create(destinationPath);
        byte[] buffer = new byte[65536];
        long written = 0;
        for (int partIndex = 1; partIndex <= package.PartCount; partIndex++)
        {
            string partUrl = package.PartCount == 1
                ? package.DownloadUrl
                : BuildPartUrl(package.DownloadUrl, partIndex);
            using HttpResponseMessage response = await httpClient.GetAsync(
                partUrl,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            await using Stream contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            int read;
            while ((read = await contentStream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken).ConfigureAwait(false)) > 0)
            {
                if (package.SizeBytes > 0 && written + read > package.SizeBytes)
                {
                    throw new InvalidOperationException(
                        $"NVIDIA AFX runtime archive exceeds the declared size of {package.SizeBytes} bytes.");
                }

                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                written += read;
                if (package.SizeBytes > 0)
                {
                    progress?.Report((double)written / package.SizeBytes * 0.9d);
                }
            }
        }
    }

    private static string BuildPartUrl(string url, int partIndex)
    {
        var builder = new UriBuilder(url);
        builder.Path += $".part{partIndex:D3}";
        return builder.Uri.AbsoluteUri;
    }

    private static async Task VerifyArchiveAsync(
        string archivePath,
        string expectedSha256,
        long expectedBytes,
        CancellationToken cancellationToken)
    {
        var fileInfo = new FileInfo(archivePath);
        if (expectedBytes > 0 && fileInfo.Length != expectedBytes)
        {
            throw new InvalidOperationException($"NVIDIA AFX runtime archive size mismatch. Expected {expectedBytes}, got {fileInfo.Length}.");
        }

        await using FileStream stream = File.OpenRead(archivePath);
        string actualHash = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false)).ToLowerInvariant();
        if (!string.Equals(actualHash, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("NVIDIA AFX runtime archive checksum verification failed.");
        }
    }

    private static async Task<string> ExtractAtomicallyAsync(
        string componentDirectory,
        string archivePath,
        CancellationToken cancellationToken)
    {
        string parent = Path.GetDirectoryName(componentDirectory) ?? componentDirectory;
        string staging = Path.Join(parent, $"{Path.GetFileName(componentDirectory)}{StagingSuffix}");
        if (Directory.Exists(staging))
        {
            Directory.Delete(staging, recursive: true);
        }

        Directory.CreateDirectory(staging);
        await Task.Run(() => ZipFile.ExtractToDirectory(archivePath, staging, overwriteFiles: true), cancellationToken).ConfigureAwait(false);

        if (Directory.Exists(componentDirectory))
        {
            Directory.Delete(componentDirectory, recursive: true);
        }

        Directory.Move(staging, componentDirectory);
        return componentDirectory;
    }
}
