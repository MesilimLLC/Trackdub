using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using Trackdub.Contracts;
using Trackdub.Infrastructure.Components;
using Trackdub.Infrastructure.Components.NvidiaAfx;

namespace Trackdub.Infrastructure.Tests.Components.NvidiaAfx;

public sealed class NvidiaAfxRuntimeDownloaderTests
{
    private const string SinglePartUrl = "https://cdn.example.com/afx/runtime.zip";

    [Fact]
    public async Task DownloadAndInstallAsync_ReassemblesAndVerifiesArchiveParts()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-afx-download-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        try
        {
            byte[] archive = CreateArchive();
            int split = archive.Length / 2;
            byte[] firstPart = archive[..split];
            byte[] secondPart = archive[split..];
            string baseUrl = SinglePartUrl;
            var handler = new PartHandler(
                new Uri(baseUrl + ".part001"), firstPart,
                new Uri(baseUrl + ".part002"), secondPart);
            using var httpClient = new HttpClient(handler);
            var store = new ComponentStore(tempRoot, new NoopLogger());
            var downloader = new NvidiaAfxRuntimeDownloader(store, httpClient, new NoopLogger(), static () => false);
            NvidiaAfxRuntimePackage package = CreatePackage(archive, archive.Length, partCount: 2);

            string installPath = await downloader.DownloadAndInstallAsync(package, null, CancellationToken.None);

            Assert.Equal("fixture runtime", await File.ReadAllTextAsync(Path.Join(installPath, "runtime.txt")));
            Assert.Equal([new Uri(baseUrl + ".part001"), new Uri(baseUrl + ".part002")], handler.RequestedUris);
            Assert.True(store.IsInstalled(NvidiaAfxRuntimeDownloader.ComponentId));
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    [Fact]
    public Task DownloadAndInstallAsync_RejectsContentLargerThanDeclaredSize() =>
        AssertInvalidArchiveAsync(exceedsDeclaredSize: true);

    [Fact]
    public Task DownloadAndInstallAsync_RemovesPartialArchiveWhenVerificationFails() =>
        AssertInvalidArchiveAsync(exceedsDeclaredSize: false);

    private static Task AssertInvalidArchiveAsync(bool exceedsDeclaredSize) =>
        WithSinglePartDownloaderAsync(
            archive => exceedsDeclaredSize
                ? CreatePackage(archive, archive.Length - 1)
                : CreatePackage(archive, archive.Length, new string('a', 64)),
            async (downloader, store, package) =>
            {
                InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
                    () => downloader.DownloadAndInstallAsync(package, null, CancellationToken.None));

                if (exceedsDeclaredSize)
                {
                    Assert.Contains("exceeds the declared size", error.Message, StringComparison.Ordinal);
                }
                else
                {
                    Assert.False(store.IsInstalled(NvidiaAfxRuntimeDownloader.ComponentId));
                }

                Assert.Empty(Directory.EnumerateFiles(store.GetComponentDirectory(NvidiaAfxRuntimeDownloader.ComponentId)));
            });

    private static async Task WithSinglePartDownloaderAsync(
        Func<byte[], NvidiaAfxRuntimePackage> createPackage,
        Func<NvidiaAfxRuntimeDownloader, ComponentStore, NvidiaAfxRuntimePackage, Task> testAsync)
    {
        string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-afx-download-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        try
        {
            byte[] archive = CreateArchive();
            using var httpClient = new HttpClient(new StaticHandler(new Uri(SinglePartUrl), archive));
            var store = new ComponentStore(tempRoot, new NoopLogger());
            var downloader = new NvidiaAfxRuntimeDownloader(store, httpClient, new NoopLogger(), static () => false);
            await testAsync(downloader, store, createPackage(archive));
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    private static NvidiaAfxRuntimePackage CreatePackage(
        byte[] archive,
        long sizeBytes,
        string? sha256 = null,
        int partCount = 1) =>
        new(
            Architecture: "ada",
            DownloadUrl: SinglePartUrl,
            Sha256: sha256 ?? Convert.ToHexString(SHA256.HashData(archive)).ToLowerInvariant(),
            SizeBytes: sizeBytes,
            RuntimeVersion: "3.0.0",
            LicenseUrl: "https://example.com/license",
            ModelRelativePaths: [],
            PartCount: partCount);

    private static void DeleteTempRoot(string tempRoot)
    {
        if (Directory.Exists(tempRoot))
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static byte[] CreateArchive()
    {
        using var stream = new MemoryStream();
        using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        using (Stream entry = archive.CreateEntry("runtime.txt").Open())
        using (var writer = new StreamWriter(entry))
        {
            writer.Write("fixture runtime");
        }

        return stream.ToArray();
    }

    private sealed class PartHandler(Uri firstUrl, byte[] firstPart, Uri secondUrl, byte[] secondPart) : HttpMessageHandler
    {
        private readonly Dictionary<Uri, byte[]> _parts = new()
        {
            [firstUrl] = firstPart,
            [secondUrl] = secondPart,
        };

        public List<Uri> RequestedUris { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Uri requestUri = request.RequestUri!;
            RequestedUris.Add(requestUri);
            HttpResponseMessage response = _parts.TryGetValue(requestUri, out byte[]? content)
                ? new(HttpStatusCode.OK) { Content = new ByteArrayContent(content) }
                : new(HttpStatusCode.NotFound);
            return Task.FromResult(response);
        }
    }

    private sealed class StaticHandler(Uri url, byte[] content) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            HttpResponseMessage response = request.RequestUri == url
                ? new(HttpStatusCode.OK) { Content = new ByteArrayContent(content) }
                : new(HttpStatusCode.NotFound);
            return Task.FromResult(response);
        }
    }

    private sealed class NoopLogger : IApplicationLogger
    {
        public void LogDebug(string message) { }
        public void LogInformation(string message) { }
        public void LogWarning(string message, Exception? exception = null) { }
        public void LogError(string message, Exception? exception = null) { }
    }
}
