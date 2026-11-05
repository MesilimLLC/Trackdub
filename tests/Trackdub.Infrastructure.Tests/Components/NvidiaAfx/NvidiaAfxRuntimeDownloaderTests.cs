using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using Trackdub.Contracts;
using Trackdub.Infrastructure.Components;
using Trackdub.Infrastructure.Components.NvidiaAfx;

namespace Trackdub.Infrastructure.Tests.Components.NvidiaAfx;

public sealed class NvidiaAfxRuntimeDownloaderTests
{
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
            string baseUrl = "https://cdn.example.com/afx/runtime.zip";
            var handler = new PartHandler(
                new Uri(baseUrl + ".part001"), firstPart,
                new Uri(baseUrl + ".part002"), secondPart);
            using var httpClient = new HttpClient(handler);
            var store = new ComponentStore(tempRoot, new NoopLogger());
            var downloader = new NvidiaAfxRuntimeDownloader(store, httpClient, new NoopLogger(), static () => false);
            var package = new NvidiaAfxRuntimePackage(
                Architecture: "ada",
                DownloadUrl: baseUrl,
                Sha256: Convert.ToHexString(SHA256.HashData(archive)).ToLowerInvariant(),
                SizeBytes: archive.Length,
                RuntimeVersion: "3.0.0",
                LicenseUrl: "https://example.com/license",
                ModelRelativePaths: [],
                PartCount: 2);

            string installPath = await downloader.DownloadAndInstallAsync(package, null, CancellationToken.None);

            Assert.Equal("fixture runtime", await File.ReadAllTextAsync(Path.Join(installPath, "runtime.txt")));
            Assert.Equal([new Uri(baseUrl + ".part001"), new Uri(baseUrl + ".part002")], handler.RequestedUris);
            Assert.True(store.IsInstalled(NvidiaAfxRuntimeDownloader.ComponentId));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task DownloadAndInstallAsync_RejectsContentLargerThanDeclaredSize()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-afx-download-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        try
        {
            byte[] archive = CreateArchive();
            string url = "https://cdn.example.com/afx/runtime.zip";
            using var httpClient = new HttpClient(new StaticHandler(new Uri(url), archive));
            var store = new ComponentStore(tempRoot, new NoopLogger());
            var downloader = new NvidiaAfxRuntimeDownloader(store, httpClient, new NoopLogger(), static () => false);
            var package = new NvidiaAfxRuntimePackage(
                Architecture: "ada",
                DownloadUrl: url,
                Sha256: Convert.ToHexString(SHA256.HashData(archive)).ToLowerInvariant(),
                SizeBytes: archive.Length - 1,
                RuntimeVersion: "3.0.0",
                LicenseUrl: "https://example.com/license",
                ModelRelativePaths: [],
                PartCount: 1);

            InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
                () => downloader.DownloadAndInstallAsync(package, null, CancellationToken.None));

            Assert.Contains("exceeds the declared size", error.Message, StringComparison.Ordinal);
            Assert.Empty(Directory.EnumerateFiles(store.GetComponentDirectory(NvidiaAfxRuntimeDownloader.ComponentId)));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task DownloadAndInstallAsync_RemovesPartialArchiveWhenVerificationFails()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-afx-download-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        try
        {
            byte[] archive = CreateArchive();
            string url = "https://cdn.example.com/afx/runtime.zip";
            using var httpClient = new HttpClient(new StaticHandler(new Uri(url), archive));
            var store = new ComponentStore(tempRoot, new NoopLogger());
            var downloader = new NvidiaAfxRuntimeDownloader(store, httpClient, new NoopLogger(), static () => false);
            var package = new NvidiaAfxRuntimePackage(
                Architecture: "ada",
                DownloadUrl: url,
                Sha256: new string('a', 64),
                SizeBytes: archive.Length,
                RuntimeVersion: "3.0.0",
                LicenseUrl: "https://example.com/license",
                ModelRelativePaths: [],
                PartCount: 1);

            await Assert.ThrowsAsync<InvalidOperationException>(
                () => downloader.DownloadAndInstallAsync(package, null, CancellationToken.None));

            Assert.False(store.IsInstalled(NvidiaAfxRuntimeDownloader.ComponentId));
            Assert.Empty(Directory.EnumerateFiles(store.GetComponentDirectory(NvidiaAfxRuntimeDownloader.ComponentId)));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
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
