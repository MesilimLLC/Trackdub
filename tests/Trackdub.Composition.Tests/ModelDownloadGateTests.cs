using System.Reflection;
using Trackdub.Infrastructure.Settings;
using Trackdub.Composition.Runtime;
using Trackdub.Contracts.Licensing;
using Trackdub.Domain;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.TestDoubles;

namespace Trackdub.Composition.Tests;

/// <summary>
/// The commercial gate on download and model-manager paths (MODEL_LICENSE_POLICY.md, accepted-risk exception): a model is
/// downloadable and auto-downloadable only when commercial use is allowed and it is verified or carries a risk acceptance.
/// </summary>
public sealed class ModelDownloadGateTests : IDisposable
{
    private const string Sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private const string RefusalText = "not cleared for commercial use";

    private readonly string tempRoot = Path.Join(Path.GetTempPath(), "Trackdub.ModelDownloadGate.Tests", Guid.NewGuid().ToString("N"));

    public enum Standing
    {
        Verified,
        RiskAccepted,
        Neither,
        VerifiedButNotCommercial,
    }

    [Theory]
    [InlineData(Standing.Verified, true)]
    [InlineData(Standing.RiskAccepted, true)]
    [InlineData(Standing.Neither, false)]
    [InlineData(Standing.VerifiedButNotCommercial, false)]
    public void Gate_clears_only_verified_or_risk_accepted_commercial_models(Standing standing, bool cleared)
    {
        (BundledModelManifestRegistry registry, _) = CreateRegistry(standing);
        BundledModelManifestEntry entry = Assert.Single(registry.Entries);

        Assert.Equal(cleared, ModelDownloadManifestFiles.IsCommerciallyCleared(entry));
        Assert.Equal(cleared, ModelDownloadManifestFiles.CanAutoDownloadAll(entry));
        Assert.Equal(cleared, ModelDownloadManifestFiles.CanAutoDownloadFile(entry, "model.onnx"));
    }

    [Theory]
    [InlineData(Standing.Neither, true)]
    [InlineData(Standing.VerifiedButNotCommercial, true)]
    [InlineData(Standing.RiskAccepted, false)]
    public async Task Download_is_refused_for_uncleared_models_before_anything_is_fetched(Standing standing, bool refused)
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths) = CreateRegistry(standing);
        IModelDownloaderContract downloader = DispatchProxy.Create<IModelDownloaderContract, CountingDownloader>();
        using ModelDownloadOrchestrator orchestrator = new(registry, new LocalModelCacheRecordStore(storagePaths), downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(refused, result.FailureReason?.Contains(RefusalText, StringComparison.Ordinal) ?? false);
        Assert.True(!refused || ((CountingDownloader)downloader).Calls == 0);
    }

    [Theory]
    [InlineData(Standing.Neither, false)]
    [InlineData(Standing.RiskAccepted, true)]
    public async Task Inventory_offers_auto_download_only_to_cleared_models(Standing standing, bool offered)
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths) = CreateRegistry(standing);
        ModelInventoryService service = new(registry, new LocalModelCacheRecordStore(storagePaths), storagePaths);

        ModelInventoryEntry? entry = await service.GetByModelIdAsync("example/model", TestContext.Current.CancellationToken);

        Assert.NotNull(entry);
        Assert.Equal(offered, entry!.CanAutoDownload);
        Assert.Equal(offered, entry.CommercialClearance);
        Assert.Equal(!offered, entry.FailureReason?.Contains(RefusalText, StringComparison.Ordinal) ?? false);
    }

    public void Dispose()
    {
        if (Directory.Exists(tempRoot))
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private (BundledModelManifestRegistry Registry, TrackdubStoragePaths StoragePaths) CreateRegistry(Standing standing)
    {
        TrackdubStoragePaths storagePaths = new(tempRoot);
        string manifestPath = Path.Join(tempRoot, "manifest", "bundled-models.manifest.json");
        Directory.CreateDirectory(Path.GetDirectoryName(manifestPath)!);
        Directory.CreateDirectory(Path.Join(storagePaths.ModelCacheDirectory, "example-model"));
        File.WriteAllText(manifestPath, ManifestJson(standing));
        return (BundledModelManifestRegistry.Load(manifestPath), storagePaths);
    }

    private static string ManifestJson(Standing standing)
    {
        string risk = standing is Standing.RiskAccepted
            ? """
              ,
                  "risk_acceptance": {
                    "decision_record": "test decision",
                    "owner": "owner",
                    "date": "2026-11-19",
                    "accepted_risks": [ "licence indemnity", "termination", "terms change", "provenance" ]
                  }
              """
            : string.Empty;
        bool verified = standing is Standing.Verified or Standing.VerifiedButNotCommercial;
        bool commercial = standing is not Standing.VerifiedButNotCommercial;
        return $$"""
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "separation",
                  "engine_family": "example",
                  "license": "custom",
                  "commercial_allowed": {{(commercial ? "true" : "false")}},
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": {{(verified ? "true" : "false")}},
                  "source_url": "https://huggingface.co/example/model",
                  "revision": "main",
                  "sha256": "{{Sha256}}",
                  "aliases": [ "example" ],
                  "root_path": "../example-model",
                  "benchmark_entry": "model.onnx",
                  "download_files": [ "model.onnx" ],
                  "variants": []{{risk}}
                }
              ]
            }
            """;
    }

    /// <summary>Counts every call made on the downloader contract and fails each one, so a refused download shows no fetch.</summary>
    public class CountingDownloader : DispatchProxy
    {
        public int Calls { get; private set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            Calls++;
            return Task.FromResult(false);
        }
    }
}
