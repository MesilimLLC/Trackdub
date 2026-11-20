using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.DeepFilterNet;

// Tensor names as consumed by DeepFilterNetOnnxInference (see
// scripts/inspect-deepfilternet-onnx.py for the original exports). Sessions rewrite the
// exports into stateful graphs (DeepFilterNetOnnxGraphTransform): GRU states become explicit
// state_in_* inputs and existing Y_h outputs, and the causal input pads are removed, so the caller
// supplies the lookback rows (2 for feat_erb/feat_spec, 4 for c0) itself.
// enc.onnx:
//   input  feat_erb  [1,1,S+2,32]  float32
//   input  feat_spec [1,2,S+2,96]  float32  (NbDf complex bins)
//   output emb, e0, e1, e2, e3, c0, lsnr
// erb_dec.onnx:
//   input  emb, e3, e2, e1, e0
//   output erb_gains [1,1,S,32]    float32
// df_dec.onnx:
//   input  emb, c0 (with lookback rows)
//   output df_coefs  [1,S,5,96,2]  float32
public sealed record DeepFilterNetModelPaths(
    string RootDirectory,
    string EncPath,
    string ErbDecPath,
    string DfDecPath,
    bool CommercialAllowed,
    bool CommercialUseVerified)
{
    public bool RiskAccepted { get; init; }

    private const string EngineAlias = "deepfilternet3";

    public static DeepFilterNetModelPaths? TryResolve(BundledModelManifestRegistry? registry) =>
        TryResolve(registry, cacheRecords: null);

    public static async Task<DeepFilterNetModelPaths?> TryResolveAsync(
        BundledModelManifestRegistry? registry,
        IModelCacheInventory? modelCacheInventory,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (registry is null ||
            !registry.TryResolve(EngineAlias, out BundledModelManifestResolution? resolution))
        {
            return null;
        }

        IReadOnlyList<LocalModelCacheRecord>? cacheRecords = null;
        if (modelCacheInventory is not null)
        {
            cacheRecords = await modelCacheInventory.LoadAsync(cancellationToken).ConfigureAwait(false);
        }

        return TryResolve(registry, cacheRecords);
    }

    internal static DeepFilterNetModelPaths? TryResolve(
        BundledModelManifestRegistry? registry,
        IReadOnlyList<LocalModelCacheRecord>? cacheRecords)
    {
        if (registry is null ||
            !registry.TryResolve(EngineAlias, out BundledModelManifestResolution? resolution))
        {
            return null;
        }

        BundledModelManifestEntry entry = resolution!.Entry;
        string? root = TryResolveRootDirectory(entry, cacheRecords);
        if (root is null)
        {
            return null;
        }

        return new DeepFilterNetModelPaths(
            root,
            Path.Join(root, "enc.onnx"),
            Path.Join(root, "erb_dec.onnx"),
            Path.Join(root, "df_dec.onnx"),
            entry.CommercialAllowed,
            entry.CommercialUseVerified)
        { RiskAccepted = entry.RiskAcceptance is not null };
    }

    public bool IsCommercialSafe => CommercialAllowed && (CommercialUseVerified || RiskAccepted);

    public bool AllFilesExist() =>
        File.Exists(EncPath) && File.Exists(ErbDecPath) && File.Exists(DfDecPath);

    private static string? TryResolveRootDirectory(
        BundledModelManifestEntry entry,
        IReadOnlyList<LocalModelCacheRecord>? cacheRecords)
    {
        if (AllOnnxFilesExist(entry.RootDirectory))
        {
            return entry.RootDirectory;
        }

        if (cacheRecords is null)
        {
            return null;
        }

        foreach (LocalModelCacheRecord cacheRecord in cacheRecords)
        {
            if (!cacheRecord.ModelId.Equals(entry.ModelId, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (AllOnnxFilesExist(cacheRecord.RootPath))
            {
                return cacheRecord.RootPath;
            }
        }

        return null;
    }

    private static bool AllOnnxFilesExist(string rootDirectory) =>
        File.Exists(Path.Join(rootDirectory, "enc.onnx")) &&
        File.Exists(Path.Join(rootDirectory, "erb_dec.onnx")) &&
        File.Exists(Path.Join(rootDirectory, "df_dec.onnx"));
}
