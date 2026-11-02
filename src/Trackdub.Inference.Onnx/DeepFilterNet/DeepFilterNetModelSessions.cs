using System.Security.Cryptography;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.DeepFilterNet;

/// <summary>
/// Sessions over the three DeepFilterNet3 ONNX exports, rewritten by
/// <see cref="DeepFilterNetOnnxGraphTransform"/> so GRU hidden states are explicit graph
/// inputs/outputs and causal input pads are neutralized. The transformed model bytes are
/// derived deterministically from the cached originals and cached under the user's local
/// application-data directory (keyed by the original content hash), so the original model
/// files are never modified. The transform still runs on every session creation; only the
/// verified cache write is skipped when the file already matches.
/// </summary>
internal sealed class DeepFilterNetModelSessions(
    OnnxExecutionSessionFactory.SingleSessionLease enc,
    OnnxExecutionSessionFactory.SingleSessionLease erbDec,
    OnnxExecutionSessionFactory.SingleSessionLease dfDec,
    IReadOnlyList<string> encStateInputs,
    IReadOnlyList<string> encStateOutputs,
    IReadOnlyList<string> erbStateInputs,
    IReadOnlyList<string> erbStateOutputs,
    IReadOnlyList<string> dfStateInputs,
    IReadOnlyList<string> dfStateOutputs) : IDisposable
{
    // Bump when the transform logic changes; the derived-model cache is keyed by this salt
    // plus the original model hash, so stale derived copies are regenerated automatically.
    private const string TransformVersion = "v2";

    public OnnxExecutionSessionFactory.SingleSessionLease Enc { get; } = enc;
    public OnnxExecutionSessionFactory.SingleSessionLease ErbDec { get; } = erbDec;
    public OnnxExecutionSessionFactory.SingleSessionLease DfDec { get; } = dfDec;

    public IReadOnlyList<string> EncStateInputs { get; } = encStateInputs;
    public IReadOnlyList<string> EncStateOutputs { get; } = encStateOutputs;
    public IReadOnlyList<string> ErbStateInputs { get; } = erbStateInputs;
    public IReadOnlyList<string> ErbStateOutputs { get; } = erbStateOutputs;
    public IReadOnlyList<string> DfStateInputs { get; } = dfStateInputs;
    public IReadOnlyList<string> DfStateOutputs { get; } = dfStateOutputs;

    public static async Task<DeepFilterNetModelSessions> CreateAsync(
        DeepFilterNetModelPaths paths,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        OnnxExecutionSessionFactory.SingleSessionLease? enc = null;
        OnnxExecutionSessionFactory.SingleSessionLease? erbDec = null;
        OnnxExecutionSessionFactory.SingleSessionLease? dfDec = null;
        try
        {
            (enc, IReadOnlyList<string> encIn, IReadOnlyList<string> encOut) = await CreateTransformedAsync(
                "deepfilternet3-enc", paths.EncPath, "enc", provider, cancellationToken).ConfigureAwait(false);
            (erbDec, IReadOnlyList<string> erbIn, IReadOnlyList<string> erbOut) = await CreateTransformedAsync(
                "deepfilternet3-erb-dec", paths.ErbDecPath, "erb", provider, cancellationToken).ConfigureAwait(false);
            (dfDec, IReadOnlyList<string> dfIn, IReadOnlyList<string> dfOut) = await CreateTransformedAsync(
                "deepfilternet3-df-dec", paths.DfDecPath, "df", provider, cancellationToken).ConfigureAwait(false);

            return new DeepFilterNetModelSessions(
                enc, erbDec, dfDec, encIn, encOut, erbIn, erbOut, dfIn, dfOut);
        }
        catch
        {
            enc?.Dispose();
            erbDec?.Dispose();
            dfDec?.Dispose();
            throw;
        }
    }

    private static async Task<(OnnxExecutionSessionFactory.SingleSessionLease Session, IReadOnlyList<string> StateInputs, IReadOnlyList<string> StateOutputs)>
        CreateTransformedAsync(
            string engineFamily,
            string modelPath,
            string prefix,
            ExecutionProviderKind provider,
            CancellationToken cancellationToken)
    {
        byte[] original = await File.ReadAllBytesAsync(modelPath, cancellationToken).ConfigureAwait(false);
        (byte[] transformed, IReadOnlyList<string> stateInputs, IReadOnlyList<string> stateOutputs) =
            DeepFilterNetOnnxGraphTransform.Transform(original, prefix);

        string cachePath = Path.Join(
            GetDerivedCacheDirectory(modelPath, original),
            Path.GetFileName(modelPath));
        Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
        if (!await CacheMatchesAsync(cachePath, transformed, cancellationToken).ConfigureAwait(false))
        {
            string tempPath = $"{cachePath}.{Guid.NewGuid():N}.tmp";
            try
            {
                await File.WriteAllBytesAsync(tempPath, transformed, cancellationToken).ConfigureAwait(false);
                File.Move(tempPath, cachePath, overwrite: true);
            }
            catch
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }

                throw;
            }
        }

        OnnxExecutionSessionFactory.SingleSessionLease lease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = engineFamily, Provider = provider, CancellationToken = cancellationToken }, cachePath)
            .ConfigureAwait(false);
        return (lease, stateInputs, stateOutputs);
    }

    private static string GetDerivedCacheDirectory(string modelPath, byte[] original)
    {
        byte[] hash = SHA256.HashData(original);
        string hashHex = Convert.ToHexStringLower(hash)[..16];
        string cacheRoot = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrEmpty(cacheRoot))
        {
            cacheRoot = Path.GetTempPath();
        }

        return Path.Join(cacheRoot, "trackdub", "deepfilternet3-stateful", TransformVersion, hashHex);
    }

    private static async Task<bool> CacheMatchesAsync(
        string cachePath, byte[] transformed, CancellationToken cancellationToken)
    {
        if (!File.Exists(cachePath) || new FileInfo(cachePath).Length != transformed.Length)
        {
            return false;
        }

        byte[] existing = await File.ReadAllBytesAsync(cachePath, cancellationToken).ConfigureAwait(false);
        return SHA256.HashData(existing).AsSpan().SequenceEqual(SHA256.HashData(transformed));
    }

    public void Dispose()
    {
        Enc.Dispose();
        ErbDec.Dispose();
        DfDec.Dispose();
    }
}
