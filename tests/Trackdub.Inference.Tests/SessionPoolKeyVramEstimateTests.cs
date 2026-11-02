using System.Text;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Tests;

/// <summary>
/// Tests for <see cref="SessionPoolKey.EstimateVramMb"/>: external-data sidecars are counted
/// and the resident factor follows the execution provider.
/// </summary>
[Collection(nameof(TensorRtRtxTeardownGuardCollection))]
public sealed class SessionPoolKeyVramEstimateTests : IDisposable
{
    private const int MiB = 1024 * 1024;

    private readonly string directory = Path.Join(Path.GetTempPath(), $"vram-est-{Guid.NewGuid():N}");

    public SessionPoolKeyVramEstimateTests()
    {
        Directory.CreateDirectory(directory);
        TensorRtRtxTeardownGuard.ResetForTests();
    }

    public void Dispose()
    {
        TensorRtRtxTeardownGuard.ResetForTests();
        OnnxExternalDataSidecars.ResetCache();
        Directory.Delete(directory, recursive: true);
    }

    [Theory]
    [InlineData(".onnx.data")]
    [InlineData(".onnx_data")]
    public void EstimateVramMb_IncludesReferencedSidecar(string sidecarSuffix)
    {
        string model = WriteModel("encoder_model.onnx", Initializer("w0", "encoder_model" + sidecarSuffix));
        WriteBytes("encoder_model" + sidecarSuffix, 10 * MiB);

        // 10 MB weights → 10 * 2 + 128.
        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_SidecarSharedByManyInitializers_CountsFileOnce()
    {
        string model = WriteModel(
            "model.onnx",
            Initializer("w0", "weights/shared.bin"),
            Initializer("w1", "weights/shared.bin"),
            Initializer("w2", "weights/shared.bin"));
        WriteBytes(Path.Join("weights", "shared.bin"), 10 * MiB);

        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_SumsDistinctSidecars()
    {
        string model = WriteModel(
            "model.onnx",
            Initializer("w0", "part0.bin"),
            Initializer("w1", "part1.bin"),
            Initializer("w2", "missing.bin"));
        WriteBytes("part0.bin", 6 * MiB);
        WriteBytes("part1.bin", 4 * MiB);

        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_IncludesSidecarReferencedFromSubgraph()
    {
        byte[] thenBranch = Graph(Initializer("then_w", "branch.bin"));
        byte[] ifNode = Concat(
            StringField(4, "If"),
            MessageField(5, Concat(StringField(1, "then_branch"), MessageField(6, thenBranch))));
        string model = WriteModel("model.onnx", MessageField(1, ifNode));
        WriteBytes("branch.bin", 10 * MiB);

        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_SkipsUnknownGroupFieldsAndKeepsReferencedSidecar()
    {
        // Field 99 as a (deprecated) protobuf group holding a varint, a nested group, and bytes.
        byte[] unknownGroup = Concat(
            Varint((99UL << 3) | 3),
            VarintField(1, 7),
            Varint((2UL << 3) | 3),
            VarintField(1, 1),
            Varint((2UL << 3) | 4),
            StringField(3, "payload"),
            Varint((99UL << 3) | 4));
        string model = WriteModel("model.onnx", unknownGroup, Initializer("w0", "weights/custom.bin"));
        WriteBytes(Path.Join("weights", "custom.bin"), 10 * MiB);

        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_IncludesSidecarsReferencedFromSparseInitializer()
    {
        // GraphProto.sparse_initializer = 15 → SparseTensorProto (onnx.proto): values = 1,
        // indices = 2, dims = 3. Packed int64 dims arrive length-delimited and must be skipped.
        byte[] sparse = Concat(
            MessageField(1, Tensor("sparse_values", "sparse_values.bin")),
            MessageField(2, Tensor("sparse_indices", "sparse_indices.bin")),
            MessageField(3, Concat(Varint(64), Varint(64))));
        string model = WriteModel("model.onnx", MessageField(15, sparse));
        WriteBytes("sparse_values.bin", 6 * MiB);
        WriteBytes("sparse_indices.bin", 4 * MiB);

        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_IgnoresLocationsOutsideModelDirectory()
    {
        string modelDirectory = Path.Join(directory, "model");
        Directory.CreateDirectory(modelDirectory);
        string model = WriteModel(
            Path.Join("model", "model.onnx"),
            Initializer("w0", "../outside.bin"),
            Initializer("w1", Path.Join(directory, "rooted.bin")));
        WriteBytes("outside.bin", 10 * MiB);
        WriteBytes("rooted.bin", 10 * MiB);

        Assert.Equal(128, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_InlineGraph_IgnoresUnreferencedAdjacentDataFile()
    {
        string model = WriteModel("model.onnx", Initializer("w0", location: null));
        WriteBytes("model.onnx.data", 10 * MiB);

        Assert.Equal(128, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_UnparseableGraph_FallsBackToAdjacentDataFiles()
    {
        string model = WriteBytes("model.onnx", 1 * MiB); // all zeros: field tag 0 is invalid
        WriteBytes("model.onnx.data", 6 * MiB);
        WriteBytes("model.onnx_data", 3 * MiB);

        // 1 + 6 + 3 MB → 10 * 2 + 128.
        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_ReplacedSidecar_IsRemeasuredWithUnchangedGraph()
    {
        string model = WriteModel("model.onnx", Initializer("w0", "model.onnx.data"));
        WriteBytes("model.onnx.data", 10 * MiB);
        Assert.Equal(148, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));

        WriteBytes("model.onnx.data", 20 * MiB);

        Assert.Equal(168, SessionPoolKey.EstimateVramMb(model, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void EstimateVramMb_MissingFile_ReturnsDefault()
    {
        Assert.Equal(
            SessionPoolKey.DefaultEstimatedVramMb,
            SessionPoolKey.EstimateVramMb(Path.Join(directory, "absent.onnx"), ExecutionProviderKind.TensorRTRtx));
    }

    [Theory]
    [InlineData(ExecutionProviderKind.TensorRTRtx, 253)] // 100 * 1.25 + 128
    [InlineData(ExecutionProviderKind.DirectMl, 253)]
    [InlineData(ExecutionProviderKind.Cpu, 328)]         // 100 * 2 + 128
    [InlineData(ExecutionProviderKind.Dnnl, 328)]
    [InlineData(ExecutionProviderKind.Cuda, 328)]
    [InlineData(ExecutionProviderKind.TensorRt, 328)]
    [InlineData(ExecutionProviderKind.OpenVino, 328)]
    public void EstimateFromWeightBytes_ScalesByProvider(ExecutionProviderKind provider, long expectedMb)
    {
        Assert.Equal(expectedMb, SessionPoolKey.EstimateFromWeightBytes(100L * MiB, provider));
    }

    [Theory]
    [InlineData(ExecutionProviderKind.TensorRTRtx)]
    [InlineData(ExecutionProviderKind.DirectMl)]
    [InlineData(ExecutionProviderKind.Cpu)]
    public void EstimateFromWeightBytes_ProviderMayFallBack_UsesConservativeFactor(ExecutionProviderKind provider)
    {
        Assert.Equal(328, SessionPoolKey.EstimateFromWeightBytes(100L * MiB, provider, providerMayFallBack: true));
    }

    [Fact]
    public void EstimateFromWeightBytes_SubMegabyteWeights_KeepPerGraphAllowance()
    {
        Assert.Equal(128, SessionPoolKey.EstimateFromWeightBytes(0, ExecutionProviderKind.TensorRTRtx));
        Assert.Equal(128, SessionPoolKey.EstimateFromWeightBytes(MiB - 1, ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void KeyHelpers_UseKeyProviderForSidecarInclusiveEstimate()
    {
        string encoder = WriteModel("encoder_model.onnx", Initializer("w0", "encoder_model.onnx_data"));
        WriteBytes("encoder_model.onnx_data", 100 * MiB);

        SessionPoolKey trt = SessionPoolKey.ForEncoder("madlad", encoder, ExecutionProviderKind.TensorRTRtx);
        SessionPoolKey cuda = SessionPoolKey.ForEncoder("madlad", encoder, ExecutionProviderKind.Cuda);

        Assert.Equal(253, trt.EstimatedVramMb);
        Assert.Equal(328, cuda.EstimatedVramMb);
    }

    [Fact]
    public async Task CreateAsync_UsesSidecarInclusiveEstimate_AndContentHashCoversGraphOnly()
    {
        string decoder = WriteModel("decoder_model.onnx", Initializer("w0", "decoder_model.onnx_data"));
        WriteBytes("decoder_model.onnx_data", 100 * MiB);

        SessionPoolKey key = await SessionPoolKey.CreateAsync(
            "madlad", null, null, ExecutionProviderKind.TensorRTRtx, decoder, deviceId: null,
            graphRole: "decoder", optionsFingerprint: null, CancellationToken.None);

        Assert.Equal(253, key.EstimatedVramMb);
        Assert.Equal(SessionPoolKey.HashModelContent(decoder), key.ModelContentHash);
    }

    [Fact]
    public async Task CreatePooledSingleAsync_TensorRtRtxWithInitFallback_ReservesForFallbackProvider()
    {
        // The session factory may still fall back from TRT RTX after the key is admitted, so the
        // reservation must cover the 2x fallback provider whichever provider the key records.
        string model = WriteModel("single.onnx", Initializer("w0", "single.onnx.data"));
        WriteBytes("single.onnx.data", 10 * MiB);
        using var pool = new InferenceSessionPool(maxSessions: 2);

        using OnnxExecutionSessionFactory.SingleSessionLease lease = await OnnxExecutionSessionFactory.CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "test-engine", Provider = ExecutionProviderKind.TensorRTRtx, CancellationToken = CancellationToken.None, Pool = pool, SessionFactory = (_, _) => new InferenceSession(IdentityModel), AllowTrtInitFallback = true }, model);

        Assert.NotNull(lease.ResolvedPoolKey);
        // Without a registered TRT RTX EP device (the Windows build in a test process), session
        // options pick DirectML/CPU up front, no init fallback can follow, and the key keeps that
        // provider's own factor. Only a key still on TRT RTX needs the fallback reservation.
        long expectedMb = lease.ResolvedPoolKey.Provider is ExecutionProviderKind.TensorRTRtx
            ? 148
            : SessionPoolKey.EstimateFromWeightBytes(10 * MiB, lease.ResolvedPoolKey.Provider);
        Assert.Equal(expectedMb, lease.ResolvedPoolKey.EstimatedVramMb);
    }

    /// <summary>Default accelerator budget on the 12 GB RTX 5070 the factors were measured on.</summary>
    private static long Rtx5070BudgetMb => InferenceSessionPool.ScaleAcceleratorBudgetMb(11_943);

    /// <summary>
    /// MADLAD-400 3B <c>trt_rtx_mixed_fp16_fp32</c> encoder + decoder, sized from the exported
    /// files. On TensorRT RTX the bundle fits the default budget of a 12 GB card (it runs there,
    /// using ~8 GB); the unmeasured 2× factor would refuse it.
    /// </summary>
    [Fact]
    public void MadladFp16Bundle_FitsTwelveGigabyteBudgetOnlyWithMeasuredFactor()
    {
        const long encoderBytes = 1_433_988L + 2_671_905_792L;
        const long decoderBytes = 2_594_639L + 3_879_930_880L;

        long trtMb = SessionPoolKey.EstimateFromWeightBytes(encoderBytes, ExecutionProviderKind.TensorRTRtx)
            + SessionPoolKey.EstimateFromWeightBytes(decoderBytes, ExecutionProviderKind.TensorRTRtx);
        long unmeasuredMb = SessionPoolKey.EstimateFromWeightBytes(encoderBytes, ExecutionProviderKind.Cuda)
            + SessionPoolKey.EstimateFromWeightBytes(decoderBytes, ExecutionProviderKind.Cuda);

        Assert.Equal(8_069, trtMb);
        Assert.True(trtMb <= Rtx5070BudgetMb, $"{trtMb} MB should fit {Rtx5070BudgetMb} MB.");
        Assert.True(unmeasuredMb > Rtx5070BudgetMb, $"2x estimate {unmeasuredMb} MB should exceed {Rtx5070BudgetMb} MB.");
    }

    /// <summary>
    /// qwen3-asr-0.6b encoder + decoder-init + decoder-step on DirectML: both decoder graphs
    /// reference the same 3.0 GB <c>decoder_weights.data</c>. At 2× the bundle needed 13274 MB
    /// and failed the ASR stage against the 8957 MB budget, while the run measured 6526 MiB.
    /// </summary>
    [Fact]
    public void Qwen3AsrBundle_FitsTwelveGigabyteBudgetOnDirectMl()
    {
        const long encoderBytes = 745_762_694L;
        const long decoderInitBytes = 286_310L + 3_006_500_864L;
        const long decoderStepBytes = 285_813L + 3_006_500_864L;

        long Bundle(ExecutionProviderKind provider) =>
            SessionPoolKey.EstimateFromWeightBytes(encoderBytes, provider)
            + SessionPoolKey.EstimateFromWeightBytes(decoderInitBytes, provider)
            + SessionPoolKey.EstimateFromWeightBytes(decoderStepBytes, provider);

        Assert.Equal(8_957, Rtx5070BudgetMb);
        Assert.Equal(8_438, Bundle(ExecutionProviderKind.DirectMl));
        Assert.Equal(13_274, Bundle(ExecutionProviderKind.Cuda));
    }

    // ── Minimal ONNX protobuf writer ─────────────────────────────────────────

    private string WriteModel(string relativePath, params byte[][] graphFields)
    {
        // ModelProto: ir_version = 1, graph = 7.
        byte[] model = Concat(VarintField(1, 8), MessageField(7, Graph(graphFields)));
        string path = Path.Join(directory, relativePath);
        File.WriteAllBytes(path, model);
        return path;
    }

    private string WriteBytes(string relativePath, int length)
    {
        string path = Path.Join(directory, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[length]);
        return path;
    }

    // GraphProto: name = 2, then the supplied node (1) / initializer (5) fields.
    private static byte[] Graph(params byte[][] fields) =>
        Concat([StringField(2, "g"), .. fields]);

    // GraphProto.initializer = 5 → TensorProto: dims = 1, data_type = 2, name = 8,
    // external_data = 13 (key = 1, value = 2), data_location = 14 (EXTERNAL = 1).
    private static byte[] Initializer(string name, string? location) => MessageField(5, Tensor(name, location));

    private static byte[] Tensor(string name, string? location) =>
        location is null
            ? Concat(VarintField(1, 4), VarintField(2, 1), StringField(8, name), MessageField(9, new byte[16]))
            : Concat(
                VarintField(1, 4),
                VarintField(2, 1),
                StringField(8, name),
                MessageField(13, Concat(StringField(1, "location"), StringField(2, location.Replace('\\', '/')))),
                MessageField(13, Concat(StringField(1, "offset"), StringField(2, "0"))),
                MessageField(13, Concat(StringField(1, "length"), StringField(2, "16"))),
                VarintField(14, 1));

    private static byte[] StringField(int field, string value) =>
        MessageField(field, Encoding.UTF8.GetBytes(value));

    private static byte[] MessageField(int field, byte[] payload) =>
        Concat(Varint(((ulong)field << 3) | 2), Varint((ulong)payload.Length), payload);

    private static byte[] VarintField(int field, ulong value) =>
        Concat(Varint((ulong)field << 3), Varint(value));

    private static byte[] Varint(ulong value)
    {
        var bytes = new List<byte>();
        do
        {
            byte next = (byte)(value & 0x7f);
            value >>= 7;
            bytes.Add(value == 0 ? next : (byte)(next | 0x80));
        }
        while (value != 0);

        return [.. bytes];
    }

    private static byte[] Concat(params byte[][] parts) => [.. parts.SelectMany(static part => part)];

    private static readonly byte[] IdentityModel =
    [
        0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
        0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
        0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
    ];
}
