using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Json.Serialization;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Worker;

/// <summary>
/// Wire format between the application and the ORT 1.30 inference worker (ADR-0017): each
/// message is a 4-byte little-endian length followed by that many bytes of UTF-8 JSON, over the
/// worker's standard input and output.
/// </summary>
public static class InferenceWorkerProtocol
{
    public const int Version = 2;
    public const int MaxMessageBytes = 256 * 1024 * 1024;

    public const string Hello = "hello";
    public const string TtsSynthesize = "tts.synthesize";
    public const string TtsWarm = "tts.warm";
    public const string Translation = "translation.run";
    public const string Cancel = "cancel";
    public const string Shutdown = "shutdown";
    public const string Result = "result";
    public const string Error = "error";

    public static JsonSerializerOptions JsonOptions { get; } = new(JsonSerializerDefaults.Web)
    {
        Converters = { new JsonStringEnumConverter() },
    };

    /// <summary>Identifies the exact build; the worker must match the application.</summary>
    public static string BuildStamp { get; } = ComputeBuildStamp(typeof(InferenceWorkerProtocol).Assembly);

    internal const string SourceRevisionMetadataKey = "TrackdubSourceRevision";

    // The worker compiles this assembly separately (TrackdubOrtHost=Stock), so module IDs never
    // match across the two compilations. Both carry the source revision the build stamped
    // (StampTrackdubSourceRevision). A build without source control has no revision to compare;
    // both compilations then fall back to the assembly version, which they also share, so a
    // packaged worker still pairs with its application.
    internal static string ComputeBuildStamp(System.Reflection.Assembly assembly)
    {
        string? revision = assembly
            .GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), inherit: false)
            .Cast<System.Reflection.AssemblyMetadataAttribute>()
            .FirstOrDefault(attribute => attribute.Key == SourceRevisionMetadataKey)?.Value;
        return string.IsNullOrWhiteSpace(revision)
            ? "unrevised:" + assembly.GetName().Version
            : "revision:" + revision;
    }

    public static async Task WriteAsync(Stream stream, WorkerMessage message, CancellationToken cancellationToken)
    {
        // One buffer holds the whole frame: the prefix is reserved, the body serialized after it and
        // its length backfilled, so a large audio payload is never held twice.
        using var frame = new MemoryStream();
        frame.Write(stackalloc byte[4]);
        JsonSerializer.Serialize(frame, message, JsonOptions);
        byte[] buffer = frame.GetBuffer();
        int length = checked((int)frame.Length);
        BinaryPrimitives.WriteInt32LittleEndian(buffer, length - 4);

        // The frame goes out in one write. A cancellation that interrupts it leaves the pipe in an
        // unknown state; callers must then retire the stream rather than send another frame.
        await stream.WriteAsync(buffer.AsMemory(0, length), cancellationToken).ConfigureAwait(false);
        await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Reads one message, or null when the peer closed the stream.</summary>
    public static async Task<WorkerMessage?> ReadAsync(Stream stream, CancellationToken cancellationToken)
    {
        byte[] header = new byte[4];
        if (!await ReadExactlyOrEndAsync(stream, header, cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        int length = BinaryPrimitives.ReadInt32LittleEndian(header);
        if (length <= 0 || length > MaxMessageBytes)
        {
            throw new InvalidDataException($"Inference worker message length {length} is out of range.");
        }

        byte[] body = new byte[length];
        if (!await ReadExactlyOrEndAsync(stream, body, cancellationToken).ConfigureAwait(false))
        {
            throw new EndOfStreamException("Inference worker stream ended inside a message.");
        }

        return JsonSerializer.Deserialize<WorkerMessage>(body, JsonOptions)
            ?? throw new InvalidDataException("Inference worker sent an empty message.");
    }

    public static JsonElement ToPayload<T>(T value) => JsonSerializer.SerializeToElement(value, JsonOptions);

    public static T FromPayload<T>(JsonElement? payload) =>
        payload is { } element
            ? element.Deserialize<T>(JsonOptions) ?? throw new InvalidDataException($"Inference worker payload is not a {typeof(T).Name}.")
            : throw new InvalidDataException($"Inference worker message has no {typeof(T).Name} payload.");

    private static async Task<bool> ReadExactlyOrEndAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = await stream.ReadAsync(buffer.AsMemory(read), cancellationToken).ConfigureAwait(false);
            if (n == 0)
            {
                if (read == 0)
                {
                    return false;
                }

                throw new EndOfStreamException("Inference worker stream ended inside a message.");
            }

            read += n;
        }

        return true;
    }
}

/// <summary>One request, response or control message. Responses reuse the request id.</summary>
public sealed record WorkerMessage(
    long Id,
    string Kind,
    JsonElement? Payload = null,
    string? Error = null,
    string? ErrorType = null);

public sealed record WorkerHelloResult(
    int ProtocolVersion,
    string BuildStamp,
    string OnnxRuntimeVersion,
    bool CudaAvailable,
    string CudaDetail);

/// <summary>
/// The runtime plan the application's planner selected, including the model paths that
/// <see cref="StageRuntimePlan"/> keeps out of JSON.
/// </summary>
public sealed record WorkerPlan(
    RuntimeStage Stage,
    string? ModelId,
    string? ModelAlias,
    string? EngineFamily,
    string? Variant,
    ExecutionProviderKind ExecutionProvider,
    bool RequirePreferredExecutionProvider,
    string? ModelEntryPath,
    string? ModelRootPath,
    string? ModelEntryRelativePath,
    string? ModelRevisionHash,
    int? DeviceIndex)
{
    public static WorkerPlan From(StageRuntimePlan plan) =>
        new(
            plan.Stage,
            plan.ModelId,
            plan.ModelAlias,
            plan.EngineFamily,
            plan.Variant,
            plan.ExecutionProvider ?? throw new InvalidOperationException("A worker plan needs an execution provider."),
            plan.RequirePreferredExecutionProvider,
            plan.ModelEntryPath,
            plan.ModelRootPath,
            plan.ModelEntryRelativePath,
            plan.ModelRevisionHash,
            plan.DeviceIndex);

    public StageRuntimePlan ToStageRuntimePlan() =>
        new()
        {
            Stage = Stage,
            Status = StageRuntimePlanStatus.Ready,
            ModelId = ModelId,
            ModelAlias = ModelAlias,
            EngineFamily = EngineFamily,
            Variant = Variant,
            ExecutionProvider = ExecutionProvider,
            RequirePreferredExecutionProvider = RequirePreferredExecutionProvider,
            ModelEntryPath = ModelEntryPath,
            ModelRootPath = ModelRootPath,
            ModelEntryRelativePath = ModelEntryRelativePath,
            ModelRevisionHash = ModelRevisionHash,
            DeviceIndex = DeviceIndex,
        };
}

public sealed record WorkerTtsRequest(
    string Text,
    string LanguageCode,
    VoiceCatalogEntry Voice,
    float Speed,
    string? PhonemeOverride,
    double? TargetDurationSeconds,
    WorkerPlan Plan);

public sealed record WorkerTtsResult(
    string WavBase64,
    int DurationSamples,
    int SampleRate,
    string ModelId,
    string VoiceId,
    string Provider,
    StageRuntimeExecutionSummary? ExecutionSummary);

public sealed record WorkerTranslationRequest(TranslationRequest Request, WorkerPlan Plan);

public sealed record WorkerTranslationResult(
    IReadOnlyList<TranslatedTextSegment> Segments,
    StageRuntimeExecutionSummary? ExecutionSummary);

/// <summary>
/// Warm request: construct the worker's Kokoro engine and create its real pooled session
/// for the supplied plan, without synthesizing. A warm success therefore means a real
/// session exists; a warm failure reports its reason and leaves the lazy inline
/// construction on the next <see cref="InferenceWorkerProtocol.TtsSynthesize"/> intact.
/// </summary>
public sealed record WorkerWarmRequest(WorkerPlan Plan);

public sealed record WorkerWarmResult(
    string? ModelId,
    string Provider,
    string? Detail = null);
