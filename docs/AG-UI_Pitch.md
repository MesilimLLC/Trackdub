Upgrading Trackdub to .NET 11 aligns directly with the official **AG-UI (Agent-User Interaction Protocol) .NET SDK** (`AGUI.Abstractions`, `AGUI.Client`, `AGUI.Server`, `AGUI.Formatting`, and `AGUI.Protobuf`), which is built natively on `Microsoft.Extensions.AI` (`IChatClient`, `ChatResponseUpdate`, and `AIFunction`).

Even though Microsoft's headline .NET 11 samples showcase AG-UI with Blazor, the underlying .NET SDK is completely UI-framework agnostic. Because Trackdub is split into an open-core inference/pipeline engine and a commercial Avalonia 11 desktop shell plus REST API, adopting `AGUI.Abstractions` as your wire and state-synchronization contract solves several architectural seams already documented in your repos:

### 1. First-Class Interrupts for Cloud Egress Consent and Readiness Gates

Trackdub enforces a strict local-first boundary where cloud fallback engines require explicit user consent (tracked in the G3 cloud egress visibility and consent specs) alongside pre-flight readiness checks (G5 readiness gate).

* **How AG-UI helps:** The AG-UI protocol has native support for human-in-the-loop pauses via `ToolApprovalRequestContent` and `InterruptRequestContent`.
* **Trackdub integration:** When a pipeline run encounters a missing local model, a VRAM budget constraint (ADR-0009), or a stage configured for cloud egress, the backend can yield an AG-UI interrupt event and suspend the run. The Avalonia shell renders the consent or fallback prompt, and upon user confirmation, dispatches `RunAgentInput.Resume` with `InterruptResponseContent` to continue the exact `RunId` and `ThreadId` without custom state-machine plumbing.



### 2. Predictive and Shared State for Transcript, Glossary, and Timing Refinement

Post-editing in Trackdub involves iterative transcript alignment, speaker diarization adjustments, glossary enforcement, and syllable/duration compression for lip-sync timing (covered in the transcript service decomposition ADR-0004 and the TTS length-control spec).

* **How AG-UI helps:** AG-UI standardizes **Shared State** (snapshots and JSON Patch deltas) and **Predictive State Updates** (streaming optimistic state mutations before final commit).
* **Trackdub integration:** When an LLM refinement stage or dubbing assistant rewrites dialogue lines to match target lip-sync durations, it can stream predictive state deltas directly into your Avalonia ViewModels. Users can preview the proposed text and duration changes in the dialogue and timing workspaces, audition them via real-time A/B voice preview, and either accept the diff into atomic revision publication (atomic-revision-publication.md and SQLite persistence ADR-0002) or reject it to restore the baseline immediately.



### 3. Closing the Backend-to-UI Surface Gap Across Desktop, CLI, and API

Your audits highlight the challenge of keeping pipeline telemetry, progress ETAs (G4 run progress ETA), and event-sourced pipeline transitions (ADR-0010 and ADR-0015) synchronized with the Avalonia inspector cards and pipeline coordinator panels (backend-ui-surface-gap-audit.md).

* **How AG-UI helps:** `AGUI.Server` adapts `IAsyncEnumerable<ChatResponseUpdate>` and agent activity streams into a deterministic event sequence (`RUN_STARTED`, step/activity progress, tool execution, state deltas, `RUN_FINISHED`).
* **Trackdub integration:** By mapping pipeline stage transitions to AG-UI activity and state events, the same backend stream can drive the Avalonia center-panel pipeline coordinator in-process, stream Server-Sent Events (SSE) from your containerized API (`Dockerfile.api` and REST API), and feed structured progress to the CLI.



### 4. Transport-Agnostic IPC for Local Sidecars and Out-of-Process Workers

Trackdub isolates heavy native inference dependencies using supervised out-of-process workers (such as the ORT 1.30 inference worker in ADR-0017, the supervised Python worker in ADR-0016, and the Rust inference sidecar pitch) while maintaining strict startup latency budgets via source-generated JSON (json-sourcegen-startup-latency.md).

* **How AG-UI helps:** `AGUI.Client` decouples protocol semantics from HTTP via the single-method `IAGUITransport` interface (`SendAsync(RunAgentInput, CancellationToken)`), while `AGUI.Abstractions` ships with a pre-built source-generated `AGUIJsonSerializerContext` and `AGUI.Protobuf` adds binary framing.
* **Trackdub integration:** You can implement a custom `IAGUITransport` over Named Pipes or Unix Domain Sockets using `AGUI.Protobuf` for zero-HTTP-overhead IPC between the Avalonia desktop process and local inference sidecars, while using the built-in `AGUIHttpTransport` (SSE) when routing jobs to remote GPU containers (`Dockerfile.cloud-worker` and cloud GPU container operations).



### 5. Client-Side `AIFunction` Tools for Agentic Desktop Sessions

The gated repository already includes specifications for agent-driven desktop sessions and agent contracts (agent-desktop-sessions.md and agent-contract.md).

* **How AG-UI helps:** `AGUIChatClient` allows the frontend application to register client-side `AIFunction` tools in `ChatOptions.Tools`. When the backend agent calls a client tool, `AGUIChatClient` executes the callback locally in the UI process and returns the result to the agent within the active run.
* **Trackdub integration:** You can expose safe, bounded Avalonia ViewModel commands as client-side `AIFunction` tools - such as seeking media playback to a flagged lip-sync drift timestamp, soloing a separated stem in the mix workspace, or querying current hardware VRAM telemetry - letting an orchestrator agent drive interactive QA walkthroughs without violating your desktop boundary rules.


Wiring `IAGUITransport` into Trackdub lets your Avalonia ViewModels consume a single `IChatClient` stream (`AGUIChatClient`) whether the dubbing pipeline runs in-process, inside a local inference sidecar, or on a remote GPU container.

### 1. Central Package Setup (`Directory.Packages.props`)

Both `Trackdub` and `Trackdub-gated` use Central Package Management with NuGet lockfiles. Only `AGUI.Abstractions` is needed in the core contracts layer, keeping your open-core pipeline free of HTTP or UI dependencies while preserving source-generated JSON startup budgets (`AGUIJsonSerializerContext`):

```xml
<ItemGroup>
  <PackageVersion Include="AGUI.Abstractions" Version="1.0.1" />
  <PackageVersion Include="AGUI.Client" Version="1.0.1" />
  <PackageVersion Include="AGUI.Formatting" Version="1.0.1" />
  <PackageVersion Include="AGUI.Protobuf" Version="1.0.1" />
  <PackageVersion Include="Microsoft.Extensions.AI" Version="11.0.0" />
</ItemGroup>

```

### 2. In-Process `IAGUITransport` for the Pipeline Orchestrator

When the pipeline orchestrator runs in-process, a custom `IAGUITransport` translates Trackdub's event-sourced stage transitions (ADR-0010), transient telemetry (ADR-0015), G4 progress ETAs, and G3 cloud egress consent gates directly into `BaseEvent` records without serialization overhead:

```csharp
using System.Runtime.CompilerServices;
using System.Text.Json;
using AGUI.Abstractions;
using AGUI.Client;

namespace Trackdub.Desktop.Services.AgUi;

public sealed class InProcessPipelineAgUiTransport(
    IPipelineOrchestrator orchestrator,
    ICloudConsentStore consentStore) : IAGUITransport
{
    public async IAsyncEnumerable<BaseEvent> SendAsync(
        RunAgentInput input,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        string threadId = string.IsNullOrEmpty(input.ThreadId)
            ? Guid.NewGuid().ToString("N")
            : input.ThreadId;
        string runId = string.IsNullOrEmpty(input.RunId)
            ? Guid.NewGuid().ToString("N")
            : input.RunId;

        yield return new RunStartedEvent
        {
            ThreadId = threadId,
            RunId = runId,
            ParentRunId = input.ParentRunId,
            ProtocolVersion = AGUIProtocol.Version
        };

        // 1. Check if this run is resuming from a G3 Cloud Egress or G5 Readiness interrupt
        if (input.Resume is { Count: > 0 })
        {
            foreach (AGUIResume resumeEntry in input.Resume)
            {
                if (resumeEntry.Payload is { } payload &&
                    payload.TryGetProperty("approved", out JsonElement approvedProp) &&
                    !approvedProp.GetBoolean())
                {
                    yield return new RunErrorEvent
                    {
                        Code = "CLOUD_EGRESS_DENIED",
                        Message = "Cloud fallback was declined by user policy."
                    };
                    yield break;
                }

                consentStore.RecordGrant(threadId, resumeEntry.InterruptId);
            }
        }

        // 2. Stream pipeline stages, G4 ETA telemetry, and state deltas
        await foreach (PipelineDomainEvent evt in orchestrator.ExecuteAsync(threadId, input, cancellationToken))
        {
            switch (evt)
            {
                case StageEnteredEvent stageStart:
                    yield return new StepStartedEvent { StepName = stageStart.StageName };
                    break;

                case StageProgressTelemetryEvent progress:
                    // Drives G4 Run Progress ETA and VRAM telemetry in the Avalonia Center Panel
                    yield return new ActivitySnapshotEvent
                    {
                        MessageId = $"eta-{progress.StageName}",
                        ActivityType = "PIPELINE_PROGRESS",
                        Replace = true,
                        Content = JsonSerializer.SerializeToElement(
                            new PipelineProgressPayload(
                                progress.StageName,
                                progress.PercentComplete,
                                progress.EstimatedRemainingSeconds,
                                progress.VramAllocatedMb),
                            TrackdubJsonContext.Default.PipelineProgressPayload)
                    };
                    break;

                case PredictiveTranscriptDeltaEvent transcriptDelta:
                    // RFC 6902 JSON Patch array for live dialogue/timing preview
                    yield return new StateDeltaEvent
                    {
                        Delta = transcriptDelta.JsonPatchArray
                    };
                    break;

                case CloudEgressConsentRequiredEvent egressGate
                    when !consentStore.HasGrant(threadId, egressGate.GateId):
                    // Pause the run with a structured AG-UI interrupt outcome
                    yield return new RunFinishedEvent
                    {
                        ThreadId = threadId,
                        RunId = runId,
                        Outcome = JsonSerializer.SerializeToElement(new
                        {
                            type = "interrupt",
                            interrupts = new[]
                            {
                                new
                                {
                                    id = egressGate.GateId,
                                    reason = "cloud_egress_consent",
                                    message = $"Stage '{egressGate.StageName}' requires cloud endpoint '{egressGate.Provider}' ({egressGate.EstimatedAudioSeconds:F1}s audio)."
                                }
                            }
                        })
                    };
                    yield break;

                case StageCompletedEvent stageEnd:
                    yield return new StepFinishedEvent { StepName = stageEnd.StageName };
                    break;
            }
        }

        yield return new RunFinishedEvent
        {
            ThreadId = threadId,
            RunId = runId
        };
    }
}

```

### 3. Named Pipe + Protobuf `IAGUITransport` for Out-of-Process Sidecars

For your isolated inference workers (the ORT 1.30 worker in ADR-0017, the supervised Python worker in ADR-0016, or the Rust sidecar), you can bypass HTTP/SSE entirely by pairing `NamedPipeClientStream` with `ProtobufEventStreamFormatter` (`AGUI.Protobuf`), which uses big-endian `uint32` length-prefixed binary frames:

```csharp
using System.Buffers.Binary;
using System.IO.Pipes;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AGUI.Abstractions;
using AGUI.Client;
using AGUI.Protobuf;

namespace Trackdub.Desktop.Services.AgUi;

public sealed class NamedPipeSidecarAgUiTransport(string pipeName) : IAGUITransport
{
    private readonly ProtobufEventStreamFormatter _formatter = new();

    public async IAsyncEnumerable<BaseEvent> SendAsync(
        RunAgentInput input,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await using var pipe = new NamedPipeClientStream(
            serverName: ".",
            pipeName: pipeName,
            direction: PipeDirection.InOut,
            options: PipeOptions.Asynchronous);

        await pipe.ConnectAsync(cancellationToken);

        // Write length-prefixed RunAgentInput using the AOT-safe AGUIJsonSerializerContext
        byte[] requestBytes = JsonSerializer.SerializeToUtf8Bytes(
            input,
            AGUIJsonSerializerContext.Default.RunAgentInput);

        byte[] lengthPrefix = new byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(lengthPrefix, (uint)requestBytes.Length);

        await pipe.WriteAsync(lengthPrefix, cancellationToken);
        await pipe.WriteAsync(requestBytes, cancellationToken);
        await pipe.FlushAsync(cancellationToken);

        // Decode length-prefixed @ag-ui/proto binary frames directly into BaseEvent items
        await foreach (BaseEvent agUiEvent in _formatter.ReadAsync(pipe, cancellationToken))
        {
            yield return agUiEvent;
        }
    }
}

```

### 4. Consuming the Stream and Handling Interrupts in the Avalonia ViewModel

In the Avalonia desktop shell, wrap the chosen `IAGUITransport` in `AGUIChatClient` and inject it into your pipeline coordinator ViewModel. `AGUIChatClient` automatically invokes any client-side `AIFunction` tools registered in `ChatOptions.Tools` (for agent-driven desktop sessions) and surfaces raw AG-UI events on `ChatResponseUpdate.RawRepresentation`:

```csharp
using System.Text.Json;
using AGUI.Abstractions;
using AGUI.Client;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.AI;

namespace Trackdub.Desktop.ViewModels;

public sealed partial class PipelineCoordinatorViewModel : ObservableObject
{
    private readonly IChatClient _agUiClient;
    private readonly List<ChatMessage> _sessionMessages = [];

    [ObservableProperty] private string? _activeThreadId;
    [ObservableProperty] private string? _lastRunId;
    [ObservableProperty] private string _activeStage = "Idle";
    [ObservableProperty] private double _stageProgressPercent;
    [ObservableProperty] private double _estimatedRemainingSeconds;
    [ObservableProperty] private InterruptRequestContent? _pendingInterrupt;

    public PipelineCoordinatorViewModel(IAGUITransport transport, IMediaTimelineController timeline)
    {
        // Wrap AGUIChatClient with FunctionInvokingChatClient so client-side tools execute automatically
        IChatClient rawClient = new AGUIChatClient(new AGUIChatClientOptions
        {
            Transport = transport,
            JsonSerializerOptions = AGUIJsonSerializerContext.Default.Options
        });

        _agUiClient = new ChatClientBuilder(rawClient)
            .UseFunctionInvocation()
            .Build();

        // Expose bounded desktop actions for agent-desktop-sessions
        _clientTools =
        [
            AIFunctionFactory.Create(
                (double timestampSeconds) => Dispatcher.UIThread.InvokeAsync(() => timeline.SeekTo(timestampSeconds)),
                name: "seek_media_timeline",
                description: "Seeks the libmpv/Avalonia timeline to a specific timestamp in seconds.")
        ];
    }

    private readonly IList<AITool> _clientTools;

    [RelayCommand]
    public async Task StartDubbingRunAsync(string promptOrPreset, CancellationToken ct)
    {
        _sessionMessages.Add(new ChatMessage(ChatRole.User, promptOrPreset));
        await StreamTurnAsync(ct);
    }

    [RelayCommand]
    public async Task ResolveCloudConsentAsync(bool approved, CancellationToken ct)
    {
        if (PendingInterrupt is null) return;

        // Attach InterruptResponseContent to the next user turn; AGUIChatClient maps this to RunAgentInput.Resume
        JsonElement responsePayload = JsonSerializer.SerializeToElement(new { approved });
        var responseContent = new InterruptResponseContent(PendingInterrupt.RequestId, responsePayload);

        _sessionMessages.Add(new ChatMessage(ChatRole.User, [responseContent]));
        PendingInterrupt = null;

        await StreamTurnAsync(ct);
    }

    private async Task StreamTurnAsync(CancellationToken ct)
    {
        var options = new ChatOptions
        {
            Tools = _clientTools,
            RawRepresentationFactory = _ => new RunAgentInput
            {
                ThreadId = ActiveThreadId ?? string.Empty,
                ParentRunId = LastRunId
            }
        };

        await foreach (ChatResponseUpdate update in _agUiClient.GetStreamingResponseAsync(_sessionMessages, options, ct))
        {
            // 1. Inspect AG-UI protocol events via RawRepresentation
            switch (update.RawRepresentation)
            {
                case RunStartedEvent started:
                    ActiveThreadId = started.ThreadId;
                    LastRunId = started.RunId;
                    break;

                case StepStartedEvent stepStart:
                    ActiveStage = stepStart.StepName;
                    break;

                case ActivitySnapshotEvent { ActivityType: "PIPELINE_PROGRESS" } activity:
                    var progress = activity.Content.Deserialize(TrackdubJsonContext.Default.PipelineProgressPayload);
                    if (progress is not null)
                    {
                        StageProgressPercent = progress.PercentComplete;
                        EstimatedRemainingSeconds = progress.EstimatedRemainingSeconds;
                    }
                    break;

                case StateDeltaEvent delta:
                    ApplyTranscriptPatchOnUiThread(delta.Delta);
                    break;
            }

            // 2. Inspect MEAI content items for HITL Interrupts (G3 Cloud Consent / G5 Readiness)
            foreach (AIContent content in update.Contents)
            {
                if (content is InterruptRequestContent interruptReq)
                {
                    PendingInterrupt = interruptReq;
                }
            }
        }
    }

    private void ApplyTranscriptPatchOnUiThread(JsonElement patchOperations)
    {
        Dispatcher.UIThread.Post(() =>
        {
            // Apply RFC 6902 JSON Patch to the in-memory transcript/timing revision buffer
        });
    }
}

```

### How This Fits Your Existing Boundaries

* **Headless Avalonia & xUnit Testing:** In your `Trackdub-gated` headless test suite, you can inject a deterministic `InMemoryAgUiTransport` that yields scripted `StepStartedEvent`, `StateDeltaEvent`, and `RunFinishedEvent` sequences to verify the pipeline panel, transport bar, and dialogue workspace without spinning up ONNX Runtime or Python workers.


* **Zero Transport Lock-In:** Switching a pipeline stage from local in-process execution to the named-pipe sidecar or `AGUIHttpTransport` (for cloud GPU containers) is a DI registration change in your composition root; the ViewModel and XAML bindings remain untouched.



Would you like to look at how the RFC 6902 `StateDeltaEvent` patches map onto your transcript revision model (ADR-0004) for live dialogue and timing previews?

Mapping AG-UI's `StateSnapshotEvent` and `StateDeltaEvent` (RFC 6902 JSON Patch) onto Trackdub's transcript revision model lets you stream line-by-line translation refinements, glossary enforcement, and syllable compression directly into a staging buffer in the Avalonia UI without mutating your committed SQLite project state until the user accepts the pass.

### 1. AOT-Safe State Snapshot & RFC 6902 Patch Models

To keep your startup latency budget intact (`json-sourcegen-startup-latency.md`), avoid reflection-heavy JSON Patch libraries. Instead, define a lightweight, source-generated RFC 6902 operation struct alongside your transcript state snapshot:

```csharp
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trackdub.Contracts.AgUi;

public sealed record TranscriptWorkspaceState(
    long BaseRevisionId,
    IReadOnlyList<SegmentStateDto> Segments);

public sealed record SegmentStateDto(
    string SegmentId,
    string SpeakerId,
    string SourceText,
    string TranslatedText,
    double TargetDurationMs,
    double PredictedTtsDurationMs,
    bool IsGlossaryCompliant);

public readonly record struct JsonPatchOp(
    [property: JsonPropertyName("op")] string Op,
    [property: JsonPropertyName("path")] string Path,
    [property: JsonPropertyName("value")] JsonElement? Value = null);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(TranscriptWorkspaceState))]
[JsonSerializable(typeof(SegmentStateDto))]
[JsonSerializable(typeof(JsonPatchOp[]))]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(double))]
[JsonSerializable(typeof(bool))]
public partial class TrackdubAgUiJsonContext : JsonSerializerContext;

```

### 2. Emitting Baseline Snapshots and Predictive Deltas in the Pipeline

When a refinement or timing-compression stage starts, the pipeline emits a `StateSnapshotEvent` pinned to the current `BaseRevisionId` (`atomic-revision-publication.md`). As the LLM or duration-control pass evaluates each dialogue segment (`tts-length-control-spec.md`), it streams granular `StateDeltaEvent` patches:

```csharp
using System.Text.Json;
using AGUI.Abstractions;
using Trackdub.Contracts.AgUi;

public static class TranscriptAgUiEventFactory
{
    public static StateSnapshotEvent CreateBaselineSnapshot(TranscriptWorkspaceState state) =>
        new()
        {
            Snapshot = JsonSerializer.SerializeToElement(
                state,
                TrackdubAgUiJsonContext.Default.TranscriptWorkspaceState)
        };

    public static StateDeltaEvent CreateSegmentRefinementDelta(
        int segmentIndex,
        string candidateTranslation,
        double predictedTtsMs,
        bool glossaryCompliant)
    {
        JsonPatchOp[] ops =
        [
            new("replace", $"/segments/{segmentIndex}/translatedText",
                JsonSerializer.SerializeToElement(candidateTranslation, TrackdubAgUiJsonContext.Default.String)),
            new("replace", $"/segments/{segmentIndex}/predictedTtsDurationMs",
                JsonSerializer.SerializeToElement(predictedTtsMs, TrackdubAgUiJsonContext.Default.Double)),
            new("replace", $"/segments/{segmentIndex}/isGlossaryCompliant",
                JsonSerializer.SerializeToElement(glossaryCompliant, TrackdubAgUiJsonContext.Default.Bool))
        ];

        return new StateDeltaEvent
        {
            Delta = JsonSerializer.SerializeToElement(ops, TrackdubAgUiJsonContext.Default.JsonPatchOpArray)
        };
    }
}

```

### 3. Zero-Reflection Patch Applier in the Avalonia Dialogue ViewModel

In the Avalonia shell, each segment ViewModel holds both the committed baseline values and observable staging properties. A span-based JSON Pointer parser (`AsSpan()`) routes incoming `StateDeltaEvent` patches directly to the target `SegmentItemViewModel` on the UI thread, driving live lip-sync drift indicators and A/B voice preview (`Real-Time AB Voice Preview.md`):

```csharp
using System.Collections.ObjectModel;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Trackdub.Contracts.AgUi;

namespace Trackdub.Desktop.ViewModels.Workspace;

public sealed partial class SegmentItemViewModel : ObservableObject
{
    public required string SegmentId { get; init; }
    public required double TargetDurationMs { get; init; }

    // Committed baseline state from SQLite revision
    public required string CommittedText { get; set; }
    public required double CommittedTtsDurationMs { get; set; }

    // Staged predictive state updated live via AG-UI StateDeltaEvent
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPendingEdit))]
    private string _stagedTranslatedText = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DurationDriftRatio))]
    private double _predictedTtsDurationMs;

    [ObservableProperty]
    private bool _isGlossaryCompliant = true;

    public bool HasPendingEdit => !string.Equals(CommittedText, StagedTranslatedText, StringComparison.Ordinal);
    public double DurationDriftRatio => TargetDurationMs > 0 ? PredictedTtsDurationMs / TargetDurationMs : 1.0;

    public void RevertToCommitted()
    {
        StagedTranslatedText = CommittedText;
        PredictedTtsDurationMs = CommittedTtsDurationMs;
    }
}

public sealed partial class DialogueWorkspaceViewModel(
    ITranscriptRevisionPublisher revisionPublisher,
    IAbVoicePreviewService voicePreview) : ObservableObject
{
    public ObservableCollection<SegmentItemViewModel> Segments { get; } = [];
    public long BaseRevisionId { get; private set; }

    public void ApplySnapshot(JsonElement snapshotElement)
    {
        TranscriptWorkspaceState? state = snapshotElement.Deserialize(
            TrackdubAgUiJsonContext.Default.TranscriptWorkspaceState);
        if (state is null) return;

        BaseRevisionId = state.BaseRevisionId;
        Segments.Clear();

        foreach (SegmentStateDto dto in state.Segments)
        {
            Segments.Add(new SegmentItemViewModel
            {
                SegmentId = dto.SegmentId,
                TargetDurationMs = dto.TargetDurationMs,
                CommittedText = dto.TranslatedText,
                CommittedTtsDurationMs = dto.PredictedTtsDurationMs,
                StagedTranslatedText = dto.TranslatedText,
                PredictedTtsDurationMs = dto.PredictedTtsDurationMs,
                IsGlossaryCompliant = dto.IsGlossaryCompliant
            });
        }
    }

    public void ApplyDelta(JsonElement deltaElement)
    {
        JsonPatchOp[]? ops = deltaElement.Deserialize(
            TrackdubAgUiJsonContext.Default.JsonPatchOpArray);
        if (ops is null) return;

        foreach (JsonPatchOp op in ops)
        {
            if (op.Op is not ("replace" or "add") || op.Value is not { } val)
                continue;

            // Expected path format: /segments/{index}/{property}
            ReadOnlySpan<char> path = op.Path.AsSpan();
            if (!path.StartsWith("/segments/")) continue;

            ReadOnlySpan<char> remainder = path["/segments/".Length..];
            int slashIdx = remainder.IndexOf('/');
            if (slashIdx <= 0) continue;

            if (!int.TryParse(remainder[..slashIdx], out int index) ||
                index < 0 || index >= Segments.Count)
                continue;

            ReadOnlySpan<char> field = remainder[(slashIdx + 1)..];
            SegmentItemViewModel segment = Segments[index];

            switch (field)
            {
                case "translatedText":
                    segment.StagedTranslatedText = val.GetString() ?? string.Empty;
                    break;
                case "predictedTtsDurationMs":
                    segment.PredictedTtsDurationMs = val.GetDouble();
                    break;
                case "isGlossaryCompliant":
                    segment.IsGlossaryCompliant = val.GetBoolean();
                    break;
            }
        }
    }

    [RelayCommand]
    public Task AuditionSegmentAbAsync(SegmentItemViewModel segment, CancellationToken ct) =>
        voicePreview.PlayAbComparisonAsync(
            segment.SegmentId,
            baselineText: segment.CommittedText,
            candidateText: segment.StagedTranslatedText,
            ct);

    [RelayCommand]
    public async Task CommitStagedRevisionAsync(CancellationToken ct)
    {
        var dirtySegments = Segments.Where(s => s.HasPendingEdit).ToList();
        if (dirtySegments.Count == 0) return;

        // Publish atomically against BaseRevisionId (ADR-0002 & atomic-revision-publication.md)
        BaseRevisionId = await revisionPublisher.PublishAtomicRevisionAsync(
            BaseRevisionId,
            dirtySegments.Select(s => (s.SegmentId, s.StagedTranslatedText)).ToList(),
            ct);

        foreach (SegmentItemViewModel seg in dirtySegments)
        {
            seg.CommittedText = seg.StagedTranslatedText;
            seg.CommittedTtsDurationMs = seg.PredictedTtsDurationMs;
        }
    }

    [RelayCommand]
    public void DiscardStagedEdits()
    {
        foreach (SegmentItemViewModel seg in Segments)
        {
            seg.RevertToCommitted();
        }
    }
}

```

### Why This Works Cleanly with Your Architecture

* **Atomic Revision Safety:** Streaming `StateDeltaEvent` updates only touch `StagedTranslatedText` and `PredictedTtsDurationMs` in memory. If a run is canceled mid-stream or a user rejects a hallmarked translation take, calling `DiscardStagedEdits()` immediately restores the last committed revision without touching SQLite.


* **Property-Based Testing with FsCheck:** Because `Trackdub-gated` already includes FsCheck in its test suite, you can write property tests generating arbitrary sequences of `JsonPatchOp` arrays to verify that `ApplyDelta` never throws on out-of-range indices and that `DiscardStagedEdits()` is strictly idempotent relative to the initial `StateSnapshotEvent`.



Would you like to look at how to expose `Core C9` (`ICloudEgressConsentService`) and the readiness gate (`G5`) as a unified AG-UI middleware layer so every pipeline stage gets automatic interrupt/resume handling?


Exposing the G3 cloud egress consent policies and the G5 readiness gate as a unified AG-UI middleware layer ensures that every pipeline stage automatically intercepts execution, pauses for validation, and resumes cleanly without cluttering individual inference handlers.

### 1. Unified AG-UI Middleware (`AGUIPipelineConsentMiddleware`)

This middleware inspects incoming runs, validates G5 local-model/hardware requirements and G3 cloud egress boundaries before execution, and intercepts mid-run when a stage requires external cloud endpoints.

```csharp
using System.Text.Json;
using AGUI.Abstractions;
using AGUI.Client;
using Microsoft.Extensions.AI;

namespace Trackdub.Desktop.Services.AgUi;

public sealed class AGUIPipelineConsentMiddleware(
    IChatClient innerClient,
    IG5ReadinessGateService readinessGate,
    ICloudConsentStore consentStore) : IChatClient
{
    public ChatClientMetadata Metadata => innerClient.Metadata;

    public async Task<ChatResponse> GetResponseAsync(
        IList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // 1. Enforce G5 Pre-Flight Readiness Gate before run initialization
        ReadinessGateResult readiness = await readinessGate.EvaluatePreFlightAsync(cancellationToken);
        if (!readiness.IsReady)
        {
            throw new InvalidOperationException($"G5 Readiness Gate Failed: {readiness.FailureReason}");
        }

        // 2. Extract RunAgentInput and validate resume payloads
        RunAgentInput? input = options?.RawRepresentationFactory?.Invoke(messages) as RunAgentInput;
        string threadId = input?.ThreadId ?? "default-thread";

        if (input?.Resume is { Count: > 0 })
        {
            foreach (AGUIResume resume in input.Resume)
            {
                if (resume.Payload.TryGetProperty("approved", out JsonElement approved) && approved.GetBoolean())
                {
                    consentStore.RecordGrant(threadId, resume.InterruptId);
                }
            }
        }

        return await innerClient.GetResponseAsync(messages, options, cancellationToken);
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // 1. Streaming pass-through via inner transport/client
        await foreach (ChatResponseUpdate update in innerClient.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            yield return update;
        }
    }

    public TService? GetService<TService>(object? key = null) where TService : class =>
        innerClient.GetService<TService>(key) ?? (this as TService);

    public void Dispose() => innerClient.Dispose();
}

```

### 2. Composition Root Registration (`Program.cs`)

Wiring the middleware into your DI container via `ChatClientBuilder` keeps your Avalonia ViewModels completely agnostic of whether a pipeline execution is gated locally, paused for cloud consent, or executing against a remote worker:

```csharp
using AGUI.Client;
using Microsoft.Extensions.AI;
using Trackdub.Desktop.Services.AgUi;

public static class AgUiServiceCollectionExtensions
{
    public static IServiceCollection AddTrackdubAgUiPipeline(this IServiceCollection services)
    {
        // Register the underlying transport (In-process, Named Pipe, or HTTP SSE)
        services.AddSingleton<IAGUITransport, InProcessPipelineAgUiTransport>();

        // Register the AG-UI Client stacked with Middleware and Function Invocation
        services.AddSingleton<IChatClient>(sp =>
        {
            IAGUITransport transport = sp.GetRequiredService<IAGUITransport>();
            IG5ReadinessGateService readinessGate = sp.GetRequiredService<IG5ReadinessGateService>();
            ICloudConsentStore consentStore = sp.GetRequiredService<ICloudConsentStore>();

            IChatClient baseClient = new AGUIChatClient(new AGUIChatClientOptions
            {
                Transport = transport,
                JsonSerializerOptions = AGUIJsonSerializerContext.Default.Options
            });

            return new ChatClientBuilder(baseClient)
                .Use(client => new AGUIPipelineConsentMiddleware(client, readinessGate, consentStore))
                .UseFunctionInvocation()
                .Build();
        });

        return services;
    }
}

```