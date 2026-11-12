using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Worker;

/// <summary>Runs engines that need ORT 1.30 in the out-of-process inference worker (ADR-0017).</summary>
public interface IInferenceWorkerClient
{
    /// <summary>Whether a worker executable was found; says nothing about whether it starts.</summary>
    bool IsInstalled { get; }

    /// <summary>Starts the worker if needed and returns its handshake, or null when it is unusable.</summary>
    Task<WorkerHelloResult?> TryGetHelloAsync(CancellationToken cancellationToken);

    Task<WorkerTranslationResult> TranslateAsync(
        TranslationRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken);

    /// <remarks>
    /// Cancelling a call while its request is still being written kills the worker, because its
    /// input pipe may then hold a partial frame. Other calls in flight on that worker fail with
    /// <see cref="InferenceWorkerException"/>, and the next call starts a fresh worker.
    /// </remarks>
    Task<TtsSynthesisResult> SynthesizeTtsAsync(
        TtsSynthesisRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken);

    /// <summary>
    /// Asks the worker to construct its Kokoro engine and real session for
    /// <paramref name="plan"/> without synthesizing. Success means a real session exists in
    /// the worker; failure throws <see cref="InferenceWorkerException"/> with the worker's
    /// reason and leaves the worker usable for a lazy inline synthesis retry.
    /// </summary>
    Task<WorkerWarmResult> WarmTtsAsync(
        StageRuntimePlan plan,
        CancellationToken cancellationToken);
}

/// <summary>
/// Supervises one worker process: lazy start, version handshake, request correlation,
/// cancellation forwarding, and restart after a crash (at most <see cref="MaxStarts"/> starts per
/// client). A failed or mismatched worker fails the call with its reason; nothing falls back to
/// in-process execution silently.
/// </summary>
public sealed class InferenceWorkerClient : IInferenceWorkerClient, IAsyncDisposable, IDisposable
{
    public const string WorkerPathEnvironmentVariable = "TRACKDUB_INFERENCE_WORKER_PATH";

    /// <summary>Cold-start evidence source label for worker-hosted TTS completions.</summary>
    internal const string SynthesisSource = "inference-worker:tts";
    public const string WorkerDirectoryName = "inference-worker";
    public const string WorkerExecutableName = "Trackdub.InferenceWorker.exe";
    private const int MaxStarts = 3;
    private const int StderrTailLines = 40;

    private readonly string? executablePath;
    private readonly SemaphoreSlim startGate = new(1, 1);
    private readonly SemaphoreSlim writeGate = new(1, 1);
    private readonly ConcurrentDictionary<long, PendingCall> pending = new();
    private readonly Queue<string> stderrTail = new();
    private Process? process;

    // Replaced workers' handles. Fast-path callers may still hold a snapshot of one and read
    // HasExited, which throws on a disposed Process, so they are released only when the client is.
    private readonly List<Process> retired = [];

    // A handshaken worker is published as one immutable snapshot so callers on the lock-free fast
    // path never pair a process with another process's handshake.
    private volatile RunningWorker? current;
    private string? unusableReason;
    private long nextId;
    private int starts;
    private int disposed;

    public InferenceWorkerClient(string? executablePath = null)
    {
        this.executablePath = executablePath ?? LocateExecutable();
    }

    public bool IsInstalled => executablePath is not null;

    public async Task<WorkerTranslationResult> TranslateAsync(
        TranslationRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        return await SendRequestAsync<WorkerTranslationRequest, WorkerTranslationResult>(
            InferenceWorkerProtocol.Translation,
            new WorkerTranslationRequest(request, WorkerPlan.From(plan)),
            cancellationToken).ConfigureAwait(false);
    }

    public static string? LocateExecutable()
    {
        string? configured = Environment.GetEnvironmentVariable(WorkerPathEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return File.Exists(configured) ? Path.GetFullPath(configured) : null;
        }

        string candidate = Path.Join(AppContext.BaseDirectory, WorkerDirectoryName, WorkerExecutableName);
        return File.Exists(candidate) ? candidate : null;
    }

    public async Task<WorkerHelloResult?> TryGetHelloAsync(CancellationToken cancellationToken)
    {
        try
        {
            return (await EnsureStartedAsync(cancellationToken).ConfigureAwait(false)).Hello;
        }
        catch (Exception ex) when (ex is InferenceWorkerException
            or System.ComponentModel.Win32Exception
            or InvalidDataException
            or JsonException
            or IOException)
        {
            // Discovery asks whether the worker is usable; a worker that cannot start or answers
            // garbage is simply unusable, not a reason to fail discovery for every provider.
            return null;
        }
    }

    public async Task<TtsSynthesisResult> SynthesizeTtsAsync(
        TtsSynthesisRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        if (request.VoiceCloneReference is not null)
        {
            throw new NotSupportedException("The inference worker does not run voice cloning.");
        }

        WorkerMessage response = await SendAsync(
            InferenceWorkerProtocol.TtsSynthesize,
            InferenceWorkerProtocol.ToPayload(new WorkerTtsRequest(
                request.Text,
                request.LanguageCode,
                request.Voice,
                request.Speed,
                request.PhonemeOverride,
                request.TargetDurationSeconds,
                WorkerPlan.From(plan))),
            cancellationToken).ConfigureAwait(false);
        // The worker process has no ambient capture of its own, so the host reports the
        // completion here: the result arrived, so the out-of-process inference completed.
        ColdStartCapture.MarkInferenceCompleted(SynthesisSource);
        WorkerTtsResult result = InferenceWorkerProtocol.FromPayload<WorkerTtsResult>(response.Payload);
        LastTtsExecutionSummary = result.ExecutionSummary;
        return new TtsSynthesisResult(
            Convert.FromBase64String(result.WavBase64),
            result.DurationSamples,
            result.SampleRate,
            result.ModelId,
            result.VoiceId,
            result.Provider);
    }

    public async Task<WorkerWarmResult> WarmTtsAsync(
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(plan);
        return await SendRequestAsync<WorkerWarmRequest, WorkerWarmResult>(
            InferenceWorkerProtocol.TtsWarm,
            new WorkerWarmRequest(WorkerPlan.From(plan)),
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<TResponse> SendRequestAsync<TRequest, TResponse>(
        string messageKind,
        TRequest request,
        CancellationToken cancellationToken)
    {
        WorkerMessage response = await SendAsync(
            messageKind,
            InferenceWorkerProtocol.ToPayload(request),
            cancellationToken).ConfigureAwait(false);
        return InferenceWorkerProtocol.FromPayload<TResponse>(response.Payload);
    }

    /// <summary>Execution summary the worker reported for the most recent synthesis.</summary>
    public StageRuntimeExecutionSummary? LastTtsExecutionSummary { get; private set; }

    private async Task<WorkerMessage> SendAsync(string kind, JsonElement payload, CancellationToken cancellationToken)
    {
        Process running = (await EnsureStartedAsync(cancellationToken).ConfigureAwait(false)).Process;
        long id = Interlocked.Increment(ref nextId);
        var completion = new TaskCompletionSource<WorkerMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = new PendingCall(running, completion);
        try
        {
            await WriteAsync(running, new WorkerMessage(id, kind, payload), cancellationToken).ConfigureAwait(false);
            using CancellationTokenRegistration registration = cancellationToken.Register(() => _ = SendCancelAsync(running, id));
            WorkerMessage response = await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
            if (response.Kind == InferenceWorkerProtocol.Error)
            {
                if (response.ErrorType == nameof(OperationCanceledException))
                {
                    throw new OperationCanceledException(response.Error, cancellationToken);
                }

                throw new InferenceWorkerException($"Inference worker failed ({response.ErrorType}): {response.Error}");
            }

            return response;
        }
        finally
        {
            pending.TryRemove(id, out _);
        }
    }

    private async Task<RunningWorker> EnsureStartedAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);
        if (current is { Process.HasExited: false } running)
        {
            return running;
        }

        await startGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (current is { Process.HasExited: false } alreadyRunning)
            {
                return alreadyRunning;
            }

            // DisposeAsync may have run while this call waited for the gate.
            ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);

            if (unusableReason is not null)
            {
                throw new InferenceWorkerException(unusableReason);
            }

            if (executablePath is null)
            {
                throw new InferenceWorkerException(
                    $"The inference worker is not installed (expected {WorkerDirectoryName}/{WorkerExecutableName} next to the application, or {WorkerPathEnvironmentVariable}).");
            }

            if (++starts > MaxStarts)
            {
                unusableReason = $"The inference worker stopped {MaxStarts} times; not restarting it again. Last output: {StderrSnapshot()}";
                throw new InferenceWorkerException(unusableReason);
            }

            Process started = Start(executablePath);

            current = null;
            if (process is not null)
            {
                retired.Add(process);
            }

            process = started;
            bool handshaken = false;
            try
            {
                RunningWorker worker = await HandshakeAsync(started, cancellationToken).ConfigureAwait(false);
                handshaken = true;
                current = worker;
                return worker;
            }
            finally
            {
                // A worker that never completed the handshake must not linger holding GPU memory.
                if (!handshaken)
                {
                    KillQuietly(started);
                }
            }
        }
        finally
        {
            startGate.Release();
        }
    }

    private async Task<RunningWorker> HandshakeAsync(Process started, CancellationToken cancellationToken)
    {
        WorkerMessage response;
        long id = Interlocked.Increment(ref nextId);
        var completion = new TaskCompletionSource<WorkerMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = new PendingCall(started, completion);
        try
        {
            await WriteAsync(started, new WorkerMessage(id, InferenceWorkerProtocol.Hello), cancellationToken).ConfigureAwait(false);
            response = await completion.Task.WaitAsync(TimeSpan.FromSeconds(60), cancellationToken).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            throw new InferenceWorkerException($"The inference worker did not answer the handshake. Output: {StderrSnapshot()}");
        }
        finally
        {
            pending.TryRemove(id, out _);
        }

        if (response.Kind == InferenceWorkerProtocol.Error)
        {
            throw new InferenceWorkerException($"Inference worker handshake failed: {response.Error}");
        }

        WorkerHelloResult result = InferenceWorkerProtocol.FromPayload<WorkerHelloResult>(response.Payload);
        if (result.ProtocolVersion != InferenceWorkerProtocol.Version ||
            !string.Equals(result.BuildStamp, InferenceWorkerProtocol.BuildStamp, StringComparison.Ordinal))
        {
            unusableReason =
                $"The inference worker at '{executablePath}' is from a different build (protocol {result.ProtocolVersion}, " +
                $"build {result.BuildStamp}; expected protocol {InferenceWorkerProtocol.Version}, build {InferenceWorkerProtocol.BuildStamp}).";
            throw new InferenceWorkerException(unusableReason);
        }

        return new RunningWorker(started, result);
    }

    // Runs from a cancellation callback that nothing awaits, so it must never fault.
    private async Task SendCancelAsync(Process target, long requestId)
    {
        try
        {
            await WriteAsync(target, new WorkerMessage(Interlocked.Increment(ref nextId), InferenceWorkerProtocol.Cancel,
                InferenceWorkerProtocol.ToPayload(requestId)), CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InferenceWorkerException or ObjectDisposedException or InvalidOperationException)
        {
            // The worker is gone or the client is disposed; the caller's wait is already cancelled.
        }
    }

    private Process Start(string path)
    {
        var startInfo = new ProcessStartInfo(path)
        {
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(path)!,
        };
        Process started = Process.Start(startInfo)
            ?? throw new InferenceWorkerException($"The inference worker at '{path}' did not start.");
        started.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null)
            {
                return;
            }

            lock (stderrTail)
            {
                stderrTail.Enqueue(e.Data);
                while (stderrTail.Count > StderrTailLines)
                {
                    stderrTail.Dequeue();
                }
            }
        };
        started.BeginErrorReadLine();
        _ = Task.Run(() => ReadLoopAsync(started));
        return started;
    }

    private async Task ReadLoopAsync(Process owner)
    {
        string reason = "The inference worker exited unexpectedly.";
        try
        {
            Stream output = owner.StandardOutput.BaseStream;
            while (await InferenceWorkerProtocol.ReadAsync(output, CancellationToken.None).ConfigureAwait(false) is { } message)
            {
                if (pending.TryGetValue(message.Id, out PendingCall? call))
                {
                    call.Completion.TrySetResult(message);
                }
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or JsonException or ObjectDisposedException)
        {
            // A broken or closed pipe is how a worker exit shows up; the calls below fail with it.
            reason = $"The inference worker stream closed ({ex.GetType().Name}: {ex.Message}).";
        }
        catch (Exception ex)
        {
            // Nothing observes this loop's task, so every failure must reach the pending calls.
            reason = $"The inference worker connection failed ({ex.GetType().Name}: {ex.Message}).";
        }

        // The worker exited or its stream broke: every outstanding call fails with what it said.
        reason = $"{reason} Output: {StderrSnapshot()}";
        // Only this process's calls: a replacement worker's requests share the map.
        foreach (PendingCall call in pending.Values.Where(call => ReferenceEquals(call.Owner, owner)))
        {
            call.Completion.TrySetException(new InferenceWorkerException(reason));
        }
    }

    private async Task WriteAsync(Process target, WorkerMessage message, CancellationToken cancellationToken)
    {
        await writeGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                await InferenceWorkerProtocol.WriteAsync(target.StandardInput.BaseStream, message, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Cancelled mid-frame (a worker that stopped draining its input can block the write
                // indefinitely): the pipe may hold a partial frame, so this worker cannot be trusted
                // with another message. Killing it ends its read loop and the next call restarts it.
                KillQuietly(target);
                throw;
            }
        }
        catch (IOException ex)
        {
            throw new InferenceWorkerException($"The inference worker is not accepting requests: {ex.Message}. Output: {StderrSnapshot()}");
        }
        finally
        {
            writeGate.Release();
        }
    }

    private string StderrSnapshot()
    {
        lock (stderrTail)
        {
            return stderrTail.Count == 0 ? "(none)" : string.Join(Environment.NewLine, stderrTail);
        }
    }

    private static void KillQuietly(Process target)
    {
        try
        {
            if (!target.HasExited)
            {
                target.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
            // The process already exited between the check and the kill; nothing is left to stop.
        }
    }

    // DI containers disposed synchronously (console hosts) require IDisposable on singletons.
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
        {
            return;
        }

        // Wait out any start in progress so the process it creates is the one killed below.
        // Callers queued behind it then see the disposed flag inside the gate.
        await startGate.WaitAsync().ConfigureAwait(false);

        if (process is { HasExited: false } running)
        {
            try
            {
                await WriteAsync(running, new WorkerMessage(Interlocked.Increment(ref nextId), InferenceWorkerProtocol.Shutdown), CancellationToken.None)
                    .ConfigureAwait(false);
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                await running.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is InferenceWorkerException or OperationCanceledException)
            {
                // The worker ignored or could not receive the shutdown request; it is killed below.
            }

            KillQuietly(running);
        }

        process?.Dispose();
        foreach (Process old in retired)
        {
            old.Dispose();
        }

        writeGate.Dispose();

        // Not disposed: queued EnsureStartedAsync callers must wake and throw, not hang.
        startGate.Release();
    }

    private sealed record RunningWorker(Process Process, WorkerHelloResult Hello);

    private sealed record PendingCall(Process Owner, TaskCompletionSource<WorkerMessage> Completion);
}

public sealed class InferenceWorkerException(string message) : InvalidOperationException(message);
