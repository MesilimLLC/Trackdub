using System.Diagnostics;
using System.Text.Json;

namespace Trackdub.SidecarHost;

/// <summary>
/// C# host-side mirror of workers/supervisor/src/supervisor.rs: spawns the worker
/// with piped stdio, speaks JSON-lines with strictly ordered id-matched requests,
/// enforces the protocol version stamp and the closed status/reason vocabularies,
/// poisons the connection — best-effort killing the child, as supervisor.rs's
/// poison() does — on timeout, unparseable response, or id-mismatch, and kills
/// the child on dispose. Requests are one-in-flight; the contract is enforced
/// by serializing the whole write-read exchange.
/// </summary>
public sealed class SupervisedSidecarWorker : IAsyncDisposable
{
    public const int ExpectedProtocolVersion = 1;
    private static readonly TimeSpan ResponseTimeout = TimeSpan.FromSeconds(120);

    /// <summary>
    /// One-in-flight request timeout. Mirrors the supervisor's gate window for
    /// short ops (health/infer), but cold model loads legitimately take several
    /// minutes, so <see cref="Request"/> applies it automatically for op "load".
    /// </summary>
    public static readonly TimeSpan LoadTimeout = TimeSpan.FromMinutes(10);

    private readonly Process _process;
    private readonly StreamReader _stdout;
    private readonly StreamWriter _stdin;
    private readonly SemaphoreSlim _requestLock = new(1, 1);
    private bool _poisoned;

    private SupervisedSidecarWorker(Process process, StreamReader stdout, StreamWriter stdin)
    {
        _process = process;
        _stdout = stdout;
        _stdin = stdin;
    }

    public int ProcessId => _process.Id;

    /// <summary>
    /// Effective per-request timeout: an explicit timeout always wins; op "load"
    /// defaults to <see cref="LoadTimeout"/> (cold model loads are minutes, not
    /// seconds); every other op defaults to the 120 s gate window.
    /// </summary>
    public static TimeSpan EffectiveTimeout(string op, TimeSpan? timeout) =>
        timeout ?? (op == "load" ? LoadTimeout : ResponseTimeout);

    public static SupervisedSidecarWorker Spawn(
        string program,
        string[] args,
        string? workingDirectory = null,
        Dictionary<string, string>? environment = null)
    {
        var psi = new ProcessStartInfo(program)
        {
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = false,   // worker stderr inherits, same as supervisor.rs
            CreateNoWindow = true,
        };
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux())
        {
            psi.KillOnParentExit = true;
        }

        foreach (string arg in args)
        {
            psi.ArgumentList.Add(arg);
        }

        if (workingDirectory is not null)
        {
            psi.WorkingDirectory = workingDirectory;
        }

        if (environment is not null)
        {
            foreach (var (key, value) in environment)
            {
                psi.Environment[key] = value;
            }
        }

        Process process = Process.Start(psi) ?? throw new InvalidOperationException($"failed to spawn {program}");
        return new SupervisedSidecarWorker(process, process.StandardOutput, process.StandardInput);
    }

    public SidecarResponse Request(SidecarRequest request, TimeSpan? timeout = null)
    {
        if (_poisoned)
        {
            throw new InvalidOperationException("worker connection is unusable after a timeout or out-of-order response; respawn first");
        }

        // One-in-flight, enforced: serialize the entire write-read exchange so
        // two concurrent callers cannot interleave writes or read each other's
        // responses. Serialization (rather than rejecting the second caller) is
        // the simpler contract: every exchange is bounded by its own timeout, so
        // a waiter blocks at most that long before failing or being served.
        _requestLock.Wait();
        try
        {
            if (_poisoned)
            {
                // Another concurrent Request may have poisoned while we waited.
                throw new InvalidOperationException("worker connection is unusable after a timeout or out-of-order response; respawn first");
            }

            TimeSpan responseTimeout = EffectiveTimeout(request.Op, timeout);

            // Apply the timeout to the stdin write too: a worker that stops
            // reading stdin can block WriteLine/Flush indefinitely, and the
            // read-side timeout would never start. A write overrun or failure
            // poisons, mirroring the read path.
            Task write = Task.Run(() =>
            {
                _stdin.WriteLine(JsonSerializer.Serialize(request, ProtocolJson.Options));
                _stdin.Flush();
            });
            if (!write.Wait(responseTimeout))
            {
                Poison();
                throw new InvalidOperationException($"worker stdin write timed out after {responseTimeout.TotalSeconds}s");
            }
            if (write.IsFaulted)
            {
                Poison();
                throw new InvalidOperationException("worker stdin write failed; connection unusable", write.Exception!.GetBaseException());
            }

            Task<string?> read = _stdout.ReadLineAsync();
            if (!read.Wait(responseTimeout))
            {
                Poison();
                throw new InvalidOperationException($"worker response timed out after {responseTimeout.TotalSeconds}s");
            }
            if (read.IsFaulted)
            {
                Poison();
                throw new InvalidOperationException("worker stdout read failed; connection unusable", read.Exception!.GetBaseException());
            }

            string? line = read.Result;
            if (line is null)
            {
                Poison();
                throw new InvalidOperationException("worker stdout closed");
            }

            SidecarResponse response;
            try
            {
                response = JsonSerializer.Deserialize<SidecarResponse>(line, ProtocolJson.Options)
                    ?? throw new InvalidOperationException($"unparseable worker response: {line}");
            }
            catch (JsonException ex)
            {
                // A non-JSON line (the stray-banner class of bug) desyncs the
                // stream: the real response may still be queued behind it.
                // Poison so no later Request reuses the invalid stream.
                Poison();
                throw new InvalidOperationException($"unparseable worker response: {line}", ex);
            }
            catch (InvalidOperationException)
            {
                // Covers the literal-null line above: same desync, same poison.
                Poison();
                throw;
            }

            if (response.ProtocolVersion != ExpectedProtocolVersion)
            {
                Poison();
                throw new InvalidOperationException($"worker protocol version {response.ProtocolVersion} != {ExpectedProtocolVersion}; refusing to serve");
            }
            if (response.Id != request.Id)
            {
                Poison();
                throw new InvalidOperationException($"worker response id {response.Id} != request id {request.Id}; stream out of order");
            }

            // Closed vocabularies (workers/PROTOCOL.md): an out-of-set status or
            // reason must not cross the host boundary — parity with the Rust
            // supervisor, which rejects these at deserialization. Like every
            // other response-validation branch, a failure poisons.
            try
            {
                ProtocolVocabulary.Validate(response);
            }
            catch (InvalidOperationException)
            {
                Poison();
                throw;
            }

            return response;
        }
        finally
        {
            _requestLock.Release();
        }
    }

    private void Poison()
    {
        // Mirror supervisor.rs poison(): mark unusable AND best-effort kill the
        // child, so a timed-out or wedged worker cannot keep consuming GPU/CPU
        // until dispose; a subsequent Request fails fast on the poisoned guard.
        _poisoned = true;
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException
            or System.ComponentModel.Win32Exception
            or NotSupportedException)
        {
            // best-effort: poison marks the connection unusable regardless
        }
    }

    public async ValueTask DisposeAsync()
    {
        // kill-on-drop, mirroring SupervisedWorker::Drop in supervisor.rs;
        // best-effort throughout: never throw from dispose.
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }

            // Kill is asynchronous: wait (bounded) so the child is really gone
            // and its pipe handles are released before disposal returns.
            _process.WaitForExit(5_000);
        }
        catch (Exception ex) when (ex is InvalidOperationException
            or System.ComponentModel.Win32Exception
            or NotSupportedException)
        {
            // best-effort: never throw from dispose
        }

        _stdout.Dispose();
        _stdin.Dispose();
        _process.Dispose();
        _requestLock.Dispose();
        await Task.CompletedTask;
    }
}
