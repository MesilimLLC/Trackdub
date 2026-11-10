using System.Globalization;
using System.Runtime.ExceptionServices;
using System.Text;
using Trackdub.Media.Process;

namespace Trackdub.Media.LipSynthesis.NvidiaMaxine;

/// <summary>One job for the persistent host: a turn's frames at a fixed size, rate and language.</summary>
internal sealed record MaxineLipSyncHostJob(
    int Width,
    int Height,
    double FramesPerSecond,
    NvidiaMaxineLipSyncLanguage Language,
    int FrameCount,
    string AudioPath)
{
    /// <summary>Matches the host's request validation; it rejects larger frames without reading them.</summary>
    public const int MaxDimension = 4096;

    public int FrameBytes => checked(Width * Height * 3);
}

/// <summary>What the host reported after returning a job's frames.</summary>
internal sealed record MaxineLipSyncHostReply(bool Succeeded, string Message);

/// <summary>The host broke the protocol (exited, closed a pipe or wrote something unexpected).</summary>
internal sealed class MaxineLipSyncHostProtocolException(string message, Exception? inner = null)
    : IOException(message, inner);

/// <summary>
/// Client for <c>tools/native/maxine-lipsync-host</c>: a process that keeps one LipSync session and
/// takes turns over stdin/stdout. A request is <c>RUN w h fps language frames wav</c> followed by
/// raw BGR24 frames; the reply is exactly that many frames, then <c>OK</c> or <c>ERR</c>.
/// </summary>
/// <remarks>
/// Frames go in and come out concurrently: the model holds back its first output until it has
/// buffered dozens of input frames, so writing everything before reading would deadlock on full
/// pipes. Any protocol break leaves the byte stream unusable, so callers dispose the session.
/// </remarks>
internal sealed class MaxineLipSyncHostSession : IDisposable
{
    public const string Handshake = "TRACKDUB-MAXINE-LIPSYNC 1 READY";

    private const int MaxLineBytes = 4096;

    private readonly IPipedProcess process;
    private bool disposed;

    private MaxineLipSyncHostSession(IPipedProcess process)
    {
        this.process = process;
    }

    public bool IsAlive => !disposed && !process.HasExited;

    /// <summary>
    /// Kills the host without the QUIT handshake. A turn in flight then fails with a protocol error, which
    /// is how shutdown frees a turn that would otherwise run until its timeout.
    /// </summary>
    public void Abort() => process.Kill();

    /// <summary>
    /// Waits for the handshake line. Returns null (and kills the process) when the executable is
    /// not a persistent host, for example the SDK's LipSyncApp sample.
    /// </summary>
    public static async Task<MaxineLipSyncHostSession?> ConnectAsync(
        IPipedProcess process,
        TimeSpan handshakeTimeout,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(process);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(handshakeTimeout);
        string? line = null;
        try
        {
            line = await ReadLineAsync(process.Output, timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            line = null;
        }
        catch (IOException)
        {
            line = null;
        }
        catch (OperationCanceledException)
        {
            process.Dispose();
            throw;
        }

        if (line == Handshake)
        {
            return new MaxineLipSyncHostSession(process);
        }

        process.Dispose();
        return null;
    }

    /// <summary>
    /// Streams <paramref name="job"/>.FrameCount frames from <paramref name="frameSource"/> through the
    /// host into <paramref name="frameSink"/>. If the source ends early its last frame is repeated, so
    /// the host always gets the count announced in the request header.
    /// </summary>
    public async Task<MaxineLipSyncHostReply> RunAsync(
        MaxineLipSyncHostJob job,
        Stream frameSource,
        Stream frameSink,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(job);
        ArgumentNullException.ThrowIfNull(frameSource);
        ArgumentNullException.ThrowIfNull(frameSink);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (job.Width <= 0 || job.Height <= 0 || job.Width > MaxineLipSyncHostJob.MaxDimension || job.Height > MaxineLipSyncHostJob.MaxDimension)
        {
            throw new ArgumentOutOfRangeException(nameof(job), $"Frames must be 1..{MaxineLipSyncHostJob.MaxDimension} pixels on each side.");
        }

        if (job.FrameCount <= 0 || job.FramesPerSecond <= 0 || job.AudioPath.Contains('\n') || job.AudioPath.Contains('\r'))
        {
            throw new ArgumentException("Invalid host job.", nameof(job));
        }

        string header = string.Create(CultureInfo.InvariantCulture,
            $"RUN {job.Width} {job.Height} {job.FramesPerSecond:0.######} {(int)job.Language} {job.FrameCount} {job.AudioPath}\n");

        // Either side failing leaves the other blocked on a pipe; cancelling kills the host to free it.
        using var failure = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        using CancellationTokenRegistration killOnFailure = failure.Token.Register(process.Kill);
        try
        {
            await process.Input.WriteAsync(Encoding.UTF8.GetBytes(header), failure.Token).ConfigureAwait(false);
            Task feed = FeedAsync(job, frameSource, failure);
            Task collect = CollectAsync(job, frameSink, failure);
            try
            {
                await Task.WhenAll(feed, collect).ConfigureAwait(false);
            }
            catch (Exception) when (!cancellationToken.IsCancellationRequested)
            {
                // The side that failed first cancels the other; report the first cause, not the cancellation.
                ExceptionDispatchInfo.Throw(RootCause(feed, collect));
            }

            string status = await ReadLineAsync(process.Output, failure.Token).ConfigureAwait(false)
                ?? throw new MaxineLipSyncHostProtocolException(Describe("The LipSync host exited before reporting a result."));
            if (status.StartsWith("OK ", StringComparison.Ordinal))
            {
                return new MaxineLipSyncHostReply(true, status);
            }

            if (status.StartsWith("ERR ", StringComparison.Ordinal))
            {
                return new MaxineLipSyncHostReply(false, status[4..]);
            }

            throw new MaxineLipSyncHostProtocolException(Describe($"Unexpected LipSync host reply: {Shorten(status)}"));
        }
        catch (Exception ex) when (cancellationToken.IsCancellationRequested)
        {
            // Cancelling kills the host, which surfaces as broken pipes; report the cancellation.
            process.Kill();
            throw ex as OperationCanceledException ?? new OperationCanceledException(cancellationToken);
        }
        catch (MaxineLipSyncHostProtocolException)
        {
            throw;
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException or OperationCanceledException)
        {
            throw new MaxineLipSyncHostProtocolException(Describe($"The LipSync host stopped mid-turn: {ex.Message}"), ex);
        }
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        disposed = true;
        try
        {
            TryQuit();
        }
        finally
        {
            process.Dispose();
        }
    }

    /// <summary>Asks the host to exit and gives it two seconds; false when the pipe was already broken.</summary>
    private bool TryQuit()
    {
        try
        {
            if (!process.HasExited)
            {
                process.Input.Write("QUIT\n"u8);
                process.Input.Flush();
                process.WaitForExitAsync(CancellationToken.None).Wait(TimeSpan.FromSeconds(2));
            }

            return true;
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException or AggregateException
                                       or InvalidOperationException)
        {
            return false;
        }
    }

    private async Task FeedAsync(MaxineLipSyncHostJob job, Stream frameSource, CancellationTokenSource failure)
    {
        try
        {
            byte[] frame = new byte[job.FrameBytes];
            byte[] previous = new byte[job.FrameBytes];
            bool havePrevious = false;
            for (int index = 0; index < job.FrameCount; index++)
            {
                int read = await frameSource.ReadAtLeastAsync(frame, frame.Length, throwOnEndOfStream: false, failure.Token)
                    .ConfigureAwait(false);
                if (read < frame.Length)
                {
                    if (!havePrevious)
                    {
                        throw new IOException("The frame source produced no complete frame.");
                    }

                    previous.CopyTo(frame, 0);
                }

                await process.Input.WriteAsync(frame, failure.Token).ConfigureAwait(false);
                (frame, previous) = (previous, frame);
                havePrevious = true;
            }

            await process.Input.FlushAsync(failure.Token).ConfigureAwait(false);
        }
        catch
        {
            await failure.CancelAsync().ConfigureAwait(false);
            throw;
        }
    }

    private async Task CollectAsync(MaxineLipSyncHostJob job, Stream frameSink, CancellationTokenSource failure)
    {
        try
        {
            byte[] frame = new byte[job.FrameBytes];
            for (int index = 0; index < job.FrameCount; index++)
            {
                await process.Output.ReadExactlyAsync(frame, failure.Token).ConfigureAwait(false);
                await frameSink.WriteAsync(frame, failure.Token).ConfigureAwait(false);
            }

            await frameSink.FlushAsync(failure.Token).ConfigureAwait(false);
        }
        catch (EndOfStreamException ex)
        {
            await failure.CancelAsync().ConfigureAwait(false);
            throw new MaxineLipSyncHostProtocolException(Describe("The LipSync host returned fewer frames than requested."), ex);
        }
        catch
        {
            await failure.CancelAsync().ConfigureAwait(false);
            throw;
        }
    }

    private static Exception RootCause(params Task[] tasks)
    {
        Exception[] causes = tasks.Where(task => task.IsFaulted).Select(task => task.Exception!.InnerException!).ToArray();
        return causes.FirstOrDefault(cause => cause is MaxineLipSyncHostProtocolException)
            ?? causes.FirstOrDefault(cause => cause is not OperationCanceledException)
            ?? causes.FirstOrDefault()
            ?? new OperationCanceledException();
    }

    /// <summary>Reads one '\n'-terminated line byte by byte, so no frame bytes are buffered past it.</summary>
    private static async Task<string?> ReadLineAsync(Stream stream, CancellationToken cancellationToken)
    {
        var bytes = new List<byte>();
        byte[] one = new byte[1];
        while (true)
        {
            int read = await stream.ReadAsync(one, cancellationToken).ConfigureAwait(false);
            if (read == 0)
            {
                return bytes.Count == 0 ? null : Encoding.UTF8.GetString(bytes.ToArray());
            }

            if (one[0] == (byte)'\n')
            {
                return Encoding.UTF8.GetString(bytes.ToArray()).TrimEnd('\r');
            }

            if (bytes.Count >= MaxLineBytes)
            {
                throw new MaxineLipSyncHostProtocolException("The LipSync host wrote an overlong line.");
            }

            bytes.Add(one[0]);
        }
    }

    private string Describe(string message)
    {
        string tail = process.ErrorTail;
        return tail.Length == 0 ? message : $"{message} Host stderr: {Shorten(tail)}";
    }

    private static string Shorten(string text) => text.Length <= 400 ? text : text[^400..];
}
