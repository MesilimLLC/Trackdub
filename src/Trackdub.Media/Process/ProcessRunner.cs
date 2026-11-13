using System.Diagnostics;
using System.ComponentModel;
using System.Text;

namespace Trackdub.Media.Process;

internal interface IProcessRunner
{
    Task<ProcessResult> RunAsync(
        string executablePath,
        IReadOnlyList<string> arguments,
        CancellationToken cancellationToken,
        ProcessRunOptions? options = null);
}

internal sealed record ProcessRunOptions(
    TimeSpan? Timeout = null,
    int MaxStandardOutputCharacters = 16 * 1024 * 1024,
    int MaxStandardErrorCharacters = 4 * 1024 * 1024,
    IReadOnlyDictionary<string, string>? EnvironmentVariables = null);

internal sealed record ProcessResult(
    int ExitCode,
    string StandardOutput,
    string StandardError,
    bool StandardOutputTruncated = false,
    bool StandardErrorTruncated = false);

internal sealed class ProcessRunnerTimeoutException(
    string executablePath,
    TimeSpan timeout,
    string standardOutput,
    string standardError,
    bool standardOutputTruncated,
    bool standardErrorTruncated)
    : TimeoutException($"Process '{executablePath}' exceeded the timeout of {timeout}.")
{
    public string ExecutablePath { get; } = executablePath;
    public TimeSpan Timeout { get; } = timeout;
    public string StandardOutput { get; } = standardOutput;
    public string StandardError { get; } = standardError;
    public bool StandardOutputTruncated { get; } = standardOutputTruncated;
    public bool StandardErrorTruncated { get; } = standardErrorTruncated;
}

internal sealed class ProcessRunner : IProcessRunner
{
    private static readonly TimeSpan PostTimeoutCleanupGracePeriod = TimeSpan.FromSeconds(1);

    public async Task<ProcessResult> RunAsync(
        string executablePath,
        IReadOnlyList<string> arguments,
        CancellationToken cancellationToken,
        ProcessRunOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(executablePath);
        ArgumentNullException.ThrowIfNull(arguments);
        options ??= new ProcessRunOptions();

        ProcessStartInfo startInfo = CreateStartInfo(executablePath, arguments, options);

        using var process = new System.Diagnostics.Process { StartInfo = startInfo };
        if (!process.Start())
        {
            throw new InvalidOperationException($"Failed to start '{executablePath}'.");
        }

        Task<BoundedProcessOutput> standardOutputTask = ReadToEndBoundedAsync(
            process.StandardOutput,
            options.MaxStandardOutputCharacters);
        Task<BoundedProcessOutput> standardErrorTask = ReadToEndBoundedAsync(
            process.StandardError,
            options.MaxStandardErrorCharacters);
        using CancellationTokenSource? timeoutSource = options.Timeout is { } timeout
            ? new CancellationTokenSource(timeout)
            : null;
        using CancellationTokenSource linkedCancellation = timeoutSource is null
            ? CancellationTokenSource.CreateLinkedTokenSource(cancellationToken)
            : CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutSource.Token);

        try
        {
            await process.WaitForExitAsync(linkedCancellation.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested || timeoutSource?.IsCancellationRequested == true)
        {
            (BoundedProcessOutput standardOutput, BoundedProcessOutput standardError) =
                await KillAndDrainAsync(process, standardOutputTask, standardErrorTask).ConfigureAwait(false);

            if (cancellationToken.IsCancellationRequested)
            {
                throw;
            }

            throw new ProcessRunnerTimeoutException(
                executablePath,
                options.Timeout!.Value,
                standardOutput.Text,
                standardError.Text,
                standardOutput.Truncated,
                standardError.Truncated);
        }

        BoundedProcessOutput completedStandardOutput = await standardOutputTask.ConfigureAwait(false);
        BoundedProcessOutput completedStandardError = await standardErrorTask.ConfigureAwait(false);
        return new ProcessResult(
            process.ExitCode,
            completedStandardOutput.Text,
            completedStandardError.Text,
            completedStandardOutput.Truncated,
            completedStandardError.Truncated);
    }

    private static ProcessStartInfo CreateStartInfo(
        string executablePath,
        IReadOnlyList<string> arguments,
        ProcessRunOptions options)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = executablePath,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux())
        {
            startInfo.KillOnParentExit = true;
        }

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        if (options.EnvironmentVariables is not null)
        {
            foreach ((string name, string value) in options.EnvironmentVariables)
            {
                startInfo.Environment[name] = value;
            }
        }

        return startInfo;
    }

    private static async Task<(BoundedProcessOutput StandardOutput, BoundedProcessOutput StandardError)> KillAndDrainAsync(
        System.Diagnostics.Process process,
        Task<BoundedProcessOutput> standardOutputTask,
        Task<BoundedProcessOutput> standardErrorTask)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
        }
        catch (Win32Exception)
        {
        }
        catch (NotSupportedException)
        {
        }

        using var waitTimeout = new CancellationTokenSource(PostTimeoutCleanupGracePeriod);
        try
        {
            await process.WaitForExitAsync(waitTimeout.Token).ConfigureAwait(false);
        }
        catch (InvalidOperationException)
        {
        }
        catch (OperationCanceledException)
        {
        }

        Task<BoundedProcessOutput[]> drainTask = Task.WhenAll(standardOutputTask, standardErrorTask);
        Task completedTask = await Task.WhenAny(drainTask, Task.Delay(PostTimeoutCleanupGracePeriod)).ConfigureAwait(false);
        if (ReferenceEquals(completedTask, drainTask))
        {
            BoundedProcessOutput[] output = await drainTask.ConfigureAwait(false);
            return (output[0], output[1]);
        }

        ObserveFault(drainTask);
        return (
            new BoundedProcessOutput(string.Empty, Truncated: true),
            new BoundedProcessOutput(string.Empty, Truncated: true));
    }

    private static void ObserveFault(Task task)
    {
        _ = task.ContinueWith(
            static completedTask => _ = completedTask.Exception,
            CancellationToken.None,
            TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);
    }

    private static async Task<BoundedProcessOutput> ReadToEndBoundedAsync(
        TextReader reader,
        int maxCharacters)
    {
        if (maxCharacters < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxCharacters), "Maximum captured output cannot be negative.");
        }

        char[] buffer = new char[8192];
        var builder = new StringBuilder(Math.Min(maxCharacters, buffer.Length));
        bool truncated = false;

        while (true)
        {
            int read = await reader.ReadAsync(buffer.AsMemory()).ConfigureAwait(false);
            if (read == 0)
            {
                break;
            }

            int remaining = maxCharacters - builder.Length;
            if (remaining > 0)
            {
                builder.Append(buffer, 0, Math.Min(read, remaining));
            }

            if (read > remaining)
            {
                truncated = true;
            }
        }

        return new BoundedProcessOutput(builder.ToString(), truncated);
    }

    private sealed record BoundedProcessOutput(string Text, bool Truncated);
}
