using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace Trackdub.Infrastructure.ModelOptimization;

internal interface IStreamingProcessRunner
{
    IAsyncEnumerable<string> RunAsync(
        string executable,
        IReadOnlyList<string> arguments,
        string workingDirectory,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? environment = null);
}

internal sealed class StreamingProcessRunner : IStreamingProcessRunner
{
    public async IAsyncEnumerable<string> RunAsync(
        string executable,
        IReadOnlyList<string> arguments,
        string workingDirectory,
        [EnumeratorCancellation] CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? environment = null)
    {
        ProcessStartInfo startInfo = CreateStartInfo(executable, arguments, workingDirectory, environment);

        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });

        using var process = new Process { StartInfo = startInfo };
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                channel.Writer.TryWrite(e.Data);
            }
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                channel.Writer.TryWrite($"[err] {e.Data}");
            }
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        _ = process.WaitForExitAsync(cancellationToken).ContinueWith(
            _ => channel.Writer.TryComplete(),
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);

        try
        {
            await foreach (string line in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return line;
            }
        }
        finally
        {
            if (!process.HasExited)
            {
                try { process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { }
                catch (NotSupportedException) { }
            }
        }
    }

    private static ProcessStartInfo CreateStartInfo(
        string executable,
        IReadOnlyList<string> arguments,
        string workingDirectory,
        IReadOnlyDictionary<string, string>? environment)
    {
        var startInfo = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = workingDirectory
        };
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux())
        {
            startInfo.KillOnParentExit = true;
        }

        foreach (string arg in arguments)
        {
            startInfo.ArgumentList.Add(arg);
        }

        if (environment is not null)
        {
            foreach ((string name, string value) in environment)
            {
                startInfo.Environment[name] = value;
            }
        }

        return startInfo;
    }
}
