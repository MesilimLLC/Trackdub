using System.ComponentModel;
using System.Diagnostics;
using System.Text;

namespace Trackdub.Media.Process;

/// <summary>A running process whose stdin and stdout are binary streams.</summary>
internal interface IPipedProcess : IDisposable
{
    Stream Input { get; }

    Stream Output { get; }

    bool HasExited { get; }

    /// <summary>The last lines the process wrote to stderr.</summary>
    string ErrorTail { get; }

    Task<int> WaitForExitAsync(CancellationToken cancellationToken);

    void Kill();
}

/// <summary>
/// Starts a process with redirected binary stdin/stdout for streaming raw data through it. Stderr is
/// drained continuously (so the process never blocks on a full pipe) and its tail kept for errors.
/// </summary>
internal sealed class PipedProcess : IPipedProcess
{
    private const int MaxErrorTailCharacters = 4096;

    private readonly System.Diagnostics.Process process;
    private readonly Task stderrDrain;
    private readonly StringBuilder errorTail = new();

    private PipedProcess(System.Diagnostics.Process process, bool redirectInput, bool redirectOutput)
    {
        this.process = process;
        Input = redirectInput ? process.StandardInput.BaseStream : Stream.Null;
        Output = redirectOutput ? process.StandardOutput.BaseStream : Stream.Null;
        stderrDrain = DrainErrorAsync(process.StandardError);
    }

    public Stream Input { get; }

    public Stream Output { get; }

    public bool HasExited
    {
        get
        {
            try
            {
                return process.HasExited;
            }
            catch (InvalidOperationException)
            {
                return true;
            }
        }
    }

    /// <summary>True when stderr ended with an I/O error instead of end of stream.</summary>
    public bool ErrorStreamClosedEarly { get; private set; }

    public string ErrorTail
    {
        get
        {
            lock (errorTail)
            {
                return errorTail.ToString().Trim();
            }
        }
    }

    public static PipedProcess Start(
        string executablePath,
        IReadOnlyList<string> arguments,
        IReadOnlyDictionary<string, string>? environment = null,
        bool redirectInput = true,
        bool redirectOutput = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(executablePath);
        var startInfo = new ProcessStartInfo
        {
            FileName = executablePath,
            RedirectStandardInput = redirectInput,
            RedirectStandardOutput = redirectOutput,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux())
        {
            startInfo.KillOnParentExit = true;
        }

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        if (environment is not null)
        {
            foreach ((string name, string value) in environment)
            {
                startInfo.Environment[name] = value;
            }
        }

        var process = new System.Diagnostics.Process { StartInfo = startInfo };
        try
        {
            if (!process.Start())
            {
                throw new InvalidOperationException($"Failed to start '{executablePath}'.");
            }
        }
        catch
        {
            process.Dispose();
            throw;
        }

        return new PipedProcess(process, redirectInput, redirectOutput);
    }

    public async Task<int> WaitForExitAsync(CancellationToken cancellationToken)
    {
        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        await stderrDrain.ConfigureAwait(false);
        return process.ExitCode;
    }

    public void Kill() => TryKill();

    /// <summary>False when the process had already gone or cannot be killed, which callers treat as done.</summary>
    private bool TryKill()
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            return true;
        }
        catch (Exception ex) when (ex is InvalidOperationException or Win32Exception or NotSupportedException)
        {
            return false;
        }
    }

    public void Dispose()
    {
        Kill();
        process.Dispose();
    }

    private async Task DrainErrorAsync(StreamReader reader)
    {
        try
        {
            while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                lock (errorTail)
                {
                    errorTail.AppendLine(line);
                    if (errorTail.Length > MaxErrorTailCharacters)
                    {
                        errorTail.Remove(0, errorTail.Length - MaxErrorTailCharacters);
                    }
                }
            }
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException)
        {
            // The pipe closed under the reader when the process was killed; what was read so far is kept.
            ErrorStreamClosedEarly = true;
        }
    }
}

internal delegate IPipedProcess PipedProcessLauncher(
    string executablePath,
    IReadOnlyList<string> arguments,
    IReadOnlyDictionary<string, string>? environment,
    bool redirectInput,
    bool redirectOutput);
