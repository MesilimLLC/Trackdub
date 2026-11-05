using System.Collections.Concurrent;
using Trackdub.Contracts;
using Trackdub.Infrastructure.Components.NvidiaAfx;

namespace Trackdub.Composition.NvidiaAfx;

/// <summary>
/// Probes whether a local AFX runtime can actually load the native library, create and load an
/// effect, and run audio through it. Filename presence alone must not be treated as Ready.
/// </summary>
public interface INvidiaAfxEffectProbe
{
    NvidiaAfxEffectProbeResult Probe(
        string runtimeRoot,
        NvidiaAfxProfileDefinition profile,
        int inputSampleRate,
        string? architectureBucket = null);
}

public sealed record NvidiaAfxEffectProbeResult(
    bool Succeeded,
    string? FailureReason,
    int? OutputSampleRate);

/// <summary>
/// Default probe: creates and loads a Maxine effect session, streams a native-valid audio span
/// through it, then disposes it. Audio is run because creating and loading an effect does not prove
/// it can process it, and a sustained run is needed because some effects accept the first frame and
/// fail later (Speaker Focus on SDK 2.1). A failed run later still falls back to DeepFilterNet.
/// </summary>
public sealed class NvidiaAfxSessionEffectProbe : INvidiaAfxEffectProbe
{
    public static NvidiaAfxSessionEffectProbe Instance { get; } = new();

    /// <summary>Frame count for common 10 ms profiles; long-frame profiles use their own count.</summary>
    internal const int ProbeFrames = 100;

    public NvidiaAfxEffectProbeResult Probe(
        string runtimeRoot,
        NvidiaAfxProfileDefinition profile,
        int inputSampleRate,
        string? architectureBucket = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(runtimeRoot);
        ArgumentNullException.ThrowIfNull(profile);

        try
        {
            using NvidiaAfxSession session = NvidiaAfxSession.Create(
                profile,
                runtimeRoot,
                inputSampleRate,
                intensityRatio: profile.SupportsIntensityRatio ? 1.0f : 0f,
                architectureBucket);

            int frameSamples = checked((int)session.NumInputSamplesPerFrame);
            float[] frame = new float[frameSamples];
            float[]? farEnd = profile.RequiresFarEndReference ? new float[frameSamples] : null;
            for (int frameIndex = 0; frameIndex < profile.ReadinessProbeFrameCount; frameIndex++)
            {
                for (int index = 0; index < frame.Length; index++)
                {
                    int sample = (frameIndex * frameSamples) + index;
                    frame[index] = 0.1f * MathF.Sin(2f * MathF.PI * 220f * sample / inputSampleRate);
                }

                session.Process(frame, farEnd);
            }

            return new NvidiaAfxEffectProbeResult(true, null, session.OutputSampleRate);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return new NvidiaAfxEffectProbeResult(
                false,
                $"AFX native probe failed for profile '{profile.Selector}': {ex.Message}",
                null);
        }
    }
}

/// <summary>
/// Remembers successful native probes so readiness checks do not create a GPU effect every time.
/// Concurrent callers for the same key share one probe, and failures are never cached. The key
/// includes the runtime root, effect, sample rate, architecture and the native library's write
/// time, so a different or replaced runtime is probed again.
/// </summary>
internal sealed class CachingNvidiaAfxEffectProbe(INvidiaAfxEffectProbe inner) : INvidiaAfxEffectProbe
{
    private readonly ConcurrentDictionary<ProbeKey, Lazy<NvidiaAfxEffectProbeResult>> _results = new();

    public NvidiaAfxEffectProbeResult Probe(
        string runtimeRoot,
        NvidiaAfxProfileDefinition profile,
        int inputSampleRate,
        string? architectureBucket = null)
    {
        var key = new ProbeKey(
            runtimeRoot,
            profile.Selector,
            inputSampleRate,
            architectureBucket,
            NativeLibraryStamp(runtimeRoot));
        Lazy<NvidiaAfxEffectProbeResult> entry = _results.GetOrAdd(
            key,
            _ => new Lazy<NvidiaAfxEffectProbeResult>(
                () => inner.Probe(runtimeRoot, profile, inputSampleRate, architectureBucket)));

        NvidiaAfxEffectProbeResult result = entry.Value;
        if (!result.Succeeded)
        {
            _results.TryRemove(key, out _);
        }

        return result;
    }

    private static long NativeLibraryStamp(string runtimeRoot)
    {
        string? path = NvidiaAfxRuntimeLayout.ResolveNativeLibraryPath(runtimeRoot);
        return path is null ? 0 : File.GetLastWriteTimeUtc(path).Ticks;
    }

    private readonly record struct ProbeKey(
        string RuntimeRoot,
        string Selector,
        int SampleRate,
        string? Architecture,
        long NativeLibraryStamp);
}
