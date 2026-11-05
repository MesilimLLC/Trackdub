using Trackdub.Inference.Onnx.Audio;
using System.Runtime.CompilerServices;
using Trackdub.Composition.NvidiaAfx;
using Trackdub.Contracts;
using Trackdub.Infrastructure.Components.NvidiaAfx;

namespace Trackdub.Composition.Tests;

/// <summary>
/// Skips unless <c>TRACKDUB_NVIDIA_AFX_RUNTIME_ROOT</c> points at an installed NVIDIA AFX runtime
/// (a folder containing <c>NVAudioEffects.dll</c>) on a Windows NVIDIA RTX machine.
/// </summary>
public sealed class NvidiaAfxLiveRuntimeFactAttribute : FactAttribute
{
    public const string RuntimeRootVariable = "TRACKDUB_NVIDIA_AFX_RUNTIME_ROOT";

    public NvidiaAfxLiveRuntimeFactAttribute(
        [CallerFilePath] string? sourceFilePath = null,
        [CallerLineNumber] int sourceLineNumber = -1)
        : base(sourceFilePath, sourceLineNumber)
    {
        string? root = Environment.GetEnvironmentVariable(RuntimeRootVariable);
        if (!OperatingSystem.IsWindows()
            || string.IsNullOrWhiteSpace(root)
            || !NvidiaAfxRuntimeLayout.HasNativeLibrary(root))
        {
            Skip = $"Set {RuntimeRootVariable} to an installed NVIDIA AFX runtime (folder containing " +
                   "NVAudioEffects.dll) on a Windows NVIDIA RTX machine to run the live create/run proofs.";
        }
    }

    public static string RuntimeRoot =>
        Environment.GetEnvironmentVariable(RuntimeRootVariable)
        ?? throw new InvalidOperationException($"{RuntimeRootVariable} is not set.");
}

/// <summary>
/// Hardware-gated proof that the Maxine native create/load/run path works against a real runtime.
/// These are the evidence behind flipping <see cref="NvidiaAfxIntegration.IsStubbed"/>; they never
/// run in default CI.
/// </summary>
public sealed class NvidiaAfxLiveRuntimeTests(Xunit.ITestOutputHelper output)
{
    private static (string Root, string Architecture) ResolveRuntime() =>
        (NvidiaAfxLiveRuntimeFactAttribute.RuntimeRoot, new NvidiaAfxArchitectureDetector().DetectArchitectureBucket());

    /// <summary>Profiles the UI offers by default: no AEC (needs a far-end reference) and no Early Access.</summary>
    private static IEnumerable<NvidiaAfxProfileDefinition> ShippingDefinitions() =>
        NvidiaAfxProfileCatalog.Definitions
            .Where(definition => !definition.RequiresFarEndReference && !definition.IsEarlyAccess);

    [NvidiaAfxLiveRuntimeFact]
    public void Probe_succeeds_for_every_selectable_profile_and_rate()
    {
        (string root, string architecture) = ResolveRuntime();
        output.WriteLine($"runtime root: {root}");
        output.WriteLine($"architecture bucket: {architecture}");

        var failures = new List<string>();
        foreach (NvidiaAfxProfileDefinition definition in ShippingDefinitions())
        {
            foreach (int sampleRate in definition.SupportedSampleRates)
            {
                NvidiaAfxEffectProbeResult result = NvidiaAfxSessionEffectProbe.Instance
                    .Probe(root, definition, sampleRate, architecture);
                output.WriteLine(
                    $"{definition.Profile} @ {sampleRate} Hz: " +
                    (result.Succeeded ? $"OK (output {result.OutputSampleRate} Hz)" : $"FAIL {result.FailureReason}"));
                if (!result.Succeeded)
                {
                    failures.Add($"{definition.Profile} @ {sampleRate} Hz: {result.FailureReason}");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Same_rate_effects_process_real_audio_at_every_supported_rate()
    {
        (string root, string architecture) = ResolveRuntime();

        var failures = new List<string>();
        foreach (NvidiaAfxProfileDefinition definition in ShippingDefinitions()
                     .Where(definition => definition.ResolveOutputSampleRate(definition.SupportedSampleRates[0])
                                          == definition.SupportedSampleRates[0]))
        {
            foreach (int sampleRate in definition.SupportedSampleRates)
            {
                try
                {
                    float[] input = BuildNoisySpeechLikeSignal(sampleRate, seconds: 2);
                    using NvidiaAfxSession session = NvidiaAfxSession.Create(
                        definition,
                        root,
                        sampleRate,
                        intensityRatio: 1.0f,
                        architecture);
                    float[] processed = session.Process(input);

                    double meanAbsDifference = input.Zip(processed, (a, b) => Math.Abs(a - b)).Average();
                    output.WriteLine(
                        $"{definition.Profile} @ {sampleRate} Hz: frame in/out " +
                        $"{session.NumInputSamplesPerFrame}/{session.NumOutputSamplesPerFrame}, " +
                        $"samples in/out {input.Length}/{processed.Length}, mean |in-out| {meanAbsDifference:F5}");

                    if (processed.Length != input.Length)
                    {
                        failures.Add($"{definition.Profile} @ {sampleRate} Hz: length {processed.Length} != {input.Length}");
                    }
                    else if (processed.Any(sample => !float.IsFinite(sample)))
                    {
                        failures.Add($"{definition.Profile} @ {sampleRate} Hz: non-finite output");
                    }
                    else if (meanAbsDifference <= 1e-4)
                    {
                        failures.Add($"{definition.Profile} @ {sampleRate} Hz: output identical to input");
                    }
                }
                catch (Exception ex) when (ex is InvalidOperationException or FileNotFoundException or DllNotFoundException)
                {
                    failures.Add($"{definition.Profile} @ {sampleRate} Hz: {ex.Message}");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Studio_voice_processes_real_audio_at_each_supported_rate()
    {
        (string root, string architecture) = ResolveRuntime();
        NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.StudioVoice);

        foreach (int sampleRate in definition.SupportedSampleRates)
        {
            float[] input = BuildNoisySpeechLikeSignal(sampleRate, seconds: 6);
            using NvidiaAfxSession session = NvidiaAfxSession.Create(
                definition,
                root,
                sampleRate,
                intensityRatio: 0f,
                architecture);
            float[] processed = session.Process(input);

            Assert.Equal(input.Length, processed.Length);
            Assert.All(processed, sample => Assert.True(float.IsFinite(sample)));
            Assert.True(
                input.Zip(processed, (source, enhanced) => Math.Abs(source - enhanced)).Average() > 1e-4,
                $"Studio Voice produced unchanged output at {sampleRate} Hz.");
        }
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Probe_agrees_with_a_full_run_for_early_access_profiles()
    {
        (string root, string architecture) = ResolveRuntime();
        var disagreements = new List<string>();

        foreach (NvidiaAfxProfileDefinition definition in NvidiaAfxProfileCatalog.Definitions
                     .Where(definition => definition.IsEarlyAccess))
        {
            foreach (int sampleRate in definition.SupportedSampleRates)
            {
                bool probeOk = NvidiaAfxSessionEffectProbe.Instance
                    .Probe(root, definition, sampleRate, architecture).Succeeded;

                bool fullRunOk;
                try
                {
                    using NvidiaAfxSession session = NvidiaAfxSession.Create(
                        definition, root, sampleRate, intensityRatio: 0f, architecture);
                    session.Process(BuildNoisySpeechLikeSignal(sampleRate, seconds: 2));
                    fullRunOk = true;
                }
                catch (Exception ex) when (ex is InvalidOperationException or FileNotFoundException or DllNotFoundException)
                {
                    fullRunOk = false;
                }

                output.WriteLine($"{definition.Profile} @ {sampleRate} Hz: probe={probeOk} fullRun={fullRunOk}");
                if (probeOk != fullRunOk)
                {
                    disagreements.Add($"{definition.Profile} @ {sampleRate} Hz: probe={probeOk} fullRun={fullRunOk}");
                }
            }
        }

        Assert.True(disagreements.Count == 0, string.Join(Environment.NewLine, disagreements));
    }

    [NvidiaAfxLiveRuntimeFact]
    public async Task Streamed_processing_matches_whole_buffer_processing_across_chunk_boundaries()
    {
        (string root, string architecture) = ResolveRuntime();
        NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.NoiseAndReverb);
        const int sampleRate = 48000;
        // 2.5 s is more than two one-second chunks and ends mid-frame, so padding and trimming are exercised.
        float[] input = BuildNoisySpeechLikeSignal(sampleRate, seconds: 3)[..((sampleRate * 5 / 2) - 7)];

        using NvidiaAfxSession whole = NvidiaAfxSession.Create(definition, root, sampleRate, 1.0f, architecture);
        float[] expected = whole.Process(input);

        using NvidiaAfxSession streamed = NvidiaAfxSession.Create(definition, root, sampleRate, 1.0f, architecture);
        var collected = new List<float>();
        long written = await streamed.ProcessStreamAsync(
            new ArrayAudioSamples(input, sampleRate),
            farEnd: null,
            (chunk, _) =>
            {
                collected.AddRange(chunk.ToArray());
                return ValueTask.CompletedTask;
            },
            CancellationToken.None);

        Assert.Equal(expected.Length, written);
        Assert.Equal(expected.Length, collected.Count);
        double maxDifference = expected.Zip(collected, (a, b) => Math.Abs(a - b)).Max();
        output.WriteLine($"whole vs streamed max |difference| = {maxDifference:G3}");
        Assert.True(maxDifference < 1e-3, $"Streamed output diverges from whole-buffer output by {maxDifference}.");
    }

    private sealed class ArrayAudioSamples(float[] data, int sampleRate) : IAudioSamples
    {
        public int SampleRate => sampleRate;

        public long SampleFrameCount => data.Length;

        public void ReadMonoSamples(long startFrame, Span<float> destination) =>
            data.AsSpan((int)startFrame, destination.Length).CopyTo(destination);

        public void Dispose()
        {
        }
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Telephony_upscale_changes_the_sample_rate_when_models_are_present()
    {
        (string root, string architecture) = ResolveRuntime();
        NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.TelephonyUpscale);
        const int inputRate = 8000;

        float[] input = BuildNoisySpeechLikeSignal(inputRate, seconds: 2);
        using NvidiaAfxSession session = NvidiaAfxSession.Create(
            definition,
            root,
            inputRate,
            intensityRatio: 1.0f,
            architecture);
        float[] processed = session.Process(input);

        output.WriteLine(
            $"telephony frame in/out {session.NumInputSamplesPerFrame}/{session.NumOutputSamplesPerFrame}, " +
            $"output rate {session.OutputSampleRate} Hz, samples in/out {input.Length}/{processed.Length}");

        Assert.Equal(16000, session.OutputSampleRate);
        Assert.Equal(input.Length * 2, processed.Length);
        Assert.All(processed, sample => Assert.True(float.IsFinite(sample)));
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Architecture_detector_finds_the_installed_nvidia_gpu()
    {
        IReadOnlyList<string> buckets = new NvidiaAfxArchitectureDetector().DetectArchitectureBuckets();
        output.WriteLine("candidate architectures: " + string.Join(", ", buckets));

        Assert.NotEmpty(buckets);
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Intensity_changes_the_output_for_every_profile_that_advertises_it()
    {
        (string root, string architecture) = ResolveRuntime();

        var ignored = new List<string>();
        foreach (NvidiaAfxProfileDefinition definition in ShippingDefinitions()
                     .Where(definition => definition.SupportsIntensityRatio))
        {
            int sampleRate = definition.SupportedSampleRates.Max();
            float[] input = BuildNoisySpeechLikeSignal(sampleRate, seconds: 2);

            float[] Run(float intensity)
            {
                using NvidiaAfxSession session = NvidiaAfxSession.Create(
                    definition,
                    root,
                    sampleRate,
                    intensity,
                    architecture);
                return session.Process(input);
            }

            float[] none = Run(0.0f);
            float[] full = Run(1.0f);
            double difference = none.Zip(full, (a, b) => Math.Abs(a - b)).Average();
            output.WriteLine($"{definition.Profile} @ {sampleRate} Hz: mean |intensity 0 - intensity 1| = {difference:F5}");
            if (difference <= 1e-3)
            {
                ignored.Add($"{definition.Profile}: output does not depend on intensity (difference {difference:F6})");
            }
        }

        Assert.True(ignored.Count == 0, string.Join(Environment.NewLine, ignored));
    }

    [NvidiaAfxLiveRuntimeFact]
    public void Readiness_service_reports_ready_for_every_selectable_profile()
    {
        string root = NvidiaAfxLiveRuntimeFactAttribute.RuntimeRoot;
        using var tempStore = new TempComponentStore();
        var service = CreateReadinessService(tempStore.Store, root);

        var notReady = new List<string>();
        foreach (NvidiaAfxProfileDefinition definition in ShippingDefinitions())
        {
            NvidiaAfxRuntimeReadiness readiness = service.GetReadiness(definition.Profile);
            output.WriteLine(
                $"{definition.Profile}: ready={readiness.IsReady} status='{readiness.StatusLabel}' " +
                $"arch={readiness.ArchitectureBucket} reason={readiness.FailureReason}");
            if (!readiness.IsReady)
            {
                notReady.Add($"{definition.Profile}: {readiness.StatusLabel} - {readiness.FailureReason}");
            }
        }

        Assert.True(notReady.Count == 0, string.Join(Environment.NewLine, notReady));
    }

    [NvidiaAfxLiveRuntimeFact]
    public async Task Enhancement_service_runs_afx_end_to_end_on_a_wav_file()
    {
        string root = NvidiaAfxLiveRuntimeFactAttribute.RuntimeRoot;
        using var tempStore = new TempComponentStore();
        var service = new NvidiaAfxSpeechAudioEnhancementService(
            CreateReadinessService(tempStore.Store, root),
            new FailingFallback());

        string sourcePath = Path.Join(tempStore.Directory, "source.wav");
        string destinationPath = Path.Join(tempStore.Directory, "enhanced.wav");
        const int sampleRate = 48000;
        WriteMonoPcm16Wav(sourcePath, BuildNoisySpeechLikeSignal(sampleRate, seconds: 2), sampleRate);

        SpeechAudioEnhancementResult result = await service.EnhanceAsync(
            new SpeechAudioEnhancementRequest(
                sourcePath,
                destinationPath,
                new SpeechAudioEnhancementOptions(true, NvidiaAfxProfile.NoiseAndReverb, 1.0f)),
            CancellationToken.None);

        output.WriteLine(
            $"backend={result.Backend} profile={result.BackendProfile} rate={result.SampleRate} " +
            $"frames={result.SampleFrames} seconds={result.DurationSeconds:F2}");

        Assert.Equal(SpeechAudioEnhancementBackend.NvidiaAfx, result.Backend);
        Assert.Equal("dereverb_denoiser", result.BackendProfile);
        Assert.Equal(sampleRate, result.SampleRate);
        Assert.True(File.Exists(destinationPath));
        Assert.True(new FileInfo(destinationPath).Length > 44, "Enhanced WAV has no audio payload.");
    }

    private static void WriteMonoPcm16Wav(string path, float[] samples, int sampleRate)
    {
        const short bitsPerSample = 16;
        int dataBytes = samples.Length * sizeof(short);
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8);
        writer.Write(36 + dataBytes);
        writer.Write("WAVEfmt "u8);
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * bitsPerSample / 8);
        writer.Write((short)(bitsPerSample / 8));
        writer.Write(bitsPerSample);
        writer.Write("data"u8);
        writer.Write(dataBytes);
        foreach (float sample in samples)
        {
            writer.Write((short)(Math.Clamp(sample, -1f, 1f) * short.MaxValue));
        }
    }

    private static NvidiaAfxRuntimeReadinessService CreateReadinessService(
        Trackdub.Infrastructure.Components.ComponentStore store,
        string runtimeRoot) =>
        new(
            store,
            new NvidiaAfxArchitectureDetector(),
            ResolveManifestPath(),
            settingsProvider: () => StudioSettings.Default with
            {
                NvidiaAfxRuntimeDirectory = runtimeRoot,
                NvidiaAfxLicenseAccepted = true,
            },
            allowEarlyAccess: () => true);

    private static string ResolveManifestPath()
    {
        string packaged = Path.Join(AppContext.BaseDirectory, "nvidiaafx-runtime.manifest.json");
        if (File.Exists(packaged))
        {
            return packaged;
        }

        for (DirectoryInfo? directory = new(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            string candidate = Path.Join(directory.FullName, "src", "Trackdub.Composition", "nvidiaafx-runtime.manifest.json");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException("nvidiaafx-runtime.manifest.json was not found next to the tests or in the repo.");
    }

    private sealed class TempComponentStore : IDisposable
    {
        public TempComponentStore()
        {
            Directory = Path.Join(Path.GetTempPath(), $"trackdub-afx-live-{Guid.NewGuid():N}");
            System.IO.Directory.CreateDirectory(Directory);
            Store = new Trackdub.Infrastructure.Components.ComponentStore(Directory, new SilentLogger());
        }

        public string Directory { get; }

        public Trackdub.Infrastructure.Components.ComponentStore Store { get; }

        public void Dispose()
        {
            if (System.IO.Directory.Exists(Directory))
            {
                System.IO.Directory.Delete(Directory, recursive: true);
            }
        }
    }

    private sealed class SilentLogger : IApplicationLogger
    {
        public void LogDebug(string message) { }
        public void LogInformation(string message) { }
        public void LogWarning(string message, Exception? exception = null) { }
        public void LogError(string message, Exception? exception = null) { }
    }

    private sealed class FailingFallback : ISpeechAudioEnhancementService
    {
        public Task<SpeechAudioEnhancementResult> EnhanceAsync(
            SpeechAudioEnhancementRequest request,
            CancellationToken cancellationToken) =>
            throw new InvalidOperationException(
                "The DeepFilterNet fallback ran, so AFX did not produce the output.");
    }

    private static float[] BuildNoisySpeechLikeSignal(int sampleRate, int seconds)
    {
        var random = new Random(1234);
        float[] signal = new float[sampleRate * seconds];
        for (int index = 0; index < signal.Length; index++)
        {
            double time = (double)index / sampleRate;
            double tone = (0.25 * Math.Sin(2 * Math.PI * 220 * time)) + (0.15 * Math.Sin(2 * Math.PI * 660 * time));
            double noise = 0.08 * ((random.NextDouble() * 2) - 1);
            signal[index] = (float)(tone + noise);
        }

        return signal;
    }
}
