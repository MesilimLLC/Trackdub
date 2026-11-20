using System.Text.Json;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.Qwen3Asr;
using Trackdub.Inference.Onnx.Whisper;

namespace Trackdub.Inference.Onnx.Tests;

// M10 golden/parity tests for the three mel implementations.
// Signals are deterministic and generated in code (sine/chirp/silence formulas below);
// the spot values in Fixtures/Mel/mel-goldens.json pin the current verified-correct
// implementation. Regenerate them only deliberately via the gated generator test:
//   GENERATE_MEL_GOLDENS=1 dotnet test --filter MelGoldenGenerator
// Analytic expectations (HTK mel centers, silence floors) are computed independently
// in this file and are not derived from the golden file.
//
// Shares a collection with NemotronAsrMelFeatureExtractorTests because the Nemotron tests
// here construct extractors, which bumps the static filterbank build counter that test asserts on.
[Collection("NemotronMelFilterBankBuildCount")]
public sealed class MelGoldenTests
{
    private const int SampleRate = 16_000;
    private const float Tolerance = 0.002f;

    // --- Analytic helpers (independent HTK/Slaney computation) ---
    private static double HzToMel(double f)
    {
        const double fSp = 200d / 3d;
        const double minLogMel = 1000d / fSp;
        double logStep = Math.Log(6.4) / 27d;
        return f >= 1000d ? minLogMel + (Math.Log(f / 1000d) / logStep) : f / fSp;
    }

    private static double MelToHz(double mel)
    {
        const double fSp = 200d / 3d;
        const double minLogMel = 1000d / fSp;
        double logStep = Math.Log(6.4) / 27d;
        return mel >= minLogMel ? 1000d * Math.Exp(logStep * (mel - minLogMel)) : mel * fSp;
    }

    private static int ExpectedPeakBin(double frequencyHz, int melBins)
    {
        double melMin = HzToMel(0), melMax = HzToMel(SampleRate / 2d);
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int i = 0; i < melBins; i++)
        {
            double center = MelToHz(melMin + ((melMax - melMin) * (i + 1) / (melBins + 1d)));
            double distance = Math.Abs(center - frequencyHz);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = i;
            }
        }

        return best;
    }

    private static float[] Sine(double frequencyHz, int samples = SampleRate, float amplitude = 0.5f)
    {
        var data = new float[samples];
        for (int i = 0; i < samples; i++)
        {
            data[i] = (float)(amplitude * Math.Sin(2 * Math.PI * frequencyHz * i / SampleRate));
        }

        return data;
    }

    private static float[] Chirp(int samples = SampleRate, float amplitude = 0.5f)
    {
        var data = new float[samples];
        double phase = 0;
        for (int i = 0; i < samples; i++)
        {
            double t = (double)i / SampleRate;
            double frequency = 100d + ((4000d - 100d) * t);
            phase += 2 * Math.PI * frequency / SampleRate;
            data[i] = (float)(amplitude * Math.Sin(phase));
        }

        return data;
    }

    private static int PeakBin(float[,] mel, int frame)
    {
        int best = 0;
        for (int m = 1; m < mel.GetLength(0); m++)
        {
            if (mel[m, frame] > mel[best, frame])
            {
                best = m;
            }
        }

        return best;
    }

    private static int PeakBin(ReadOnlySpan<float> flat, int melBins, int frames, int frame)
    {
        int best = 0;
        for (int m = 1; m < melBins; m++)
        {
            if (flat[(m * frames) + frame] > flat[(best * frames) + frame])
            {
                best = m;
            }
        }

        return best;
    }

    private static Dictionary<string, GoldenCase> LoadGoldens()
    {
        string path = Path.Join(AppContext.BaseDirectory, "Fixtures", "Mel", "mel-goldens.json");
        Assert.True(File.Exists(path), $"Missing golden fixture: {path}");
        return JsonSerializer.Deserialize<Dictionary<string, GoldenCase>>(File.ReadAllText(path))!;
    }

    private static void AssertGolden(
        Dictionary<string, GoldenCase> goldens,
        string key,
        int melBins,
        int frames,
        Func<int, int, float> valueAt,
        (int Mel, int Frame)[] peaks)
    {
        GoldenCase golden = goldens[key];
        Assert.Equal(golden.Frames, frames);
        foreach (GoldenSpot spot in golden.Spots)
        {
            Assert.Equal(spot.Value, valueAt(spot.Mel, spot.Frame), Tolerance);
        }

        foreach ((int mel, int frame) in peaks)
        {
            int actualPeak = 0;
            for (int candidate = 1; candidate < melBins; candidate++)
            {
                if (valueAt(candidate, frame) > valueAt(actualPeak, frame))
                {
                    actualPeak = candidate;
                }
            }

            Assert.True(
                actualPeak == mel,
                $"{key} frame {frame}: expected peak mel bin {mel}, actual argmax {actualPeak}");
        }
    }

    // --- Whisper (80 bins, fixed 3000 frames) ---

    [Fact]
    public void Whisper_Silence_IsAnalyticLogMelFloor()
    {
        var tensor = new WhisperFeatureExtractor(80).Extract(new float[SampleRate]);
        Assert.Equal([1, 80, 3000], tensor.Dimensions.ToArray());
        // Silence: clamp 1e-10 -> log10 = -10 = max, so (max+4)/4 = -1.5 everywhere.
        foreach (float value in tensor.Buffer.Span)
        {
            Assert.Equal(-1.5f, value, 1e-6f);
        }
    }

    [Fact]
    public void Whisper_SineAndChirp_MatchAnalyticPeakAndGoldens()
    {
        Dictionary<string, GoldenCase> goldens = LoadGoldens();
        var extractor = new WhisperFeatureExtractor(80);

        var sine = extractor.Extract(Sine(1000));
        float[] sineData = sine.Buffer.ToArray();
        Assert.Equal(26, ExpectedPeakBin(1000, 80)); // analytic HTK center 1005.6 Hz
        Assert.Equal(ExpectedPeakBin(1000, 80), PeakBin(sineData, 80, 3000, 50));
        AssertGolden(goldens, "whisper-sine1000", 80, 3000, (m, f) => sineData[(m * 3000) + f], [(ExpectedPeakBin(1000, 80), 50), (ExpectedPeakBin(1000, 80), 10)]);

        var chirp = extractor.Extract(Chirp());
        float[] chirpData = chirp.Buffer.ToArray();
        int early = PeakBin(chirpData, 80, 3000, 10);
        int late = PeakBin(chirpData, 80, 3000, 90);
        Assert.True(early < late, $"Chirp peak should rise: early {early}, late {late}");
        // Chirp is ~490 Hz at frame 10 and ~3610 Hz at frame 90 (100->4000 Hz over 1 s).
        AssertGolden(goldens, "whisper-chirp", 80, 3000, (m, f) => chirpData[(m * 3000) + f], [(ExpectedPeakBin(490, 80), 10), (ExpectedPeakBin(3610, 80), 90)]);
    }

    // --- Qwen (QwenMelSpectrogramMath via Qwen3Asr extractor, 128 bins) ---

    [Fact]
    public void Qwen_Silence_IsAnalyticLogMelFloor()
    {
        var tensor = new Qwen3AsrMelFeatureExtractor().Extract(new float[SampleRate]);
        Assert.Equal([1, 128, 100], tensor.Dimensions.ToArray());
        foreach (float value in tensor.Buffer.Span)
        {
            Assert.Equal(-1.5f, value, 1e-6f);
        }
    }

    [Fact]
    public void Qwen_FrameCountsAndReflectPadding()
    {
        var extractor = new Qwen3AsrMelFeatureExtractor();
        Assert.Equal(0, extractor.Extract(ReadOnlySpan<float>.Empty).Dimensions[2]);
        Assert.Equal(10, extractor.Extract(new float[1600]).Dimensions[2]);
        Assert.Equal(100, extractor.Extract(new float[SampleRate]).Dimensions[2]);

        float[] source = [1, 2, 3, 4];
        ReadOnlySpan<float> padded = QwenMelSpectrogramMath.ReflectPad(source, 2, out float[]? rented);
        try
        {
            Assert.Equal(new float[] { 3, 2, 1, 2, 3, 4, 3, 2 }, padded.ToArray());
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    [Fact]
    public void Qwen_MelFilterBankCentersMatchAnalyticHtk()
    {
        float[,] bank = QwenMelSpectrogramMath.BuildMelFilterBank(SampleRate, 400, 128);
        Assert.Equal(128, bank.GetLength(0));
        Assert.Equal(201, bank.GetLength(1));
        for (int mel = 0; mel < 128; mel++)
        {
            int peakBin = 0;
            for (int bin = 1; bin < 201; bin++)
            {
                if (bank[mel, bin] > bank[mel, peakBin])
                {
                    peakBin = bin;
                }
            }

            double peakHz = peakBin * (SampleRate / 2d) / 200d;
            double melMin = HzToMel(0), melMax = HzToMel(8000);
            double centerHz = MelToHz(melMin + ((melMax - melMin) * (mel + 1) / 129d));
            // Filter peak must sit within one FFT bin (40 Hz) of the analytic center.
            Assert.True(Math.Abs(peakHz - centerHz) <= 40d, $"mel {mel}: peak {peakHz} vs center {centerHz}");
        }
    }

    [Fact]
    public void Qwen_SineAndChirp_MatchAnalyticPeakAndGoldens()
    {
        Dictionary<string, GoldenCase> goldens = LoadGoldens();
        var extractor = new Qwen3AsrMelFeatureExtractor();

        var sine = extractor.Extract(Sine(1000));
        int frames = sine.Dimensions[2];
        float[] sineData = sine.Buffer.ToArray();
        Assert.Equal(ExpectedPeakBin(1000, 128), PeakBin(sineData, 128, frames, 50));
        AssertGolden(goldens, "qwen-sine1000", 128, frames, (m, f) => sineData[(m * frames) + f], [(ExpectedPeakBin(1000, 128), 50), (ExpectedPeakBin(1000, 128), 10)]);

        var chirp = extractor.Extract(Chirp());
        int chirpFrames = chirp.Dimensions[2];
        float[] chirpData = chirp.Buffer.ToArray();
        Assert.True(
            PeakBin(chirpData, 128, chirpFrames, 10) < PeakBin(chirpData, 128, chirpFrames, 90),
            "Chirp peak should rise");
        AssertGolden(goldens, "qwen-chirp", 128, chirpFrames, (m, f) => chirpData[(m * chirpFrames) + f], [(ExpectedPeakBin(490, 128), 10), (ExpectedPeakBin(3610, 128), 90)]);
    }

    // --- Nemotron (128 bins, pre-emphasis, natural log) ---

    [Fact]
    public void Nemotron_Silence_IsAnalyticLogGuardFloor()
    {
        float[,] mel = new NemotronAsrMelFeatureExtractor().Extract(new float[SampleRate]);
        Assert.Equal(128, mel.GetLength(0));
        Assert.Equal(100, mel.GetLength(1));
        // Silence: log(0 + 5.9604645e-8) with natural log.
        float expected = MathF.Log(5.9604645e-8f);
        Assert.Equal((float)Math.Log(5.9604645e-8), expected, 1e-5f);
        for (int m = 0; m < mel.GetLength(0); m++)
        {
            for (int f = 0; f < mel.GetLength(1); f++)
            {
                Assert.Equal(expected, mel[m, f], 1e-4f);
            }
        }
    }

    [Fact]
    public void Nemotron_FrameCounts()
    {
        var extractor = new NemotronAsrMelFeatureExtractor();
        Assert.Equal(0, extractor.Extract(ReadOnlySpan<float>.Empty).GetLength(1));
        Assert.Equal(0, extractor.Extract(new float[159]).GetLength(1));
        Assert.Equal(10, extractor.Extract(new float[1600]).GetLength(1));
        Assert.Equal(100, extractor.Extract(new float[SampleRate]).GetLength(1));
    }

    [Fact]
    public void Nemotron_SineAndChirp_MatchAnalyticPeakAndGoldens()
    {
        Dictionary<string, GoldenCase> goldens = LoadGoldens();
        var extractor = new NemotronAsrMelFeatureExtractor();

        float[,] sine = extractor.Extract(Sine(1000));
        Assert.Equal(ExpectedPeakBin(1000, 128), PeakBin(sine, 50));
        AssertGolden(goldens, "nemotron-sine1000", 128, sine.GetLength(1), (m, f) => sine[m, f], [(ExpectedPeakBin(1000, 128), 50), (ExpectedPeakBin(1000, 128), 10)]);

        float[,] chirp = extractor.Extract(Chirp());
        Assert.True(PeakBin(chirp, 10) < PeakBin(chirp, 90), "Chirp peak should rise");
        // Nemotron framing lands the early chirp argmax one bin below the analytic center.
        AssertGolden(goldens, "nemotron-chirp", 128, chirp.GetLength(1), (m, f) => chirp[m, f], [(19, 10), (95, 90)]);
    }

    // --- Gated golden generator (not a normal test) ---

    [Fact]
    public void MelGoldenGenerator_RegeneratesFixtureOnlyWhenRequested()
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable("GENERATE_MEL_GOLDENS"), "1", StringComparison.Ordinal))
        {
            return;
        }

        var goldens = new Dictionary<string, GoldenCase>();

        void AddWhisper(string key, float[] signal)
        {
            var tensor = new WhisperFeatureExtractor(80).Extract(signal);
            float[] data = tensor.Buffer.ToArray();
            goldens[key] = MakeGolden(3000, data,
                [(26, 50), (0, 50), (79, 50), (26, 10), (62, 90), (12, 10)]);
        }

        void AddQwen(string key, float[] signal)
        {
            var tensor = new Qwen3AsrMelFeatureExtractor().Extract(signal);
            int frames = tensor.Dimensions[2];
            float[] data = tensor.Buffer.ToArray();
            goldens[key] = MakeGolden(frames,
                (m, f) => data[(m * frames) + f],
                [(42, 50), (0, 50), (127, 50), (42, 10), (99, 90), (20, 10)]);
        }

        void AddNemotron(string key, float[] signal)
        {
            float[,] mel = new NemotronAsrMelFeatureExtractor().Extract(signal);
            goldens[key] = MakeGolden(mel.GetLength(1),
                (m, f) => mel[m, f],
                [(42, 50), (0, 50), (127, 50), (42, 10), (99, 90), (20, 10)]);
        }

        AddWhisper("whisper-sine1000", Sine(1000));
        AddWhisper("whisper-chirp", Chirp());
        AddQwen("qwen-sine1000", Sine(1000));
        AddQwen("qwen-chirp", Chirp());
        AddNemotron("nemotron-sine1000", Sine(1000));
        AddNemotron("nemotron-chirp", Chirp());

        string path = Path.Join(AppContext.BaseDirectory, "Fixtures", "Mel", "mel-goldens.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(goldens,
            new JsonSerializerOptions { WriteIndented = true }));
    }

    private static GoldenCase MakeGolden(
        int frames, float[] flat, (int Mel, int Frame)[] spots) =>
        MakeGolden(frames, (m, f) => flat[(m * frames) + f], spots);

    private static GoldenCase MakeGolden(int frames, Func<int, int, float> at, (int Mel, int Frame)[] spots) =>
        new(frames, spots.Select(s => new GoldenSpot(s.Mel, s.Frame, at(s.Mel, s.Frame))).ToArray());

    private sealed record GoldenCase(int Frames, GoldenSpot[] Spots);

    private sealed record GoldenSpot(int Mel, int Frame, float Value);
}
