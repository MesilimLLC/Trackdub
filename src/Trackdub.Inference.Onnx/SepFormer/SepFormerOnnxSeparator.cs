// Tensor names confirmed from scripts/export-sepformer-onnx.py output.
// Update if shapes differ after running the export script.
//
// SepFormer (sepformer.onnx):
//   input  "mix"      : [1, time]    float32
//   output "source_0" : [time]       float32  (speaker 1)
//   output "source_1" : [time]       float32  (speaker 2)
//
// OSD (osd.onnx) — pyannote/segmentation-3.0:
//   input  "waveform"     : [1, time]         float32
//   output "segmentation" : [1, frames, 7]    float32  (pyannote/segmentation-3.0)
//     class indices: 0=non-speech, 1=spk1, 2=spk2, 3=overlap (others unused)
//   At SR=16000, 10s window frame count is measured at runtime (~589 frames in export).
//   OsdSamplesPerFrame = SR * OsdWindowSeconds / OsdFrameCount — must be measured at runtime.

using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Globalization;

namespace Trackdub.Inference.Onnx.SepFormer;

internal sealed class SepFormerOnnxSeparator : ISepFormerSeparator
{
    private const int OsdWindowSamples = 16000 * 10;   // 10-second OSD window
    private const int SepChunkSamples = 16000 * 4;     // 4-second SepFormer chunks
    private const int OverlapSamples = 16000 / 2;      // 0.5-second crossfade overlap
    private const float OsdOverlapThreshold = 0.5f;
    private const int OsdOverlapClassIndex = 3;         // class 3 = overlapping speech
    private const string SepModelFileName = "sepformer.onnx";
    private const string OsdModelFileName = "osd.onnx";

    public async Task<SepFormerSeparation> SeparateAsync(
        SepFormerSeparatorRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        PooledProviderRequest providerRequest = PooledProviderRequest.ForPlan(request.Plan);
        string sepPath = Path.Join(request.ModelRootPath, SepModelFileName);
        string osdPath = Path.Join(request.ModelRootPath, OsdModelFileName);

        float[] samples = request.Samples;

        // 1. OSD pass — find overlap regions in sample space
        progress?.Report(new StemSeparationProgress(0, 3, 0d, 1d));
        (List<(int Start, int End)> overlapRegions, string osdSelectedProvider, string? osdBootstrapDetail) = await DetectOverlapRegionsAsync(
            samples, osdPath, providerRequest, cancellationToken).ConfigureAwait(false);

        // 2. Chunked SepFormer with OSD gate
        progress?.Report(new StemSeparationProgress(1, 3, 0.33d, 1d));
        (float[] source0, float[] source1, int chunkCount, _, string separatorSelectedProvider, string? separatorBootstrapDetail) = await RunChunkedSepFormerAsync(
            samples, overlapRegions, sepPath, providerRequest, cancellationToken).ConfigureAwait(false);

        progress?.Report(new StemSeparationProgress(3, 3, 1d, 1d));

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["runner"] = "onnx",
            ["engine_family"] = SepFormerOverlapRescueEngine.EngineFamilyName,
            ["selected_provider"] = ResolveSelectedProvider(osdSelectedProvider, separatorSelectedProvider),
            ["overlap_regions"] = overlapRegions.Count.ToString(CultureInfo.InvariantCulture)
        };
        AddBootstrapDetail(metadata, osdBootstrapDetail, separatorBootstrapDetail);

        return new SepFormerSeparation(source0, source1, request.SampleRate, chunkCount, false, metadata);
    }

    public async Task<SepFormerSeparation> SeparateRegionAsync(
        SepFormerRegionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        PooledProviderRequest providerRequest = PooledProviderRequest.ForPlan(request.Plan);
        string sepPath = Path.Join(request.ModelRootPath, SepModelFileName);
        float[] samples = request.Samples;
        var overlapRegions = new List<(int Start, int End)> { (0, samples.Length) };

        (float[] source0, float[] source1, int chunkCount, bool permutationWarning, string separatorSelectedProvider, string? separatorBootstrapDetail) =
            await RunChunkedSepFormerAsync(
                samples, overlapRegions, sepPath, providerRequest, cancellationToken)
                .ConfigureAwait(false);

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["runner"] = "onnx",
            ["engine_family"] = SepFormerOverlapRescueEngine.EngineFamilyName,
            ["selected_provider"] = separatorSelectedProvider,
            ["mode"] = "region",
            ["permutation_warning"] = permutationWarning.ToString(CultureInfo.InvariantCulture)
        };
        AddBootstrapDetail(metadata, separatorBootstrapDetail);

        return new SepFormerSeparation(
            source0,
            source1,
            request.SampleRate,
            chunkCount,
            permutationWarning,
            metadata);
    }

    private static async Task<(List<(int Start, int End)> Regions, string SelectedProvider, string? BootstrapDetail)> DetectOverlapRegionsAsync(
        float[] samples,
        string osdPath,
        PooledProviderRequest providerRequest,
        CancellationToken cancellationToken)
    {
        var overlapRegions = new List<(int Start, int End)>();
        int totalSamples = samples.Length;

        using OnnxExecutionSessionFactory.SingleSessionLease osdLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(providerRequest.ToSessionRequest("sepformer-osd", cancellationToken), osdPath)
            .ConfigureAwait(false);
        string selectedProvider = osdLease.SelectedProvider;
        string? bootstrapDetail = osdLease.BootstrapDetail;

        int windowStart = 0;
        while (windowStart < totalSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int windowEnd = Math.Min(windowStart + OsdWindowSamples, totalSamples);
            int windowLen = windowEnd - windowStart;

            // Pad to OsdWindowSamples if needed
            float[] windowSamples;
            if (windowLen < OsdWindowSamples)
            {
                windowSamples = new float[OsdWindowSamples];
                samples.AsSpan(windowStart, windowLen).CopyTo(windowSamples);
            }
            else
            {
                windowSamples = samples[windowStart..windowEnd];
            }

            var inputTensor = new DenseTensor<float>(windowSamples, [1, OsdWindowSamples]);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = osdLease.Session.RunWithRetry(
                [NamedOnnxValue.CreateFromTensor("waveform", inputTensor)],
                cancellationToken: cancellationToken);

            // segmentation: [1, frames, num_classes]
            Tensor<float> segTensor = outputs.First().AsTensor<float>();
            float[] segData = segTensor.ToArray();
            int numFrames = segTensor.Dimensions[1];
            int numClasses = segTensor.Dimensions[2];

            // Map frames to sample offsets within this window
            // samplesPerFrame = OsdWindowSamples / numFrames (approximate)
            float samplesPerFrame = (float)OsdWindowSamples / numFrames;

            bool inOverlap = false;
            int overlapStart = 0;

            for (int f = 0; f < numFrames; f++)
            {
                // batch=0; offset = f * numClasses + classIdx (first term of [1,F,C] flat index)
                float overlapProb = numClasses > OsdOverlapClassIndex
                    ? segData[(f * numClasses) + OsdOverlapClassIndex]
                    : 0f;

                int frameSampleStart = windowStart + (int)(f * samplesPerFrame);
                int frameSampleEnd = windowStart + (int)((f + 1) * samplesPerFrame);
                frameSampleEnd = Math.Min(frameSampleEnd, totalSamples);

                if (overlapProb >= OsdOverlapThreshold)
                {
                    if (!inOverlap)
                    {
                        overlapStart = frameSampleStart;
                        inOverlap = true;
                    }
                }
                else
                {
                    if (inOverlap)
                    {
                        overlapRegions.Add((overlapStart, frameSampleEnd));
                        inOverlap = false;
                    }
                }
            }

            if (inOverlap)
            {
                overlapRegions.Add((overlapStart, windowEnd));
            }

            windowStart += OsdWindowSamples;
        }

        return (MergeAdjacentRegions(overlapRegions), selectedProvider, bootstrapDetail);
    }

    private static List<(int Start, int End)> MergeAdjacentRegions(List<(int Start, int End)> regions)
    {
        if (regions.Count <= 1)
        {
            return regions;
        }

        var merged = new List<(int Start, int End)>();
        (int curStart, int curEnd) = regions[0];

        for (int i = 1; i < regions.Count; i++)
        {
            (int s, int e) = regions[i];
            if (s <= curEnd + OverlapSamples)
            {
                curEnd = Math.Max(curEnd, e);
            }
            else
            {
                merged.Add((curStart, curEnd));
                (curStart, curEnd) = (s, e);
            }
        }

        merged.Add((curStart, curEnd));
        return merged;
    }

    private static async Task<(float[] Source0, float[] Source1, int ChunkCount, bool PermutationWarning, string SelectedProvider, string? BootstrapDetail)> RunChunkedSepFormerAsync(
        float[] samples,
        List<(int Start, int End)> overlapRegions,
        string sepPath,
        PooledProviderRequest providerRequest,
        CancellationToken cancellationToken)
    {
        int totalSamples = samples.Length;
        float[] source0 = new float[totalSamples];
        float[] source1 = new float[totalSamples];
        float[] normWeights = new float[totalSamples];

        int chunkCount = 0;
        int chunkStart = 0;
        bool permutationWarning = false;
        float[]? previousSource0Tail = null;
        float[]? previousSource1Tail = null;

        // Linear crossfade ramp for overlap-add
        float[] fadeIn = BuildLinearRamp(OverlapSamples, rising: true);
        float[] fadeOut = BuildLinearRamp(OverlapSamples, rising: false);

        using OnnxExecutionSessionFactory.SingleSessionLease sepLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(providerRequest.ToSessionRequest("sepformer", cancellationToken), sepPath)
            .ConfigureAwait(false);
        string selectedProvider = sepLease.SelectedProvider;
        string? bootstrapDetail = sepLease.BootstrapDetail;

        while (chunkStart < totalSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int chunkEnd = Math.Min(chunkStart + SepChunkSamples, totalSamples);
            int chunkLen = chunkEnd - chunkStart;

            bool isOverlap = overlapRegions.Any(r => r.Start < chunkEnd && r.End > chunkStart);

            float[] s0chunk;
            float[] s1chunk;

            if (isOverlap)
            {
                // Pad to SepChunkSamples
                float[] chunkInput;
                if (chunkLen < SepChunkSamples)
                {
                    chunkInput = new float[SepChunkSamples];
                    samples.AsSpan(chunkStart, chunkLen).CopyTo(chunkInput);
                }
                else
                {
                    chunkInput = samples[chunkStart..chunkEnd];
                }

                var inputTensor = new DenseTensor<float>(chunkInput, [1, SepChunkSamples]);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = sepLease.Session.RunWithRetry(
                    [NamedOnnxValue.CreateFromTensor("mix", inputTensor)],
                cancellationToken: cancellationToken);

                float[] rawS0 = outputs.First(o => string.Equals(o.Name, "source_0", StringComparison.Ordinal))
                    .AsTensor<float>().ToArray();
                float[] rawS1 = outputs.First(o => string.Equals(o.Name, "source_1", StringComparison.Ordinal))
                    .AsTensor<float>().ToArray();

                // Trim to actual chunk length
                s0chunk = rawS0[..chunkLen];
                s1chunk = rawS1[..chunkLen];

                (s0chunk, s1chunk, bool ambiguous) = SourcePermutationCorrelator.AlignChunk(
                    s0chunk,
                    s1chunk,
                    previousSource0Tail,
                    previousSource1Tail,
                    permutationWarning);
                permutationWarning |= ambiguous;
                previousSource0Tail = SourcePermutationCorrelator.TakeTail(s0chunk);
                previousSource1Tail = SourcePermutationCorrelator.TakeTail(s1chunk);
            }
            else
            {
                // Pass-through: no overlapping speech in this chunk
                s0chunk = samples[chunkStart..chunkEnd];
                s1chunk = samples[chunkStart..chunkEnd];
            }

            // Overlap-add with linear crossfade weights
            for (int i = 0; i < chunkLen; i++)
            {
                int globalIdx = chunkStart + i;
                float w;

                if (i < OverlapSamples)
                {
                    w = fadeIn[i];
                }
                else if (i >= chunkLen - OverlapSamples)
                {
                    int fadeIdx = i - (chunkLen - OverlapSamples);
                    w = fadeIdx < OverlapSamples ? fadeOut[fadeIdx] : 0f;
                }
                else
                {
                    w = 1f;
                }

                source0[globalIdx] += s0chunk[i] * w;
                source1[globalIdx] += s1chunk[i] * w;
                normWeights[globalIdx] += w;
            }

            chunkCount++;
            chunkStart += SepChunkSamples - OverlapSamples;
        }

        // Normalize by accumulated weights
        for (int i = 0; i < totalSamples; i++)
        {
            if (normWeights[i] > 1e-6f)
            {
                source0[i] /= normWeights[i];
                source1[i] /= normWeights[i];
            }
        }

        return (source0, source1, chunkCount, permutationWarning, selectedProvider, bootstrapDetail);
    }

    private static float[] BuildLinearRamp(int length, bool rising)
    {
        float[] ramp = new float[length];
        for (int i = 0; i < length; i++)
        {
            ramp[i] = rising
                ? (float)i / length
                : (float)(length - i) / length;
        }

        return ramp;
    }

    private static string FormatProvider(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.Cpu => "cpu",
            ExecutionProviderKind.DirectMl => "dml",
            ExecutionProviderKind.Migraphx => "migraphx",
            ExecutionProviderKind.TensorRTRtx => "tensorrt-rtx",
            _ => provider.ToString().ToLowerInvariant()
        };

    private static string ResolveSelectedProvider(string first, string second) =>
        string.Equals(first, second, StringComparison.OrdinalIgnoreCase) ? first : "cpu";

    private static void AddBootstrapDetail(Dictionary<string, string> metadata, params string?[] details)
    {
        string[] nonEmptyDetails = details
            .Where(static detail => !string.IsNullOrWhiteSpace(detail))
            .Select(static detail => detail!)
            .ToArray();
        if (nonEmptyDetails.Length > 0)
        {
            metadata["bootstrap_detail"] = string.Join(" ", nonEmptyDetails);
        }
    }
}
