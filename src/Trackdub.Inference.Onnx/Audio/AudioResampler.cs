using System;
using System.Numerics;

namespace Trackdub.Inference.Onnx.Audio;

public sealed class ResampledAudioReader : IAudioSamples
{
    private readonly IAudioSamples source;
    private readonly int targetSampleRate;
    private readonly double ratio;
    private readonly float[]? lowPassFilter;
    private readonly int lowPassHalfLength;

    public int SampleRate => targetSampleRate;
    public long SampleFrameCount { get; }

    internal ResampledAudioReader(IAudioSamples source, int targetSampleRate)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));
        this.targetSampleRate = targetSampleRate;
        if (targetSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSampleRate), "Sample rate must be positive.");
        }

        ratio = (double)source.SampleRate / targetSampleRate;
        // M-H5: downsampling without a low-pass aliases above-Nyquist content into the
        // passband. Pipeline-normalized audio is 48 kHz (FfmpegAudioExtractionService)
        // while most consumers target 16 kHz, so this path is live, not theoretical.
        if (ratio > 1d)
        {
            lowPassFilter = AudioResampler.BuildAntiAliasFilter(ratio);
            lowPassHalfLength = lowPassFilter.Length / 2;
        }

        SampleFrameCount = source.SampleFrameCount == 0
            ? 0
            : Math.Max(1, (long)Math.Round(source.SampleFrameCount / ratio));
    }

    public void ReadMonoSamples(long startFrame, Span<float> destination)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        // Match IAudioSamples contract used by MemoryMappedWaveAudioReader:
        // an out-of-range startFrame zero-fills the destination instead of throwing,
        // so callers can safely request reads at or past EOF as silence.
        if (startFrame < 0 || startFrame >= SampleFrameCount)
        {
            destination.Clear();
            return;
        }

        int framesToCompute = (int)Math.Min(destination.Length, SampleFrameCount - startFrame);

        // Find the source boundaries needed for this destination chunk
        double startSourcePosition = startFrame * ratio;
        long startSourceLeftIndex = (long)Math.Floor(startSourcePosition);

        double endSourcePosition = (startFrame + framesToCompute - 1) * ratio;
        long endSourceRightIndex = Math.Min((long)Math.Floor(endSourcePosition) + 1, source.SampleFrameCount - 1);

        int sourceFramesNeeded = (int)(endSourceRightIndex - startSourceLeftIndex + 1);

        // Rent a buffer to hold the source chunk (anti-alias filtered when downsampling)
        float[]? filteredBuffer = lowPassFilter is null
            ? null
            : AudioResampler.ReadFilteredRange(
                startSourceLeftIndex, sourceFramesNeeded, lowPassHalfLength, lowPassFilter,
                source.SampleFrameCount, source.ReadMonoSamples);
        float[]? sourceBuffer = filteredBuffer ?? System.Buffers.ArrayPool<float>.Shared.Rent(sourceFramesNeeded);
        try
        {
            Span<float> sourceSpan = sourceBuffer.AsSpan(0, sourceFramesNeeded);
            if (filteredBuffer is null)
            {
                source.ReadMonoSamples(startSourceLeftIndex, sourceSpan);
            }

            double sourcePosition = startSourcePosition;
            for (int i = 0; i < framesToCompute; i++, sourcePosition += ratio)
            {
                long globalLeftIndex = (long)Math.Floor(sourcePosition);
                long globalRightIndex = Math.Min(globalLeftIndex + 1, source.SampleFrameCount - 1);
                double fraction = sourcePosition - globalLeftIndex;

                int localLeftIndex = (int)(globalLeftIndex - startSourceLeftIndex);
                int localRightIndex = (int)(globalRightIndex - startSourceLeftIndex);

                float left = sourceSpan[localLeftIndex];
                float right = sourceSpan[localRightIndex];
                destination[i] = (float)(left + ((right - left) * fraction));
            }

            if (framesToCompute < destination.Length)
            {
                destination.Slice(framesToCompute).Clear();
            }
        }
        finally
        {
            if (filteredBuffer is null)
            {
                System.Buffers.ArrayPool<float>.Shared.Return(sourceBuffer);
            }
        }
    }

    public void Dispose()
    {
        source.Dispose();
    }
}

internal sealed class ResampledChannelAudioReader : IAudioChannelSamples
{
    private readonly IAudioChannelSamples source;
    private readonly int targetSampleRate;
    private readonly double ratio;
    private readonly float[]? lowPassFilter;
    private readonly int lowPassHalfLength;

    public int SampleRate => targetSampleRate;
    public int ChannelCount => source.ChannelCount;
    public long SampleFrameCount { get; }

    internal ResampledChannelAudioReader(IAudioChannelSamples source, int targetSampleRate)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));
        this.targetSampleRate = targetSampleRate;
        if (targetSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSampleRate), "Sample rate must be positive.");
        }

        ratio = (double)source.SampleRate / targetSampleRate;
        if (ratio > 1d)
        {
            lowPassFilter = AudioResampler.BuildAntiAliasFilter(ratio);
            lowPassHalfLength = lowPassFilter.Length / 2;
        }

        SampleFrameCount = source.SampleFrameCount == 0
            ? 0
            : Math.Max(1, (long)Math.Round(source.SampleFrameCount / ratio));
    }

    public void ReadMonoSamples(long startFrame, Span<float> destination)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        if (ChannelCount == 1)
        {
            ReadChannelSamples(startFrame, 0, destination);
            return;
        }

        float[] channelBuffer = System.Buffers.ArrayPool<float>.Shared.Rent(destination.Length);
        try
        {
            destination.Clear();
            Span<float> channelSpan = channelBuffer.AsSpan(0, destination.Length);
            for (int channel = 0; channel < ChannelCount; channel++)
            {
                channelSpan.Clear();
                ReadChannelSamples(startFrame, channel, channelSpan);
                for (int index = 0; index < destination.Length; index++)
                {
                    destination[index] += channelSpan[index] / ChannelCount;
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(channelBuffer);
        }
    }

    public void ReadChannelSamples(long startFrame, int channelIndex, Span<float> destination)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        if (channelIndex < 0 || channelIndex >= ChannelCount)
        {
            throw new ArgumentOutOfRangeException(nameof(channelIndex), "Channel index is outside the source channel range.");
        }

        if (startFrame < 0 || startFrame >= SampleFrameCount)
        {
            destination.Clear();
            return;
        }

        int framesToCompute = (int)Math.Min(destination.Length, SampleFrameCount - startFrame);
        double startSourcePosition = startFrame * ratio;
        long startSourceLeftIndex = (long)Math.Floor(startSourcePosition);
        double endSourcePosition = (startFrame + framesToCompute - 1) * ratio;
        long endSourceRightIndex = Math.Min((long)Math.Floor(endSourcePosition) + 1, source.SampleFrameCount - 1);
        int sourceFramesNeeded = (int)(endSourceRightIndex - startSourceLeftIndex + 1);

        float[]? filteredBuffer = lowPassFilter is null
            ? null
            : AudioResampler.ReadFilteredRange(
                startSourceLeftIndex, sourceFramesNeeded, lowPassHalfLength, lowPassFilter,
                source.SampleFrameCount,
                (start, dest) => source.ReadChannelSamples(start, channelIndex, dest));
        float[]? sourceBuffer = filteredBuffer ?? System.Buffers.ArrayPool<float>.Shared.Rent(sourceFramesNeeded);
        try
        {
            Span<float> sourceSpan = sourceBuffer.AsSpan(0, sourceFramesNeeded);
            if (filteredBuffer is null)
            {
                source.ReadChannelSamples(startSourceLeftIndex, channelIndex, sourceSpan);
            }

            double sourcePosition = startSourcePosition;
            for (int index = 0; index < framesToCompute; index++, sourcePosition += ratio)
            {
                long globalLeftIndex = (long)Math.Floor(sourcePosition);
                long globalRightIndex = Math.Min(globalLeftIndex + 1, source.SampleFrameCount - 1);
                double fraction = sourcePosition - globalLeftIndex;

                int localLeftIndex = (int)(globalLeftIndex - startSourceLeftIndex);
                int localRightIndex = (int)(globalRightIndex - startSourceLeftIndex);

                float left = sourceSpan[localLeftIndex];
                float right = sourceSpan[localRightIndex];
                destination[index] = (float)(left + ((right - left) * fraction));
            }

            if (framesToCompute < destination.Length)
            {
                destination.Slice(framesToCompute).Clear();
            }
        }
        finally
        {
            if (filteredBuffer is null)
            {
                System.Buffers.ArrayPool<float>.Shared.Return(sourceBuffer);
            }
        }
    }

    public void Dispose()
    {
        source.Dispose();
    }
}

internal static class AudioResampler
{
    public static float[] Resample(float[] input, int inputSampleRate, int outputSampleRate)
    {
        // Legacy array-based fallback (if still needed)
        ArgumentNullException.ThrowIfNull(input);
        if (inputSampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(inputSampleRate), "Sample rate must be positive.");
        if (outputSampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(outputSampleRate), "Sample rate must be positive.");
        if (input.Length == 0 || inputSampleRate == outputSampleRate) return input.ToArray();

        // M-H5 (live): pipeline-normalized audio is 48 kHz and most consumers target
        // 16 kHz, so downsampling must low-pass at the output Nyquist before decimation
        // or above-Nyquist content folds into the passband. Upsampling needs no filter.
        float[] sourceSamples = inputSampleRate > outputSampleRate
            ? ApplyFilter(input, BuildAntiAliasFilter((double)inputSampleRate / outputSampleRate))
            : input;

        int outputLength = Math.Max(1, (int)Math.Round(input.Length * (double)outputSampleRate / inputSampleRate));
        var output = new float[outputLength];
        double ratio = (double)inputSampleRate / outputSampleRate;

        double sourcePosition = 0;
        for (int index = 0; index < outputLength; index++, sourcePosition += ratio)
        {
            int leftIndex = (int)Math.Floor(sourcePosition);
            int rightIndex = Math.Min(leftIndex + 1, input.Length - 1);
            double fraction = sourcePosition - leftIndex;

            float left = sourceSamples[Math.Min(leftIndex, input.Length - 1)];
            float right = sourceSamples[rightIndex];
            output[index] = (float)(left + ((right - left) * fraction));
        }

        return output;
    }

    public static IAudioSamples CreateResampledStream(IAudioSamples source, int targetSampleRate)
    {
        ArgumentNullException.ThrowIfNull(source);

        // Ownership is transferred to the returned stream. It may be the original
        // source when rates already match, or a wrapper that disposes the source.
        if (source.SampleRate == targetSampleRate)
        {
            return source;
        }

        return new ResampledAudioReader(source, targetSampleRate);
    }

    public static IAudioChannelSamples CreateResampledChannelStream(IAudioChannelSamples source, int targetSampleRate)
    {
        ArgumentNullException.ThrowIfNull(source);

        if (source.SampleRate == targetSampleRate)
        {
            return source;
        }

        return new ResampledChannelAudioReader(source, targetSampleRate);
    }

    // Blackman-windowed sinc low-pass at the output Nyquist (cutoff 0.5/ratio cycles
    // per input sample), unity DC gain. Length scales with ratio so the transition
    // band stays a similar fraction of the output Nyquist for any downsample factor.
    internal static float[] BuildAntiAliasFilter(double ratio)
    {
        int halfLength = Math.Clamp((int)Math.Ceiling(16d * ratio), 16, 96);
        int length = (halfLength * 2) + 1;
        double cutoff = 0.5d / ratio;
        var filter = new float[length];
        double sum = 0;
        for (int index = 0; index < length; index++)
        {
            double offset = index - halfLength;
            double sinc = offset == 0
                ? 2d * cutoff
                : Math.Sin(2d * Math.PI * cutoff * offset) / (Math.PI * offset);
            double window = 0.42d
                - (0.5d * Math.Cos(2d * Math.PI * index / (length - 1)))
                + (0.08d * Math.Cos(4d * Math.PI * index / (length - 1)));
            double value = sinc * window;
            filter[index] = (float)value;
            sum += value;
        }

        for (int index = 0; index < length; index++)
        {
            filter[index] = (float)(filter[index] / sum);
        }

        return filter;
    }

    internal static float[] ApplyFilter(ReadOnlySpan<float> input, float[] filter)
    {
        int halfLength = filter.Length / 2;
        var output = new float[input.Length];
        // Output indices in [interiorStart, interiorEnd) read only in-range source
        // samples for every tap, so the per-tap clamp can be skipped and the window
        // is a contiguous slice. Edge indices keep the clamped loop.
        int interiorStart = Math.Min(halfLength, input.Length);
        int interiorEnd = Math.Clamp(input.Length - filter.Length + halfLength + 1, interiorStart, input.Length);
        for (int index = 0; index < interiorStart; index++)
        {
            output[index] = ClampedWindowDotProduct(filter, input, index - halfLength);
        }

        for (int index = interiorStart; index < interiorEnd; index++)
        {
            output[index] = DotProduct(filter, input.Slice(index - halfLength, filter.Length));
        }

        for (int index = interiorEnd; index < input.Length; index++)
        {
            output[index] = ClampedWindowDotProduct(filter, input, index - halfLength);
        }

        return output;
    }

    internal static float[] ReadFilteredRange(
        long startIndex,
        int count,
        int halfLength,
        float[] filter,
        long totalFrames,
        Action<long, Span<float>> read)
    {
        long expandedStart = Math.Max(0, startIndex - halfLength);
        long expandedEnd = Math.Min(totalFrames - 1, startIndex + count - 1 + halfLength);
        int expandedCount = (int)(expandedEnd - expandedStart + 1);
        float[] expanded = System.Buffers.ArrayPool<float>.Shared.Rent(expandedCount);
        try
        {
            Span<float> expandedSpan = expanded.AsSpan(0, expandedCount);
            expandedSpan.Clear();
            read(expandedStart, expandedSpan);
            var filtered = new float[count];
            // For output i the unclamped tap window is expandedSpan[(i - leftPad) .. +filter.Length).
            // leftPad/rightLimit mark the outputs whose full tap window stays inside the expanded
            // read; those take the contiguous dot product, boundary outputs keep the clamped loop.
            long leftPadLong = expandedStart - startIndex + halfLength;
            long rightLimitLong = expandedEnd - startIndex - halfLength;
            int leftPad = (int)Math.Clamp(leftPadLong, 0, count);
            int rightLimit = (int)Math.Clamp(rightLimitLong, -1, count - 1);
            for (int index = 0; index < leftPad; index++)
            {
                filtered[index] = ClampedWindowDotProduct(
                    filter, expandedSpan, (int)(startIndex + index - expandedStart - halfLength));
            }

            for (int index = leftPad; index <= rightLimit; index++)
            {
                filtered[index] = DotProduct(filter, expandedSpan.Slice(index - leftPad, filter.Length));
            }

            for (int index = Math.Max(leftPad, rightLimit + 1); index < count; index++)
            {
                filtered[index] = ClampedWindowDotProduct(
                    filter, expandedSpan, (int)(startIndex + index - expandedStart - halfLength));
            }

            return filtered;
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(expanded);
        }
    }

    private static float ClampedWindowDotProduct(float[] filter, ReadOnlySpan<float> samples, int firstSampleIndex)
    {
        double sum = 0;
        int lastIndex = samples.Length - 1;
        for (int tap = 0; tap < filter.Length; tap++)
        {
            sum += filter[tap] * samples[Math.Clamp(firstSampleIndex + tap, 0, lastIndex)];
        }

        return (float)sum;
    }

    private static float DotProduct(float[] filter, ReadOnlySpan<float> input) =>
        Vector.IsHardwareAccelerated
            ? VectorDotProduct(filter, input)
            : ScalarDotProduct(filter, input);

    private static float VectorDotProduct(float[] filter, ReadOnlySpan<float> input)
    {
        var acc = Vector<float>.Zero;
        int vectorLength = Vector<float>.Count;
        int length = filter.Length;
        int index = 0;
        for (; index + vectorLength <= length; index += vectorLength)
        {
            acc += new Vector<float>(filter, index) * new Vector<float>(input.Slice(index, vectorLength));
        }

        float sum = 0;
        for (int lane = 0; lane < vectorLength; lane++)
        {
            sum += acc[lane];
        }

        for (; index < length; index++)
        {
            sum += filter[index] * input[index];
        }

        return sum;
    }

    private static float ScalarDotProduct(float[] filter, ReadOnlySpan<float> input)
    {
        double sum = 0;
        for (int index = 0; index < filter.Length; index++)
        {
            sum += filter[index] * input[index];
        }

        return (float)sum;
    }
}
