using System.Runtime.InteropServices;
using Trackdub.Contracts;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Infrastructure.Components.NvidiaAfx;

namespace Trackdub.Composition.NvidiaAfx;

internal sealed class NvidiaAfxSession : IDisposable
{
    // Most AFX effects process 10 ms frames. Studio Voice High Quality uses 6-second frames.
    private const int FramesPerSecond = 100;

    private readonly NvidiaAfxEffectHandle _handle;
    private readonly string _selector;
    private readonly uint _numInputChannels;
    private readonly uint _numInputSamplesPerFrame;
    private readonly uint _numOutputSamplesPerFrame;
    private readonly int _outputSampleRate;
    private readonly bool _requiresFarEndReference;

    private NvidiaAfxSession(
        NvidiaAfxEffectHandle handle,
        string selector,
        uint numInputChannels,
        uint numInputSamplesPerFrame,
        uint numOutputSamplesPerFrame,
        int outputSampleRate,
        bool requiresFarEndReference)
    {
        _handle = handle;
        _selector = selector;
        _numInputChannels = numInputChannels;
        _numInputSamplesPerFrame = numInputSamplesPerFrame;
        _numOutputSamplesPerFrame = numOutputSamplesPerFrame;
        _outputSampleRate = outputSampleRate;
        _requiresFarEndReference = requiresFarEndReference;
    }

    public uint NumInputSamplesPerFrame => _numInputSamplesPerFrame;

    public uint NumOutputSamplesPerFrame => _numOutputSamplesPerFrame;

    /// <summary>
    /// Native or profile-declared output sample rate. May differ from the input rate for
    /// rate-changing chained effects such as telephony upscale.
    /// </summary>
    public int OutputSampleRate => _outputSampleRate;

    public bool RequiresFarEndReference => _requiresFarEndReference;

    public static NvidiaAfxSession Create(
        NvidiaAfxProfileDefinition profile,
        string runtimeRoot,
        int sampleRate,
        float intensityRatio,
        string? architectureBucket = null)
    {
        NvidiaAfxNativeLoader.EnsureLoaded(runtimeRoot);
        IntPtr effectHandle;
        int status;
        if (profile.IsChainedEffect)
        {
            status = NvidiaAfxNative.NvAFX_CreateChainedEffect(profile.Selector, out effectHandle);
        }
        else
        {
            status = NvidiaAfxNative.NvAFX_CreateEffect(profile.Selector, out effectHandle);
        }

        EnsureSuccess(status, profile.Selector, "CreateEffect");
        var safeHandle = new NvidiaAfxEffectHandle(effectHandle);
        try
        {
            NvidiaAfxRequiredModel[] requiredModels = profile.ResolveRequiredModels(sampleRate);
            if (requiredModels.Length > 0)
            {
                string[] modelPaths = requiredModels
                    .Select(model =>
                    {
                        string? resolved = NvidiaAfxRuntimeLayout.ResolveModelFile(
                            runtimeRoot,
                            model.FeatureFolder,
                            model.ModelStem,
                            architectureBucket);
                        if (resolved is null)
                        {
                            throw new FileNotFoundException(
                                $"NVIDIA AFX model '{model.ModelStem}' was not found under feature " +
                                $"'{model.FeatureFolder}' in runtime root '{runtimeRoot}' " +
                                $"(architecture bucket '{architectureBucket ?? "any"}', " +
                                $"input rate {sampleRate} Hz).");
                        }

                        return resolved;
                    })
                    .ToArray();
                if (modelPaths.Length == 1)
                {
                    EnsureSuccess(
                        NvidiaAfxNative.NvAFX_SetString(
                            safeHandle.DangerousGetHandle(),
                            NvidiaAfxNativeParameters.ModelPath,
                            modelPaths[0]),
                        profile.Selector,
                        "Set model path");
                }
                else
                {
                    EnsureSuccess(
                        NvidiaAfxNative.NvAFX_SetStringList(
                            safeHandle.DangerousGetHandle(),
                            NvidiaAfxNativeParameters.ModelPath,
                            modelPaths,
                            (uint)modelPaths.Length),
                        profile.Selector,
                        "Set model path list");
                }
            }

            // A chained selector encodes its sample rates (superres8kto16k_...), and SDK 2.x rejects
            // input_sample_rate on it with NVAFX_STATUS_INVALID_PARAM.
            if (!profile.IsChainedEffect)
            {
                EnsureSuccess(
                    NvidiaAfxNative.NvAFX_SetU32(
                        safeHandle.DangerousGetHandle(),
                        NvidiaAfxNativeParameters.InputSampleRate,
                        (uint)sampleRate),
                    profile.Selector,
                    "Set input sample rate");
            }

            int expectedOutputSampleRate = profile.ResolveOutputSampleRate(sampleRate);
            // Maxine docs: NVAFX_PARAM_OUTPUT_SAMPLE_RATE is Windows-only and not supported by
            // chained effects. Only set (and require success) for non-chained rate changes.
            if (!profile.IsChainedEffect && expectedOutputSampleRate != sampleRate)
            {
                EnsureSuccess(
                    NvidiaAfxNative.NvAFX_SetU32(
                        safeHandle.DangerousGetHandle(),
                        NvidiaAfxNativeParameters.OutputSampleRate,
                        (uint)expectedOutputSampleRate),
                    profile.Selector,
                    "Set output sample rate");
            }

            if (profile.SupportsIntensityRatio)
            {
                EnsureSuccess(
                    NvidiaAfxNative.NvAFX_SetFloat(
                        safeHandle.DangerousGetHandle(),
                        NvidiaAfxNativeParameters.IntensityRatio,
                        intensityRatio),
                    profile.Selector,
                    "Set intensity ratio");
            }

            EnsureSuccess(
                NvidiaAfxNative.NvAFX_Load(safeHandle.DangerousGetHandle()),
                profile.Selector,
                "Load");

            uint expectedInputChannels = profile.RequiresFarEndReference ? 2u : 1u;
            uint numInputChannels = QueryU32OrDefault(
                safeHandle,
                NvidiaAfxNativeParameters.NumInputChannels,
                expectedInputChannels);
            if (numInputChannels != expectedInputChannels)
            {
                throw new InvalidOperationException(
                    $"NVIDIA AFX profile '{profile.Selector}' reported {numInputChannels} input channel(s); " +
                    $"expected exactly {expectedInputChannels}" +
                    (profile.RequiresFarEndReference ? " (near-end + far-end)." : "."));
            }

            uint numInputSamples = QueryU32OrDefault(
                safeHandle,
                NvidiaAfxNativeParameters.NumInputSamplesPerFrame,
                fallbackParameter: NvidiaAfxNativeParameters.SamplesPerFrameLegacy,
                defaultValue: checked((uint)((long)sampleRate * profile.InputFrameDurationMilliseconds / 1000)));
            int outputSampleRate = QueryOutputSampleRate(
                safeHandle,
                expectedOutputSampleRate);
            uint numOutputSamples = ResolveOutputSamplesPerFrame(
                safeHandle,
                profile,
                sampleRate,
                outputSampleRate,
                numInputSamples);

            return new NvidiaAfxSession(
                safeHandle,
                profile.Selector,
                numInputChannels,
                numInputSamples,
                numOutputSamples,
                outputSampleRate,
                profile.RequiresFarEndReference);
        }
        catch
        {
            safeHandle.Dispose();
            throw;
        }
    }

    public float[] Process(float[] nearEnd, float[]? farEnd = null)
    {
        ArgumentNullException.ThrowIfNull(nearEnd);

        if (_requiresFarEndReference)
        {
            ArgumentNullException.ThrowIfNull(farEnd);
        }

        int inputFrame = checked((int)_numInputSamplesPerFrame);
        int outputFrame = checked((int)_numOutputSamplesPerFrame);
        if (inputFrame <= 0 || outputFrame <= 0)
        {
            throw new InvalidOperationException("Invalid AFX frame size.");
        }

        // Always process the full near-end stream. When far-end is shorter, zero-pad it rather
        // than truncating speech; when far-end is longer, ignore the excess.
        int nearSampleCount = nearEnd.Length;
        int alignedInput = ((nearSampleCount + inputFrame - 1) / inputFrame) * inputFrame;
        float[] nearAligned = new float[alignedInput];
        Array.Copy(nearEnd, nearAligned, nearSampleCount);
        float[]? farAligned = null;
        if (_requiresFarEndReference)
        {
            farAligned = new float[alignedInput];
            int farCopy = Math.Min(farEnd!.Length, nearSampleCount);
            Array.Copy(farEnd, farAligned, farCopy);
        }

        int outputLength = (alignedInput / inputFrame) * outputFrame;
        float[] output = new float[outputLength];
        float[] nearFrame = new float[inputFrame];
        float[] farFrame = _requiresFarEndReference ? new float[inputFrame] : [];
        float[] outFrame = new float[outputFrame];

        for (int inputOffset = 0, outputOffset = 0;
             inputOffset < alignedInput;
             inputOffset += inputFrame, outputOffset += outputFrame)
        {
            Array.Copy(nearAligned, inputOffset, nearFrame, 0, inputFrame);
            if (_requiresFarEndReference)
            {
                Array.Copy(farAligned!, inputOffset, farFrame, 0, inputFrame);
            }

            RunFrame(nearFrame, farFrame, outFrame);
            Array.Copy(outFrame, 0, output, outputOffset, outputFrame);
        }

        // Trim frame-alignment padding to source duration × (outputFrame/inputFrame).
        // Same-rate effects reduce to nearSampleCount; rate-changing effects keep the ratio.
        int expectedOutputSamples = ComputeTrimmedOutputLength(nearSampleCount, inputFrame, outputFrame);
        if (expectedOutputSamples < output.Length)
        {
            float[] trimmed = new float[expectedOutputSamples];
            Array.Copy(output, trimmed, expectedOutputSamples);
            return trimmed;
        }

        return output;
    }

    /// <summary>
    /// Streams <paramref name="nearEnd"/> through the effect in one-second chunks so long recordings
    /// never need full-length input and output arrays. Output is identical to <see cref="Process"/>:
    /// the last frame is zero-padded and the output trimmed to <see cref="GetOutputSampleCount"/>.
    /// Returns the number of samples handed to <paramref name="sink"/>.
    /// </summary>
    public async Task<long> ProcessStreamAsync(
        IAudioSamples nearEnd,
        IAudioSamples? farEnd,
        Func<ReadOnlyMemory<float>, CancellationToken, ValueTask> sink,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(nearEnd);
        ArgumentNullException.ThrowIfNull(sink);
        if (_requiresFarEndReference)
        {
            ArgumentNullException.ThrowIfNull(farEnd);
        }

        int inputFrame = checked((int)_numInputSamplesPerFrame);
        int outputFrame = checked((int)_numOutputSamplesPerFrame);
        if (inputFrame <= 0 || outputFrame <= 0)
        {
            throw new InvalidOperationException("Invalid AFX frame size.");
        }

        int framesPerChunk = Math.Clamp(nearEnd.SampleRate / inputFrame, 1, FramesPerSecond);
        long nearTotal = nearEnd.SampleFrameCount;
        long expected = GetOutputSampleCount(nearTotal, inputFrame, outputFrame);
        float[] nearChunk = new float[inputFrame * framesPerChunk];
        float[] farChunk = _requiresFarEndReference ? new float[inputFrame * framesPerChunk] : [];
        float[] outChunk = new float[outputFrame * framesPerChunk];
        float[] nearFrame = new float[inputFrame];
        float[] farFrame = _requiresFarEndReference ? new float[inputFrame] : [];
        float[] outFrame = new float[outputFrame];

        long position = 0;
        long written = 0;
        while (position < nearTotal)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = (int)Math.Min(nearChunk.Length, nearTotal - position);
            Array.Clear(nearChunk);
            nearEnd.ReadMonoSamples(position, nearChunk.AsSpan(0, count));
            if (_requiresFarEndReference)
            {
                // A shorter far-end is zero-padded, a longer one ignored, as in Process.
                Array.Clear(farChunk);
                IAudioSamples far = farEnd
                    ?? throw new ArgumentNullException(nameof(farEnd), "This effect requires a far-end reference.");
                int farAvailable = (int)Math.Clamp(far.SampleFrameCount - position, 0, count);
                if (farAvailable > 0)
                {
                    far.ReadMonoSamples(position, farChunk.AsSpan(0, farAvailable));
                }
            }

            int frames = (count + inputFrame - 1) / inputFrame;
            for (int frame = 0; frame < frames; frame++)
            {
                Array.Copy(nearChunk, frame * inputFrame, nearFrame, 0, inputFrame);
                if (_requiresFarEndReference)
                {
                    Array.Copy(farChunk, frame * inputFrame, farFrame, 0, inputFrame);
                }

                RunFrame(nearFrame, farFrame, outFrame);
                Array.Copy(outFrame, 0, outChunk, frame * outputFrame, outputFrame);
            }

            int outCount = (int)Math.Min((long)frames * outputFrame, expected - written);
            await sink(outChunk.AsMemory(0, outCount), cancellationToken).ConfigureAwait(false);
            written += outCount;
            position += count;
        }

        return written;
    }

    /// <summary>Output samples for a stream of <paramref name="nearSampleCount"/> input samples.</summary>
    public long GetOutputSampleCount(long nearSampleCount) =>
        GetOutputSampleCount(nearSampleCount, checked((int)_numInputSamplesPerFrame), checked((int)_numOutputSamplesPerFrame));

    private static long GetOutputSampleCount(long nearSampleCount, int inputFrame, int outputFrame) =>
        checked(nearSampleCount * outputFrame / inputFrame);

    /// <summary>
    /// Output sample count after dropping frame-alignment padding, preserving the
    /// input→output frame ratio (source duration × rate ratio for Maxine chained effects).
    /// </summary>
    internal static int ComputeTrimmedOutputLength(int nearSampleCount, int inputFrame, int outputFrame)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(nearSampleCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputFrame);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputFrame);

        return checked((int)(((long)nearSampleCount * outputFrame) / inputFrame));
    }

    /// <summary>
    /// Aligns AEC far-end audio to the near-end length by zero-padding a shorter far-end
    /// (or truncating a longer one). Exposed for unit tests.
    /// </summary>
    internal static float[] AlignFarEndToNearEnd(float[] nearEnd, float[] farEnd)
    {
        ArgumentNullException.ThrowIfNull(nearEnd);
        ArgumentNullException.ThrowIfNull(farEnd);

        if (farEnd.Length == nearEnd.Length)
        {
            return farEnd;
        }

        float[] aligned = new float[nearEnd.Length];
        Array.Copy(farEnd, aligned, Math.Min(farEnd.Length, nearEnd.Length));
        return aligned;
    }

    public void Reset()
    {
        EnsureSuccess(
            NvidiaAfxNative.NvAFX_Reset(_handle.DangerousGetHandle()),
            _selector,
            "Reset");
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void RunFrame(float[] nearFrame, float[] farFrame, float[] outFrame)
    {
        // Marshal exactly the validated channel count (1 non-AEC / 2 AEC). Never pass a larger
        // native-reported channel count than the number of pinned input pointers.
        uint marshalledChannels = _requiresFarEndReference ? 2u : 1u;
        if (_numInputChannels != marshalledChannels)
        {
            throw new InvalidOperationException(
                $"NVIDIA AFX RunFrame channel mismatch for '{_selector}': session has {_numInputChannels}, " +
                $"marshalled {marshalledChannels}.");
        }

        GCHandle nearHandle = GCHandle.Alloc(nearFrame, GCHandleType.Pinned);
        GCHandle farHandle = default;
        GCHandle outHandle = GCHandle.Alloc(outFrame, GCHandleType.Pinned);
        try
        {
            IntPtr[] inputs;
            if (_requiresFarEndReference)
            {
                farHandle = GCHandle.Alloc(farFrame, GCHandleType.Pinned);
                inputs = [nearHandle.AddrOfPinnedObject(), farHandle.AddrOfPinnedObject()];
            }
            else
            {
                inputs = [nearHandle.AddrOfPinnedObject()];
            }

            if (inputs.Length != marshalledChannels)
            {
                throw new InvalidOperationException(
                    $"NVIDIA AFX RunFrame marshalled {inputs.Length} pointer(s) but expected {marshalledChannels}.");
            }

            IntPtr[] outputs = [outHandle.AddrOfPinnedObject()];
            EnsureSuccess(
                NvidiaAfxNative.NvAFX_Run(
                    _handle.DangerousGetHandle(),
                    inputs,
                    outputs,
                    _numInputSamplesPerFrame,
                    marshalledChannels),
                _selector,
                "Run");
        }
        finally
        {
            if (farHandle.IsAllocated)
            {
                farHandle.Free();
            }

            nearHandle.Free();
            outHandle.Free();
        }
    }

    private static int QueryOutputSampleRate(NvidiaAfxEffectHandle handle, int fallback)
    {
        if (TryQueryU32(handle, NvidiaAfxNativeParameters.OutputSampleRate, out uint value) && value != 0)
        {
            return checked((int)value);
        }

        return fallback;
    }

    private static uint ResolveOutputSamplesPerFrame(
        NvidiaAfxEffectHandle handle,
        NvidiaAfxProfileDefinition profile,
        int inputSampleRate,
        int outputSampleRate,
        uint numInputSamples)
    {
        bool rateChanging = outputSampleRate != inputSampleRate || profile.IsChainedEffect;
        if (TryQueryU32(handle, NvidiaAfxNativeParameters.NumOutputSamplesPerFrame, out uint numOutputSamples)
            && numOutputSamples != 0)
        {
            if (rateChanging && numInputSamples > 0)
            {
                ValidateOutputFrameRatio(
                    profile.Selector,
                    inputSampleRate,
                    outputSampleRate,
                    numInputSamples,
                    numOutputSamples);
            }

            return numOutputSamples;
        }

        if (rateChanging)
        {
            // SDK 2.x chained handles answer every Get* with NVAFX_STATUS_FAILED, so the output
            // frame cannot be queried. Derive it from the fixed 10 ms framing and the declared rate.
            if (profile.IsChainedEffect)
            {
                uint derived = checked((uint)(outputSampleRate / FramesPerSecond));
                ValidateOutputFrameRatio(
                    profile.Selector,
                    inputSampleRate,
                    outputSampleRate,
                    numInputSamples,
                    derived);
                return derived;
            }

            throw new InvalidOperationException(
                $"NVIDIA AFX rate-changing profile '{profile.Selector}' did not report " +
                $"{NvidiaAfxNativeParameters.NumOutputSamplesPerFrame}; refusing to fall back to the input frame size.");
        }

        return QueryU32OrDefault(
            handle,
            NvidiaAfxNativeParameters.NumOutputSamplesPerFrame,
            fallbackParameter: NvidiaAfxNativeParameters.SamplesPerFrameLegacy,
            defaultValue: numInputSamples);
    }

    /// <summary>
    /// Ensures native output-frame size roughly matches the sample-rate ratio for rate-changing effects.
    /// </summary>
    internal static void ValidateOutputFrameRatio(
        string selector,
        int inputSampleRate,
        int outputSampleRate,
        uint numInputSamples,
        uint numOutputSamples)
    {
        if (inputSampleRate <= 0 || outputSampleRate <= 0 || numInputSamples == 0 || numOutputSamples == 0)
        {
            throw new InvalidOperationException(
                $"NVIDIA AFX profile '{selector}' reported invalid frame/sample-rate metadata.");
        }

        if (inputSampleRate == outputSampleRate)
        {
            return;
        }

        // expectedOut ≈ numInput * outRate / inRate (integer ratio for Maxine chained effects).
        long expected = ((long)numInputSamples * outputSampleRate) / inputSampleRate;
        if (expected <= 0 || Math.Abs(expected - numOutputSamples) > 1)
        {
            throw new InvalidOperationException(
                $"NVIDIA AFX profile '{selector}' output frame size {numOutputSamples} does not match " +
                $"input frame {numInputSamples} at {inputSampleRate}→{outputSampleRate} Hz (expected ~{expected}).");
        }
    }

    private static bool TryQueryU32(NvidiaAfxEffectHandle handle, string parameter, out uint value)
    {
        int status = NvidiaAfxNative.NvAFX_GetU32(handle.DangerousGetHandle(), parameter, out value);
        return status == 0;
    }

    private static uint QueryU32OrDefault(
        NvidiaAfxEffectHandle handle,
        string parameter,
        uint defaultValue,
        string? fallbackParameter = null)
    {
        if (TryQueryU32(handle, parameter, out uint value) && value != 0)
        {
            return value;
        }

        if (fallbackParameter is not null
            && TryQueryU32(handle, fallbackParameter, out value)
            && value != 0)
        {
            return value;
        }

        return defaultValue;
    }

    private static void EnsureSuccess(int status, string selector, string operation)
    {
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"NVIDIA AFX operation failed. Selector='{selector}', Operation='{operation}', Status={status}.");
        }
    }
}
