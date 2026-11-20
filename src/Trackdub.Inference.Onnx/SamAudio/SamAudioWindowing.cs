namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>How a signal of <see cref="Length"/> samples is cut into windows of <see cref="Window"/> that overlap by <see cref="Overlap"/>.</summary>
internal readonly record struct WindowLayout(int Length, int Window, int Overlap)
{
    /// <summary>Start sample of each window; the last window ends exactly at the end of the signal.</summary>
    public int[] Starts()
    {
        if (Window <= Overlap)
        {
            throw new ArgumentOutOfRangeException(nameof(Overlap), "Overlap must be smaller than the window.");
        }

        if (Length <= Window)
        {
            return [0];
        }

        List<int> starts = [];
        for (int start = 0; start < Length - Window; start += Window - Overlap)
        {
            starts.Add(start);
        }

        starts.Add(Length - Window);
        return [.. starts];
    }
}

/// <summary>
/// Overlap-adds separated windows (one channel each) with linear ramps on every overlapping edge, then divides by the
/// summed ramp so the join is gain-neutral.
/// </summary>
internal sealed class SamAudioCrossfade(WindowLayout layout)
{
    private const float MinimumWeight = 1e-6f;

    private readonly float[] output = new float[layout.Length];
    private readonly float[] weight = new float[layout.Length];

    public static float[] Join(IReadOnlyList<float[]> pieces, WindowLayout layout)
    {
        ArgumentNullException.ThrowIfNull(pieces);
        int[] starts = layout.Starts();
        if (pieces.Count != starts.Length)
        {
            throw new ArgumentException($"Expected {starts.Length} pieces for the layout but got {pieces.Count}.", nameof(pieces));
        }

        SamAudioCrossfade crossfade = new(layout);
        for (int k = 0; k < pieces.Count; k++)
        {
            crossfade.Add(pieces[k], starts[k], k, pieces.Count);
        }

        return crossfade.Normalized();
    }

    /// <summary>Adds window <paramref name="index"/> of <paramref name="count"/>, so windows can be accumulated as they are produced.</summary>
    public void Add(float[] piece, int start, int index, int count)
    {
        float[] ramp = Ramp(piece.Length, new Edges(HasPrevious: index > 0, HasNext: index < count - 1));
        for (int i = 0; i < piece.Length; i++)
        {
            output[start + i] += piece[i] * ramp[i];
            weight[start + i] += ramp[i];
        }
    }

    public float[] Normalized()
    {
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= Math.Max(weight[i], MinimumWeight);
        }

        return output;
    }

    private float[] Ramp(int width, Edges edges)
    {
        float[] ramp = new float[width];
        Array.Fill(ramp, 1f);
        int fade = Math.Min(layout.Overlap, width);
        for (int i = 0; i < fade; i++)
        {
            ramp[i] = edges.HasPrevious ? Position(i, fade) : 1f;
            int tail = width - fade + i;
            ramp[tail] = edges.HasNext ? Math.Min(ramp[tail], 1f - Position(i, fade)) : ramp[tail];
        }

        return ramp;
    }

    /// <summary>Element <paramref name="index"/> of <c>linspace(0, 1, count)</c>, endpoints included.</summary>
    private static float Position(int index, int count) => count <= 1 ? 0f : (float)index / (count - 1);

    private readonly record struct Edges(bool HasPrevious, bool HasNext);
}
