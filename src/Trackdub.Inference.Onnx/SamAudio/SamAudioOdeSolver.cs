namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>The explicit midpoint ODE solver SAM-Audio integrates its flow with, from t = 0 to t = 1.</summary>
internal static class SamAudioOdeSolver
{
    /// <summary>
    /// Integrates dx/dt = <paramref name="velocity"/>(t, x) over <paramref name="steps"/> equal steps:
    /// k1 = f(t, x); k2 = f(t + h/2, x + h/2 * k1); x += h * k2.
    /// </summary>
    public static float[] Midpoint(float[] initial, int steps, Func<float, float[], float[]> velocity)
    {
        ArgumentNullException.ThrowIfNull(initial);
        ArgumentNullException.ThrowIfNull(velocity);
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);

        float h = 1f / steps;
        float[] x = (float[])initial.Clone();
        for (int step = 0; step < steps; step++)
        {
            float t = step * h;
            float[] k1 = velocity(t, x);
            float[] midpoint = Advance(x, k1, h / 2f);
            float[] k2 = velocity(t + (h / 2f), midpoint);
            x = Advance(x, k2, h);
        }

        return x;
    }

    private static float[] Advance(float[] x, float[] slope, float stepSize)
    {
        if (slope.Length != x.Length)
        {
            throw new InvalidOperationException(
                $"Velocity length {slope.Length} does not match state length {x.Length}.");
        }

        float[] next = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            next[i] = x[i] + (stepSize * slope[i]);
        }

        return next;
    }
}
