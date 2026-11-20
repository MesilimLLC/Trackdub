namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Seeded random inputs for a separation, so a rerun of the same job gives the same stems.</summary>
internal sealed class SamAudioRandom(int seed)
{
    private readonly Random random = new(seed);

    /// <summary>Standard-normal samples (Box-Muller), the flow's starting noise.</summary>
    public float[] Gaussian(int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            values[i] = (float)(radius * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < count)
            {
                values[i + 1] = (float)(radius * Math.Sin(2.0 * Math.PI * u2));
            }
        }

        return values;
    }

    /// <summary>
    /// The 16-bit watermark message DACVAE draws at random for each decoded stem (target and residual),
    /// as 0/1 floats shaped [2, 16]. Meta's output watermark is kept, not stripped.
    /// </summary>
    public float[] WatermarkMessage()
    {
        float[] bits = new float[2 * SamAudioModelConstants.WatermarkBits];
        for (int i = 0; i < bits.Length; i++)
        {
            bits[i] = random.Next(2);
        }

        return bits;
    }
}
