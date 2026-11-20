namespace Trackdub.Contracts.Pipeline;

public sealed partial record StemSeparationResult
{
    /// <summary>
    /// Channel count of a stem whose layout differs from <see cref="ChannelCount"/>, keyed by stem name
    /// (vocals, ambiance, music, sfx). A stem not listed here has <see cref="ChannelCount"/> channels.
    /// </summary>
    public IReadOnlyDictionary<string, int>? StemChannelCounts { get; init; }

    public int ChannelCountOf(string stemName) =>
        StemChannelCounts is not null && StemChannelCounts.TryGetValue(stemName, out int channels) ? channels : ChannelCount;
}
