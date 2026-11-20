using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

public sealed class SamAudioPlanningTests
{
    private static readonly string ManifestPath = Path.Join(
        FindRepoRoot(), "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

    [Fact]
    public void Automatic_separation_ranking_never_selects_sam_audio_and_keeps_spleeter_first()
    {
        RankedManifestEntry[] ranked = Rank(new StageRuntimePlanningRequest(RuntimeStage.Separation));

        Assert.DoesNotContain(ranked, candidate => IsSamAudio(candidate));
        Assert.Equal("spleeter", ranked[0].Entry.EngineFamily);
    }

    [Theory]
    [InlineData("sam-audio")]
    [InlineData("sam-audio-small-onnx")]
    public void An_explicit_sam_audio_alias_reaches_the_sam_audio_entry(string alias)
    {
        RankedManifestEntry[] ranked = Rank(new StageRuntimePlanningRequest(RuntimeStage.Separation, PreferredModelAlias: alias));

        Assert.Contains(ranked, candidate => IsSamAudio(candidate));
    }

    private static RankedManifestEntry[] Rank(StageRuntimePlanningRequest request) =>
        new RuntimePlannerRankingStrategy(BundledModelManifestRegistry.Load(ManifestPath))
            .RankEntries(request, StageRuntimeRequirementsCatalog.All[RuntimeStage.Separation]);

    private static bool IsSamAudio(RankedManifestEntry candidate) =>
        candidate.Entry.EngineFamily.Equals("sam-audio", StringComparison.OrdinalIgnoreCase);

    private static string FindRepoRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "Trackdub.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName ?? throw new DirectoryNotFoundException("Repository root (Trackdub.slnx) not found.");
    }
}
