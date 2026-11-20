using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

/// <summary>
/// SAM-Audio is the preferred separator where the hardware meets its declared minimum VRAM; Spleeter is the fallback
/// (owner decision). An explicit alias is honoured regardless of hardware.
/// </summary>
public sealed class SamAudioPlanningTests
{
    private static readonly string ManifestPath = Path.Join(
        FindRepoRoot(), "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

    private static readonly StageRuntimePlanningRequest Automatic = new(RuntimeStage.Separation);

    private static readonly StageRuntimeRequirements Requirements = StageRuntimeRequirementsCatalog.All[RuntimeStage.Separation];

    [Fact]
    public void Automatic_separation_ranking_puts_sam_audio_first_and_keeps_spleeter_as_the_fallback()
    {
        RankedManifestEntry[] ranked = Rank(Automatic);

        Assert.Equal("sam-audio", ranked[0].Entry.EngineFamily);
        Assert.Contains(ranked, candidate => candidate.Entry.EngineFamily == "spleeter");
    }

    [Theory]
    [InlineData(true, 12288, true, "sam-audio")]
    [InlineData(true, 8192, true, "sam-audio")]
    [InlineData(true, 6144, true, "spleeter")]
    [InlineData(false, 0, false, "spleeter")]
    [InlineData(true, 16384, false, "spleeter")]
    public void Hardware_that_cannot_run_sam_audio_falls_back_to_spleeter(bool hasGpu, long vramMb, bool nvidia, string expectedFamily)
    {
        RankedManifestEntry[] ranked = RuntimePlannerRankingStrategy.FilterEntriesTheHardwareCanRun(
            Automatic, Requirements, Hardware(hasGpu, vramMb, nvidia), Rank(Automatic));

        Assert.Equal(expectedFamily, ranked[0].Entry.EngineFamily);
    }

    [Theory]
    [InlineData("sam-audio")]
    [InlineData("sam-audio-small-onnx")]
    public void An_explicit_sam_audio_alias_is_honoured_even_on_weak_hardware(string alias)
    {
        StageRuntimePlanningRequest request = new(RuntimeStage.Separation, PreferredModelAlias: alias);
        RankedManifestEntry[] ranked = Rank(request);

        RankedManifestEntry[] kept = RuntimePlannerRankingStrategy.FilterEntriesTheHardwareCanRun(request, Requirements, Hardware(false, 0, nvidia: false), ranked);

        Assert.Equal("sam-audio", kept[0].Entry.EngineFamily);
    }

    [Fact]
    public void Sam_audio_is_planned_only_on_the_providers_it_was_verified_on()
    {
        // CUDA and CPU only: TensorRT RTX and the other providers are untested for its attention-heavy flow graph.
        Assert.Equal(
            [ExecutionProviderKind.Cuda, ExecutionProviderKind.Cpu],
            Requirements.AllowedProvidersByEngineFamily!["sam-audio"]);
    }

    [Fact]
    public void The_ranking_is_kept_when_no_entry_would_meet_the_hardware()
    {
        RankedManifestEntry[] onlySam = [.. Rank(Automatic).Where(candidate => candidate.Entry.EngineFamily == "sam-audio")];

        RankedManifestEntry[] kept = RuntimePlannerRankingStrategy.FilterEntriesTheHardwareCanRun(Automatic, Requirements, Hardware(false, 0, nvidia: false), onlySam);

        Assert.Equal(onlySam, kept);
    }

    private static HardwareProfile Hardware(bool hasGpu, long vramMb, bool nvidia) =>
        new("Windows", "x64", hasGpu, DedicatedVramMb: vramMb,
            NvidiaGpuArchitecture: nvidia ? NvidiaGpuArchitectureBucket.Ada : NvidiaGpuArchitectureBucket.Unknown);

    private static RankedManifestEntry[] Rank(StageRuntimePlanningRequest request) =>
        new RuntimePlannerRankingStrategy(BundledModelManifestRegistry.Load(ManifestPath))
            .RankEntries(request, Requirements);

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
