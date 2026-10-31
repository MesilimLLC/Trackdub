using Trackdub.Application.Editing;
using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;
using Trackdub.Domain.Media;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class SequenceExportDeliveryServiceTests
{
    [Fact]
    public async Task Selected_export_keeps_source_range_and_publishes_only_after_probe_and_tier_check()
    {
        using var scenario = new Scenario(
            new FakeMediaProbe { Snapshot = OutputSnapshot(durationSeconds: 4.5d) },
            new RecordingTierGate(maxDurationSeconds: 60, requiresWatermark: true));

        SequenceExportDeliveryResult result = await scenario.ExportSelectedAsync();

        AssertPublishedResult(result, scenario.Plan, TimeSpan.FromSeconds(4.5d));
        AssertRenderedFileContents(scenario.Plan);
        AssertSelectedPlacement(scenario.Renderer, scenario.SelectedPlacementId);
        AssertScopedRenderOptions(scenario.Renderer, scenario.Plan);
        AssertDeliveryChecks(scenario.TierGate!, scenario.Probe);
    }

    [Fact]
    public async Task Planned_duration_block_skips_rendering()
    {
        using var scenario = new Scenario(
            tierGate: new RecordingTierGate(maxDurationSeconds: 5, requiresWatermark: true));

        SequenceExportDeliveryResult result = await scenario.ExportAsync();

        AssertDurationBlocked(result);
        AssertNoRenderOccurred(scenario.Renderer, scenario.Probe);
        Assert.False(File.Exists(scenario.Plan.OutputPath));
        Assert.Equal([9.5d], scenario.TierGate!.CheckedDurations);
    }

    [Fact]
    public async Task Rendered_duration_block_preserves_previous_output_and_cleans_temporary_file()
    {
        using var scenario = new Scenario(
            new FakeMediaProbe { Snapshot = OutputSnapshot(durationSeconds: 301d) },
            new RecordingTierGate(maxDurationSeconds: 60, requiresWatermark: true));
        await File.WriteAllTextAsync(scenario.Plan.OutputPath, "previous", scenario.CancellationToken);

        SequenceExportDeliveryResult result = await scenario.ExportAsync();

        AssertDurationBlocked(result);
        AssertPriorOutputPreservedAndTemporaryFileRemoved(scenario.Plan, scenario.Files);
        Assert.Equal([9.5d, 301d], scenario.TierGate!.CheckedDurations);
        Assert.Single(scenario.Probe.Calls);
    }

    [Fact]
    public async Task Render_failure_preserves_previous_output_and_cleans_temporary_file()
    {
        using var scenario = new Scenario(
            renderFailure: new InvalidOperationException("render failed"));
        await File.WriteAllTextAsync(scenario.Plan.OutputPath, "previous", scenario.CancellationToken);

        await Assert.ThrowsAsync<InvalidOperationException>(() => scenario.ExportAsync());

        Assert.Equal("previous", await File.ReadAllTextAsync(scenario.Plan.OutputPath, scenario.CancellationToken));
        Assert.Empty(Directory.EnumerateFiles(scenario.Files.Root, "*.partial.mp4"));
    }

    [Fact]
    public async Task Output_cannot_replace_an_unselected_sequence_source()
    {
        using var scenario = new Scenario();
        string unselectedSourcePath = scenario.Plan.PlacementAssets[scenario.Plan.Sequence.Placements[1].Id].SourceMediaPath;

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            scenario.ExportSelectedAsync(scenario.Plan with { OutputPath = unselectedSourcePath }));

        Assert.Empty(scenario.Renderer.Plans);
    }

    [Fact]
    public async Task Output_extension_must_match_selected_container()
    {
        using var scenario = new Scenario();

        await Assert.ThrowsAsync<ArgumentException>(() =>
            scenario.ExportAsync(scenario.Plan with { OutputPath = Path.Join(scenario.Files.Root, "sequence.mkv") }));

        Assert.Empty(scenario.Renderer.Plans);
    }

    [Fact]
    public async Task Selected_export_still_requires_a_complete_asset_map_for_safe_output_validation()
    {
        using var scenario = new Scenario();
        Guid unselectedPlacementId = scenario.Plan.Sequence.Placements[0].Id;

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            scenario.ExportSelectedAsync(
                scenario.Plan with
                {
                    PlacementAssets = new Dictionary<Guid, SequenceRenderPlacementAssets>
                    {
                        [scenario.SelectedPlacementId] = scenario.Plan.PlacementAssets[scenario.SelectedPlacementId]
                    }
                }));

        Assert.True(scenario.Plan.PlacementAssets.ContainsKey(unselectedPlacementId));
        Assert.Empty(scenario.Renderer.Plans);
    }

    [Fact]
    public async Task Selected_export_rejects_burn_in_subtitles_until_they_are_rebased()
    {
        using var scenario = new Scenario();

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            scenario.ExportSelectedAsync(
                scenario.Plan with { BurnInSubtitlePath = scenario.Files.Create("sequence.srt") }));

        Assert.Empty(scenario.Renderer.Plans);
    }

    [Fact]
    public async Task Rendered_output_shorter_than_planned_sequence_is_blocked()
    {
        using var scenario = new Scenario(
            new FakeMediaProbe { Snapshot = OutputSnapshot(durationSeconds: 4.5d) },
            new RecordingTierGate(maxDurationSeconds: 60, requiresWatermark: false));

        SequenceExportDeliveryResult result = await scenario.ExportAsync();

        AssertDurationBlocked(result);
        Assert.False(File.Exists(scenario.Plan.OutputPath));
        Assert.Empty(Directory.EnumerateFiles(scenario.Files.Root, "*.partial.mp4"));
    }

    [Theory]
    [InlineData(60, 9.46, true)]
    [InlineData(60, 9.48, false)]
    [InlineData(10, 9.31, false)]
    [InlineData(10, 9.29, true)]
    public async Task Rendered_duration_uses_two_frame_tolerance(int frameRate, double duration, bool blocked)
    {
        using var scenario = new Scenario(new FakeMediaProbe { Snapshot = OutputSnapshot(duration) });
        SequenceExportDeliveryResult result = await scenario.ExportAsync(scenario.Plan with { OutputFrameRate = frameRate });
        Assert.Equal(blocked, result.IsBlocked);
    }

    private static SequenceExportPlan CreatePlan(TemporaryFiles files, out Guid selectedPlacementId)
    {
        Guid projectId = Guid.NewGuid();
        var firstSource = new SequenceSource(Guid.NewGuid(), projectId, Guid.NewGuid(), null, 20d, 1920, 1080);
        var secondSource = new SequenceSource(Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid(), "sequence-sources/second", 40d, 1280, 720);
        var firstPlacement = new SequencePlacement(Guid.NewGuid(), firstSource.Id, 0d, 5d, new SequenceCrop());
        selectedPlacementId = Guid.NewGuid();
        var selectedPlacement = new SequencePlacement(selectedPlacementId, secondSource.Id, 12d, 16.5d, new SequenceCrop());
        SequenceDocument sequence = SequenceDocument.Create(projectId);
        sequence = SequenceEditor.AddSource(sequence, firstSource);
        sequence = SequenceEditor.AddSource(sequence, secondSource);
        sequence = SequenceEditor.AddPlacement(sequence, firstPlacement);
        sequence = SequenceEditor.AddPlacement(sequence, selectedPlacement);

        string firstVideo = files.Create("first.mp4");
        string secondVideo = files.Create("second.mp4");
        string firstAudio = files.Create("first.wav");
        string secondAudio = files.Create("second.wav");
        return new SequenceExportPlan(
            sequence,
            new Dictionary<Guid, SequenceRenderPlacementAssets>
            {
                [firstPlacement.Id] = new(firstVideo, firstAudio),
                [selectedPlacement.Id] = new(secondVideo, secondAudio)
            },
            Path.Join(files.Root, "sequence.mp4"),
            ExportOutputContainer.Mp4,
            SequenceExportAudioMode.Dubbed);
    }

    private static void AssertPublishedResult(
        SequenceExportDeliveryResult result,
        SequenceExportPlan plan,
        TimeSpan expectedDuration)
    {
        Assert.False(result.IsBlocked);
        Assert.Equal(plan.OutputPath, result.OutputPath);
        Assert.Equal(expectedDuration.TotalSeconds, result.DurationSeconds);
    }

    private static void AssertRenderedFileContents(SequenceExportPlan plan) =>
        Assert.Equal("rendered", File.ReadAllText(plan.OutputPath));

    private static void AssertSelectedPlacement(RecordingSequenceRenderer renderer, Guid expectedPlacementId)
    {
        SequencePlacement placement = Assert.Single(Assert.Single(renderer.Plans).Sequence.Placements);
        Assert.Equal(expectedPlacementId, placement.Id);
        Assert.Equal(12d, placement.SourceInSeconds);
        Assert.Equal(16.5d, placement.SourceOutSeconds);
    }

    private static void AssertScopedRenderOptions(RecordingSequenceRenderer renderer, SequenceExportPlan plan)
    {
        SequenceExportPlan renderedPlan = Assert.Single(renderer.Plans);
        Assert.Single(renderedPlan.PlacementAssets);
        Assert.NotEqual(plan.OutputPath, renderedPlan.OutputPath);
        Assert.True(renderedPlan.RequiresWatermark);
    }

    private static void AssertDeliveryChecks(RecordingTierGate tierGate, FakeMediaProbe probe)
    {
        Assert.Equal([4.5d, 4.5d], tierGate.CheckedDurations);
        Assert.Single(probe.Calls);
    }

    private static void AssertDurationBlocked(SequenceExportDeliveryResult result)
    {
        Assert.True(result.IsBlocked);
        Assert.Contains("duration", result.BlockedReason, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertNoRenderOccurred(RecordingSequenceRenderer renderer, FakeMediaProbe probe)
    {
        Assert.Empty(renderer.Plans);
        Assert.Empty(probe.Calls);
    }

    private static void AssertPriorOutputPreservedAndTemporaryFileRemoved(SequenceExportPlan plan, TemporaryFiles files)
    {
        Assert.Equal("previous", File.ReadAllText(plan.OutputPath));
        Assert.Empty(Directory.EnumerateFiles(files.Root, "*.partial.mp4"));
    }

    private static MediaProbeSnapshot OutputSnapshot(double durationSeconds) => new(
        "mp4",
        "MP4",
        durationSeconds,
        BitRate: null,
        AudioStreams: [],
        VideoStreams: [new MediaVideoStream(0, "h264", 1920, 1080, 30, durationSeconds)]);

    private sealed class Scenario : IDisposable
    {
        public Scenario(
            FakeMediaProbe? probe = null,
            RecordingTierGate? tierGate = null,
            Exception? renderFailure = null)
        {
            Files = new TemporaryFiles();
            Plan = CreatePlan(Files, out Guid selectedPlacementId);
            SelectedPlacementId = selectedPlacementId;
            Renderer = new RecordingSequenceRenderer { Failure = renderFailure };
            Probe = probe ?? new FakeMediaProbe();
            TierGate = tierGate;
            Service = tierGate is null
                ? new SequenceExportDeliveryService(Renderer, Probe)
                : new SequenceExportDeliveryService(Renderer, Probe, tierGate);
        }

        public TemporaryFiles Files { get; }

        public SequenceExportPlan Plan { get; }

        public Guid SelectedPlacementId { get; }

        public RecordingSequenceRenderer Renderer { get; }

        public FakeMediaProbe Probe { get; }

        public RecordingTierGate? TierGate { get; }

        public SequenceExportDeliveryService Service { get; }

        public CancellationToken CancellationToken { get; } = TestContext.Current.CancellationToken;

        public Task<SequenceExportDeliveryResult> ExportAsync(SequenceExportPlan? plan = null) =>
            Service.ExportAsync(plan ?? Plan, cancellationToken: CancellationToken);

        public Task<SequenceExportDeliveryResult> ExportSelectedAsync(SequenceExportPlan? plan = null) =>
            Service.ExportAsync(plan ?? Plan, SelectedPlacementId, CancellationToken);

        public void Dispose() => Files.Dispose();
    }

    private sealed class RecordingSequenceRenderer : ISequenceExportRenderer
    {
        public List<SequenceExportPlan> Plans { get; } = [];

        public Exception? Failure { get; init; }

        public async Task<SequenceExportRenderResult> RenderAsync(
            SequenceExportPlan plan,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Plans.Add(plan);
            await File.WriteAllTextAsync(plan.OutputPath, "rendered", cancellationToken);
            if (Failure is not null)
            {
                throw Failure;
            }

            return new SequenceExportRenderResult(plan.OutputPath, ["test warning"]);
        }
    }

    private sealed class RecordingTierGate(double maxDurationSeconds, bool requiresWatermark) : IExportTierGate
    {
        public bool RequiresWatermark { get; } = requiresWatermark;

        public List<double> CheckedDurations { get; } = [];

        public string? CheckDurationGate(TimeSpan sourceDuration)
        {
            CheckedDurations.Add(sourceDuration.TotalSeconds);
            return sourceDuration.TotalSeconds > maxDurationSeconds
                ? $"Export duration exceeds {maxDurationSeconds:0} seconds."
                : null;
        }
    }

    private sealed class TemporaryFiles : IDisposable
    {
        public string Root { get; } = Directory.CreateTempSubdirectory("trackdub-sequence-delivery-").FullName;

        public string Create(string name)
        {
            string path = Path.Join(Root, name);
            File.WriteAllText(path, "test input");
            return path;
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
