using Trackdub.Cli;
using Trackdub.Contracts.Jobs;
using System.Text.Json;
using Trackdub.Contracts.Pipeline;

namespace Trackdub.Sdk.Tests;

public sealed class CliMachineBoundaryTests
{
    [Fact]
    public void PocPresetModelsResolveInBundledInventory()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Join(root, "AGENTS.md"))) root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Join(root, "src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json")));
        var models = manifest.RootElement.GetProperty("models").EnumerateArray().ToArray();
        foreach (string alias in PocServerPreset.Create().Models!.Values)
            Assert.Contains(models, model => model.GetProperty("aliases").EnumerateArray().Any(a => a.GetString() == alias));
        Assert.Equal("quantized", PocServerPreset.Variants["translation"]);
        Assert.Equal("cpu", PocServerPreset.Providers["vad"]);
    }

    [Fact]
    public void QualityPresetModelsResolveInBundledInventory()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Join(root, "AGENTS.md"))) root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Join(root, "src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json")));
        var models = manifest.RootElement.GetProperty("models").EnumerateArray().ToArray();
        foreach (string alias in PocServerQualityPreset.Create().Models!.Values)
            Assert.Contains(models, model => model.GetProperty("aliases").EnumerateArray().Any(a => a.GetString() == alias));
        Assert.Equal("qwen3-tts-1.7b-customvoice", PocServerQualityPreset.Create().Models!["tts"]);
        Assert.Equal("quantized", PocServerQualityPreset.Variants["translation"]);
        Assert.Equal("cuda", PocServerQualityPreset.Providers["tts"]);
        Assert.True(PocPresets.TryResolve(PocServerQualityPreset.Name, out var qualityPreset, out var qualityProviders, out var qualityVariants));
        Assert.NotNull(qualityPreset);
        Assert.Equal("qwen3-tts-1.7b-customvoice", qualityPreset!.Models!["tts"]);
        Assert.Equal("cuda", qualityProviders!["tts"]);
        Assert.Equal("default", qualityVariants!["tts"]);
        Assert.True(PocPresets.TryResolve(PocServerPreset.Name, out var serverPreset, out var serverProviders, out var serverVariants));
        Assert.NotNull(serverPreset);
        Assert.Equal("kokoro", serverPreset!.Models!["tts"]);
        Assert.Equal("cpu", serverProviders!["vad"]);
        Assert.Equal("quantized", serverVariants!["translation"]);
        Assert.False(PocPresets.TryResolve("not-a-preset", out var unknownPreset, out var unknownProviders, out var unknownVariants));
        Assert.Null(unknownPreset);
        Assert.Null(unknownProviders);
        Assert.Null(unknownVariants);
    }
    [Fact]
    public void PipelineAcceptsDedicatedMachineFiles()
    {
        var result = Program.BuildRootCommand().Parse([
            "run", "pipeline", "--media", "input.mp4", "--target-language", "es",
            "--result-file", "result.json", "--progress-file", "progress.jsonl"]);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task AtomicResultCannotOverwritePriorAttemptEvidence()
    {
        string directory = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        string path = Path.Join(directory, "result.json");
        try
        {
            var outcome = new WorkerRunOutcome(1, "failed", null, "PIPELINE_FAILED", "Failed.", []);
            await CliMachineResultWriter.WriteAsync(outcome, path, CancellationToken.None);
            Assert.Equal(outcome.Status, JsonSerializer.Deserialize<WorkerRunOutcome>(await File.ReadAllTextAsync(path), WorkerRunOutcome.JsonOptions)!.Status);
            await Assert.ThrowsAsync<IOException>(() => CliMachineResultWriter.WriteAsync(outcome, path, CancellationToken.None));
            Assert.Single(Directory.GetFiles(directory));
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task ArtifactOutsideProjectIsRejected()
    {
        await Assert.ThrowsAsync<IOException>(() => CliMachineResultWriter.DescribeAsync(
            Path.Join(Path.GetTempPath(), "project"), "../outside.mp4", "video", CancellationToken.None));
    }

    [Fact]
    public void ConcurrentProgressRemainsDedicatedValidJsonLines()
    {
        string path = Path.Join(Path.GetTempPath(), Guid.NewGuid() + ".jsonl");
        try
        {
            using (var sink = new CliProgressFileSink(path, null))
            {
                Parallel.For(0, 100, i => sink.Report(new PipelineProgressEvent("ASR", PipelineProgressEventKind.Progress,
                    PercentComplete: null, RunId: Guid.Empty, SequenceNumber: i)));
            }
            var lines = File.ReadAllLines(path);
            Assert.Equal(100, lines.Length);
            Assert.All(lines, line =>
            {
                using var parsed = JsonDocument.Parse(line);
                Assert.Equal(JsonValueKind.Null, parsed.RootElement.GetProperty("percentComplete").ValueKind);
            });
        }
        finally { File.Delete(path); }
    }
}
