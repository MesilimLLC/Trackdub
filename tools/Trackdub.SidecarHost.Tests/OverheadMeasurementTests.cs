using System.Text.Json;
using Trackdub.SidecarHost;
using Xunit;

namespace Trackdub.SidecarHost.Tests;

public sealed class OverheadMeasurementTests
{
    [SidecarWorkerFact]
    public async Task WarmInfer_MeasuresAndWritesEvidence()
    {
        string repoRoot = FindRepoRoot();
        string workerDir = ChatterboxLiveWorkerTests.WorkerDir;
        Assert.True(Directory.Exists(ChatterboxLiveWorkerTests.ModelSnapshotDir),
            $"Chatterbox snapshot not staged at {ChatterboxLiveWorkerTests.ModelSnapshotDir}.");

        await using var worker = SupervisedSidecarWorker.Spawn(
            "uv", ["run", "--extra", "model", "python", "worker.py"],
            workingDirectory: workerDir);
        SidecarResponse load = worker.Request(new SidecarRequest("l1", "load",
            Plan: new SidecarLoadPlan(ChatterboxLiveWorkerTests.ModelSnapshotDir, ["CUDA", "CPU"], RequirePreferred: false)),
            timeout: SupervisedSidecarWorker.LoadTimeout);
        Assert.Equal("loaded", load.Status);

        (double median, double[] samples) = SidecarBenchmark.MeasureWarmInfer(worker);
        Assert.True(median > 0, "median must be positive");

        // Audio duration of the last measured response, for the evidence record.
        SidecarResponse last = worker.Request(new SidecarRequest("dur", "infer",
            Inputs: new() { ["text"] = SidecarTensor.Utf8("The quick brown fox jumps over the lazy dog.") }));
        Assert.Equal("ok", last.Status);
        Assert.NotNull(last.Outputs);
        double audioSeconds = last.Outputs!["audio"].Shape[0] / (double)(last.SampleRate ?? 24000);

        var evidence = new
        {
            host = Environment.MachineName,
            tfm = "net11.0",
            timestamp = DateTimeOffset.UtcNow,
            n = samples.Length,
            samplesMs = samples,
            medianMs = median,
            audioDurationSeconds = Math.Round(audioSeconds, 3),
        };
        string evidenceDir = Path.Join(repoRoot, "tools", "sidecar-evidence");
        Directory.CreateDirectory(evidenceDir);
        string path = Path.Join(evidenceDir, $"warm-infer-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}.json");
        await File.WriteAllTextAsync(
            path,
            JsonSerializer.Serialize(evidence, new JsonSerializerOptions { WriteIndented = true }));
        Assert.True(File.Exists(path), $"evidence file not written: {path}");
        await Task.CompletedTask;

        static string FindRepoRoot()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Join(dir.FullName, "Directory.Packages.props")))
            {
                dir = dir.Parent;
            }

            return dir?.FullName ?? throw new InvalidOperationException("repo root not found (Directory.Packages.props missing)");
        }
    }
}
