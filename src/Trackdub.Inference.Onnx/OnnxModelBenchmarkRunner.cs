using System.Diagnostics;
using System.Text.Json;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Domain.Benchmarking;
using Trackdub.Inference;
using Trackdub.Inference.Onnx.Benchmarking;
using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.NativeCudaTensorRt;
using Trackdub.Inference.Runtime.TensorRtRtx;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx;

public sealed class OnnxModelBenchmarkRunner : IModelBenchmarkRunner
{
    private readonly ITensorRtRtxProviderBootstrap _tensorRtRtxProviderBootstrap;
    private static readonly AsyncLocal<WindowsMlExecutionDevicePolicy?> RunWindowsMlDevicePolicy = new();
    private static readonly AsyncLocal<BenchmarkSessionPlan?> ActiveSessionPlan = new();

    public OnnxModelBenchmarkRunner()
        : this(TensorRtRtxProviderBootstrapFactory.Shared)
    {
    }

    public OnnxModelBenchmarkRunner(ITensorRtRtxProviderBootstrap tensorRtRtxProviderBootstrap)
    {
        _tensorRtRtxProviderBootstrap = tensorRtRtxProviderBootstrap
            ?? throw new ArgumentNullException(nameof(tensorRtRtxProviderBootstrap));
    }

    private static WindowsMlExecutionDevicePolicy ActiveWindowsMlDevicePolicy =>
        RunWindowsMlDevicePolicy.Value ?? WindowsMlExecutionDevicePolicy.Explicit;

    private static WindowsMlExecutionDevicePolicy ResolveWindowsMlDevicePolicy(BenchmarkRequest request) =>
        string.IsNullOrWhiteSpace(request.WindowsMlDevicePolicyKey)
            ? WindowsMlExecutionDevicePolicy.Explicit
            : WindowsMlExecutionDevicePolicySettings.FromKey(request.WindowsMlDevicePolicyKey);

    public async Task<BenchmarkReport> RunAsync(BenchmarkRequest request, CancellationToken cancellationToken)
    {
#if WINDOWS
        WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
        cancellationToken.ThrowIfCancellationRequested();

        RunWindowsMlDevicePolicy.Value = ResolveWindowsMlDevicePolicy(request);
        try
        {
            return await RunCoreAsync(request, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            RunWindowsMlDevicePolicy.Value = null;
            ActiveSessionPlan.Value = null;
        }
    }

    private async Task<BenchmarkReport> RunCoreAsync(BenchmarkRequest request, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var fullModelPath = request.ModelPath;
        var fullReportPath = Path.GetFullPath(request.ReportPath);
        var requestedProvider = FormatProviderPreference(request.ProviderPreference);
        var selectedProvider = requestedProvider;
        var modelSizeBytes = File.Exists(fullModelPath) ? new FileInfo(fullModelPath).Length : 0L;
        var notes = new List<string>();
#if WINDOWS
        notes.Add($"ORT package version: {typeof(OrtEnv).Assembly.GetName().Version}");
#endif

        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            BenchmarkExecutionSettings settings = request.Execution ?? BenchmarkExecutionSettings.Default;
            settings.Validate(request.RunCount);
            bool deviceBound = settings.IoMode is BenchmarkIoMode.DeviceBound;
            if (deviceBound && request.ProviderPreference is not BenchmarkProviderPreference.TensorRtRtx)
            {
                throw new NotSupportedException(
                    "Device-bound I/O timing requires the explicit TensorRT-RTX provider route; "
                    + "request managed I/O or the trt-rtx provider.");
            }

            fullModelPath = ResolveModelPath(fullModelPath, notes);
            bool isCompositeGraph = LooksLikeWhisperEncoder(fullModelPath) || LooksLikeOpusEncoder(fullModelPath);
            if (deviceBound && isCompositeGraph)
            {
                throw new NotSupportedException(
                    "Device-bound I/O timing supports single-graph models only in this build; "
                    + "encoder/decoder pipelines keep managed I/O.");
            }

            await TryRegisterWindowsMlProvidersAsync(request.ProviderPreference, notes, cancellationToken).ConfigureAwait(false);

            BenchmarkSessionPlan plan = CreateSessionPlan(
                request, settings, fullReportPath, notes,
                deviceBoundEligible: deviceBound && !isCompositeGraph);
            ActiveSessionPlan.Value = plan;

            modelSizeBytes = new FileInfo(fullModelPath).Length;
            notes.Add($"Model file discovered at '{fullModelPath}'.");
            if (settings.InputShapeOverrides is { Count: > 0 } shapeOverrides)
            {
                notes.Add("Input shape overrides applied: " + string.Join(
                    ", ",
                    shapeOverrides.Select(pair => $"{pair.Key}=[{string.Join('x', pair.Value)}]")));
            }

            BenchmarkExecution execution;
            if (settings.EffectiveTensorRtRtx.CudaGraphMode is BenchmarkCudaGraphMode.Compare)
            {
                execution = RunCudaGraphCompareExecution(
                    fullModelPath, request.ProviderPreference, request.RunCount, settings, notes, cancellationToken);
            }
            else if (isCompositeGraph)
            {
                execution = LooksLikeWhisperEncoder(fullModelPath)
                    ? RunWhisperExecution(fullModelPath, request.ProviderPreference, request.RunCount, settings, notes, cancellationToken)
                    : RunOpusExecution(fullModelPath, request.ProviderPreference, request.RunCount, settings, notes, cancellationToken);
            }
            else
            {
                execution = RunSingleSessionExecution(
                    fullModelPath, request.ProviderPreference, request.RunCount, settings, notes, cancellationToken);
            }

            requestedProvider = execution.RequestedProvider;
            selectedProvider = execution.SelectedProvider;
            modelSizeBytes = execution.ModelSizeBytes;

            BenchmarkMeasurements measurements = BenchmarkMeasurementFactory.Create(
                execution.Sampling,
                execution.SessionInitializationMilliseconds,
                execution.ThroughputUnit,
                execution.AudioDurationSeconds) with
            {
                EngineCacheOutcome = execution.Attribution.Outcome,
                EngineCacheBytesAdded = execution.Attribution.BytesAdded,
                EngineCacheFilesAdded = execution.Attribution.FilesAdded,
                ColdLoadDominantPhase = execution.Attribution.DominantPhase
            };

            bool succeeded = execution.Sampling.PolicyCompleted && execution.Sampling.Failure is null;
            var metadataWarnings = new List<string>(execution.MetadataWarnings);
            if (!succeeded)
            {
                if (execution.Sampling.LatencySamplesMilliseconds.Count > 0)
                {
                    notes.Add(
                        $"Sampling stopped early: {execution.Sampling.LatencySamplesMilliseconds.Count} measured pass(es) retained as partial evidence.");
                }

                metadataWarnings.Add(
                    $"sampling policy did not complete: {execution.Sampling.Failure?.Message ?? "measurement floor not reached"}.");
            }

            var configuration = new BenchmarkConfigurationInfo
            {
                RequestedProvider = requestedProvider,
                SelectedProvider = selectedProvider,
                WindowsMlDevicePolicyKey = request.WindowsMlDevicePolicyKey,
                RequestedRunCount = request.RunCount,
                Execution = settings,
                TimingMode = settings.IoMode is BenchmarkIoMode.DeviceBound
                    ? BenchmarkTimingMode.DeviceBoundSynchronized
                    : BenchmarkTimingMode.ManagedOrtRunEndToEnd,
                InputProfileId = settings.InputMode switch
                {
                    BenchmarkInputMode.Synthetic => BenchmarkMetadataBuilder.SyntheticInputProfileId,
                    BenchmarkInputMode.Profile => BenchmarkMetadataBuilder.ProfileInputProfileId,
                    _ => BenchmarkMetadataBuilder.ZerosInputProfileId
                },
                ModelProfileId = FormatModelProfile(execution.ModelProfile)
            };
            BenchmarkEnvironmentInfo environment = BenchmarkMetadataBuilder.CaptureEnvironment();
            string? comparisonKey = BenchmarkMetadataBuilder.BuildComparisonKey(
                configuration, execution.Graphs, environment, metadataWarnings);
            var metadata = new BenchmarkRunMetadata
            {
                Environment = environment,
                Configuration = configuration,
                WorkDefinition = new BenchmarkWorkDefinition(
                    execution.Scenario,
                    $"{execution.Scenario} benchmark workload",
                    execution.ThroughputUnit,
                    execution.AudioDurationSeconds),
                Graphs = execution.Graphs,
                CudaGraphLanes = execution.CudaGraphLanes ?? [],
                Sampling = new BenchmarkSamplingInfo(
                    execution.Sampling.PolicyCompleted,
                    execution.Sampling.WarmupRunCount,
                    execution.Sampling.WarmupMilliseconds,
                    execution.Sampling.LatencySamplesMilliseconds.Count,
                    execution.Sampling.MeasuredWallTimeMilliseconds,
                    comparisonKey is not null ? "EnvironmentAndWorkload" : "WarmupPolicyOnly"),
                ComparisonKey = comparisonKey,
                Warnings = metadataWarnings
            };

            return CreateReport(
                scenario: execution.Scenario,
                status: succeeded ? BenchmarkStatus.Completed : BenchmarkStatus.Failed,
                modelPath: fullModelPath,
                reportPath: fullReportPath,
                requestedProvider: requestedProvider,
                selectedProvider: selectedProvider,
                runCount: request.RunCount,
                supportsExecution: succeeded || execution.Sampling.FirstRunMilliseconds is not null,
                modelSizeBytes: modelSizeBytes,
                measurements: measurements,
                failureReason: succeeded
                    ? null
                    : execution.Sampling.Failure?.Message ?? "Measurement policy did not complete.",
                notes: notes,
                metadata: metadata);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            notes.Add($"Benchmark failed: {ex.GetType().Name}");
            return CreateReport(
                scenario: ResolveFailureScenario(fullModelPath),
                status: BenchmarkStatus.Failed,
                modelPath: fullModelPath,
                reportPath: fullReportPath,
                requestedProvider: requestedProvider,
                selectedProvider: selectedProvider,
                runCount: request.RunCount,
                supportsExecution: false,
                modelSizeBytes: modelSizeBytes,
                measurements: EmptyMeasurements(),
                failureReason: ex.Message,
                notes: notes);
        }
    }

    private static ExecutionProviderKind MapPreferenceToExecutionProvider(BenchmarkProviderPreference preference) =>
        preference switch
        {
            BenchmarkProviderPreference.Cpu => ExecutionProviderKind.Cpu,
            BenchmarkProviderPreference.Dml => ExecutionProviderKind.DirectMl,
            BenchmarkProviderPreference.TensorRtRtx => ExecutionProviderKind.TensorRTRtx,
            BenchmarkProviderPreference.Migraphx => ExecutionProviderKind.Migraphx,
            BenchmarkProviderPreference.Cuda => ExecutionProviderKind.Cuda,
            BenchmarkProviderPreference.TensorRt => ExecutionProviderKind.TensorRt,
            BenchmarkProviderPreference.Qnn => ExecutionProviderKind.Qnn,
            BenchmarkProviderPreference.Auto => ExecutionProviderKind.DirectMl,
            _ => throw new ArgumentOutOfRangeException(nameof(preference), preference, "Unknown provider preference.")
        };

    private async Task TryRegisterWindowsMlProvidersAsync(
        BenchmarkProviderPreference preference,
        ICollection<string> notes,
        CancellationToken cancellationToken)
    {
        if (preference is BenchmarkProviderPreference.Cuda or BenchmarkProviderPreference.TensorRt)
        {
            notes.Add("Skipping WinML catalog registration for native ORT CUDA/TensorRT benchmark.");
            return;
        }

        ExecutionProviderKind provider = MapPreferenceToExecutionProvider(preference);

        if (provider is ExecutionProviderKind.Migraphx)
        {
            var catalog = new WindowsMlMigraphxCatalogService();
            MigraphxBootstrapResult bootstrap = await catalog
                .EnsureRegisteredAsync(allowProviderDownloads: true, cancellationToken)
                .ConfigureAwait(false);
            notes.Add(bootstrap.Detail);
            return;
        }

        if (provider is ExecutionProviderKind.TensorRTRtx)
        {
            TensorRtRtxBootstrapResult bootstrap = await _tensorRtRtxProviderBootstrap
                .EnsureRegisteredAsync(allowProviderDownloads: true, cancellationToken)
                .ConfigureAwait(false);
            notes.Add(bootstrap.Detail);
            return;
        }

        WindowsMlProviderRegistrationResult result = await WindowsMlProviderRegistrationPolicy.Shared
            .RegisterForReadinessAsync(provider, cancellationToken)
            .ConfigureAwait(false);
        notes.Add(result.Detail);
    }

    private static BenchmarkExecution RunSingleSessionExecution(
        string modelPath,
        BenchmarkProviderPreference preference,
        int runCount,
        BenchmarkExecutionSettings settings,
        ICollection<string> notes,
        CancellationToken cancellationToken)
    {
        BenchmarkGraphPlan graph = CreateGraphPlan(modelPath, preference, "single", notes);
        string? probeDirectory = ActiveSessionPlan.Value?.RuntimeCacheDirectory;
        EngineCacheProbe.Snapshot cacheBefore = EngineCacheProbe.Capture(probeDirectory);
        var sessionInitStopwatch = Stopwatch.StartNew();
        using var sessionLease = CreateSession(graph, preference, notes);
        sessionInitStopwatch.Stop();
        ColdLoadAttribution attribution = AttributeColdLoad(
            sessionInitStopwatch.Elapsed.TotalMilliseconds, sessionLease.SelectedProvider, cacheBefore, probeDirectory, notes);

        var inputSet = CreateInputs(modelPath, sessionLease.Session.InputMetadata, settings, "single");

        var deviceBoundLimitations = new List<string>();
        using DeviceBoundIoBinding? deviceBinding = settings.IoMode is BenchmarkIoMode.DeviceBound
            ? DeviceBoundIoBinding.Create(
                sessionLease.Session,
                inputSet.Values,
                sessionLease.Session.InputMetadata,
                sessionLease.Session.OutputMetadata,
                notes,
                deviceBoundLimitations)
            : null;
        foreach (string limitation in deviceBoundLimitations)
        {
            notes.Add($"Device-bound I/O limitation: {limitation}");
        }

        Action<CancellationToken> pass = deviceBinding is { } bound
            ? _ => bound.RunPass()
            : _ => { using var results = sessionLease.Session.Run(inputSet.Values); };

        var phaseSnapshots = new List<(BenchmarkSamplingPhase, EngineCacheProbe.Snapshot)>();
        BenchmarkSamplingResult sampling = BenchmarkSamplingLoop.Run(
            pass,
            runCount,
            settings,
            cancellationToken: cancellationToken,
            phaseObserver: phase => phaseSnapshots.Add((phase, EngineCacheProbe.Capture(probeDirectory))));

        ReportPostInitCacheActivity(attribution.CacheAfter, phaseSnapshots.Select(pair => pair.Item2).ToArray(), notes);

        var warnings = new List<string>();
        List<BenchmarkCacheObservation> cacheObservations = BuildCacheObservations(
            cacheBefore, attribution.CacheAfter, attribution.Outcome, phaseSnapshots);
        var graphs = new List<BenchmarkGraphMetadata>
        {
            DescribeGraph(graph, sessionLease.Session, cacheObservations, warnings)
        };
        foreach (string warning in warnings)
        {
            notes.Add(warning);
        }

        return new BenchmarkExecution(
            Scenario: "onnx-model",
            RequestedProvider: sessionLease.RequestedProvider,
            SelectedProvider: sessionLease.SelectedProvider,
            ModelSizeBytes: new FileInfo(modelPath).Length,
            SessionInitializationMilliseconds: sessionInitStopwatch.Elapsed.TotalMilliseconds,
            Attribution: attribution,
            Sampling: sampling,
            ThroughputUnit: inputSet.AudioDurationSeconds is not null
                ? BenchmarkThroughputUnit.AudioSecondsPerSecond
                : BenchmarkThroughputUnit.PassesPerSecond,
            AudioDurationSeconds: inputSet.AudioDurationSeconds,
            Graphs: graphs,
            MetadataWarnings: warnings,
            ModelProfile: inputSet.ModelProfile);
    }

    private static BenchmarkSessionPlan CreateCompareLanePlan(
        BenchmarkSessionPlan basePlan,
        bool enableCudaGraph,
        string laneName)
    {
        string? laneCacheDirectory = basePlan.RuntimeCacheDirectory;
        if (basePlan.IsolatedRuntimeCache && laneCacheDirectory is not null)
        {
            laneCacheDirectory = Path.Join(laneCacheDirectory, laneName);
            Directory.CreateDirectory(laneCacheDirectory);
        }

        return basePlan with { EnableCudaGraph = enableCudaGraph, RuntimeCacheDirectory = laneCacheDirectory };
    }

    /// <summary>
    /// Compare mode: runs the same single-graph benchmark twice under otherwise identical
    /// policy, once with enable_cuda_graph=0 and once with =1, each in a fresh session so the
    /// provider option is honestly applied. The on-lane becomes the primary measurement; both
    /// lanes are recorded in metadata for attribution.
    /// </summary>
    private static BenchmarkExecution RunCudaGraphCompareExecution(
        string modelPath,
        BenchmarkProviderPreference preference,
        int runCount,
        BenchmarkExecutionSettings settings,
        ICollection<string> notes,
        CancellationToken cancellationToken)
    {
        BenchmarkSessionPlan basePlan = ActiveSessionPlan.Value ?? BenchmarkSessionPlan.Default;
        BenchmarkExecution offLane;
        BenchmarkExecution onLane;
        try
        {
            notes.Add("CUDA graph compare: measuring graphs-off baseline lane.");
            ActiveSessionPlan.Value = CreateCompareLanePlan(basePlan, enableCudaGraph: false, "graphs-off");
            offLane = RunSingleSessionExecution(
                modelPath, preference, runCount, settings, notes, cancellationToken);

            notes.Add("CUDA graph compare: measuring graphs-on lane.");
            ActiveSessionPlan.Value = CreateCompareLanePlan(basePlan, enableCudaGraph: true, "graphs-on");
            onLane = RunSingleSessionExecution(
                modelPath, preference, runCount, settings, notes, cancellationToken);
        }
        finally
        {
            ActiveSessionPlan.Value = basePlan;
        }

        BenchmarkGraphMetadata? offGraph = offLane.Graphs.Count > 0 ? offLane.Graphs[0] : null;
        BenchmarkGraphMetadata? onGraph = onLane.Graphs.Count > 0 ? onLane.Graphs[0] : null;
        string? offLoadKind = offGraph?.LoadKind;
        string? onLoadKind = onGraph?.LoadKind;
        string? offArtifact = offGraph?.ArtifactIdentity;
        string? onArtifact = onGraph?.ArtifactIdentity;
        if (!string.Equals(offLoadKind, onLoadKind, StringComparison.Ordinal)
            || !string.Equals(offArtifact, onArtifact, StringComparison.Ordinal))
        {
            notes.Add(
                $"CUDA graph compare lanes loaded different artifacts: " +
                $"off=[{offLoadKind ?? "none"}:{offArtifact ?? "none"}] " +
                $"on=[{onLoadKind ?? "none"}:{onArtifact ?? "none"}].");
        }

        return onLane with
        {
            CudaGraphLanes =
            [
                ToCudaGraphLaneResult("graphs-off", offLane),
                ToCudaGraphLaneResult("graphs-on", onLane)
            ]
        };
    }

    private static BenchmarkCudaGraphLaneResult ToCudaGraphLaneResult(string lane, BenchmarkExecution execution)
    {
        BenchmarkGraphMetadata? graph = execution.Graphs.Count > 0 ? execution.Graphs[0] : null;
        return new(
            Lane: lane,
            EnableCudaGraph: execution.Graphs.Any(g => g.EnableCudaGraph),
            SessionInitializationMilliseconds: execution.SessionInitializationMilliseconds,
            TimingMode: BenchmarkTimingMode.DeviceBoundSynchronized,
            Sampling: new BenchmarkSamplingInfo(
                execution.Sampling.PolicyCompleted,
                execution.Sampling.WarmupRunCount,
                execution.Sampling.WarmupMilliseconds,
                execution.Sampling.LatencySamplesMilliseconds.Count,
                execution.Sampling.MeasuredWallTimeMilliseconds),
            Statistics: BenchmarkLatencyCalculator.Calculate(execution.Sampling.LatencySamplesMilliseconds),
            UnavailableReason: execution.Sampling.Failure?.Message,
            LoadKind: graph?.LoadKind,
            ArtifactIdentity: graph?.ArtifactIdentity);
    }

    /// <summary>
    /// Run-scoped benchmark plan for TensorRT-RTX option normalization, artifact selection,
    /// and runtime-cache isolation. Stored in <see cref="ActiveSessionPlan"/> so every graph
    /// in a composite workload resolves consistently.
    /// </summary>
    private sealed record BenchmarkSessionPlan(
        BenchmarkArtifactMode ArtifactMode,
        bool EnableCudaGraph,
        string? RuntimeCacheDirectory,
        bool IsolatedRuntimeCache = false)
    {
        public static BenchmarkSessionPlan Default { get; } = new(
            BenchmarkArtifactMode.Auto, EnableCudaGraph: false, RuntimeCacheDirectory: null);
    }

    /// <summary>Per-graph resolution: source path, the actual load path, and normalized TRT options.</summary>
    private sealed record BenchmarkGraphPlan(
        string SourcePath,
        string LoadPath,
        string LoadKind,
        string Role,
        IReadOnlyDictionary<string, string>? TrtOptions,
        IReadOnlyDictionary<string, string> EffectiveProviderOptions,
        string? CompileOptionsIdentity,
        string? ArtifactIdentity);

    private static BenchmarkSessionPlan CreateSessionPlan(
        BenchmarkRequest request,
        BenchmarkExecutionSettings settings,
        string fullReportPath,
        ICollection<string> notes,
        bool deviceBoundEligible = false)
    {
        BenchmarkTensorRtRtxSettings trtSettings = settings.EffectiveTensorRtRtx;
        if (request.ProviderPreference is not BenchmarkProviderPreference.TensorRtRtx)
        {
            if (trtSettings.EnableCudaGraph)
            {
                throw new ArgumentException("CUDA graph options require the trt-rtx provider.");
            }

            if (trtSettings.ArtifactMode is not BenchmarkArtifactMode.Auto
                || trtSettings.CacheMode is not BenchmarkRuntimeCacheMode.Reuse
                || trtSettings.RuntimeCacheDirectory is not null)
            {
                notes.Add("TensorRT-RTX benchmark options are ignored for a non-trt-rtx provider.");
            }

            return BenchmarkSessionPlan.Default;
        }

        if (ShouldUseCatalogDevicePolicyForPreference(request.ProviderPreference))
        {
            bool unsupported = trtSettings.EnableCudaGraph
                || trtSettings.ArtifactMode is not BenchmarkArtifactMode.Auto
                || trtSettings.CacheMode is not BenchmarkRuntimeCacheMode.Reuse
                || trtSettings.RuntimeCacheDirectory is not null;
            if (unsupported)
            {
                throw new NotSupportedException(
                    "The WinML catalog device-policy route cannot honor TensorRT-RTX artifact, runtime-cache, or CUDA graph benchmark settings.");
            }

            return BenchmarkSessionPlan.Default;
        }

        if (trtSettings.CudaGraphMode is not BenchmarkCudaGraphMode.Auto
            && OnnxExecutionSessionFactory.IsCudaGraphOverrideSet())
        {
            throw new NotSupportedException(
                "--cuda-graphs on, off and compare choose the capture mode for each run, so they cannot run while "
                + $"{OnnxExecutionSessionFactory.CudaGraphsOverrideEnvironmentVariable} is set: it would give every lane "
                + "the same capture mode and report a comparison that compares nothing. Unset the variable.");
        }

        string? cacheDirectory = trtSettings.CacheMode switch
        {
            BenchmarkRuntimeCacheMode.Isolated =>
                ResolveIsolatedRuntimeCacheDirectory(trtSettings, fullReportPath, notes),
            BenchmarkRuntimeCacheMode.Reuse =>
                ResolveReusedRuntimeCacheDirectory(trtSettings, notes),
            _ => null
        };

        bool enableCudaGraph = trtSettings.ResolveCudaGraphEnable(deviceBoundEligible);
        if (trtSettings.CudaGraphMode is BenchmarkCudaGraphMode.Auto)
        {
            notes.Add(enableCudaGraph
                ? "CUDA graph auto mode: enabled for device-bound single-graph trt-rtx run."
                : "CUDA graph auto mode: disabled; the run does not meet the persistent device-bound I/O precondition.");
        }

        return new BenchmarkSessionPlan(
            trtSettings.ArtifactMode,
            enableCudaGraph,
            cacheDirectory,
            IsolatedRuntimeCache: trtSettings.CacheMode is BenchmarkRuntimeCacheMode.Isolated);
    }

    private static string? ResolveReusedRuntimeCacheDirectory(
        BenchmarkTensorRtRtxSettings trtSettings,
        ICollection<string> notes)
    {
        if (trtSettings.RuntimeCacheDirectory is null)
        {
            return null;
        }

        Directory.CreateDirectory(trtSettings.RuntimeCacheDirectory);
        notes.Add($"Runtime cache directory: '{trtSettings.RuntimeCacheDirectory}'.");
        return trtSettings.RuntimeCacheDirectory;
    }

    private static string ResolveIsolatedRuntimeCacheDirectory(
        BenchmarkTensorRtRtxSettings trtSettings,
        string fullReportPath,
        ICollection<string> notes)
    {
        string directory = trtSettings.RuntimeCacheDirectory
            ?? Path.Join(
                Path.GetDirectoryName(fullReportPath) ?? Environment.CurrentDirectory,
                $"benchmark-runtime-cache-{Guid.NewGuid():N}");
        if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
        {
            throw new InvalidOperationException(
                $"Isolated runtime cache directory '{directory}' already exists and is not empty; " +
                "refusing to reuse another run's artifacts.");
        }

        Directory.CreateDirectory(directory);
        notes.Add($"Isolated runtime cache directory: '{directory}'.");
        return directory;
    }

    /// <summary>
    /// Resolves the per-graph TRT-RTX options (production profile plus benchmark overrides)
    /// and selects the actual load path through the same compatibility checks production uses.
    /// Artifact validation and profile resolution run before the session-init timer.
    /// </summary>
    private static BenchmarkGraphPlan CreateGraphPlan(
        string sourceModelPath,
        BenchmarkProviderPreference preference,
        string role,
        ICollection<string> notes)
    {
        BenchmarkSessionPlan plan = ActiveSessionPlan.Value ?? BenchmarkSessionPlan.Default;
        if (preference is not BenchmarkProviderPreference.TensorRtRtx
            || ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            return new BenchmarkGraphPlan(
                sourceModelPath, sourceModelPath, "source", role, null,
                new Dictionary<string, string>(StringComparer.Ordinal), null, null);
        }

        IReadOnlyDictionary<string, string> additional = BuildBenchmarkTrtAdditionalOptions(sourceModelPath, plan);
        IReadOnlyDictionary<string, string> effective =
            OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(additional, plan.EnableCudaGraph);
        string compileOptionsIdentity = EpContextLoadPathResolver.BuildCompileOptionsIdentity(sourceModelPath, effective);

        if (plan.ArtifactMode is BenchmarkArtifactMode.Source)
        {
            notes.Add($"Artifact policy 'source': bypassing EP-context artifacts for '{sourceModelPath}'.");
            return new BenchmarkGraphPlan(
                sourceModelPath, sourceModelPath, "source", role, additional,
                effective, compileOptionsIdentity, null);
        }

        string? artifactPath = EpContextLoadPathResolver.TryResolveLoadPath(sourceModelPath, effective);
        if (artifactPath is not null)
        {
            notes.Add($"EP-context artifact selected for {role}: '{artifactPath}'.");
            return new BenchmarkGraphPlan(
                sourceModelPath, artifactPath, "ep-context-artifact", role, additional,
                effective, compileOptionsIdentity,
                $"{Path.GetFileName(artifactPath)}#{compileOptionsIdentity}");
        }

        if (plan.ArtifactMode is BenchmarkArtifactMode.Require)
        {
            throw new InvalidOperationException(
                $"EP-context artifact required but no compatible artifact was found for '{sourceModelPath}' ({role}).");
        }

        return new BenchmarkGraphPlan(
            sourceModelPath, sourceModelPath, "source", role, additional,
            effective, compileOptionsIdentity, null);
    }

    private static IReadOnlyDictionary<string, string> BuildBenchmarkTrtAdditionalOptions(
        string sourceModelPath,
        BenchmarkSessionPlan plan)
    {
        var additional = new Dictionary<string, string>(StringComparer.Ordinal);
        if (EpContextTrtProfiles.Resolve(sourceModelPath) is { } profile)
        {
            foreach ((string key, string value) in profile)
            {
                // The benchmark's explicit graph selection wins over production profile
                // defaults such as the MADLAD graph opt-in.
                if (!key.Equals("enable_cuda_graph", StringComparison.OrdinalIgnoreCase))
                {
                    additional[key] = value;
                }
            }
        }

        if (plan.RuntimeCacheDirectory is not null)
        {
            additional["nv_runtime_cache_path"] = plan.RuntimeCacheDirectory;
        }

        return additional;
    }

    private static void ReportPostInitCacheActivity(
        EngineCacheProbe.Snapshot cacheAfterInit,
        IReadOnlyList<EngineCacheProbe.Snapshot> phaseSnapshots,
        ICollection<string> notes)
    {
        if (phaseSnapshots.Count == 0)
        {
            return;
        }

        EngineCacheProbe.Snapshot last = phaseSnapshots[^1];
        long bytesAdded = Math.Max(0, last.TotalBytes - cacheAfterInit.TotalBytes);
        int filesAdded = Math.Max(0, last.FileCount - cacheAfterInit.FileCount);
        if (bytesAdded > 0 || filesAdded > 0)
        {
            notes.Add(
                $"Runtime cache grew after session initialization (+{filesAdded} file(s), +{bytesAdded} byte(s)); " +
                "first-run/warmup/measurement cache writes are recorded as activity, not a proven hit.");
        }
    }

    /// <summary>Cache probes at session-init and after each sampling phase, for one execution.</summary>
    private static List<BenchmarkCacheObservation> BuildCacheObservations(
        EngineCacheProbe.Snapshot cacheBefore,
        EngineCacheProbe.Snapshot cacheAfterInit,
        string coldLoadOutcome,
        IReadOnlyList<(BenchmarkSamplingPhase Phase, EngineCacheProbe.Snapshot Snapshot)> phaseSnapshots)
    {
        var observations = new List<BenchmarkCacheObservation>
        {
            new(BenchmarkCachePhase.BeforeSession,
                cacheBefore.DirectoryExists, cacheBefore.FileCount, cacheBefore.TotalBytes),
            new(BenchmarkCachePhase.SessionInitialized,
                cacheAfterInit.DirectoryExists, cacheAfterInit.FileCount, cacheAfterInit.TotalBytes,
                coldLoadOutcome)
        };

        BenchmarkCachePhase MapPhase(BenchmarkSamplingPhase phase) => phase switch
        {
            BenchmarkSamplingPhase.FirstRun => BenchmarkCachePhase.FirstRun,
            BenchmarkSamplingPhase.Warmup => BenchmarkCachePhase.Warmup,
            _ => BenchmarkCachePhase.Measurement
        };

        foreach ((BenchmarkSamplingPhase phase, EngineCacheProbe.Snapshot snapshot) in phaseSnapshots)
        {
            observations.Add(new BenchmarkCacheObservation(
                MapPhase(phase), snapshot.DirectoryExists, snapshot.FileCount, snapshot.TotalBytes));
        }

        return observations;
    }

    /// <summary>
    /// Graph identity + tensor surface + effective options recorded for report metadata.
    /// Runs after session creation, outside every timed region.
    /// </summary>
    private static BenchmarkGraphMetadata DescribeGraph(
        BenchmarkGraphPlan plan,
        InferenceSession session,
        IReadOnlyList<BenchmarkCacheObservation> cacheObservations,
        ICollection<string> warnings)
    {
        string? sourceSha256 = BenchmarkMetadataBuilder.ComputeModelSha256(plan.SourcePath);
        if (sourceSha256 is null)
        {
            warnings.Add($"model identity unavailable: could not hash '{plan.SourcePath}'.");
        }

        bool externalDataVerified = BenchmarkMetadataBuilder.TryVerifyExternalDataIdentity(
            plan.SourcePath, out string? externalDataSha256, out string? externalReason);
        if (externalReason is not null)
        {
            warnings.Add($"external weights identity unavailable for '{plan.SourcePath}': {externalReason}.");
        }

        IReadOnlyList<string> devices = BenchmarkMetadataBuilder.DescribeSessionDevices(session);
        string? selectedDevice = null;
        foreach (string device in devices)
        {
            if (device.Contains("GPU", StringComparison.OrdinalIgnoreCase))
            {
                selectedDevice = device;
                break;
            }
        }

        selectedDevice ??= devices.Count > 0 ? devices[0] : null;
        if (selectedDevice is null)
        {
            warnings.Add($"selected-device identity unavailable for graph '{plan.Role}'.");
        }

        return new BenchmarkGraphMetadata
        {
            Role = plan.Role,
            SourceModelPath = plan.SourcePath,
            SourceModelSha256 = sourceSha256,
            ExternalDataIdentityVerified = externalDataVerified,
            ExternalDataSha256 = externalDataSha256,
            LoadedModelPath = plan.LoadPath,
            LoadKind = plan.LoadKind,
            ArtifactIdentity = plan.ArtifactIdentity,
            CompileOptionsIdentity = plan.CompileOptionsIdentity,
            RuntimeCacheDirectoryIdentity = plan.EffectiveProviderOptions.TryGetValue(
                "nv_runtime_cache_path", out string? cachePath) ? cachePath : null,
            EnableCudaGraph = plan.EffectiveProviderOptions.TryGetValue(
                "enable_cuda_graph", out string? graphFlag) && graphFlag == "1",
            SelectedDeviceIdentity = selectedDevice,
            EffectiveProviderOptions = plan.EffectiveProviderOptions,
            Inputs = BenchmarkMetadataBuilder.DescribeTensors(session.InputMetadata),
            Outputs = BenchmarkMetadataBuilder.DescribeTensors(session.OutputMetadata),
            CacheObservations = cacheObservations
        };
    }

    private readonly record struct ColdLoadAttribution(
        string Outcome,
        long BytesAdded,
        int FilesAdded,
        string DominantPhase,
        EngineCacheProbe.Snapshot CacheAfter);

    private static readonly string[] EngineCacheRelevantProviderLabels = ["tensorrt-rtx", "tensorrt", "migraphx"];

    private static ColdLoadAttribution AttributeColdLoad(
        double coldLoadMilliseconds,
        string selectedProvider,
        EngineCacheProbe.Snapshot cacheBefore,
        string? probeDirectory,
        ICollection<string> notes)
    {
        // Classify from the provider the session actually selected, not the requested
        // preference — a TRT-RTX request that fell back to CPU never touches the engine
        // cache, so scoring it as a GPU cache run can produce a false "hit".
        bool isEngineCacheRelevant = EngineCacheRelevantProviderLabels.Contains(
            selectedProvider, StringComparer.OrdinalIgnoreCase);
        EngineCacheProbe.Snapshot cacheAfter = EngineCacheProbe.Capture(probeDirectory);
        string outcome = EngineCacheProbe.ClassifyOutcome(cacheBefore, cacheAfter, isEngineCacheRelevant);
        string dominant = EngineCacheProbe.ClassifyDominantPhase(outcome);
        long bytesAdded = Math.Max(0, cacheAfter.TotalBytes - cacheBefore.TotalBytes);
        int filesAdded = Math.Max(0, cacheAfter.FileCount - cacheBefore.FileCount);
        notes.Add(EngineCacheProbe.FormatNote(coldLoadMilliseconds, outcome, cacheBefore, cacheAfter, dominant));
        return new ColdLoadAttribution(outcome, bytesAdded, filesAdded, dominant, cacheAfter);
    }

    private static BenchmarkExecution RunWhisperExecution(
        string encoderModelPath,
        BenchmarkProviderPreference preference,
        int runCount,
        BenchmarkExecutionSettings settings,
        ICollection<string> notes,
        CancellationToken cancellationToken)
    {
        var decoderModelPath = ResolveWhisperDecoderPath(encoderModelPath);
        var configPath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "..", "config.json");
        var fullConfigPath = Path.GetFullPath(configPath);

        notes.Add($"Whisper decoder discovered at '{decoderModelPath}'.");

        BenchmarkGraphPlan encoderGraph = CreateGraphPlan(encoderModelPath, preference, "encoder", notes);
        BenchmarkGraphPlan decoderGraph = CreateGraphPlan(decoderModelPath, preference, "decoder", notes);
        string? probeDirectory = ActiveSessionPlan.Value?.RuntimeCacheDirectory;
        EngineCacheProbe.Snapshot cacheBefore = EngineCacheProbe.Capture(probeDirectory);
        var sessionInitStopwatch = Stopwatch.StartNew();
        using var whisperLease = CreateWhisperSessionLease(encoderGraph, decoderGraph, preference, notes);
        sessionInitStopwatch.Stop();
        ColdLoadAttribution attribution = AttributeColdLoad(
            sessionInitStopwatch.Elapsed.TotalMilliseconds, whisperLease.SelectedProvider, cacheBefore, probeDirectory, notes);

        var encoderInputSet = CreateInputs(encoderModelPath, whisperLease.EncoderSession.InputMetadata, settings, "encoder");
        var decoderStartTokenId = ResolveWhisperDecoderStartTokenId(fullConfigPath);

        var phaseSnapshots = new List<(BenchmarkSamplingPhase, EngineCacheProbe.Snapshot)>();
        BenchmarkSamplingResult sampling = BenchmarkSamplingLoop.Run(
            _ => RunWhisperPass(whisperLease, encoderInputSet, decoderStartTokenId),
            runCount,
            settings,
            cancellationToken: cancellationToken,
            phaseObserver: phase => phaseSnapshots.Add((phase, EngineCacheProbe.Capture(probeDirectory))));

        ReportPostInitCacheActivity(attribution.CacheAfter, phaseSnapshots.Select(pair => pair.Item2).ToArray(), notes);

        var warnings = new List<string>();
        List<BenchmarkCacheObservation> cacheObservations = BuildCacheObservations(
            cacheBefore, attribution.CacheAfter, attribution.Outcome, phaseSnapshots);
        var graphs = new List<BenchmarkGraphMetadata>
        {
            DescribeGraph(encoderGraph, whisperLease.EncoderSession, cacheObservations, warnings),
            DescribeGraph(decoderGraph, whisperLease.DecoderSession, [], warnings)
        };
        foreach (string warning in warnings)
        {
            notes.Add(warning);
        }

        return new BenchmarkExecution(
            Scenario: "whisper-encoder-decoder",
            RequestedProvider: whisperLease.RequestedProvider,
            SelectedProvider: whisperLease.SelectedProvider,
            ModelSizeBytes: new FileInfo(encoderModelPath).Length + new FileInfo(decoderModelPath).Length,
            SessionInitializationMilliseconds: sessionInitStopwatch.Elapsed.TotalMilliseconds,
            Attribution: attribution,
            Sampling: sampling,
            ThroughputUnit: encoderInputSet.AudioDurationSeconds is not null
                ? BenchmarkThroughputUnit.AudioSecondsPerSecond
                : BenchmarkThroughputUnit.PassesPerSecond,
            AudioDurationSeconds: encoderInputSet.AudioDurationSeconds,
            Graphs: graphs,
            MetadataWarnings: warnings,
            ModelProfile: encoderInputSet.ModelProfile);
    }

    private static BenchmarkExecution RunOpusExecution(
        string encoderModelPath,
        BenchmarkProviderPreference preference,
        int runCount,
        BenchmarkExecutionSettings settings,
        ICollection<string> notes,
        CancellationToken cancellationToken)
    {
        var decoderModelPath = ResolveOpusDecoderPath(encoderModelPath);
        var configPath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "..", "config.json");
        var fullConfigPath = Path.GetFullPath(configPath);

        notes.Add($"Opus decoder discovered at '{decoderModelPath}'.");

        BenchmarkGraphPlan encoderGraph = CreateGraphPlan(encoderModelPath, preference, "encoder", notes);
        BenchmarkGraphPlan decoderGraph = CreateGraphPlan(decoderModelPath, preference, "decoder", notes);
        string? probeDirectory = ActiveSessionPlan.Value?.RuntimeCacheDirectory;
        EngineCacheProbe.Snapshot cacheBefore = EngineCacheProbe.Capture(probeDirectory);
        var sessionInitStopwatch = Stopwatch.StartNew();
        using var opusLease = CreateOpusSessionLease(encoderGraph, decoderGraph, preference, notes);
        sessionInitStopwatch.Stop();
        ColdLoadAttribution attribution = AttributeColdLoad(
            sessionInitStopwatch.Elapsed.TotalMilliseconds, opusLease.SelectedProvider, cacheBefore, probeDirectory, notes);

        var encoderInputSet = CreateInputs(encoderModelPath, opusLease.EncoderSession.InputMetadata, settings, "encoder");
        var decoderStartTokenId = ResolveOpusDecoderStartTokenId(fullConfigPath);

        var phaseSnapshots = new List<(BenchmarkSamplingPhase, EngineCacheProbe.Snapshot)>();
        BenchmarkSamplingResult sampling = BenchmarkSamplingLoop.Run(
            _ => RunOpusPass(opusLease, encoderInputSet, decoderStartTokenId),
            runCount,
            settings,
            cancellationToken: cancellationToken,
            phaseObserver: phase => phaseSnapshots.Add((phase, EngineCacheProbe.Capture(probeDirectory))));

        ReportPostInitCacheActivity(attribution.CacheAfter, phaseSnapshots.Select(pair => pair.Item2).ToArray(), notes);

        var warnings = new List<string>();
        List<BenchmarkCacheObservation> cacheObservations = BuildCacheObservations(
            cacheBefore, attribution.CacheAfter, attribution.Outcome, phaseSnapshots);
        var graphs = new List<BenchmarkGraphMetadata>
        {
            DescribeGraph(encoderGraph, opusLease.EncoderSession, cacheObservations, warnings),
            DescribeGraph(decoderGraph, opusLease.DecoderSession, [], warnings)
        };
        foreach (string warning in warnings)
        {
            notes.Add(warning);
        }

        return new BenchmarkExecution(
            Scenario: "opus-mt-encoder-decoder",
            RequestedProvider: opusLease.RequestedProvider,
            SelectedProvider: opusLease.SelectedProvider,
            ModelSizeBytes: new FileInfo(encoderModelPath).Length + new FileInfo(decoderModelPath).Length,
            SessionInitializationMilliseconds: sessionInitStopwatch.Elapsed.TotalMilliseconds,
            Attribution: attribution,
            Sampling: sampling,
            ThroughputUnit: BenchmarkThroughputUnit.PassesPerSecond,
            AudioDurationSeconds: null,
            Graphs: graphs,
            MetadataWarnings: warnings,
            ModelProfile: encoderInputSet.ModelProfile);
    }

    private static bool ShouldUseCatalogDevicePolicyForPreference(BenchmarkProviderPreference preference)
    {
#if WINDOWS
        return ProviderNames.ShouldUseCatalogDevicePolicy(
            ActiveWindowsMlDevicePolicy,
            MapPreferenceToExecutionProvider(preference));
#else
        return false;
#endif
    }

    private static string GetCatalogGpuRouteNote(BenchmarkProviderPreference preference, string explicitRouteDescription) =>
        ShouldUseCatalogDevicePolicyForPreference(preference)
            ? "Provider route: WinML catalog device policy (ORT auto-selects among registered catalog EPs)."
            : $"Provider route: {explicitRouteDescription}";

    private static string ResolveBenchmarkEffectiveProviderLabel(
        InferenceSession session,
        BenchmarkProviderPreference preference)
    {
        ExecutionProviderKind optionsSelected = MapPreferenceToExecutionProvider(preference);
        bool useCatalog = ShouldUseCatalogDevicePolicyForPreference(preference);
        ExecutionProviderKind effective = OnnxExecutionSessionFactory.ResolveEffectiveProviderKindFromSession(
            session,
            optionsSelected,
            useCatalog);
        return FormatExecutionProvider(effective);
    }

    private static BenchmarkSessionLease CreateSession(
        BenchmarkGraphPlan graph,
        BenchmarkProviderPreference preference,
        ICollection<string> notes) =>
        preference switch
        {
            BenchmarkProviderPreference.Cpu => CreateCpuSession(graph, notes),
            BenchmarkProviderPreference.Dml => CreateDirectMlSession(graph, notes),
            BenchmarkProviderPreference.TensorRtRtx => CreateTensorRtRtxSession(graph, notes),
            BenchmarkProviderPreference.Migraphx => CreateMigraphxSession(graph, notes),
            BenchmarkProviderPreference.Cuda => CreateNativeCudaSession(graph, notes),
            BenchmarkProviderPreference.TensorRt => CreateNativeTensorRtSession(graph, notes),
            BenchmarkProviderPreference.Qnn => CreateQnnSession(graph, notes),
            BenchmarkProviderPreference.Auto => CreateAutoSession(graph, notes),
            _ => throw new ArgumentOutOfRangeException(nameof(preference), preference, "Unknown provider preference.")
        };

    private static BenchmarkSessionLease CreateCpuSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        notes.Add("Provider route: explicit CPU execution provider.");
        using SessionOptions options = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
        var session = new InferenceSession(graph.LoadPath, options);
        return new BenchmarkSessionLease(session, "cpu", "cpu", [graph]);
    }

    private static BenchmarkSessionLease CreateDirectMlSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Dml;
        notes.Add(GetCatalogGpuRouteNote(preference, "explicit WinML catalog DirectML execution provider."));
        using SessionOptions options = CreateSessionOptions(preference);
        if (!ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            OnnxExecutionSessionFactory.AppendDirectMlProvider(options);
        }

        var session = new InferenceSession(graph.LoadPath, options);
        string selectedProviderLabel = ResolveBenchmarkEffectiveProviderLabel(session, preference);
        return new BenchmarkSessionLease(session, "dml", selectedProviderLabel, [graph]);
    }

    private static BenchmarkSessionLease CreateTensorRtRtxSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        notes.Add(GetCatalogGpuRouteNote(
            BenchmarkProviderPreference.TensorRtRtx,
            "explicit TensorRT RTX EP ABI plugin execution provider."));
        ProviderSession providerSession = CreateTensorRtRtxProviderSession(graph, notes);
        return new BenchmarkSessionLease(providerSession.Session, "trt-rtx", providerSession.SelectedProvider, [graph]);
    }

    private static BenchmarkSessionLease CreateMigraphxSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        notes.Add(GetCatalogGpuRouteNote(
            BenchmarkProviderPreference.Migraphx,
            "explicit MIGraphX execution provider."));
        var session = CreateMigraphxInferenceSession(graph.LoadPath, notes);
        string selectedProviderLabel = ResolveBenchmarkEffectiveProviderLabel(session, BenchmarkProviderPreference.Migraphx);
        return new BenchmarkSessionLease(session, "migraphx", selectedProviderLabel, [graph]);
    }

    private static InferenceSession CreateMigraphxInferenceSession(string modelPath, ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Migraphx;
        using SessionOptions options = CreateSessionOptions(preference);
        ExecutionProviderKind selectedProvider;
        if (ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            selectedProvider = ExecutionProviderKind.Migraphx;
            notes.Add("Selected provider: catalog device policy (ORT auto-selects among registered catalog EPs).");
        }
        else
        {
            selectedProvider = MigraphxSessionOptionsExtensions.AppendMigraphxOrFallback(options);
            string selectedProviderLabel = FormatExecutionProvider(selectedProvider);
            if (selectedProvider is not ExecutionProviderKind.Migraphx)
            {
                throw new InvalidOperationException(
                    "MIGraphX execution provider is not visible to ONNX Runtime; refusing CPU fallback for explicit migraphx benchmark.");
            }
            notes.Add($"Selected provider: {selectedProviderLabel}.");
        }

        return new InferenceSession(modelPath, options);
    }

    private static BenchmarkSessionLease CreateQnnSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        notes.Add(GetCatalogGpuRouteNote(
            BenchmarkProviderPreference.Qnn,
            "explicit WinML catalog QNN execution provider."));
        var session = CreateQnnInferenceSession(graph.LoadPath, notes);
        string selectedProviderLabel = ResolveBenchmarkEffectiveProviderLabel(session, BenchmarkProviderPreference.Qnn);
        return new BenchmarkSessionLease(session, "qnn", selectedProviderLabel, [graph]);
    }

    private static InferenceSession CreateQnnInferenceSession(string modelPath, ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Qnn;
        using SessionOptions options = CreateSessionOptions(preference);
        if (ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            notes.Add("Selected provider: catalog device policy (ORT auto-selects among registered catalog EPs).");
        }
        else
        {
            ExecutionProviderKind selectedProvider = WinMlCatalog.WinMlCatalogSessionOptionsExtensions.AppendQnnOrFallback(options);
            if (selectedProvider is not ExecutionProviderKind.Qnn)
            {
                throw new InvalidOperationException(
                    "QNN execution provider is not visible to ONNX Runtime; refusing CPU fallback for explicit qnn benchmark.");
            }

            notes.Add($"Selected provider: {FormatExecutionProvider(selectedProvider)}.");
        }

        var session = new InferenceSession(modelPath, options);
        try
        {
            string effectiveProviderLabel = ResolveBenchmarkEffectiveProviderLabel(session, preference);
            if (!string.Equals(effectiveProviderLabel, "qnn", StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Explicit qnn benchmark resolved to '{effectiveProviderLabel}'; refusing to report it as qnn.");
            }

            return session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    private static BenchmarkSessionLease CreateNativeCudaSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        notes.Add("Provider route: explicit native ORT CUDA execution provider.");
        var session = CreateNativeCudaInferenceSession(graph.LoadPath, notes);
        return new BenchmarkSessionLease(session, "cuda", "cuda", [graph]);
    }

    private static BenchmarkSessionLease CreateNativeTensorRtSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        notes.Add("Provider route: explicit native ORT TensorRT execution provider.");
        var session = CreateNativeTensorRtInferenceSession(graph.LoadPath, notes);
        return new BenchmarkSessionLease(session, "tensorrt", "tensorrt", [graph]);
    }

    private static InferenceSession CreateNativeCudaInferenceSession(string modelPath, ICollection<string> notes)
    {
        using SessionOptions options = CreateSessionOptions(BenchmarkProviderPreference.Cuda);
        ExecutionProviderKind selectedProvider = AppendNativeCudaOrFallback(options);
        if (selectedProvider is not ExecutionProviderKind.Cuda)
        {
            throw new InvalidOperationException(
                $"{NativeCudaTensorRtWindowsProviderConstants.CudaOrtExecutionProviderName} is not available to ONNX Runtime; "
                + "refusing CPU fallback for explicit cuda benchmark.");
        }

        notes.Add("Selected provider: cuda.");
        return new InferenceSession(modelPath, options);
    }

    private static InferenceSession CreateNativeTensorRtInferenceSession(string modelPath, ICollection<string> notes)
    {
        using SessionOptions options = CreateSessionOptions(BenchmarkProviderPreference.TensorRt);
        ExecutionProviderKind selectedProvider = AppendNativeTensorRtOrFallback(options);
        if (selectedProvider is not ExecutionProviderKind.TensorRt)
        {
            throw new InvalidOperationException(
                $"{TensorRtRtxProviderConstants.NativeOrtExecutionProviderName} is not available to ONNX Runtime; "
                + "refusing CPU fallback for explicit tensorrt benchmark.");
        }

        notes.Add("Selected provider: tensorrt.");
        return new InferenceSession(modelPath, options);
    }

    private static ExecutionProviderKind AppendNativeCudaOrFallback(SessionOptions options) =>
        TryAppendNativeCudaProvider(options, out _)
            ? ExecutionProviderKind.Cuda
            : ExecutionProviderKind.Cpu;

    private static ExecutionProviderKind AppendNativeTensorRtOrFallback(SessionOptions options) =>
        TryAppendNativeTensorRtProvider(options, out _)
            ? ExecutionProviderKind.TensorRt
            : ExecutionProviderKind.Cpu;

    private static bool TryAppendNativeCudaProvider(SessionOptions options, out string? failureReason)
    {
        failureReason = null;
        if (!CudaOrtProbe.IsCudaProviderListed())
        {
            failureReason =
                $"{NativeCudaTensorRtWindowsProviderConstants.CudaOrtExecutionProviderName} is not listed by ONNX Runtime.";
            return false;
        }

        try
        {
            options.AppendExecutionProvider_CUDA(deviceId: 0);
            return true;
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            failureReason = ex.Message;
            return false;
        }
    }

    private static bool TryAppendNativeTensorRtProvider(SessionOptions options, out string? failureReason)
    {
        failureReason = null;
        if (!NativeTensorRtLibraryProbe.IsNativeTensorRtAvailable())
        {
            failureReason = "Native TensorRT libraries were not found.";
            return false;
        }

        if (!TensorRtRtxOrtProbe.IsNativeTensorRtProviderListed())
        {
            failureReason =
                $"{TensorRtRtxProviderConstants.NativeOrtExecutionProviderName} is not listed by ONNX Runtime.";
            return false;
        }

        try
        {
            options.AppendExecutionProvider_Tensorrt(deviceId: 0);
            return true;
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            failureReason = ex.Message;
            return false;
        }
    }

    private static BenchmarkSessionLease CreateAutoSession(BenchmarkGraphPlan graph, ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Auto;
        notes.Add("Provider route: auto selected.");

        if (ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            notes.Add("Auto route: catalog device policy (ORT auto-selects among registered catalog EPs).");
            using SessionOptions options = CreateSessionOptions(preference);
            var session = new InferenceSession(graph.LoadPath, options);
            string selectedProviderLabel = ResolveBenchmarkEffectiveProviderLabel(session, preference);
            return new BenchmarkSessionLease(session, "auto", selectedProviderLabel, [graph]);
        }

        try
        {
            using SessionOptions options = CreateSessionOptions(preference);
            OnnxExecutionSessionFactory.AppendDirectMlProvider(options);
            var session = new InferenceSession(graph.LoadPath, options);
            notes.Add("Auto route resolved to DirectML.");
            return new BenchmarkSessionLease(session, "auto", "dml", [graph]);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or EntryPointNotFoundException or DllNotFoundException)
        {
            notes.Add($"Auto route fell back to CPU because DirectML was unavailable: {ex.Message}");
            using SessionOptions cpuOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
            var session = new InferenceSession(graph.LoadPath, cpuOptions);
            return new BenchmarkSessionLease(session, "auto", "cpu", [graph]);
        }
    }

    private static SessionOptions CreateSessionOptions(BenchmarkProviderPreference preference)
    {
        SessionOptions options = new()
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };
#if WINDOWS
        if (ProviderNames.ShouldUseCatalogDevicePolicy(
                ActiveWindowsMlDevicePolicy,
                MapPreferenceToExecutionProvider(preference)))
        {
            WindowsMlExecutionDevicePolicyMapper.ApplyIfNeeded(options, ActiveWindowsMlDevicePolicy);
        }
#endif
        return options;
    }

    private static ProviderSession CreateTensorRtRtxProviderSession(
        BenchmarkGraphPlan graph,
        ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.TensorRtRtx;
        BenchmarkSessionPlan plan = ActiveSessionPlan.Value ?? BenchmarkSessionPlan.Default;
        using SessionOptions options = CreateSessionOptions(preference);
        if (ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            var session = new InferenceSession(graph.LoadPath, options);
            string effectiveProviderLabel = ResolveBenchmarkEffectiveProviderLabel(session, preference);
            notes.Add($"Selected provider: {effectiveProviderLabel} (catalog device policy).");
            return new ProviderSession(session, effectiveProviderLabel);
        }

        ExecutionProviderKind selectedProvider = OnnxExecutionSessionFactory.AppendTensorRtRtxOrFallbackProvider(
            options, graph.TrtOptions, plan.EnableCudaGraph);
        string selectedProviderLabel = FormatExecutionProvider(selectedProvider);
        if (selectedProvider is not ExecutionProviderKind.TensorRTRtx)
        {
            throw new InvalidOperationException(
                $"TensorRT RTX EP ABI plugin device is not visible in OrtEnv.GetEpDevices(); refusing fallback to {selectedProviderLabel} for explicit trt-rtx benchmark.");
        }

        return new ProviderSession(new InferenceSession(graph.LoadPath, options), selectedProviderLabel);
    }

    private static string FormatProviderPreference(BenchmarkProviderPreference preference) =>
        preference switch
        {
            BenchmarkProviderPreference.Auto => "auto",
            BenchmarkProviderPreference.Cpu => "cpu",
            BenchmarkProviderPreference.Dml => "dml",
            BenchmarkProviderPreference.TensorRtRtx => "trt-rtx",
            BenchmarkProviderPreference.Migraphx => "migraphx",
            BenchmarkProviderPreference.Cuda => "cuda",
            BenchmarkProviderPreference.TensorRt => "tensorrt",
            BenchmarkProviderPreference.Qnn => "qnn",
            _ => throw new ArgumentOutOfRangeException(nameof(preference), preference, "Unknown provider preference.")
        };

    private static string FormatExecutionProvider(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.Cpu => "cpu",
            ExecutionProviderKind.DirectMl => "dml",
            ExecutionProviderKind.Migraphx => "migraphx",
            ExecutionProviderKind.TensorRTRtx => "trt-rtx",
            ExecutionProviderKind.OpenVino => "openvino",
            ExecutionProviderKind.CoreMl => "coreml",
            ExecutionProviderKind.Cuda => "cuda",
            ExecutionProviderKind.TensorRt => "tensorrt",
            ExecutionProviderKind.Dnnl => "dnnl",
            ExecutionProviderKind.Qnn => "qnn",
            ExecutionProviderKind.OpenVinoCatalog => "openvino-catalog",
            ExecutionProviderKind.VitisAi => "vitisai",
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unknown execution provider.")
        };

    private static BenchmarkMeasurements EmptyMeasurements() =>
        new(
            ColdLoadMilliseconds: null,
            WarmupMilliseconds: null,
            WarmLatencyAverageMilliseconds: null,
            WarmLatencyMinimumMilliseconds: null,
            WarmLatencyMaximumMilliseconds: null,
            AudioDurationSeconds: null,
            RealTimeFactorAverage: null);

    private static double? CalculateRealTimeFactorAverage(
        double? audioDurationSeconds,
        double? averageLatencyMilliseconds) =>
        audioDurationSeconds is > 0 && averageLatencyMilliseconds is { } average
            ? (average / 1000d) / audioDurationSeconds.Value
            : null;

    private static string FormatModelProfile(BenchmarkModelProfile modelProfile) =>
        modelProfile switch
        {
            BenchmarkModelProfile.SileroVad => "silero-vad",
            BenchmarkModelProfile.WhisperEncoder => "whisper-encoder",
            BenchmarkModelProfile.OpusEncoder => "opus-encoder",
            _ => "generic"
        };

    private static BenchmarkReport CreateReport(
        string scenario,
        BenchmarkStatus status,
        string modelPath,
        string reportPath,
        string requestedProvider,
        string selectedProvider,
        int runCount,
        bool supportsExecution,
        long modelSizeBytes,
        BenchmarkMeasurements measurements,
        string? failureReason,
        IReadOnlyList<string> notes,
        BenchmarkRunMetadata? metadata = null) =>
        new(
            Scenario: scenario,
            ModelPath: modelPath,
            ReportPath: reportPath,
            Status: status,
            RequestedProvider: requestedProvider,
            SelectedProvider: selectedProvider,
            RunCount: runCount,
            SupportsExecution: supportsExecution,
            ModelSizeBytes: modelSizeBytes,
            Measurements: measurements,
            FailureReason: failureReason,
            Notes: notes,
            GeneratedAtUtc: DateTimeOffset.UtcNow,
            Metadata: metadata);

    private static BenchmarkInputSet CreateInputs(
        string modelPath,
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        BenchmarkExecutionSettings settings,
        string graphRole)
    {
        if (settings.InputMode is BenchmarkInputMode.Profile)
        {
            settings = settings with
            {
                InputShapeOverrides = ResolveProfileShapeOverrides(modelPath, settings.InputShapeOverrides)
            };
        }

        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        var modelProfile = DetermineModelProfile(modelPath, inputMetadata);
        double? audioDurationSeconds = null;

        foreach (var pair in inputMetadata)
        {
            values.Add(CreateInputValue(modelProfile, pair.Key, pair.Value, settings, graphRole));
        }

        if (modelProfile is BenchmarkModelProfile.SileroVad)
        {
            audioDurationSeconds = 512d / 16000d;
        }
        else if (modelProfile is BenchmarkModelProfile.WhisperEncoder)
        {
            audioDurationSeconds = ResolveWhisperAudioDurationSeconds(inputMetadata);
        }

        return new BenchmarkInputSet(values, audioDurationSeconds, modelProfile);
    }

    /// <summary>
    /// Merges registered production profile opt shapes with explicit caller overrides; the explicit
    /// flags win. Fails honestly when the model family has no registered profile.
    /// </summary>
    private static IReadOnlyDictionary<string, int[]> ResolveProfileShapeOverrides(
        string modelPath,
        IReadOnlyDictionary<string, int[]>? explicitOverrides)
    {
        if (EpContextTrtProfiles.ResolveOptShapes(modelPath) is not { } profileShapes)
        {
            throw new NotSupportedException(
                $"Input mode 'profile' requires a registered trt-rtx optimization profile; none is registered for '{modelPath}'. "
                + "Use --input-shape overrides or another input mode.");
        }

        var merged = new Dictionary<string, int[]>(profileShapes, StringComparer.Ordinal);
        if (explicitOverrides is not null)
        {
            foreach ((string key, int[] value) in explicitOverrides)
            {
                merged[key] = value;
            }
        }

        return merged;
    }

    private static NamedOnnxValue CreateInputValue(
        BenchmarkModelProfile modelProfile,
        string inputName,
        NodeMetadata metadata,
        BenchmarkExecutionSettings settings,
        string graphRole)
    {
        if (!metadata.IsTensor)
        {
            throw new NotSupportedException($"Input '{inputName}' is not a tensor input.");
        }

        return metadata.ElementDataType switch
        {
            TensorElementType.Float => NamedOnnxValue.CreateFromTensor(inputName, CreateFloatTensor(modelProfile, inputName, metadata, settings, graphRole)),
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(inputName, CreateFloat16Tensor(modelProfile, inputName, metadata, settings, graphRole)),
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(inputName, CreateInt32Tensor(modelProfile, inputName, metadata, settings)),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(inputName, CreateInt64Tensor(modelProfile, inputName, metadata, settings)),
            _ => throw new NotSupportedException($"Input '{inputName}' uses unsupported tensor element type '{metadata.ElementDataType}'.")
        };
    }

    private static float ResolveSyntheticScale(BenchmarkModelProfile modelProfile, string inputName) =>
        modelProfile is BenchmarkModelProfile.SileroVad
        && inputName.Equals("input", StringComparison.Ordinal)
            ? 0.1f
            : 1f;

    private static DenseTensor<float> CreateFloatTensor(
        BenchmarkModelProfile modelProfile,
        string inputName,
        NodeMetadata metadata,
        BenchmarkExecutionSettings settings,
        string graphRole)
    {
        var dimensions = ResolveDimensions(modelProfile, inputName, metadata, settings);
        var count = CountElements(dimensions);
        var data = new float[count];

        if (settings.InputMode is BenchmarkInputMode.Synthetic)
        {
            BenchmarkInputGenerator.Fill(
                data, settings.InputSeed, graphRole, inputName, ResolveSyntheticScale(modelProfile, inputName));
        }
        else if (modelProfile is BenchmarkModelProfile.SileroVad && inputName.Equals("input", StringComparison.Ordinal))
        {
            if (data.Length > 0)
            {
                data[0] = 0.1f;
            }
        }
        else if (modelProfile is BenchmarkModelProfile.WhisperEncoder && IsWhisperAudioFeaturesInput(inputName))
        {
            for (var index = 0; index < data.Length; index++)
            {
                data[index] = (index % 11) * 0.01f;
            }
        }

        return new DenseTensor<float>(data, dimensions);
    }

    private static DenseTensor<Float16> CreateFloat16Tensor(
        BenchmarkModelProfile modelProfile,
        string inputName,
        NodeMetadata metadata,
        BenchmarkExecutionSettings settings,
        string graphRole)
    {
        var dimensions = ResolveDimensions(modelProfile, inputName, metadata, settings);
        var count = CountElements(dimensions);
        var data = new Float16[count];

        if (settings.InputMode is BenchmarkInputMode.Synthetic)
        {
            var floats = new float[count];
            BenchmarkInputGenerator.Fill(
                floats, settings.InputSeed, graphRole, inputName, ResolveSyntheticScale(modelProfile, inputName));
            for (var index = 0; index < data.Length; index++)
            {
                data[index] = (Float16)floats[index];
            }
        }
        else if (modelProfile is BenchmarkModelProfile.WhisperEncoder && IsWhisperAudioFeaturesInput(inputName))
        {
            for (var index = 0; index < data.Length; index++)
            {
                data[index] = (Float16)((index % 11) * 0.01f);
            }
        }

        return new DenseTensor<Float16>(data, dimensions);
    }

    private static DenseTensor<long> CreateInt64Tensor(
        BenchmarkModelProfile modelProfile,
        string inputName,
        NodeMetadata metadata,
        BenchmarkExecutionSettings settings)
    {
        var dimensions = ResolveDimensions(modelProfile, inputName, metadata, settings);
        var count = CountElements(dimensions);
        var data = new long[count];

        if (modelProfile is BenchmarkModelProfile.SileroVad && inputName.Equals("sr", StringComparison.Ordinal) && data.Length > 0)
        {
            data[0] = 16000L;
        }
        else if (modelProfile is BenchmarkModelProfile.OpusEncoder)
        {
            FillOpusEncoderTensor(inputName, data);
        }

        return new DenseTensor<long>(data, dimensions);
    }

    private static DenseTensor<int> CreateInt32Tensor(
        BenchmarkModelProfile modelProfile,
        string inputName,
        NodeMetadata metadata,
        BenchmarkExecutionSettings settings)
    {
        var dimensions = ResolveDimensions(modelProfile, inputName, metadata, settings);
        var count = CountElements(dimensions);
        var data = new int[count];

        if (modelProfile is BenchmarkModelProfile.OpusEncoder)
        {
            var longData = new long[count];
            FillOpusEncoderTensor(inputName, longData);
            for (var index = 0; index < data.Length; index++)
            {
                data[index] = checked((int)longData[index]);
            }
        }

        return new DenseTensor<int>(data, dimensions);
    }

    private static int[] ResolveDimensions(
        BenchmarkModelProfile modelProfile,
        string inputName,
        NodeMetadata metadata,
        BenchmarkExecutionSettings settings)
    {
        if (settings.InputShapeOverrides is { } overrides
            && overrides.TryGetValue(inputName, out int[]? overrideShape))
        {
            return (int[])overrideShape.Clone();
        }

        if (modelProfile is BenchmarkModelProfile.SileroVad)
        {
            if (inputName.Equals("input", StringComparison.Ordinal))
            {
                return [1, 512];
            }

            if (inputName.Equals("state", StringComparison.Ordinal))
            {
                return [2, 1, 128];
            }

            if (inputName.Equals("sr", StringComparison.Ordinal))
            {
                return [1];
            }
        }
        else if (modelProfile is BenchmarkModelProfile.WhisperEncoder && IsWhisperAudioFeaturesInput(inputName))
        {
            return [1, 80, 3000];
        }
        else if (modelProfile is BenchmarkModelProfile.OpusEncoder)
        {
            return inputName switch
            {
                "input_ids" => [1, 8],
                "attention_mask" => [1, 8],
                _ => metadata.Dimensions.Select(ToPositiveDimension).ToArray()
            };
        }

        return metadata.Dimensions.Select(ToPositiveDimension).ToArray();
    }

    private static BenchmarkModelProfile DetermineModelProfile(
        string modelPath,
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        if (LooksLikeSileroVad(modelPath, inputMetadata))
        {
            return BenchmarkModelProfile.SileroVad;
        }

        if (LooksLikeWhisperEncoder(modelPath))
        {
            return BenchmarkModelProfile.WhisperEncoder;
        }

        if (LooksLikeOpusEncoder(modelPath))
        {
            return BenchmarkModelProfile.OpusEncoder;
        }

        return BenchmarkModelProfile.Generic;
    }

    private static bool LooksLikeSileroVad(
        string modelPath,
        IReadOnlyDictionary<string, NodeMetadata>? inputMetadata)
    {
        var normalizedPath = modelPath.Replace('/', '\\');
        if (normalizedPath.Contains("silero", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (inputMetadata is null)
        {
            return false;
        }

        return inputMetadata.ContainsKey("input")
            && inputMetadata.ContainsKey("state")
            && inputMetadata.ContainsKey("sr");
    }

    private static string ResolveModelPath(string suppliedPath, ICollection<string> notes)
    {
        BenchmarkModelCandidate resolution = BenchmarkModelPathResolver.CreateDefault().ResolveSingle(suppliedPath);
        notes.Add(resolution.ResolutionNote);
        return resolution.ModelPath;
    }

    private static bool LooksLikeWhisperEncoder(string modelPath)
    {
        var fileName = Path.GetFileName(modelPath);
        var parentDirectory = Path.GetDirectoryName(modelPath) ?? string.Empty;
        return (fileName.StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase)
                || fileName.Equals("encoder.onnx", StringComparison.OrdinalIgnoreCase))
            && parentDirectory.Contains("whisper", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsWhisperAudioFeaturesInput(string inputName)
    {
        return inputName.Equals("input_features", StringComparison.Ordinal)
            || inputName.Equals("audio_features", StringComparison.Ordinal);
    }

    private static bool LooksLikeOpusEncoder(string modelPath)
    {
        var fileName = Path.GetFileName(modelPath);
        var parentDirectory = Path.GetDirectoryName(modelPath) ?? string.Empty;
        return fileName.StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase)
            && parentDirectory.Contains("opus", StringComparison.OrdinalIgnoreCase);
    }

    private static string ResolveWhisperDecoderPath(string encoderModelPath)
    {
        var modelDirectory = Path.GetDirectoryName(encoderModelPath)!;
        foreach (var fileName in new[] { "decoder_model.onnx", "decoder.onnx" })
        {
            var decoderModelPath = Path.Join(modelDirectory, fileName);
            if (File.Exists(decoderModelPath))
            {
                return Path.GetFullPath(decoderModelPath);
            }
        }

        throw new FileNotFoundException(
            "Whisper decoder model was not found next to the encoder model.",
            Path.Join(modelDirectory, "decoder_model.onnx"));
    }

    private static int ResolveWhisperDecoderStartTokenId(string configPath)
    {
        if (!File.Exists(configPath))
        {
            return 50258;
        }

        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        if (document.RootElement.TryGetProperty("decoder_start_token_id", out var tokenIdElement) &&
            tokenIdElement.TryGetInt32(out var tokenId))
        {
            return tokenId;
        }

        return 50258;
    }

    private static string ResolveOpusDecoderPath(string encoderModelPath)
    {
        var decoderModelPath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "decoder_model.onnx");
        if (File.Exists(decoderModelPath))
        {
            return Path.GetFullPath(decoderModelPath);
        }

        var mergedDecoderModelPath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "decoder_model_merged.onnx");
        if (!File.Exists(mergedDecoderModelPath))
        {
            throw new FileNotFoundException("Opus decoder model was not found next to the encoder model.", decoderModelPath);
        }

        return Path.GetFullPath(mergedDecoderModelPath);
    }

    private static int ResolveOpusDecoderStartTokenId(string configPath)
    {
        if (!File.Exists(configPath))
        {
            return 65000;
        }

        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        if (document.RootElement.TryGetProperty("decoder_start_token_id", out var tokenIdElement) &&
            tokenIdElement.TryGetInt32(out var tokenId))
        {
            return tokenId;
        }

        return 65000;
    }

    private static double? ResolveWhisperAudioDurationSeconds(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        if (!inputMetadata.TryGetValue("input_features", out var metadata)
            && !inputMetadata.TryGetValue("audio_features", out metadata))
        {
            return null;
        }

        var dimensions = metadata.Dimensions.Select(ToPositiveDimension).ToArray();
        if (dimensions.Length < 3)
        {
            return null;
        }

        return dimensions[2] / 100d;
    }

    private static void RunWhisperPass(
        WhisperSessionLease lease,
        BenchmarkInputSet encoderInputSet,
        int decoderStartTokenId)
    {
        using var encoderResults = lease.EncoderSession.Run(encoderInputSet.Values);
        using var decoderInputs = CreateWhisperDecoderInputs(
            lease.DecoderSession.InputMetadata,
            encoderResults,
            decoderStartTokenId);
        using var decoderResults = lease.DecoderSession.Run(decoderInputs.Values);
    }

    private static void RunOpusPass(
        OpusSessionLease lease,
        BenchmarkInputSet encoderInputSet,
        int decoderStartTokenId)
    {
        using var encoderResults = lease.EncoderSession.Run(encoderInputSet.Values);
        DisposableNamedOnnxValue encoderHidden = encoderResults.FirstOrDefault()
            ?? throw new InvalidOperationException("Opus encoder produced no outputs.");
        Tensor<float>? encoderHiddenStates = encoderHidden.AsTensor<float>();
        if (encoderHiddenStates is null)
        {
            throw new InvalidOperationException(
                $"Opus encoder output '{encoderHidden.Name}' is not a float tensor.");
        }

        using var decoderInputs = CreateOpusDecoderInputs(
            lease.DecoderSession.InputMetadata,
            encoderHiddenStates,
            decoderStartTokenId,
            encoderInputSet.Values);
        using var decoderResults = lease.DecoderSession.Run(decoderInputs.Values);
    }

    private static DecoderInputSet CreateWhisperDecoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        IEnumerable<DisposableNamedOnnxValue> encoderResults,
        int decoderStartTokenId)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        var encoderOutputByName = encoderResults.ToDictionary(static value => value.Name, StringComparer.Ordinal);

        foreach (var pair in inputMetadata)
        {
            values.Add(pair.Key switch
            {
                "input_ids" => CreateWhisperDecoderInputIds(pair.Key, pair.Value, decoderStartTokenId),
                "encoder_hidden_states" => CreateWhisperDecoderTensorFromEncoderOutput(
                    pair.Key,
                    pair.Value,
                    encoderOutputByName),
                _ when TryResolveWhisperEncoderOutputName(pair.Key, out _) => CreateWhisperDecoderTensorFromEncoderOutput(
                    pair.Key,
                    pair.Value,
                    encoderOutputByName),
                _ when IsWhisperSelfPastInput(pair.Key) => CreateEmptyPastInput(pair.Key, pair.Value),
                _ => throw new NotSupportedException($"Whisper decoder input '{pair.Key}' is not supported by the benchmark harness yet.")
            });
        }

        return new DecoderInputSet(values);
    }

    private static NamedOnnxValue CreateWhisperDecoderInputIds(
        string inputName,
        NodeMetadata metadata,
        int decoderStartTokenId)
    {
        if (!metadata.IsTensor)
        {
            throw new NotSupportedException($"Whisper decoder input '{inputName}' is not a tensor input.");
        }

        return metadata.ElementDataType switch
        {
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<int>(new[] { decoderStartTokenId }, [1, 1])),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<long>(new[] { (long)decoderStartTokenId }, [1, 1])),
            _ => throw new NotSupportedException(
                $"Whisper decoder input '{inputName}' uses unsupported token tensor element type '{metadata.ElementDataType}'.")
        };
    }

    private static NamedOnnxValue CreateWhisperDecoderTensorFromEncoderOutput(
        string inputName,
        NodeMetadata metadata,
        IReadOnlyDictionary<string, DisposableNamedOnnxValue> encoderOutputByName)
    {
        if (!metadata.IsTensor)
        {
            throw new NotSupportedException($"Whisper decoder input '{inputName}' is not a tensor input.");
        }

        if (!TryResolveWhisperEncoderOutputName(inputName, out var outputName)
            || !encoderOutputByName.TryGetValue(outputName, out var encoderOutput))
        {
            throw new NotSupportedException(
                $"Whisper decoder input '{inputName}' could not be matched to an encoder output.");
        }

        return metadata.ElementDataType switch
        {
            TensorElementType.Float => NamedOnnxValue.CreateFromTensor(inputName, encoderOutput.AsTensor<float>()),
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(inputName, encoderOutput.AsTensor<Float16>()),
            _ => throw new NotSupportedException(
                $"Whisper decoder input '{inputName}' uses unsupported encoder-output tensor element type '{metadata.ElementDataType}'.")
        };
    }

    private static NamedOnnxValue CreateEmptyPastInput(string inputName, NodeMetadata metadata)
    {
        if (!metadata.IsTensor) { throw new NotSupportedException($"Decoder input '{inputName}' is not a tensor input."); }

        var dimensions = ResolveEmptyPastDimensions(metadata);
        return metadata.ElementDataType switch
        {
            TensorElementType.Float => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<float>(new float[CountElements(dimensions)], dimensions)),
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<Float16>(new Float16[CountElements(dimensions)], dimensions)),
            _ => throw new NotSupportedException(
                $"Decoder input '{inputName}' uses unsupported past-cache tensor element type '{metadata.ElementDataType}'.")
        };
    }

    private static bool TryResolveWhisperEncoderOutputName(string decoderInputName, out string encoderOutputName)
    {
        if (decoderInputName.Equals("encoder_hidden_states", StringComparison.Ordinal))
        {
            encoderOutputName = "hidden_states";
            return true;
        }

        if (decoderInputName.StartsWith("past_key_cross_", StringComparison.Ordinal))
        {
            encoderOutputName = "present_key_cross_" + decoderInputName["past_key_cross_".Length..];
            return true;
        }

        if (decoderInputName.StartsWith("past_value_cross_", StringComparison.Ordinal))
        {
            encoderOutputName = "present_value_cross_" + decoderInputName["past_value_cross_".Length..];
            return true;
        }

        encoderOutputName = string.Empty;
        return false;
    }

    private static bool IsWhisperSelfPastInput(string inputName)
    {
        return inputName.StartsWith("past_key_self_", StringComparison.Ordinal)
            || inputName.StartsWith("past_value_self_", StringComparison.Ordinal);
    }

    private static int[] ResolveEmptyPastDimensions(NodeMetadata metadata)
    {
        var dimensions = metadata.Dimensions.Select(ToPositiveDimension).ToArray();
        if (dimensions.Length >= 3)
        {
            dimensions[2] = 0;
        }

        return dimensions;
    }

    private static DecoderInputSet CreateOpusDecoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        Tensor<float> encoderHiddenStates,
        int decoderStartTokenId,
        IReadOnlyList<NamedOnnxValue> encoderInputs)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        var encoderAttentionMask = encoderInputs
            .FirstOrDefault(static value => value.Name.Equals("attention_mask", StringComparison.Ordinal))
            ?.AsTensor<long>();

        foreach (var pair in inputMetadata)
        {
            values.Add(pair.Key switch
            {
                "input_ids" => NamedOnnxValue.CreateFromTensor(
                    "input_ids",
                    new DenseTensor<long>(new long[] { decoderStartTokenId }, [1, 1])),
                "encoder_hidden_states" => NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                "attention_mask" when encoderAttentionMask is not null => NamedOnnxValue.CreateFromTensor("attention_mask", encoderAttentionMask),
                "encoder_attention_mask" when encoderAttentionMask is not null => NamedOnnxValue.CreateFromTensor("encoder_attention_mask", encoderAttentionMask),
                // Merged Opus decoders take the no-cache branch first: empty past, use_cache_branch=false.
                "use_cache_branch" => NamedOnnxValue.CreateFromTensor("use_cache_branch", new DenseTensor<bool>(new[] { false }, [1])),
                _ when pair.Key.StartsWith("past_key_values.", StringComparison.Ordinal) => CreateEmptyPastInput(pair.Key, pair.Value),
                _ => throw new NotSupportedException($"Opus decoder input '{pair.Key}' is not supported by the benchmark harness yet.")
            });
        }

        return new DecoderInputSet(values);
    }

    private static WhisperSessionLease CreateWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        BenchmarkProviderPreference preference,
        ICollection<string> notes) =>
        preference switch
        {
            BenchmarkProviderPreference.Cpu => CreateCpuWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Dml => CreateDirectMlWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.TensorRtRtx => CreateTensorRtRtxWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Migraphx => CreateMigraphxWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Cuda => CreateNativeCudaWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.TensorRt => CreateNativeTensorRtWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Qnn => CreateQnnWhisperSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Auto => CreateAutoWhisperSessionLease(encoderGraph, decoderGraph, notes),
            _ => throw new ArgumentOutOfRangeException(nameof(preference), preference, "Unknown provider preference.")
        };

    private static OpusSessionLease CreateOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        BenchmarkProviderPreference preference,
        ICollection<string> notes) =>
        preference switch
        {
            BenchmarkProviderPreference.Cpu => CreateCpuOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Dml => CreateDirectMlOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.TensorRtRtx => CreateTensorRtRtxOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Migraphx => CreateMigraphxOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Cuda => CreateNativeCudaOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.TensorRt => CreateNativeTensorRtOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Qnn => CreateQnnOpusSessionLease(encoderGraph, decoderGraph, notes),
            BenchmarkProviderPreference.Auto => CreateAutoOpusSessionLease(encoderGraph, decoderGraph, notes),
            _ => throw new ArgumentOutOfRangeException(nameof(preference), preference, "Unknown provider preference.")
        };

    private static WhisperSessionLease CreateCpuWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit CPU execution provider.");
        using SessionOptions encoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
        using SessionOptions decoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
        return new WhisperSessionLease(
            new InferenceSession(encoderGraph.LoadPath, encoderOptions),
            new InferenceSession(decoderGraph.LoadPath, decoderOptions),
            "cpu",
            "cpu",
            [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateDirectMlWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Dml;
        notes.Add(GetCatalogGpuRouteNote(preference, "explicit WinML catalog DirectML execution provider."));
        var encoderSession = CreateDirectMlSession(encoderGraph);
        var decoderSession = CreateDirectMlSession(decoderGraph);
        string selectedProviderLabel = ResolvePairSelectedProvider(
            ResolveBenchmarkEffectiveProviderLabel(encoderSession, preference),
            ResolveBenchmarkEffectiveProviderLabel(decoderSession, preference),
            notes);
        return new WhisperSessionLease(encoderSession, decoderSession, "dml", selectedProviderLabel, [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateTensorRtRtxWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit TensorRT RTX EP ABI plugin execution provider.");
        ProviderSession encoderSession = CreateTensorRtRtxProviderSession(encoderGraph, notes);
        ProviderSession decoderSession = CreateTensorRtRtxProviderSession(decoderGraph, notes);
        return new WhisperSessionLease(
            encoderSession.Session,
            decoderSession.Session,
            "trt-rtx",
            ResolvePairSelectedProvider(encoderSession.SelectedProvider, decoderSession.SelectedProvider, notes),
            [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateNativeCudaWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit native ORT CUDA execution provider.");
        return new WhisperSessionLease(
            CreateNativeCudaInferenceSession(encoderGraph.LoadPath, notes),
            CreateNativeCudaInferenceSession(decoderGraph.LoadPath, notes),
            "cuda",
            "cuda",
            [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateNativeTensorRtWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit native ORT TensorRT execution provider.");
        return new WhisperSessionLease(
            CreateNativeTensorRtInferenceSession(encoderGraph.LoadPath, notes),
            CreateNativeTensorRtInferenceSession(decoderGraph.LoadPath, notes),
            "tensorrt",
            "tensorrt",
            [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateMigraphxWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit MIGraphX execution provider.");
        return new WhisperSessionLease(
            CreateMigraphxInferenceSession(encoderGraph.LoadPath, notes),
            CreateMigraphxInferenceSession(decoderGraph.LoadPath, notes),
            "migraphx",
            "migraphx",
            [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateQnnWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit WinML catalog QNN execution provider.");
        return new WhisperSessionLease(
            CreateQnnInferenceSession(encoderGraph.LoadPath, notes),
            CreateQnnInferenceSession(decoderGraph.LoadPath, notes),
            "qnn",
            "qnn",
            [encoderGraph, decoderGraph]);
    }

    private static WhisperSessionLease CreateAutoWhisperSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: auto selected.");

        try
        {
            var encoderSession = CreateDirectMlSession(encoderGraph);
            var decoderSession = CreateDirectMlSession(decoderGraph);
            notes.Add("Auto route resolved to DirectML.");
            return new WhisperSessionLease(encoderSession, decoderSession, "auto", "dml", [encoderGraph, decoderGraph]);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or EntryPointNotFoundException or DllNotFoundException)
        {
            notes.Add($"Auto route fell back to CPU because DirectML was unavailable: {ex.Message}");
            using SessionOptions encoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
            using SessionOptions decoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
            return new WhisperSessionLease(
                new InferenceSession(encoderGraph.LoadPath, encoderOptions),
                new InferenceSession(decoderGraph.LoadPath, decoderOptions),
                "auto",
                "cpu",
                [encoderGraph, decoderGraph]);
        }
    }

    private static OpusSessionLease CreateCpuOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit CPU execution provider.");
        using SessionOptions encoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
        using SessionOptions decoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
        return new OpusSessionLease(
            new InferenceSession(encoderGraph.LoadPath, encoderOptions),
            new InferenceSession(decoderGraph.LoadPath, decoderOptions),
            "cpu",
            "cpu",
            [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateDirectMlOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Dml;
        notes.Add(GetCatalogGpuRouteNote(preference, "explicit WinML catalog DirectML execution provider."));
        var encoderSession = CreateDirectMlSession(encoderGraph);
        var decoderSession = CreateDirectMlSession(decoderGraph);
        string selectedProviderLabel = ResolvePairSelectedProvider(
            ResolveBenchmarkEffectiveProviderLabel(encoderSession, preference),
            ResolveBenchmarkEffectiveProviderLabel(decoderSession, preference),
            notes);
        return new OpusSessionLease(encoderSession, decoderSession, "dml", selectedProviderLabel, [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateTensorRtRtxOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit TensorRT RTX EP ABI plugin execution provider.");
        ProviderSession encoderSession = CreateTensorRtRtxProviderSession(encoderGraph, notes);
        ProviderSession decoderSession = CreateTensorRtRtxProviderSession(decoderGraph, notes);
        return new OpusSessionLease(
            encoderSession.Session,
            decoderSession.Session,
            "trt-rtx",
            ResolvePairSelectedProvider(encoderSession.SelectedProvider, decoderSession.SelectedProvider, notes),
            [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateMigraphxOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit MIGraphX execution provider.");
        return new OpusSessionLease(
            CreateMigraphxInferenceSession(encoderGraph.LoadPath, notes),
            CreateMigraphxInferenceSession(decoderGraph.LoadPath, notes),
            "migraphx",
            "migraphx",
            [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateQnnOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit WinML catalog QNN execution provider.");
        return new OpusSessionLease(
            CreateQnnInferenceSession(encoderGraph.LoadPath, notes),
            CreateQnnInferenceSession(decoderGraph.LoadPath, notes),
            "qnn",
            "qnn",
            [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateNativeCudaOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit native ORT CUDA execution provider.");
        return new OpusSessionLease(
            CreateNativeCudaInferenceSession(encoderGraph.LoadPath, notes),
            CreateNativeCudaInferenceSession(decoderGraph.LoadPath, notes),
            "cuda",
            "cuda",
            [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateNativeTensorRtOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: explicit native ORT TensorRT execution provider.");
        return new OpusSessionLease(
            CreateNativeTensorRtInferenceSession(encoderGraph.LoadPath, notes),
            CreateNativeTensorRtInferenceSession(decoderGraph.LoadPath, notes),
            "tensorrt",
            "tensorrt",
            [encoderGraph, decoderGraph]);
    }

    private static OpusSessionLease CreateAutoOpusSessionLease(
        BenchmarkGraphPlan encoderGraph,
        BenchmarkGraphPlan decoderGraph,
        ICollection<string> notes)
    {
        notes.Add("Provider route: auto selected.");

        try
        {
            var encoderSession = CreateDirectMlSession(encoderGraph);
            var decoderSession = CreateDirectMlSession(decoderGraph);
            notes.Add("Auto route resolved to DirectML.");
            return new OpusSessionLease(encoderSession, decoderSession, "auto", "dml", [encoderGraph, decoderGraph]);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or EntryPointNotFoundException or DllNotFoundException)
        {
            notes.Add($"Auto route fell back to CPU because DirectML was unavailable: {ex.Message}");
            using SessionOptions encoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
            using SessionOptions decoderOptions = CreateSessionOptions(BenchmarkProviderPreference.Cpu);
            return new OpusSessionLease(
                new InferenceSession(encoderGraph.LoadPath, encoderOptions),
                new InferenceSession(decoderGraph.LoadPath, decoderOptions),
                "auto",
                "cpu",
                [encoderGraph, decoderGraph]);
        }
    }

    private static InferenceSession CreateDirectMlSession(BenchmarkGraphPlan graph)
    {
        const BenchmarkProviderPreference preference = BenchmarkProviderPreference.Dml;
        using SessionOptions options = CreateSessionOptions(preference);
        if (!ShouldUseCatalogDevicePolicyForPreference(preference))
        {
            OnnxExecutionSessionFactory.AppendDirectMlProvider(options);
        }

        return new InferenceSession(graph.LoadPath, options);
    }

    private static string ResolvePairSelectedProvider(
        string firstSelectedProvider,
        string secondSelectedProvider,
        ICollection<string> notes)
    {
        if (firstSelectedProvider.Equals(secondSelectedProvider, StringComparison.Ordinal))
        {
            return firstSelectedProvider;
        }

        notes.Add($"Paired model sessions resolved to mixed providers ({firstSelectedProvider}, {secondSelectedProvider}); reporting cpu.");
        return "cpu";
    }

    private static void FillOpusEncoderTensor(string inputName, long[] data)
    {
        if (data.Length == 0)
        {
            return;
        }

        if (inputName.Equals("attention_mask", StringComparison.Ordinal))
        {
            Array.Fill(data, 1L);
            return;
        }

        if (inputName.Equals("input_ids", StringComparison.Ordinal))
        {
            long[] sampleTokens = [250, 142, 77, 901, 54, 12, 0, 65000];
            for (var index = 0; index < data.Length; index++)
            {
                data[index] = sampleTokens[index % sampleTokens.Length];
            }
        }
    }

    private static string ResolveFailureScenario(string modelPath)
    {
        if (LooksLikeWhisperEncoder(modelPath))
        {
            return "whisper-encoder-decoder";
        }

        if (LooksLikeOpusEncoder(modelPath))
        {
            return "opus-mt-encoder-decoder";
        }

        return "onnx-model";
    }

    private static int ToPositiveDimension(int dimension) => dimension <= 0 ? 1 : dimension;

    private static int CountElements(IEnumerable<int> dimensions)
    {
        var count = 1;
        foreach (var dimension in dimensions)
        {
            checked
            {
                count *= dimension;
            }
        }

        return count;
    }

    private sealed record BenchmarkInputSet(
        IReadOnlyList<NamedOnnxValue> Values,
        double? AudioDurationSeconds,
        BenchmarkModelProfile ModelProfile = BenchmarkModelProfile.Generic);

    private sealed class DecoderInputSet : IDisposable
    {
        public DecoderInputSet(IReadOnlyList<NamedOnnxValue> values)
        {
            Values = values;
        }

        public IReadOnlyList<NamedOnnxValue> Values { get; }

        public void Dispose()
        {
            foreach (var value in Values.OfType<IDisposable>())
            {
                value.Dispose();
            }
        }
    }

    private sealed record BenchmarkSessionLease(
        InferenceSession Session,
        string RequestedProvider,
        string SelectedProvider,
        IReadOnlyList<BenchmarkGraphPlan> Graphs) : IDisposable
    {
        public void Dispose() => Session.Dispose();
    }

    private sealed record ProviderSession(
        InferenceSession Session,
        string SelectedProvider);

    private sealed record WhisperSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderSession,
        string RequestedProvider,
        string SelectedProvider,
        IReadOnlyList<BenchmarkGraphPlan> Graphs) : IDisposable
    {
        public void Dispose()
        {
            DecoderSession.Dispose();
            EncoderSession.Dispose();
        }
    }

    private sealed record OpusSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderSession,
        string RequestedProvider,
        string SelectedProvider,
        IReadOnlyList<BenchmarkGraphPlan> Graphs) : IDisposable
    {
        public void Dispose()
        {
            DecoderSession.Dispose();
            EncoderSession.Dispose();
        }
    }

    private sealed record BenchmarkExecution(
        string Scenario,
        string RequestedProvider,
        string SelectedProvider,
        long ModelSizeBytes,
        double SessionInitializationMilliseconds,
        ColdLoadAttribution Attribution,
        BenchmarkSamplingResult Sampling,
        BenchmarkThroughputUnit ThroughputUnit,
        double? AudioDurationSeconds,
        IReadOnlyList<BenchmarkGraphMetadata> Graphs,
        IReadOnlyList<string> MetadataWarnings,
        BenchmarkModelProfile ModelProfile = BenchmarkModelProfile.Generic,
        IReadOnlyList<BenchmarkCudaGraphLaneResult>? CudaGraphLanes = null);

    private enum BenchmarkModelProfile
    {
        Generic,
        SileroVad,
        WhisperEncoder,
        OpusEncoder
    }
}
