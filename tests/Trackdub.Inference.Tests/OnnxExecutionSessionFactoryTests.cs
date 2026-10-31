using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Pool;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using System.Reflection;

namespace Trackdub.Inference.Tests;

[Collection(nameof(OnnxExecutionSessionFactoryStateCollection))]
public sealed class OnnxExecutionSessionFactoryTests
{
    private static string CompleteTensorRtFingerprint(
        string architecture = "Ampere",
        string model = "RTX-4090",
        string driver = "535.104.05",
        char tensorRtHashCharacter = 'A') =>
        $"{architecture}|{model}|{driver}|trt-sha256-{new string(tensorRtHashCharacter, 64)}|ort-1.31.0";

    private static string ClassicTensorRtCacheKey(string fingerprint)
    {
        byte[] digest = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(fingerprint));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    [Fact]
    public async Task CreatePooledSingleAsync_reuses_pool_hit_without_invoking_session_factory_again()
    {
        using var pool = new InferenceSessionPool(maxSessions: 2);
        int createCount = 0;

        using (await OnnxExecutionSessionFactory.CreatePooledSingleAsync(
            "test-engine",
            "single-model.onnx",
            ExecutionProviderKind.Cpu,
            CancellationToken.None,
            pool,
            sessionFactory: CreateCountingSession))
        {
        }

        using (await OnnxExecutionSessionFactory.CreatePooledSingleAsync(
            "test-engine",
            "single-model.onnx",
            ExecutionProviderKind.Cpu,
            CancellationToken.None,
            pool,
            sessionFactory: CreateCountingSession))
        {
        }

        Assert.Equal(1, createCount);

        InferenceSession CreateCountingSession(string modelPath, SessionOptions options)
        {
            createCount++;
            return CreateMinimalSession();
        }
    }

    [Fact]
    public async Task CreatePooledSingleAsync_separates_trt_init_fallback_policies()
    {
        using var pool = new InferenceSessionPool(maxSessions: 2);
        int createCount = 0;

        using (await OnnxExecutionSessionFactory.CreatePooledSingleAsync(
            "test-engine",
            "single-model.onnx",
            ExecutionProviderKind.Cpu,
            CancellationToken.None,
            pool,
            sessionFactory: CreateCountingSession,
            allowTrtInitFallback: true))
        {
        }

        using (await OnnxExecutionSessionFactory.CreatePooledSingleAsync(
            "test-engine",
            "single-model.onnx",
            ExecutionProviderKind.Cpu,
            CancellationToken.None,
            pool,
            sessionFactory: CreateCountingSession,
            allowTrtInitFallback: false))
        {
        }

        Assert.Equal(2, createCount);

        InferenceSession CreateCountingSession(string modelPath, SessionOptions options)
        {
            createCount++;
            return CreateMinimalSession();
        }
    }

    [Fact]
    public async Task CreatePooledOpusAsync_reuses_pair_pool_hits_without_invoking_session_factory_again()
    {
        using var pool = new InferenceSessionPool(maxSessions: 4);
        int createCount = 0;

        using (await OnnxExecutionSessionFactory.CreatePooledOpusAsync(
            "test-engine",
            "encoder-model.onnx",
            "decoder-model.onnx",
            ExecutionProviderKind.Cpu,
            CancellationToken.None,
            pool,
            sessionFactory: CreateCountingSession))
        {
        }

        using (await OnnxExecutionSessionFactory.CreatePooledOpusAsync(
            "test-engine",
            "encoder-model.onnx",
            "decoder-model.onnx",
            ExecutionProviderKind.Cpu,
            CancellationToken.None,
            pool,
            sessionFactory: CreateCountingSession))
        {
        }

        Assert.Equal(2, createCount);

        InferenceSession CreateCountingSession(string modelPath, SessionOptions options)
        {
            createCount++;
            return CreateMinimalSession();
        }
    }

    [Fact]
    public async Task CreateSingleAsync_honors_pre_canceled_token_before_directml_session_creation()
    {
        using var cancellationSource = new CancellationTokenSource();
        await cancellationSource.CancelAsync();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            OnnxExecutionSessionFactory.CreateSingleAsync(
                Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx"),
                ExecutionProviderKind.DirectMl,
                cancellationSource.Token));
    }

    [Fact]
    public void ResolveSharedSelectedProvider_preserves_directml_when_tensorrt_falls_back_to_gpu_for_both_sessions()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ResolveSharedSelectedProvider", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate shared provider resolver.");

        object? rawResult = method.Invoke(null,
            [ExecutionProviderKind.TensorRTRtx, ExecutionProviderKind.DirectMl, ExecutionProviderKind.DirectMl]);
        ExecutionProviderKind selectedProvider = Assert.IsType<ExecutionProviderKind>(rawResult);

        Assert.Equal(ExecutionProviderKind.DirectMl, selectedProvider);
    }

    [Fact]
    public void IsTensorRtRtxDeviceCandidate_requires_exact_provider_name_and_gpu_hardware_type()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("IsTensorRtRtxDeviceCandidate", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX device filter helper.");

        object? gpuMatch = method.Invoke(null, ["NvTensorRTRTXExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? windowsMlSpellingMatch = method.Invoke(null, ["NvTensorRtRtxExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? legacyNvTrt = method.Invoke(null, ["NvTensorRtExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? legacyTrt = method.Invoke(null, ["TensorRTExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? wrongName = method.Invoke(null, ["RandomOtherProvider", OrtHardwareDeviceType.GPU]);
        object? wrongDeviceType = method.Invoke(null, ["NvTensorRTRTXExecutionProvider", OrtHardwareDeviceType.CPU]);

        Assert.True(Assert.IsType<bool>(gpuMatch));
        Assert.False(Assert.IsType<bool>(windowsMlSpellingMatch));
        Assert.False(Assert.IsType<bool>(legacyNvTrt));
        Assert.False(Assert.IsType<bool>(legacyTrt));
        Assert.False(Assert.IsType<bool>(wrongName));
        Assert.False(Assert.IsType<bool>(wrongDeviceType));
    }

    [Fact]
    public void IsDirectMlDeviceCandidate_requires_directml_provider_name_and_gpu_hardware_type()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("IsDirectMlDeviceCandidate", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate DirectML device filter helper.");

        object? dmlMatch = method.Invoke(null, ["DmlExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? directMlSpellingMatch = method.Invoke(null, ["DirectMLExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? trtRtx = method.Invoke(null, ["NvTensorRtRtxExecutionProvider", OrtHardwareDeviceType.GPU]);
        object? wrongDeviceType = method.Invoke(null, ["DmlExecutionProvider", OrtHardwareDeviceType.CPU]);

        Assert.True(Assert.IsType<bool>(dmlMatch));
        Assert.True(Assert.IsType<bool>(directMlSpellingMatch));
        Assert.False(Assert.IsType<bool>(trtRtx));
        Assert.False(Assert.IsType<bool>(wrongDeviceType));
    }

    [Fact]
    public void ResolveSessionOptionsProvider_keeps_directml_when_bootstrap_selected_cpu_on_windows()
    {
        Assert.Equal(
            OperatingSystem.IsWindows() ? ExecutionProviderKind.DirectMl : ExecutionProviderKind.Cpu,
            OnnxExecutionSessionFactory.ResolveSessionOptionsProvider(
                ExecutionProviderKind.DirectMl,
                ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void ResolveSessionOptionsProvider_preserves_non_directml_cpu_selection()
    {
        Assert.Equal(
            ExecutionProviderKind.Cpu,
            OnnxExecutionSessionFactory.ResolveSessionOptionsProvider(
                ExecutionProviderKind.TensorRTRtx,
                ExecutionProviderKind.Cpu));
    }

    [Fact]
    public void ResolveSessionOptionsProvider_preserves_native_cuda_selection()
    {
        Assert.Equal(
            ExecutionProviderKind.Cuda,
            OnnxExecutionSessionFactory.ResolveSessionOptionsProvider(
                ExecutionProviderKind.Cuda,
                ExecutionProviderKind.Cuda));
    }

    [Fact]
    public void ResolveSessionOptionsProvider_maps_windows_cuda_cpu_bootstrap_to_directml()
    {
        ExecutionProviderKind resolved = OnnxExecutionSessionFactory.ResolveSessionOptionsProvider(
            ExecutionProviderKind.Cuda,
            ExecutionProviderKind.Cpu);

        Assert.Equal(
            OperatingSystem.IsWindows() ? ExecutionProviderKind.DirectMl : ExecutionProviderKind.Cpu,
            resolved);
    }

    [RequiresDirectMlFact]
    public async Task CreateSingleAsync_directml_selects_dml_not_cpu()
    {
        string path = Path.Join(Path.GetTempPath(), $"trackdub-dml-{Guid.NewGuid():N}.onnx");
        await File.WriteAllBytesAsync(path, BuildIdentityOnnxModel());
        try
        {
            using OnnxExecutionSessionFactory.SingleSessionLease lease =
                await OnnxExecutionSessionFactory.CreateSingleAsync(
                    path,
                    ExecutionProviderKind.DirectMl,
                    CancellationToken.None);

            Assert.Equal("dml", lease.RequestedProvider);
            Assert.Equal("dml", lease.SelectedProvider);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [RequiresDirectMlFact]
    public async Task CreateSingleAsync_directml_appends_even_when_bootstrap_selected_cpu()
    {
        OnnxExecutionProviderBootstrapperRegistry.ResetForTests();
        OnnxExecutionProviderBootstrapperRegistry.Initialize(new CpuSelectingDirectMlBootstrapper());

        string path = Path.Join(Path.GetTempPath(), $"trackdub-dml-cpu-bootstrap-{Guid.NewGuid():N}.onnx");
        await File.WriteAllBytesAsync(path, BuildIdentityOnnxModel());
        try
        {
            using OnnxExecutionSessionFactory.SingleSessionLease lease =
                await OnnxExecutionSessionFactory.CreateSingleAsync(
                    path,
                    ExecutionProviderKind.DirectMl,
                    CancellationToken.None);

            Assert.Equal("dml", lease.RequestedProvider);
            Assert.Equal("dml", lease.SelectedProvider);
        }
        finally
        {
            File.Delete(path);
            OnnxExecutionProviderBootstrapperRegistry.ResetForTests();
        }
    }

    [Fact]
    public void BuildTensorRtRtxOptions_includes_runtime_cache_by_default_and_omits_cuda_graph()
    {
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");

        try
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", null);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", null);

            MethodInfo method = typeof(OnnxExecutionSessionFactory)
                .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

            object? rawResult = method.Invoke(null, [null, false]);
            var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

            Assert.Equal(
                Path.Join(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "Trackdub",
                    "EngineCache",
                    "trt-rtx",
                    Trackdub.Inference.Runtime.TensorRtRtx.TensorRtRtxProviderConstants.BundledFingerprintVersion),
                options["nv_runtime_cache_path"]);
            // Cuda-graph capture requires stable device buffer addresses across Run() calls; most
            // call sites build fresh input tensors every call, so the flag must default to off.
            // It has to be an explicit "0": the TensorRT RTX EP treats a missing key as true.
            Assert.Equal("0", options["enable_cuda_graph"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    [Fact]
    public void BuildTensorRtRtxOptions_drops_device_id_removed_in_ep_abi_0_5()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        var additional = new Dictionary<string, string> { ["device_id"] = "0", ["nv_max_workspace_size"] = "1024" };
        object? rawResult = method.Invoke(null, [additional, false]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.False(options.ContainsKey("device_id"));
        Assert.Equal("1024", options["nv_max_workspace_size"]);
    }

    [Fact]
    public void BuildTensorRtRtxOptions_enable_cuda_graph_true_sets_the_flag()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(null, [null, true]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("1", options["enable_cuda_graph"]);
    }

    [Fact]
    public void BuildTensorRtRtxOptions_uses_environment_cache_root_by_default()
    {
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");
        string cacheRoot = Path.Join(
            Path.GetTempPath(),
            "Trackdub.Inference.Tests",
            Guid.NewGuid().ToString("N"));

        try
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", null);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", cacheRoot);

            MethodInfo method = typeof(OnnxExecutionSessionFactory)
                .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

            object? rawResult = method.Invoke(null, [null, false]);
            var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

            Assert.Equal(
                Path.Join(
                    Path.GetFullPath(cacheRoot),
                    "EngineCache",
                    "trt-rtx",
                    Trackdub.Inference.Runtime.TensorRtRtx.TensorRtRtxProviderConstants.BundledFingerprintVersion),
                options["nv_runtime_cache_path"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    [Fact]
    public void BuildTensorRtRtxOptions_uses_environment_engine_cache_root_by_default()
    {
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");
        string engineCacheRoot = Path.Join(
            Path.GetTempPath(),
            "Trackdub.Inference.Tests",
            Guid.NewGuid().ToString("N"),
            "engine");

        try
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", engineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", Path.Join(Path.GetTempPath(), "ignored"));

            MethodInfo method = typeof(OnnxExecutionSessionFactory)
                .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

            object? rawResult = method.Invoke(null, [null, false]);
            var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

            Assert.Equal(
                Path.Join(
                    Path.GetFullPath(engineCacheRoot),
                    "trt-rtx",
                    Trackdub.Inference.Runtime.TensorRtRtx.TensorRtRtxProviderConstants.BundledFingerprintVersion),
                options["nv_runtime_cache_path"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    [Fact]
    public void BuildTensorRtRtxOptions_allows_callers_to_override_defaults()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["enable_cuda_graph"] = "0",
                    ["nv_runtime_cache_path"] = @"D:\cache"
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("0", options["enable_cuda_graph"]);
        Assert.Equal(@"D:\cache", options["nv_runtime_cache_path"]);
    }

    [Fact]
    public void BuildTensorRtRtxOptions_remaps_classic_trt_profile_keys()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["trt_profile_min_shapes"] = "mel:1x128x1",
                    ["trt_profile_max_shapes"] = "mel:1x128x3000",
                    ["trt_profile_opt_shapes"] = "mel:1x128x3000",
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("mel:1x128x1", options["nv_profile_min_shapes"]);
        Assert.Equal("mel:1x128x3000", options["nv_profile_max_shapes"]);
        Assert.Equal("mel:1x128x3000", options["nv_profile_opt_shapes"]);
        Assert.False(options.ContainsKey("trt_profile_min_shapes"));
        Assert.False(options.ContainsKey("trt_profile_max_shapes"));
        Assert.False(options.ContainsKey("trt_profile_opt_shapes"));
    }

    [Fact]
    public void BuildTensorRtRtxOptions_prefers_explicit_nv_profile_keys_over_classic()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["trt_profile_min_shapes"] = "mel:1x128x1",
                    ["nv_profile_min_shapes"] = "mel:1x128x8",
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("mel:1x128x8", options["nv_profile_min_shapes"]);
        Assert.False(options.ContainsKey("trt_profile_min_shapes"));
    }

    [Fact]
    public void BuildTensorRtRtxOptions_round_trips_whisper_encoder_trt_profile_shapes()
    {
        // Mirrors WhisperOnnxAudioTranscriptionEngine.TrtEncoderOptions. If this test ever fails,
        // the Whisper TRT-RTX smoke failures will not be fixable by editing the engine code alone —
        // the C# -> ORT options translation is broken.
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["trt_profile_min_shapes"] = "input_features:1x80x1",
                    ["trt_profile_max_shapes"] = "input_features:1x80x3000",
                    ["trt_profile_opt_shapes"] = "input_features:1x80x3000",
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("input_features:1x80x1", options["nv_profile_min_shapes"]);
        Assert.Equal("input_features:1x80x3000", options["nv_profile_max_shapes"]);
        Assert.Equal("input_features:1x80x3000", options["nv_profile_opt_shapes"]);
    }

    [Fact]
    public void BuildTensorRtRtxOptions_round_trips_sortformer_trt_profile_shapes()
    {
        // Mirrors SortFormerDiarizationEngine.TrtOptions. The streaming feature path uses
        // "chunk" / "spkcache" / "fifo" instead of "waveform"; both must reach ORT as nv_profile_*.
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["trt_profile_min_shapes"] = "waveform:1x16000",
                    ["trt_profile_max_shapes"] = "waveform:1x57600000",
                    ["trt_profile_opt_shapes"] = "waveform:1x160000",
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("waveform:1x16000", options["nv_profile_min_shapes"]);
        Assert.Equal("waveform:1x57600000", options["nv_profile_max_shapes"]);
        Assert.Equal("waveform:1x160000", options["nv_profile_opt_shapes"]);
        Assert.False(options.ContainsKey("trt_profile_min_shapes"));
        Assert.False(options.ContainsKey("trt_profile_max_shapes"));
        Assert.False(options.ContainsKey("trt_profile_opt_shapes"));
    }

    [Fact]
    public void BuildTensorRtRtxOptions_round_trips_nemotron_encoder_trt_profile_shapes()
    {
        // Mirrors NemotronAsrEncoderTrtProfiles.ProfileWithoutPromptIndex / PromptIndexSuffix.
        // The encoder uses five dynamic inputs plus a prompt index; all six names must survive.
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["trt_profile_min_shapes"] =
                        "processed_signal:1x128x65,processed_signal_length:1,cache_last_channel:24x1x56x1024,cache_last_time:24x1x1024x8,cache_last_channel_len:1,prompt_index:1",
                    ["trt_profile_max_shapes"] =
                        "processed_signal:1x128x65,processed_signal_length:1,cache_last_channel:24x1x56x1024,cache_last_time:24x1x1024x8,cache_last_channel_len:1,prompt_index:1",
                    ["trt_profile_opt_shapes"] =
                        "processed_signal:1x128x65,processed_signal_length:1,cache_last_channel:24x1x56x1024,cache_last_time:24x1x1024x8,cache_last_channel_len:1,prompt_index:1",
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        string expectedShapes =
            "processed_signal:1x128x65,processed_signal_length:1,cache_last_channel:24x1x56x1024,cache_last_time:24x1x1024x8,cache_last_channel_len:1,prompt_index:1";
        Assert.Equal(expectedShapes, options["nv_profile_min_shapes"]);
        Assert.Equal(expectedShapes, options["nv_profile_max_shapes"]);
        Assert.Equal(expectedShapes, options["nv_profile_opt_shapes"]);
    }

    [Fact]
    public void BuildTensorRtRtxOptions_preserves_arbitrary_caller_keys()
    {
        // TRT-RTX accepts more than just nv_profile_*. Engine code may pass extra hints like
        // nv_cuda_graph_enable, nv_max_workspace_size, or custom EP options; BuildTensorRtRtxOptions
        // must pass them through verbatim rather than whitelist-dropping unknown keys.
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildTensorRtRtxOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TensorRT RTX provider-options helper.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, string>
                {
                    ["nv_max_workspace_size"] = "2147483648",
                    ["trt_profile_min_shapes"] = "waveform:1x16000",
                },
                false
            ]);
        var options = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(rawResult);

        Assert.Equal("2147483648", options["nv_max_workspace_size"]);
        Assert.Equal("waveform:1x16000", options["nv_profile_min_shapes"]);
    }

    [Fact]
    public void BuildSessionOptionsFingerprint_distinguishes_tensorrt_option_overrides()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildSessionOptionsFingerprint", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session-options fingerprint helper.");

        object? first = method.Invoke(
            null,
            [
                ExecutionProviderKind.TensorRTRtx,
                WindowsMlExecutionDevicePolicy.Explicit,
                new Dictionary<string, string> { ["enable_cuda_graph"] = "1" },
                false
            ]);
        object? second = method.Invoke(
            null,
            [
                ExecutionProviderKind.TensorRTRtx,
                WindowsMlExecutionDevicePolicy.Explicit,
                new Dictionary<string, string> { ["enable_cuda_graph"] = "0" },
                false
            ]);
        object? cpuExplicit = method.Invoke(
            null,
            [ExecutionProviderKind.Cpu, WindowsMlExecutionDevicePolicy.Explicit, null, false]);
        object? cpuMaxPerf = method.Invoke(
            null,
            [ExecutionProviderKind.Cpu, WindowsMlExecutionDevicePolicy.MaxPerformance, null, false]);

        Assert.NotEqual(Assert.IsType<string>(first), Assert.IsType<string>(second));
        Assert.Equal(Assert.IsType<string>(cpuExplicit), Assert.IsType<string>(cpuMaxPerf));
    }

    [Fact]
    public void BuildSessionOptionsFingerprint_distinguishes_enable_cuda_graph_parameter_from_override_dictionary()
    {
        // Guards against the enableCudaGraph parameter and the additionalTrtOptions dictionary
        // silently collapsing onto the same fingerprint when neither mentions enable_cuda_graph.
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildSessionOptionsFingerprint", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session-options fingerprint helper.");

        object? cudaGraphOff = method.Invoke(
            null,
            [ExecutionProviderKind.TensorRTRtx, WindowsMlExecutionDevicePolicy.Explicit, null, false]);
        object? cudaGraphOn = method.Invoke(
            null,
            [ExecutionProviderKind.TensorRTRtx, WindowsMlExecutionDevicePolicy.Explicit, null, true]);

        Assert.NotEqual(Assert.IsType<string>(cudaGraphOff), Assert.IsType<string>(cudaGraphOn));
    }

    [Fact]
    public void ShouldUseCatalogDevicePolicy_is_false_for_explicit_policy_and_native_cuda()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ShouldUseCatalogDevicePolicy", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate device policy helper.");

        object? explicitDirectMl = method.Invoke(
            null,
            [WindowsMlExecutionDevicePolicy.Explicit, ExecutionProviderKind.DirectMl]);
        object? policyNativeCuda = method.Invoke(
            null,
            [WindowsMlExecutionDevicePolicy.MaxPerformance, ExecutionProviderKind.Cuda]);

        Assert.False(Assert.IsType<bool>(explicitDirectMl));
        Assert.False(Assert.IsType<bool>(policyNativeCuda));
    }

    [Fact]
    public void BuildSessionOptionsFingerprint_ignores_policy_for_cpu()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildSessionOptionsFingerprint", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session options fingerprint helper.");

        object? explicitCpu = method.Invoke(
            null,
            [ExecutionProviderKind.Cpu, WindowsMlExecutionDevicePolicy.Explicit, null, false]);
        object? maxPerfCpu = method.Invoke(
            null,
            [ExecutionProviderKind.Cpu, WindowsMlExecutionDevicePolicy.MaxPerformance, null, false]);

        Assert.Equal(Assert.IsType<string>(explicitCpu), Assert.IsType<string>(maxPerfCpu));
        Assert.Equal("default", Assert.IsType<string>(explicitCpu));
    }

    [Fact]
    public void BuildSessionOptionsFingerprint_includes_policy_only_for_catalog_providers_that_apply_it()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildSessionOptionsFingerprint", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session options fingerprint helper.");

        object? explicitDml = method.Invoke(
            null,
            [ExecutionProviderKind.DirectMl, WindowsMlExecutionDevicePolicy.Explicit, null, false]);
        object? maxPerfDml = method.Invoke(
            null,
            [ExecutionProviderKind.DirectMl, WindowsMlExecutionDevicePolicy.MaxPerformance, null, false]);
        object? defaultRenderDml = method.Invoke(
            null,
            [ExecutionProviderKind.DirectMl, WindowsMlExecutionDevicePolicy.DefaultRender, null, false]);
        object? minPowerDml = method.Invoke(
            null,
            [ExecutionProviderKind.DirectMl, WindowsMlExecutionDevicePolicy.MinPower, null, false]);
        object? maxPerfTrt = method.Invoke(
            null,
            [ExecutionProviderKind.TensorRTRtx, WindowsMlExecutionDevicePolicy.MaxPerformance, null, false]);
        object? explicitTrt = method.Invoke(
            null,
            [ExecutionProviderKind.TensorRTRtx, WindowsMlExecutionDevicePolicy.Explicit, null, false]);

        Assert.Equal("default", Assert.IsType<string>(explicitDml));
        Assert.Equal(Assert.IsType<string>(explicitDml), Assert.IsType<string>(maxPerfDml));
        Assert.Equal(Assert.IsType<string>(explicitDml), Assert.IsType<string>(defaultRenderDml));
        Assert.Equal(Assert.IsType<string>(explicitDml), Assert.IsType<string>(minPowerDml));
        Assert.Equal(Assert.IsType<string>(explicitTrt), Assert.IsType<string>(maxPerfTrt));
        Assert.NotEqual(Assert.IsType<string>(maxPerfTrt), Assert.IsType<string>(explicitDml));

        object? explicitMigraphx = method.Invoke(
            null,
            [ExecutionProviderKind.Migraphx, WindowsMlExecutionDevicePolicy.Explicit, null, false]);
        object? defaultRenderMigraphx = method.Invoke(
            null,
            [ExecutionProviderKind.Migraphx, WindowsMlExecutionDevicePolicy.DefaultRender, null, false]);
        if (OperatingSystem.IsWindows())
        {
            Assert.NotEqual(Assert.IsType<string>(explicitMigraphx), Assert.IsType<string>(defaultRenderMigraphx));
        }
        else
        {
            Assert.Equal(Assert.IsType<string>(explicitMigraphx), Assert.IsType<string>(defaultRenderMigraphx));
        }
    }

    [Fact]
    public void BuildTensorRtOptions_enables_engine_cache_under_classic_trt_fingerprint_directory()
    {
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");
        string engineCacheRoot = Path.Join(
            Path.GetTempPath(), "Trackdub.Inference.Tests", Guid.NewGuid().ToString("N"), "engine");

        try
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", engineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", null);

            IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtOptions(
                null, CompleteTensorRtFingerprint());

            Assert.Equal("1", options["trt_engine_cache_enable"]);
            Assert.Equal(
                Path.Join(Path.GetFullPath(engineCacheRoot), "classic-trt", ClassicTensorRtCacheKey(CompleteTensorRtFingerprint())),
                options["trt_engine_cache_path"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    [Fact]
    public void BuildTensorRtOptions_uses_cache_root_fallback_when_engine_cache_root_is_unset()
    {
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");
        string cacheRoot = Path.Join(
            Path.GetTempPath(), "Trackdub.Inference.Tests", Guid.NewGuid().ToString("N"));

        try
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", null);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", cacheRoot);

            IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtOptions(
                null, CompleteTensorRtFingerprint(architecture: "Ada", driver: "551.86"));

            Assert.Equal(
                Path.Join(Path.GetFullPath(cacheRoot), "EngineCache", "classic-trt",
                    ClassicTensorRtCacheKey(CompleteTensorRtFingerprint(architecture: "Ada", driver: "551.86"))),
                options["trt_engine_cache_path"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    [Fact]
    public void BuildTensorRtOptions_passes_through_caller_profile_options()
    {
        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            new Dictionary<string, string>
            {
                ["trt_profile_min_shapes"] = "waveform:1x16000",
                ["trt_profile_max_shapes"] = "waveform:1x57600000",
                ["trt_profile_opt_shapes"] = "waveform:1x160000",
            },
            CompleteTensorRtFingerprint());

        Assert.Equal("waveform:1x16000", options["trt_profile_min_shapes"]);
        Assert.Equal("waveform:1x57600000", options["trt_profile_max_shapes"]);
        Assert.Equal("waveform:1x160000", options["trt_profile_opt_shapes"]);
    }

    [Fact]
    public void BuildTensorRtOptions_caller_cannot_override_cache_keys()
    {
        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            new Dictionary<string, string>
            {
                ["trt_engine_cache_enable"] = "0",
                ["trt_engine_cache_path"] = @"D:\attacker-cache",
            },
            CompleteTensorRtFingerprint());

        Assert.Equal("1", options["trt_engine_cache_enable"]);
        Assert.NotEqual(@"D:\attacker-cache", options["trt_engine_cache_path"]);
        Assert.Contains("classic-trt", options["trt_engine_cache_path"]);
    }

    [Fact]
    public void BuildTensorRtOptionsForModel_uses_distinct_stable_authoritative_graph_prefixes()
    {
        string modelPath = Path.Join(Path.GetTempPath(), "trackdub-prefix-test", "encoder.onnx");
        string otherPath = Path.Join(Path.GetTempPath(), "trackdub-prefix-test", "decoder.onnx");
        var callerOptions = new Dictionary<string, string>
        {
            ["trt_engine_cache_prefix"] = "another-model",
            ["trt_profile_min_shapes"] = "audio:1x80x1",
        };

        IReadOnlyDictionary<string, string> options =
            OnnxExecutionSessionFactory.BuildTensorRtOptionsAfterProviderLoad(
                callerOptions, modelPath, static () => { }, () => CompleteTensorRtFingerprint());
        string prefix = options["trt_engine_cache_prefix"];
        Assert.Equal(OnnxExecutionSessionFactory.BuildClassicTensorRtModelCachePrefix(modelPath), prefix);
        Assert.Equal(prefix, OnnxExecutionSessionFactory.BuildClassicTensorRtModelCachePrefix(
            Path.Join(Path.GetDirectoryName(modelPath)!, ".", "encoder.onnx")));
        Assert.NotEqual(prefix, OnnxExecutionSessionFactory.BuildClassicTensorRtModelCachePrefix(otherPath));
        Assert.Equal("audio:1x80x1", options["trt_profile_min_shapes"]);
    }

    [Fact]
    public void BuildTensorRtOptionsAfterProviderLoad_first_model_uses_the_same_cache_across_cold_processes()
    {
        string loadedFingerprint = CompleteTensorRtFingerprint();
        string modelPath = Path.Join(Path.GetTempPath(), "trackdub-load-sequence-test", "encoder.onnx");
        var callerOptions = new Dictionary<string, string> { ["trt_profile_min_shapes"] = "audio:1x80x1" };

        IReadOnlyDictionary<string, string> BuildForColdProcess()
        {
            bool providerLoaded = false;
            string unresolvedFingerprint = $"unknown-process-{Guid.NewGuid():N}";
            return OnnxExecutionSessionFactory.BuildTensorRtOptionsAfterProviderLoad(
                callerOptions, modelPath,
                () => providerLoaded = true,
                () => providerLoaded ? loadedFingerprint : unresolvedFingerprint);
        }

        IReadOnlyDictionary<string, string> firstProcess = BuildForColdProcess();
        IReadOnlyDictionary<string, string> secondProcess = BuildForColdProcess();
        string expectedDirectory = OnnxExecutionSessionFactory.ResolveClassicTensorRtEngineCacheDirectory(loadedFingerprint);
        Assert.Equal(expectedDirectory, firstProcess["trt_engine_cache_path"]);
        Assert.Equal(expectedDirectory, secondProcess["trt_engine_cache_path"]);
        Assert.Equal("1", firstProcess["trt_engine_cache_enable"]);
        Assert.Equal("audio:1x80x1", firstProcess["trt_profile_min_shapes"]);
        Assert.Equal(OnnxExecutionSessionFactory.BuildClassicTensorRtModelCachePrefix(modelPath),
            firstProcess["trt_engine_cache_prefix"]);
    }

    [Fact]
    public void BuildTensorRtOptionsAfterProviderLoad_does_not_resolve_cache_identity_when_loading_fails()
    {
        var failure = new InvalidOperationException("TensorRT provider could not be loaded.");
        bool fingerprintResolved = false;
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
            OnnxExecutionSessionFactory.BuildTensorRtOptionsAfterProviderLoad(
                null, null,
                () => throw failure,
                () => { fingerprintResolved = true; return "unverified"; })));
        Assert.False(fingerprintResolved);
    }

    [Fact]
    public void BuildTensorRtOptions_cache_path_changes_when_environment_fingerprint_changes()
    {
        IReadOnlyDictionary<string, string> first = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            null, CompleteTensorRtFingerprint());
        IReadOnlyDictionary<string, string> otherDriver = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            null, CompleteTensorRtFingerprint(driver: "551.86"));
        IReadOnlyDictionary<string, string> otherArch = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            null, CompleteTensorRtFingerprint(architecture: "Blackwell", driver: "551.86"));
        IReadOnlyDictionary<string, string> otherTrt = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            null, CompleteTensorRtFingerprint(tensorRtHashCharacter: 'B'));

        Assert.NotEqual(first["trt_engine_cache_path"], otherDriver["trt_engine_cache_path"]);
        Assert.NotEqual(first["trt_engine_cache_path"], otherArch["trt_engine_cache_path"]);
        Assert.NotEqual(first["trt_engine_cache_path"], otherTrt["trt_engine_cache_path"]);
    }

    [Fact]
    public void BuildTensorRtOptions_disables_cache_when_any_environment_identity_is_uncertain()
    {
        string complete = CompleteTensorRtFingerprint();
        string[] incompleteFingerprints =
        [
            CompleteTensorRtFingerprint(architecture: "unknown"),
            CompleteTensorRtFingerprint(model: "unknown"),
            CompleteTensorRtFingerprint(driver: "unknown"),
            complete.Replace("trt-sha256-" + new string('A', 64), "trt-unknown", StringComparison.Ordinal),
            complete.Replace("ort-1.31.0", "ort-unknown", StringComparison.Ordinal),
        ];

        foreach (string fingerprint in incompleteFingerprints)
        {
            Assert.False(OnnxExecutionSessionFactory.HasCompleteClassicTensorRtCacheIdentity(fingerprint));
            IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtOptions(
                new Dictionary<string, string>
                {
                    ["trt_engine_cache_enable"] = "1",
                    ["trt_engine_cache_path"] = "caller-cache",
                    ["trt_engine_cache_prefix"] = "caller-prefix",
                },
                fingerprint);

            Assert.Equal("0", options["trt_engine_cache_enable"]);
            Assert.False(options.ContainsKey("trt_engine_cache_path"));
            Assert.False(options.ContainsKey("trt_engine_cache_prefix"));
        }
    }

    [RequiresClassicTensorRtFact]
    public void BuildTensorRtProviderOptions_carries_engine_cache_options_and_creates_cache_directory()
    {
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");
        string engineCacheRoot = Path.Join(
            Path.GetTempPath(), "Trackdub.Inference.Tests", Guid.NewGuid().ToString("N"), "engine");

        try
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", engineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", null);

            using OrtTensorRTProviderOptions providerOptions = OnnxExecutionSessionFactory.BuildTensorRtProviderOptions(
                null, CompleteTensorRtFingerprint());

            string ortOptions = providerOptions.GetOptions();
            string expectedCachePath = Path.Join(
                Path.GetFullPath(engineCacheRoot), "classic-trt", ClassicTensorRtCacheKey(CompleteTensorRtFingerprint()));
            Assert.Contains("trt_engine_cache_enable=1", ortOptions);
            Assert.Contains($"trt_engine_cache_path={expectedCachePath}", ortOptions);
            Assert.True(Directory.Exists(expectedCachePath));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    [Fact]
    public void BuildSessionOptionsFingerprint_distinguishes_classic_tensorrt_option_overrides()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildSessionOptionsFingerprint", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session-options fingerprint helper.");

        object? first = method.Invoke(
            null,
            [
                ExecutionProviderKind.TensorRt,
                WindowsMlExecutionDevicePolicy.Explicit,
                new Dictionary<string, string> { ["trt_profile_min_shapes"] = "waveform:1x16000" },
                false
            ]);
        object? second = method.Invoke(
            null,
            [
                ExecutionProviderKind.TensorRt,
                WindowsMlExecutionDevicePolicy.Explicit,
                new Dictionary<string, string> { ["trt_profile_min_shapes"] = "waveform:1x32000" },
                false
            ]);
        object? repeat = method.Invoke(
            null,
            [
                ExecutionProviderKind.TensorRt,
                WindowsMlExecutionDevicePolicy.Explicit,
                new Dictionary<string, string> { ["trt_profile_min_shapes"] = "waveform:1x16000" },
                false
            ]);

        Assert.NotEqual(Assert.IsType<string>(first), Assert.IsType<string>(second));
        Assert.Equal(Assert.IsType<string>(first), Assert.IsType<string>(repeat));
        Assert.NotEqual("default", Assert.IsType<string>(first));
    }

    [Theory]
    [InlineData("/usr/lib/libnvinfer.so.10.13.3")]
    [InlineData("nvinfer.dll")]
    [InlineData("nvinfer_10.dll")]
    public void ResolveTensorRtVersionIdentity_uses_only_the_mapped_core_binary(string loadedPath)
    {
        string identity = OnnxExecutionSessionFactory.ResolveTensorRtVersionIdentity(
            [loadedPath, loadedPath, "libnvinfer_plugin.so.10", "nvinfer_plugin_10.dll"],
            path => path == loadedPath ? "loaded-binary-identity" : throw new InvalidOperationException());

        Assert.Equal("loaded-binary-identity", identity);
    }

    [Fact]
    public void ResolveTensorRtVersionIdentity_isolates_missing_or_ambiguous_runtimes()
    {
        string missing = OnnxExecutionSessionFactory.ResolveTensorRtVersionIdentity(
            [], _ => throw new InvalidOperationException("Must not probe an installed library."));
        string ambiguous = OnnxExecutionSessionFactory.ResolveTensorRtVersionIdentity(
            ["/a/libnvinfer.so.10", "/b/libnvinfer.so.10"],
            _ => throw new InvalidOperationException("Must not guess which runtime ORT uses."));

        Assert.StartsWith("unknown-process-", missing, StringComparison.Ordinal);
        Assert.Equal(missing, ambiguous);
    }

    [Fact]
    public void ResolveTensorRtVersionIdentity_changes_when_the_loaded_binary_changes()
    {
        string first = OnnxExecutionSessionFactory.ResolveTensorRtVersionIdentity(
            ["/a/libnvinfer.so.10"], _ => "binary-10.9.0");
        string second = OnnxExecutionSessionFactory.ResolveTensorRtVersionIdentity(
            ["/a/libnvinfer.so.10"], _ => "binary-10.13.3");
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void BuildTensorRtOptions_translates_enable_cuda_graph_and_drops_rtx_only_keys_for_classic_trt()
    {
        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            new Dictionary<string, string>
            {
                ["enable_cuda_graph"] = "1",
                ["nv_profile_min_shapes"] = "input_ids:1x1",
                ["nv_runtime_cache_path"] = @"C:\rtx-cache",
                ["trt_profile_max_shapes"] = "input_ids:1x512",
            },
            CompleteTensorRtFingerprint());

        Assert.Equal("1", options["trt_cuda_graph_enable"]);
        Assert.Equal("input_ids:1x512", options["trt_profile_max_shapes"]);
        Assert.False(options.ContainsKey("enable_cuda_graph"));
        Assert.DoesNotContain(options.Keys, key => key.StartsWith("nv_", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(WindowsMlExecutionDevicePolicy.Explicit, ExecutionProviderKind.DirectMl, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, ExecutionProviderKind.DirectMl, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, ExecutionProviderKind.TensorRTRtx, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, ExecutionProviderKind.Migraphx, true)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, ExecutionProviderKind.Cuda, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, ExecutionProviderKind.TensorRt, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.PreferNpu, ExecutionProviderKind.DirectMl, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxEfficiency, ExecutionProviderKind.DirectMl, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinOverallPower, ExecutionProviderKind.DirectMl, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.DefaultRender, ExecutionProviderKind.DirectMl, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinPower, ExecutionProviderKind.DirectMl, false)]
    public void ShouldUseCatalogDevicePolicy_matrix(
        WindowsMlExecutionDevicePolicy policy,
        ExecutionProviderKind provider,
        bool expected)
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ShouldUseCatalogDevicePolicy", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate device policy helper.");

        object? raw = method.Invoke(null, [policy, provider]);
        bool adjustedExpected = expected && OperatingSystem.IsWindows();
        Assert.Equal(adjustedExpected, Assert.IsType<bool>(raw));
    }

    [Theory]
    [InlineData("NvTensorRTRTXExecutionProvider", ExecutionProviderKind.TensorRTRtx)]
    [InlineData("MIGraphXExecutionProvider", ExecutionProviderKind.Migraphx)]
    [InlineData("DnnlExecutionProvider", ExecutionProviderKind.Dnnl)]
    [InlineData("DNNLExecutionProvider", ExecutionProviderKind.Dnnl)]
    [InlineData("DmlExecutionProvider", ExecutionProviderKind.DirectMl)]
    [InlineData("DirectMLExecutionProvider", ExecutionProviderKind.DirectMl)]
    [InlineData("CPUExecutionProvider", ExecutionProviderKind.Cpu)]
    public void TryMapCatalogEpNameToExecutionProviderKind_maps_known_catalog_names(
        string epName,
        ExecutionProviderKind expected)
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("TryMapCatalogEpNameToExecutionProviderKind", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate catalog EP name mapper.");

        object?[] args = [epName, null];
        bool mapped = Assert.IsType<bool>(method.Invoke(null, args));
        Assert.True(mapped);
        Assert.Equal(expected, Assert.IsType<ExecutionProviderKind>(args[1]));
    }

    [Theory]
    [InlineData(WindowsMlExecutionDevicePolicy.DefaultRender, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinPower, false)]
    public void BuildDevicePolicyFallbackReason_reports_truthful_fallback_when_extended_policy_not_applied(
        WindowsMlExecutionDevicePolicy devicePolicy,
        bool devicePolicyApplied)
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildDevicePolicyFallbackReason", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate device policy fallback reason helper.");

        object? reason = method.Invoke(null, [devicePolicy, devicePolicyApplied]);

        // The mapper can silently skip SetEpSelectionPolicy when the loaded ORT binding lacks
        // DEFAULT_RENDER/MIN_POWER; this helper must surface that truthfully instead of letting
        // callers assume the requested policy took effect.
        string message = Assert.IsType<string>(reason);
        Assert.Contains("was not applied", message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(WindowsMlExecutionDevicePolicySettings.ToKey(devicePolicy), message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(WindowsMlExecutionDevicePolicy.DefaultRender, true)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinPower, true)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.PreferNpu, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.Explicit, false)]
    public void BuildDevicePolicyFallbackReason_returns_null_when_policy_applied_or_not_extended(
        WindowsMlExecutionDevicePolicy devicePolicy,
        bool devicePolicyApplied)
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("BuildDevicePolicyFallbackReason", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate device policy fallback reason helper.");

        object? reason = method.Invoke(null, [devicePolicy, devicePolicyApplied]);

        Assert.Null(reason);
    }

    [Theory]
    [InlineData(WindowsMlExecutionDevicePolicy.DefaultRender)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinPower)]
    public void CreateSessionOptions_for_dml_never_claims_extended_policy_active_when_not_applied(
        WindowsMlExecutionDevicePolicy devicePolicy)
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("CreateSessionOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session options factory.");

        object result = method.Invoke(
            null,
            [new OnnxExecutionSessionFactory.SessionOptionsRequest(ExecutionProviderKind.DirectMl, devicePolicy)])
            ?? throw new InvalidOperationException("Session options factory returned null.");
        Type resultType = result.GetType();
        using var options = Assert.IsAssignableFrom<SessionOptions>(
            resultType.GetProperty("Options")?.GetValue(result));
        PropertyInfo fallbackReasonProp = resultType.GetProperty("FallbackReason")
            ?? throw new InvalidOperationException("SessionOptionsSelection.FallbackReason property not found.");
        string? fallbackReason = fallbackReasonProp.GetValue(result) as string;

        if (!OperatingSystem.IsWindows())
        {
            // ShouldUseCatalogDevicePolicy gates catalog device policies on Windows; off-Windows
            // the extended policy is never requested against ORT. A non-null fallback here is a
            // provider-level CPU fallback (e.g. DirectML unavailable), which must not falsely claim
            // the requested extended device policy was applied.
            if (fallbackReason is not null)
            {
                Assert.DoesNotContain("was applied", fallbackReason, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain(WindowsMlExecutionDevicePolicySettings.DefaultRenderKey, fallbackReason, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain(WindowsMlExecutionDevicePolicySettings.MinPowerKey, fallbackReason, StringComparison.OrdinalIgnoreCase);
            }

            return;
        }

        // Packaged DirectML is appended explicitly. Catalog device policies do not apply, so a
        // fallback reason is a DirectML append failure, not a silent "policy was applied" claim.
        if (fallbackReason is not null)
        {
            Assert.DoesNotContain("was applied", fallbackReason, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("DirectML", fallbackReason, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void CreateSessionOptions_for_dnnl_records_cpu_fallback_reason_when_append_fails()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("CreateSessionOptions", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate session options factory.");

        object result = method.Invoke(
            null,
            [new OnnxExecutionSessionFactory.SessionOptionsRequest(ExecutionProviderKind.Dnnl, WindowsMlExecutionDevicePolicy.Explicit)])
            ?? throw new InvalidOperationException("Session options factory returned null.");
        Type resultType = result.GetType();
        using var options = Assert.IsAssignableFrom<SessionOptions>(
            resultType.GetProperty("Options")?.GetValue(result));
        var selectedProvider = Assert.IsType<ExecutionProviderKind>(
            resultType.GetProperty("SelectedProvider")?.GetValue(result));
        string? fallbackReason = resultType.GetProperty("FallbackReason")?.GetValue(result) as string;

        if (selectedProvider is ExecutionProviderKind.Dnnl)
        {
            Assert.Null(fallbackReason);
            return;
        }

        Assert.Equal(ExecutionProviderKind.Cpu, selectedProvider);
        Assert.NotNull(fallbackReason);
        Assert.Contains("Requested dnnl", fallbackReason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("AppendExecutionProvider_Dnnl", fallbackReason, StringComparison.Ordinal);
    }

    [Fact]
    public void TryMapCatalogEpNameToExecutionProviderKind_rejects_unknown_names()
    {
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("TryMapCatalogEpNameToExecutionProviderKind", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate catalog EP name mapper.");

        object?[] args = ["TensorrtExecutionProvider", null];
        Assert.False(Assert.IsType<bool>(method.Invoke(null, args)));
    }

    [Fact]
    public void ResolveEffectiveProviderKindFromSession_returns_options_provider_when_probe_not_needed()
    {
        using InferenceSession session = CreateMinimalSession();
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ResolveEffectiveProviderKindFromSession", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate effective provider resolver.");

        object? raw = method.Invoke(
            null,
            [session, ExecutionProviderKind.Cpu, false]);
        Assert.Equal(ExecutionProviderKind.Cpu, Assert.IsType<ExecutionProviderKind>(raw));
    }

    [Fact]
    public void ResolveEffectiveProviderKindFromSession_probes_gpu_providers_even_without_catalog_policy()
    {
        using InferenceSession session = CreateMinimalSession();
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ResolveEffectiveProviderKindFromSession", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate effective provider resolver.");

        object? raw = method.Invoke(
            null,
            [session, ExecutionProviderKind.TensorRTRtx, false]);
        Assert.Equal(ExecutionProviderKind.Cpu, Assert.IsType<ExecutionProviderKind>(raw));
    }

    [Theory]
    [InlineData("Kernel not found for op: Squeeze", true)]
    [InlineData("ModelImporter failed to import graph", true)]
    [InlineData("No graph will run on TensorRT-RTX", true)]
    [InlineData("NvTensorRTRTXExecutionProvider registration failed", true)]
    // #329: exact managed message ORT surfaced for silero-vad's build-time squeeze shape error
    // (RTX 5070, EP ABI 0.4.2/cu13, TensorRT-RTX 1.6.1).
    [InlineData(SileroVadTrtBuildShapeErrorMessage, true)]
    // #329 sweep: whisper-large-v3 engine deserialize failure after CUDA OOM on 12 GB VRAM.
    [InlineData("[ErrorCode:ShapeInferenceNotRegistered] [NvTensorRTRTX EP] NvTensorRTRTX EP failed to deserialize engine for fused node: NvTensorRTRTXExecutionProvider_NvTensorRTRTXExecutionProvider_4350742136944386634_0_0", true)]
    [InlineData("unrelated IO error", false)]
    public void LooksLikeTrtSessionInitFailure_detects_known_trt_messages(string message, bool expected)
    {
        var exception = new InvalidOperationException(message);
        Assert.Equal(expected, OnnxExecutionSessionFactory.LooksLikeTrtSessionInitFailure(exception));
    }

    [Fact]
    public void LooksLikeTrtSessionInitFailure_does_not_treat_bare_onnx_runtime_exception_as_trt_init()
    {
        // Generic ORT failures (corrupt model, OOM, etc.) must not trigger soft TRT→DML retry.
        // Only TRT-specific importer/kernel evidence qualifies.
        Exception ex = Assert.ThrowsAny<Exception>(() => new InferenceSession(Array.Empty<byte>()));
        Assert.False(OnnxExecutionSessionFactory.LooksLikeTrtSessionInitFailure(ex));
    }

    [Fact]
    public void CreateInferenceSessionWithTrtInitFallback_retries_cpu_when_trt_init_fails()
    {
        var trtOptions = new SessionOptions();
        var initialSelection = new OnnxExecutionSessionFactory.SessionOptionsSelection(
            trtOptions,
            ExecutionProviderKind.TensorRTRtx);

        int createAttempts = 0;
        (InferenceSession session, OnnxExecutionSessionFactory.SessionOptionsSelection selection) =
            OnnxExecutionSessionFactory.CreateInferenceSessionWithTrtInitFallback(
                "fake-model.onnx",
                initialSelection,
                WindowsMlExecutionDevicePolicy.Explicit,
                additionalTrtOptions: null,
                sessionFactory: (_, _) =>
                {
                    createAttempts++;
                    if (createAttempts == 1)
                    {
                        throw new InvalidOperationException(
                            "Kernel not found for op: Squeeze(13) under NvTensorRTRTXExecutionProvider");
                    }

                    return CreateMinimalSession();
                },
                CancellationToken.None);

        using (session)
        using (selection.Options)
        {
            // Windows tries DirectML then CPU; Linux goes straight to CPU. Fake factory succeeds on
            // the first non-TRT attempt, so createAttempts is 2 either way. SelectedProvider follows
            // CreateSessionOptions (DirectML may itself fall back to CPU when DML is unavailable).
            Assert.Equal(2, createAttempts);
            Assert.True(
                selection.SelectedProvider is ExecutionProviderKind.DirectMl or ExecutionProviderKind.Cpu);
            Assert.NotNull(selection.FallbackReason);
            Assert.Contains("TensorRT RTX session init failed", selection.FallbackReason, StringComparison.Ordinal);
            Assert.Contains("fell back to", selection.FallbackReason, StringComparison.OrdinalIgnoreCase);
        }
    }

    private const string SileroVadTrtBuildShapeErrorMessage =
        "[ErrorCode:ShapeInferenceNotRegistered] [NvTensorRTRTX EP] Failed to create serialized engine for fused node: NvTensorRTRTXExecutionProvider_NvTensorRTRTXExecutionProvider_12112723482072567427_1_1";

    [Fact]
    public void CreateInferenceSessionWithTrtInitFallback_degrades_on_trt_build_time_shape_error()
    {
        // #329 acceptance: a TensorRT RTX build-time shape error must degrade to the next EP
        // instead of propagating. ORT surfaces it as OnnxRuntimeException (internal ctor), so the
        // fake factory throws the exact observed message; classification is message-based.
        var trtOptions = new SessionOptions();
        var initialSelection = new OnnxExecutionSessionFactory.SessionOptionsSelection(
            trtOptions,
            ExecutionProviderKind.TensorRTRtx);

        int createAttempts = 0;
        (InferenceSession session, OnnxExecutionSessionFactory.SessionOptionsSelection selection) =
            OnnxExecutionSessionFactory.CreateInferenceSessionWithTrtInitFallback(
                "silero-vad.onnx",
                initialSelection,
                WindowsMlExecutionDevicePolicy.Explicit,
                additionalTrtOptions: null,
                sessionFactory: (_, _) =>
                {
                    createAttempts++;
                    if (createAttempts == 1)
                    {
                        throw new InvalidOperationException(SileroVadTrtBuildShapeErrorMessage);
                    }

                    return CreateMinimalSession();
                },
                CancellationToken.None);

        using (session)
        using (selection.Options)
        {
            Assert.Equal(2, createAttempts);
            Assert.NotEqual(ExecutionProviderKind.TensorRTRtx, selection.SelectedProvider);
            Assert.True(
                selection.SelectedProvider is ExecutionProviderKind.DirectMl or ExecutionProviderKind.Cpu);
            Assert.NotNull(selection.FallbackReason);
            Assert.Contains("Failed to create serialized engine", selection.FallbackReason, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CreateInferenceSessionWithTrtInitFallback_hard_pin_does_not_retry()
    {
        using var trtOptions = new SessionOptions();
        var initialSelection = new OnnxExecutionSessionFactory.SessionOptionsSelection(
            trtOptions,
            ExecutionProviderKind.TensorRTRtx);

        int createAttempts = 0;
        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() =>
            OnnxExecutionSessionFactory.CreateInferenceSessionWithTrtInitFallback(
                "fake-model.onnx",
                initialSelection,
                WindowsMlExecutionDevicePolicy.Explicit,
                additionalTrtOptions: null,
                sessionFactory: (_, _) =>
                {
                    createAttempts++;
                    throw new InvalidOperationException(
                        "Kernel not found for op: Squeeze(13) under NvTensorRTRTXExecutionProvider");
                },
                CancellationToken.None,
                allowTrtInitFallback: false));

        Assert.Equal(1, createAttempts);
        Assert.Contains("Kernel not found", thrown.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ResolveEffectiveProviderKindFromSession_keeps_native_cuda_without_catalog_probe()
    {
        using InferenceSession session = CreateMinimalSession();
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ResolveEffectiveProviderKindFromSession", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate effective provider resolver.");

        object? raw = method.Invoke(
            null,
            [session, ExecutionProviderKind.Cuda, false]);
        Assert.Equal(ExecutionProviderKind.Cuda, Assert.IsType<ExecutionProviderKind>(raw));
    }

    [Fact]
    public void ResolveEffectiveProviderKindFromSession_checks_dnnl_placement_even_without_catalog_policy()
    {
        using InferenceSession session = CreateMinimalSession();
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ResolveEffectiveProviderKindFromSession", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate effective provider resolver.");

        object? raw = method.Invoke(
            null,
            [session, ExecutionProviderKind.Dnnl, false]);
        Assert.Equal(ExecutionProviderKind.Cpu, Assert.IsType<ExecutionProviderKind>(raw));
    }

    [Fact]
    public void ResolveCatalogEffectiveProviderFromSession_returns_cpu_for_minimal_session()
    {
        using InferenceSession session = CreateMinimalSession();
        MethodInfo method = typeof(OnnxExecutionSessionFactory)
            .GetMethod("ResolveCatalogEffectiveProviderFromSession", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate catalog effective provider resolver.");

        object? raw = method.Invoke(null, [session]);
        Assert.Equal(ExecutionProviderKind.Cpu, Assert.IsType<ExecutionProviderKind>(raw));
    }

    [Fact]
    public void Initialize_parallel_calls_do_not_throw()
    {
        OnnxExecutionProviderBootstrapperRegistry.ResetForTests();
        Parallel.For(
            0,
            8,
            _ => OnnxExecutionProviderBootstrapperRegistry.Initialize(
                new PortableExecutionProviderBootstrapper()));
    }

    [Fact]
    public void ReportUnmappedCatalogEpName_logs_once_per_distinct_ep_name()
    {
        OnnxExecutionProviderBootstrapperRegistry.ResetForTests();
        var logger = new TestLogger();
        OnnxExecutionProviderBootstrapperRegistry.Initialize(
            new PortableExecutionProviderBootstrapper(),
            logger: logger);

        OnnxExecutionSessionFactory.ReportUnmappedCatalogEpName(
            "UnknownEp",
            OnnxExecutionSessionFactory.CatalogEpNameSource.GetEpDeviceForInputs,
            deviceCount: 2);
        OnnxExecutionSessionFactory.ReportUnmappedCatalogEpName(
            "UnknownEp",
            OnnxExecutionSessionFactory.CatalogEpNameSource.GetEpDeviceForInputs,
            deviceCount: 2);
        OnnxExecutionSessionFactory.ReportUnmappedCatalogEpName(
            "AnotherUnknownEp",
            OnnxExecutionSessionFactory.CatalogEpNameSource.GetEpDevices,
            deviceCount: 5);

        Assert.Equal(2, logger.WarningCount);
        Assert.Contains(
            logger.Messages,
            message => message.Contains("UnknownEp", StringComparison.Ordinal));
        Assert.Contains(
            logger.Messages,
            message => message.Contains("AnotherUnknownEp", StringComparison.Ordinal));
    }

    [Fact]
    public void ResetForTests_clears_unmapped_ep_name_dedupe()
    {
        OnnxExecutionProviderBootstrapperRegistry.ResetForTests();
        var logger = new TestLogger();
        OnnxExecutionProviderBootstrapperRegistry.Initialize(
            new PortableExecutionProviderBootstrapper(),
            logger: logger);

        OnnxExecutionSessionFactory.ReportUnmappedCatalogEpName(
            "UnknownEp",
            OnnxExecutionSessionFactory.CatalogEpNameSource.GetEpDeviceForInputs,
            deviceCount: 1);
        OnnxExecutionProviderBootstrapperRegistry.ResetForTests();

        var loggerAfterReset = new TestLogger();
        OnnxExecutionProviderBootstrapperRegistry.Initialize(
            new PortableExecutionProviderBootstrapper(),
            logger: loggerAfterReset);
        OnnxExecutionSessionFactory.ReportUnmappedCatalogEpName(
            "UnknownEp",
            OnnxExecutionSessionFactory.CatalogEpNameSource.GetEpDeviceForInputs,
            deviceCount: 1);

        Assert.Equal(1, logger.WarningCount);
        Assert.Equal(1, loggerAfterReset.WarningCount);
    }

    private sealed class TestLogger : ILogger
    {
        public int WarningCount { get; private set; }

        public List<string> Messages { get; } = [];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Warning;

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            if (logLevel < LogLevel.Warning)
            {
                return;
            }

            WarningCount++;
            Messages.Add(formatter(state, exception));
        }
    }

    private sealed class MutableDevicePolicyProvider : IWindowsMlEpDevicePolicyProvider
    {
        public WindowsMlExecutionDevicePolicy Policy { get; set; }

        public Task<WindowsMlExecutionDevicePolicy> GetPolicyAsync(CancellationToken cancellationToken = default) =>
            cancellationToken.IsCancellationRequested
                ? Task.FromCanceled<WindowsMlExecutionDevicePolicy>(cancellationToken)
                : Task.FromResult(Policy);

        public void InvalidateCache()
        {
        }
    }

    private sealed class FixedSelectingBootstrapper(ExecutionProviderKind selected) : IExecutionProviderBootstrapper
    {
        public Task<ExecutionProviderBootstrapResult> BootstrapAsync(
            ExecutionProviderKind provider,
            bool allowDownloads,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _ = allowDownloads;
            return Task.FromResult(new ExecutionProviderBootstrapResult(
                provider,
                selected,
                Succeeded: true,
                Detail: "test bootstrap"));
        }

        public Task<ExecutionProviderBootstrapResult> CheckReadinessAsync(
            ExecutionProviderKind provider,
            CancellationToken cancellationToken) =>
            BootstrapAsync(provider, allowDownloads: false, cancellationToken);
    }

    private sealed class CpuSelectingDirectMlBootstrapper : IExecutionProviderBootstrapper
    {
        public Task<ExecutionProviderBootstrapResult> BootstrapAsync(
            ExecutionProviderKind provider,
            bool allowDownloads,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _ = allowDownloads;
            return Task.FromResult(new ExecutionProviderBootstrapResult(
                provider,
                ExecutionProviderKind.Cpu,
                Succeeded: false,
                Detail: "catalog unavailable",
                FailureReason: "catalog unavailable"));
        }

        public Task<ExecutionProviderBootstrapResult> CheckReadinessAsync(
            ExecutionProviderKind provider,
            CancellationToken cancellationToken) =>
            BootstrapAsync(provider, allowDownloads: false, cancellationToken);
    }

    private static InferenceSession CreateMinimalSession()
    {
        byte[] model = BuildIdentityOnnxModel();
        return new InferenceSession(model);
    }

    private static byte[] BuildIdentityOnnxModel() =>
    [
        0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
        0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
        0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
    ];

    [Fact]
    public async Task CreatePooledLatentSyncAsync_does_not_reuse_sessions_when_only_device_policy_changes()
    {
        string directory = Path.Join(Path.GetTempPath(), "trackdub-latentsync-policy-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        byte[] model = BuildIdentityOnnxModel();
        string unet = Path.Join(directory, "unet.onnx");
        string vaeEncoder = Path.Join(directory, "vae-encoder.onnx");
        string vaeDecoder = Path.Join(directory, "vae-decoder.onnx");
        string whisperEncoder = Path.Join(directory, "whisper-encoder.onnx");
        File.WriteAllBytes(unet, model);
        File.WriteAllBytes(vaeEncoder, model);
        File.WriteAllBytes(vaeDecoder, model);
        File.WriteAllBytes(whisperEncoder, model);

        var policy = new MutableDevicePolicyProvider
        {
            Policy = WindowsMlExecutionDevicePolicy.MaxPerformance,
        };
        OnnxExecutionSessionFactory.ResetForTests();
        OnnxExecutionSessionFactory.Initialize(
            new FixedSelectingBootstrapper(ExecutionProviderKind.Migraphx),
            policy);

        try
        {
            using var pool = new InferenceSessionPool(maxSessions: 16, enableMemoryAdmission: false);
            int creations = 0;

            InferenceSession CreateSession(string _, SessionOptions __)
            {
                Interlocked.Increment(ref creations);
                return new InferenceSession(model);
            }

            using (await OnnxExecutionSessionFactory.CreatePooledLatentSyncAsync(
                "latentsync",
                unet,
                vaeEncoder,
                vaeDecoder,
                whisperEncoder,
                ExecutionProviderKind.Migraphx,
                CancellationToken.None,
                pool,
                sessionFactory: CreateSession))
            {
            }

            Assert.Equal(4, creations);

            using (await OnnxExecutionSessionFactory.CreatePooledLatentSyncAsync(
                "latentsync",
                unet,
                vaeEncoder,
                vaeDecoder,
                whisperEncoder,
                ExecutionProviderKind.Migraphx,
                CancellationToken.None,
                pool,
                sessionFactory: CreateSession))
            {
            }

            Assert.Equal(4, creations);

            policy.Policy = WindowsMlExecutionDevicePolicy.MaxEfficiency;
            using (await OnnxExecutionSessionFactory.CreatePooledLatentSyncAsync(
                "latentsync",
                unet,
                vaeEncoder,
                vaeDecoder,
                whisperEncoder,
                ExecutionProviderKind.Migraphx,
                CancellationToken.None,
                pool,
                sessionFactory: CreateSession))
            {
            }

            // Catalog providers on Windows include the policy in the pool key. Elsewhere the
            // fingerprint stays default, so the same provider and model may reuse.
            Assert.Equal(OperatingSystem.IsWindows() ? 8 : 4, creations);
        }
        finally
        {
            OnnxExecutionSessionFactory.ResetForTests();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void GetPlatformBootstrapper_selects_the_platform_bootstrapper_at_run_time()
    {
        if (OperatingSystem.IsWindows())
        {
            return; // The Windows TFM always uses the Windows bootstrapper.
        }

        IExecutionProviderBootstrapper bootstrapper = OnnxExecutionSessionFactory.GetPlatformBootstrapper();

        if (OperatingSystem.IsLinux())
        {
            Assert.IsType<Trackdub.Inference.Onnx.ExecutionProviders.Linux.LinuxExecutionProviderBootstrapper>(bootstrapper);
        }
        else if (OperatingSystem.IsMacOSVersionAtLeast(10, 15))
        {
            Assert.IsType<Trackdub.Inference.Onnx.ExecutionProviders.Mac.MacExecutionProviderBootstrapper>(bootstrapper);
        }
        else
        {
            Assert.IsType<Trackdub.Inference.Onnx.ExecutionProviders.PortableExecutionProviderBootstrapper>(bootstrapper);
        }
    }
}

/// <summary>
/// These tests reset and replace the process-global bootstrapper (e.g. one that always selects
/// MIGraphX), which makes concurrent smoke/session tests see the wrong provider. xUnit runs
/// non-parallel collections only after every parallel collection has finished, so the swaps
/// cannot overlap any other test in the assembly.
/// </summary>
[CollectionDefinition(nameof(OnnxExecutionSessionFactoryStateCollection), DisableParallelization = true)]
public sealed class OnnxExecutionSessionFactoryStateCollection;
