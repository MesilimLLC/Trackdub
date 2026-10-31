using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Serving;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Serving;

/// <summary>
/// Resident models pinned in the session pool. Each model is opened through
/// <see cref="OnnxExecutionSessionFactory.PinPooledSingleAsync"/>, so EP bootstrap, EP-context
/// artifacts, the TensorRT RTX runtime cache, TRT init fallback and pool admission all behave as
/// they do for pipeline stages; the residency pin keeps the session out of idle eviction.
/// </summary>
internal sealed class ResidentInferenceHost : IResidentInferenceHost
{
    internal const string EngineFamily = "serve";

    private readonly Func<CancellationToken, Task<IReadOnlyList<ExecutionProviderKind>>> availableProviders;
    private readonly InferenceSessionPool? pool;
    private readonly Func<string, SessionOptions, InferenceSession>? sessionFactory;
    private readonly ConcurrentDictionary<string, ResidentEntry> entries = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim loadGate = new(1, 1);
    private bool disposed;

    /// <param name="availableProviders">
    /// Providers usable on this machine; consulted only when a spec has no preferred provider.
    /// The first one in <see cref="ExecutionProviderProbeOrder.Global"/> order wins.
    /// </param>
    public ResidentInferenceHost(
        Func<CancellationToken, Task<IReadOnlyList<ExecutionProviderKind>>>? availableProviders = null,
        InferenceSessionPool? pool = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null)
    {
        this.availableProviders = availableProviders
            ?? (static _ => Task.FromResult<IReadOnlyList<ExecutionProviderKind>>([ExecutionProviderKind.Cpu]));
        this.pool = pool;
        this.sessionFactory = sessionFactory;
    }

    /// <summary>Test seam: invoked with the model name immediately before each run.</summary>
    internal Action<string>? BeforeRun { get; set; }

    /// <summary>Test seam: invoked with the model name after its entry resolves, before the lease is acquired.</summary>
    internal Action<string>? BeforeAcquire { get; set; }

    public async Task<ResidentModelInfo> LoadAsync(ResidentModelSpec spec, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (string.IsNullOrWhiteSpace(spec.Name))
        {
            throw new ResidentInferenceException(ResidentInferenceErrorReasons.BadInputs, "A resident model needs a name.");
        }

        if (!File.Exists(spec.ModelPath))
        {
            throw new ResidentInferenceException(
                ResidentInferenceErrorReasons.ModelNotFound, $"Model file not found: {spec.ModelPath}");
        }

        await loadGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Release the replaced model first: once unpinned it would otherwise stay pooled,
            // holding device memory until LRU eviction reached it.
            if (entries.TryRemove(spec.Name, out ResidentEntry? replaced))
            {
                replaced.Dispose();
                await EvictAsync(spec.Name).ConfigureAwait(false);
            }

            ResidentEntry entry = await OpenAsync(spec, cancellationToken).ConfigureAwait(false);
            entries[spec.Name] = entry;
            return entry.Describe();
        }
        finally
        {
            loadGate.Release();
        }
    }

    public async Task<ResidentRunResult> RunAsync(
        string modelName,
        IReadOnlyList<ServeTensor> inputs,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelName);
        ArgumentNullException.ThrowIfNull(inputs);
        while (true)
        {
            ResidentEntry entry = await GetRunnableAsync(modelName, cancellationToken).ConfigureAwait(false);
            BeforeAcquire?.Invoke(modelName);
            using OnnxExecutionSessionFactory.SingleSessionLease lease =
                await entry.Pin.AcquireAsync(cancellationToken).ConfigureAwait(false);
            // A concurrent load, unload or poisoned reopen can swap the entry while the lease was
            // acquired; the pin would resurrect the removed model, so run only on the current one.
            if (!entries.TryGetValue(modelName, out ResidentEntry? current) || !ReferenceEquals(current, entry))
            {
                continue;
            }

            InferenceSession session = lease.Session;
            ServeTensorConverter.Validate(session, inputs);

            try
            {
                BeforeRun?.Invoke(entry.Spec.Name);
                var stopwatch = Stopwatch.StartNew();
                IReadOnlyList<ServeTensor> outputs = Run(session, inputs, cancellationToken);
                stopwatch.Stop();
                return new ResidentRunResult(entry.Spec.Name, lease.SelectedProvider, outputs, stopwatch.Elapsed);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (ex is not ResidentInferenceException)
            {
                // Device errors (sticky CUDA faults, lost devices) can leave the session unusable;
                // the next request reopens it instead of failing forever.
                entry.Poisoned = true;
                throw new ResidentInferenceException(
                    ResidentInferenceErrorReasons.InferFailed,
                    $"Inference failed on {lease.SelectedProvider}: {ex.Message}",
                    ex);
            }
        }
    }

    public async Task<ResidentRunResult?> WarmAsync(
        string modelName,
        IReadOnlyList<ServeTensor>? inputs = null,
        CancellationToken cancellationToken = default)
    {
        if (inputs is null)
        {
            ResidentEntry entry = await GetRunnableAsync(modelName, cancellationToken).ConfigureAwait(false);
            inputs = ServeTensorConverter.SynthesizeZeroInputs(entry.Inputs);
            if (inputs is null)
            {
                return null;
            }
        }

        return await RunAsync(modelName, inputs, cancellationToken).ConfigureAwait(false);
    }

    public async Task<bool> UnloadAsync(string modelName)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Serialized with load and poisoned-reopen so neither can re-add the model after
        // the caller was told it is unloaded.
        await loadGate.WaitAsync().ConfigureAwait(false);
        try
        {
            if (!entries.TryRemove(modelName, out ResidentEntry? entry))
            {
                return false;
            }

            entry.Dispose();
            await EvictAsync(modelName).ConfigureAwait(false);
            return true;
        }
        finally
        {
            loadGate.Release();
        }
    }

    public IReadOnlyList<ResidentModelInfo> Describe() =>
        entries.Values.OrderBy(entry => entry.Spec.Name, StringComparer.Ordinal).Select(entry => entry.Describe()).ToList();

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        disposed = true;
        foreach (ResidentEntry entry in entries.Values)
        {
            entry.Dispose();
        }

        entries.Clear();
        loadGate.Dispose();
    }

    private async Task<ResidentEntry> GetRunnableAsync(string modelName, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (!entries.TryGetValue(modelName, out ResidentEntry? entry))
        {
            throw new ResidentInferenceException(
                ResidentInferenceErrorReasons.UnknownModel, $"No resident model named '{modelName}'.");
        }

        if (!entry.Poisoned)
        {
            return entry;
        }

        await loadGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Another request may have reopened (or unloaded) it while this one waited.
            if (!entries.TryGetValue(modelName, out ResidentEntry? current))
            {
                throw new ResidentInferenceException(
                    ResidentInferenceErrorReasons.UnknownModel, $"No resident model named '{modelName}'.");
            }

            if (!current.Poisoned)
            {
                return current;
            }

            // The poisoned session is still pooled under the same key; evict it so the reopen
            // builds a fresh one instead of handing it back.
            current.Dispose();
            await EvictAsync(modelName).ConfigureAwait(false);
            ResidentEntry reopened = await OpenAsync(current.Spec, cancellationToken).ConfigureAwait(false);
            entries[modelName] = reopened;
            return reopened;
        }
        finally
        {
            loadGate.Release();
        }
    }

    private Task<int> EvictAsync(string modelName) =>
        (pool ?? InferenceSessionPool.Shared).EvictModelAsync(EngineFamily, modelName);

    private async Task<ResidentEntry> OpenAsync(ResidentModelSpec spec, CancellationToken cancellationToken)
    {
        List<ExecutionProviderKind> candidates = await BuildCandidatesAsync(spec, cancellationToken).ConfigureAwait(false);
        var fallbackChain = new List<ResidentFallbackStep>();
        string requestedLabel = OnnxExecutionSessionFactory.FormatProviderLabel(candidates[0]);

        foreach (ExecutionProviderKind candidate in candidates)
        {
            string candidateLabel = OnnxExecutionSessionFactory.FormatProviderLabel(candidate);
            OnnxExecutionSessionFactory.PooledSingleSessionPin pin;
            try
            {
                pin = await OnnxExecutionSessionFactory.PinPooledSingleAsync(
                    EngineFamily,
                    spec.ModelPath,
                    candidate,
                    cancellationToken,
                    pool,
                    modelId: spec.Name,
                    additionalTrtOptions: spec.ProviderOptions,
                    sessionFactory: sessionFactory,
                    allowTrtInitFallback: !spec.RequireProvider).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                fallbackChain.Add(new ResidentFallbackStep(candidateLabel, ex.Message));
                if (spec.RequireProvider)
                {
                    throw new ResidentInferenceException(
                        ResidentInferenceErrorReasons.ProviderUnavailable,
                        $"Required provider '{candidateLabel}' failed to load '{spec.Name}': {ex.Message}",
                        ex);
                }

                continue;
            }

            if (!string.Equals(pin.SelectedProvider, candidateLabel, StringComparison.Ordinal))
            {
                // The factory settled on another provider (bootstrap or TRT init fallback).
                string reason = pin.BootstrapDetail ?? $"Provider '{candidateLabel}' was unavailable.";
                if (spec.RequireProvider)
                {
                    pin.Dispose();
                    throw new ResidentInferenceException(
                        ResidentInferenceErrorReasons.ProviderUnavailable,
                        $"Required provider '{candidateLabel}' is unavailable (settled on '{pin.SelectedProvider}'): {reason}");
                }

                fallbackChain.Add(new ResidentFallbackStep(candidateLabel, reason));
            }

            try
            {
                using OnnxExecutionSessionFactory.SingleSessionLease lease =
                    await pin.AcquireAsync(cancellationToken).ConfigureAwait(false);
                return new ResidentEntry(
                    spec,
                    pin,
                    requestedLabel,
                    fallbackChain,
                    ServeTensorConverter.Describe(lease.Session.InputMetadata),
                    ServeTensorConverter.Describe(lease.Session.OutputMetadata));
            }
            catch
            {
                pin.Dispose();
                throw;
            }
        }

        string attempted = string.Join("; ", fallbackChain.Select(step => $"{step.Provider}: {step.Reason}"));
        throw new ResidentInferenceException(
            ResidentInferenceErrorReasons.LoadFailed,
            $"Every provider failed to load '{spec.Name}' ({attempted}).");
    }

    private async Task<List<ExecutionProviderKind>> BuildCandidatesAsync(
        ResidentModelSpec spec,
        CancellationToken cancellationToken)
    {
        ExecutionProviderKind first;
        if (spec.PreferredProvider is ExecutionProviderKind preferred)
        {
            first = ExecutionProviderTokens.ResolvePlatformPin(preferred);
        }
        else
        {
            IReadOnlyList<ExecutionProviderKind> available =
                await availableProviders(cancellationToken).ConfigureAwait(false);
            first = ExecutionProviderProbeOrder.Global.FirstOrDefault(available.Contains, ExecutionProviderKind.Cpu);
        }

        return spec.RequireProvider || first is ExecutionProviderKind.Cpu
            ? [first]
            : [first, ExecutionProviderKind.Cpu];
    }

    private static IReadOnlyList<ServeTensor> Run(
        InferenceSession session,
        IReadOnlyList<ServeTensor> inputs,
        CancellationToken cancellationToken)
    {
        using var inputValues = new DisposableList<OrtValue>(inputs.Count);
        try
        {
            foreach (ServeTensor input in inputs)
            {
                inputValues.Add(ServeTensorConverter.ToOrtValue(input));
            }

            using var runOptions = new RunOptions();
            using CancellationTokenRegistration registration = cancellationToken.UnsafeRegister(
                static state => ((RunOptions)state!).Terminate = true, runOptions);
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(
                runOptions,
                inputs.Select(input => input.Name).ToList(),
                inputValues,
                session.OutputNames);

            // ORT honors Terminate between nodes and may still finish; never return results
            // produced after cancellation.
            cancellationToken.ThrowIfCancellationRequested();
            var results = new List<ServeTensor>(outputs.Count);
            for (int i = 0; i < outputs.Count; i++)
            {
                results.Add(ServeTensorConverter.FromOrtValue(session.OutputNames[i], outputs[i]));
            }

            return results;
        }
        catch (OnnxRuntimeException) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }
    }

    private sealed class DisposableList<T>(int capacity) : List<T>(capacity), IDisposable
        where T : IDisposable
    {
        public void Dispose()
        {
            foreach (T item in this)
            {
                item.Dispose();
            }
        }
    }

    private sealed class ResidentEntry(
        ResidentModelSpec spec,
        OnnxExecutionSessionFactory.PooledSingleSessionPin pin,
        string requestedProvider,
        IReadOnlyList<ResidentFallbackStep> fallbackChain,
        IReadOnlyList<ServeTensorSpec> inputs,
        IReadOnlyList<ServeTensorSpec> outputs) : IDisposable
    {
        private volatile bool poisoned;

        public ResidentModelSpec Spec => spec;
        public OnnxExecutionSessionFactory.PooledSingleSessionPin Pin => pin;
        public string SelectedProvider => pin.SelectedProvider;
        public IReadOnlyList<ServeTensorSpec> Inputs => inputs;

        public bool Poisoned
        {
            get => poisoned;
            set => poisoned = value;
        }

        public ResidentModelInfo Describe() => new(
            spec.Name,
            spec.ModelPath,
            requestedProvider,
            pin.SelectedProvider,
            fallbackChain,
            pin.BootstrapDetail,
            inputs,
            outputs,
            poisoned);

        public void Dispose() => pin.Dispose();
    }
}
