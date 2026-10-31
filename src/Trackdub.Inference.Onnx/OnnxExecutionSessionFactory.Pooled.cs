using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Pool;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.TensorRtRtx;
using Trackdub.Inference.Runtime.WinMlCatalog;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Onnx;

internal static partial class OnnxExecutionSessionFactory
{
    /// <summary>
    /// Engine-lifetime residency pin (audit §3A). Keeps the pooled session warm without
    /// holding an execution lease; call <see cref="AcquireAsync"/> only around each Run().
    /// </summary>
    internal sealed class PooledSingleSessionPin : IDisposable
    {
        private readonly SessionResidency residency;
        private readonly Func<CancellationToken, Task<SingleSessionLease>> acquire;
        private readonly string requestedProvider;
        private readonly string selectedProvider;
        private readonly string? bootstrapDetail;

        internal PooledSingleSessionPin(
            SessionResidency residency,
            Func<CancellationToken, Task<SingleSessionLease>> acquire,
            string requestedProvider,
            string selectedProvider,
            string? bootstrapDetail)
        {
            this.residency = residency;
            this.acquire = acquire;
            this.requestedProvider = requestedProvider;
            this.selectedProvider = selectedProvider;
            this.bootstrapDetail = bootstrapDetail;
        }

        public string RequestedProvider => requestedProvider;
        public string SelectedProvider => selectedProvider;
        public string? BootstrapDetail => bootstrapDetail;

        public Task<SingleSessionLease> AcquireAsync(CancellationToken cancellationToken) =>
            acquire(cancellationToken);

        public void Dispose() => residency.Dispose();
    }
}
