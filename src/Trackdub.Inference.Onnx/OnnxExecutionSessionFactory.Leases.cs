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
    /// <summary>Disposes the pool lease when present, else the raw session.</summary>
    private static void DisposeLeaseOrSession(SessionLease? poolLease, InferenceSession session)
    {
        if (poolLease is not null)
        {
            poolLease.Dispose();
        }
        else
        {
            TensorRtRtxTeardownGuard.DisposeSafely(session);
        }
    }

    /// <summary>Disposes the shared bundle lease when present, else each lease/session pair.</summary>
    private static void DisposeBundleOrLeases(
        SessionLeaseBundle? bundle,
        params (SessionLease? Lease, InferenceSession Session)[] pairs)
    {
        if (bundle is not null)
        {
            bundle.Dispose();
            return;
        }

        foreach ((SessionLease? lease, InferenceSession session) in pairs)
        {
            DisposeLeaseOrSession(lease, session);
        }
    }

    internal sealed record SingleSessionLease(
        InferenceSession Session,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLease? PoolLease { get; init; }

        /// <summary>Set when the lease came from the shared pool — used for residency pinning.</summary>
        internal SessionPoolKey? ResolvedPoolKey { get; init; }

        public void Dispose()
        {
            DisposeLeaseOrSession(PoolLease, Session);
        }
    }

    internal sealed record Qwen3AsrSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderInitSession,
        InferenceSession DecoderStepSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLeaseBundle? PoolBundle { get; init; }
        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderInitPoolLease { get; init; }
        internal SessionLease? DecoderStepPoolLease { get; init; }

        public void Dispose()
        {
            DisposeBundleOrLeases(
                PoolBundle,
                (EncoderPoolLease, EncoderSession),
                (DecoderInitPoolLease, DecoderInitSession),
                (DecoderStepPoolLease, DecoderStepSession));
        }
    }

    internal sealed record LatentSyncSessionLease(
        InferenceSession UNetSession,
        InferenceSession VaeEncoderSession,
        InferenceSession VaeDecoderSession,
        InferenceSession WhisperEncoderSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLeaseBundle? PoolBundle { get; init; }
        internal SessionLease? UNetPoolLease { get; init; }
        internal SessionLease? VaeEncoderPoolLease { get; init; }
        internal SessionLease? VaeDecoderPoolLease { get; init; }
        internal SessionLease? WhisperEncoderPoolLease { get; init; }

        public void Dispose()
        {
            DisposeBundleOrLeases(
                PoolBundle,
                (UNetPoolLease, UNetSession),
                (VaeEncoderPoolLease, VaeEncoderSession),
                (VaeDecoderPoolLease, VaeDecoderSession),
                (WhisperEncoderPoolLease, WhisperEncoderSession));
        }
    }

    internal sealed record NemotronAsrSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderJointSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderJointPoolLease { get; init; }

        public void Dispose()
        {
            DisposeLeaseOrSession(EncoderPoolLease, EncoderSession);
            DisposeLeaseOrSession(DecoderJointPoolLease, DecoderJointSession);
        }
    }

    internal sealed record WhisperSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderPoolLease { get; init; }

        public void Dispose()
        {
            DisposeLeaseOrSession(EncoderPoolLease, EncoderSession);
            DisposeLeaseOrSession(DecoderPoolLease, DecoderSession);
        }
    }

    internal sealed record OpusSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        /// <summary>Provider the sessions actually run on, after any downgrade.</summary>
        internal ExecutionProviderKind SelectedProviderKind { get; init; } = ExecutionProviderKind.Cpu;

        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderPoolLease { get; init; }

        public void Dispose()
        {
            DisposeLeaseOrSession(EncoderPoolLease, EncoderSession);
            DisposeLeaseOrSession(DecoderPoolLease, DecoderSession);
        }
    }
}
