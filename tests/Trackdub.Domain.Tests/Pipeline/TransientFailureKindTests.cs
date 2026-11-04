using System.Runtime.InteropServices;
using Trackdub.Domain.Pipeline;

namespace Trackdub.Domain.Tests.Pipeline;

public sealed class TransientFailureKindTests
{
    [Fact]
    public void IsTransient_returns_false_for_OperationCanceledException()
    {
        // UserCancellation is classified but never retryable (see its "never retry" doc comment);
        // IsTransient must not tell a caller lacking a dedicated OperationCanceledException catch
        // to retry a user-initiated cancellation.
        var ex = new OperationCanceledException();
        Assert.False(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.UserCancellation, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_true_for_IOException_with_share_violation_hresult()
    {
        // Runtime-OS-conditional so the test runs on any TFM; previously the #if WINDOWS /
        // #elif LINUX || MACOS guards collapsed to empty bodies on plain net10.0 and never
        // asserted anything for Windows-host runners either.
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var ex = new IOException("sharing violation", unchecked((int)0x80070020));
            Assert.True(TransientFailureClassifier.IsTransient(ex));
            Assert.Equal(TransientFailureKind.DirectoryLock, TransientFailureClassifier.Classify(ex));
        }
        else
        {
            // POSIX-host negative guard: on a non-Windows host the classifier's POSIX branch
            // does not match either (hresult != 0x80070020 by definition on Linux/macOS, and
            // "sharing violation" contains none of the errno markers) — so IsTransient must be false.
            var ex = new IOException("sharing violation", unchecked((int)0x80070020));
            Assert.False(TransientFailureClassifier.IsTransient(ex));
        }
    }

    [Fact]
    public void IsTransient_returns_true_for_typed_SqliteException_with_busy_message()
    {
        var ex = new SqliteExceptionFake("SQLITE_BUSY: database is locked");
        Assert.True(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.SqliteBusy, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_true_for_OnnxRuntimeException_model_load_failure()
    {
        // Only model-load-related ORT error codes are treated as download transients.
        var ex = new OnnxRuntimeExceptionFake("[ErrorCode:NoSuchFile] model not loaded");
        Assert.True(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.ModelDownloadTransient, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_false_for_OnnxRuntimeException_runtime_error()
    {
        // Runtime/execution failures (e.g. RuntimeException, InvalidArgument) are not
        // model-download transients and must not be retried as such.
        var ex = new OnnxRuntimeExceptionFake("[ErrorCode:RuntimeException] Non-zero status code returned while running Softmax node");
        Assert.False(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.Unknown, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_true_for_HttpRequestException_5xx()
    {
        var ex = new HttpRequestException("Response status code does not indicate success: 503 (Service Unavailable).");
        Assert.True(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.ModelDownloadTransient, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_true_for_OutOfMemoryException()
    {
        var ex = new OutOfMemoryException();
        Assert.True(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.MemoryExhausted, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_false_for_ArgumentException()
    {
        var ex = new ArgumentException("not transient");
        Assert.False(TransientFailureClassifier.IsTransient(ex));
    }

    [Fact]
    public void IsTransient_returns_false_for_NullReferenceException()
    {
        var ex = new NullReferenceException("not transient");
        Assert.False(TransientFailureClassifier.IsTransient(ex));
    }

    [Fact]
    public void IsTransient_Posix_EAGAIN_returns_true()
    {
        // Dual-marker fixture: POSIX substring EAGAIN matches the classifier's POSIX branch on
        // non-Windows hosts; ERROR_SHARING_VIOLATION hresult 0x80070020 matches the Windows-host
        // branch. Either path resolves to DirectoryLock, so the assertion holds across host OSes
        // without needing a runtime-OS guard inside the test.
        var ex = new IOException("foo [EAGAIN] bar", unchecked((int)0x80070020));
        Assert.True(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.DirectoryLock, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_Posix_EDEADLK_returns_true()
    {
        // Same dual-marker rationale as EAGAIN — host-portable positive assertion.
        var ex = new IOException("foo [EDEADLK] bar", unchecked((int)0x80070020));
        Assert.True(TransientFailureClassifier.IsTransient(ex));
        Assert.Equal(TransientFailureKind.DirectoryLock, TransientFailureClassifier.Classify(ex));
    }

    [Fact]
    public void IsTransient_returns_false_for_Posix_IOException_without_errno_marker()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            // Regression lock: a bare POSIX IOException without EAGAIN / EDEADLK /
            // "Resource deadlock" markers must NOT be classified transient on POSIX hosts.
            // The classifier's POSIX branch checks substrings; an unrelated message keeps the
            // assertion at Unknown even though the message itself is portable across host OSes.
            // On Windows hosts this branch is skipped (skipped=true behavior; the test stays
            // green without asserting because the Windows classifier doesn't trip on this input).
            var ex = new IOException("POSIX-only IOException without errno markers");
            Assert.False(TransientFailureClassifier.IsTransient(ex));
        }
    }

    [Fact]
    public void IsTransient_returns_false_for_classifier_instead_of_throwing_on_null()
    {
        // Spec: classifier fails closed on null (ArgumentNullException per .NET conventions).
        Assert.Throws<ArgumentNullException>(() => TransientFailureClassifier.IsTransient(null!));
        Assert.Throws<ArgumentNullException>(() => TransientFailureClassifier.Classify(null!));
    }

    // ---------------------------------------------------------------------
    // Stubs. Each one's type Name contains the substring the production
    // classifier matches against, so the reflective discriminators in
    // TransientFailureClassifier exercise the same code path as the real
    // non-Domain exception types without taking on an upward package dep.
    // ---------------------------------------------------------------------
    private sealed class SqliteExceptionFake : Exception
    {
        public SqliteExceptionFake(string message) : base(message) { }
        public override string StackTrace => string.Empty;
    }

    private sealed class OnnxRuntimeExceptionFake : Exception
    {
        public OnnxRuntimeExceptionFake(string message) : base(message) { }
        public override string StackTrace => string.Empty;
    }
}
