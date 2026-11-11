using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Trackdub.Contracts.Pipeline;

namespace Trackdub.Cloud.Jobs;

public sealed class JobService(IJobStore store, IArtifactStore artifacts, TimeProvider clock)
{
    public async Task<JobRecord> SubmitAsync(string owner, string key, CreateJobRequest request, CancellationToken token)
    {
        if (request.SchemaVersion != 1 || string.IsNullOrWhiteSpace(request.ProjectName) || request.ProjectName.Length > 120
            || request.Input is null || request.Input.UploadId == Guid.Empty || request.SourceLanguage != "en"
            || !CloudPresets.TryGet(request.Preset, request.TargetLanguage, out _)
            || string.IsNullOrWhiteSpace(key) || key.Length > 128 || key.Any(char.IsControl))
            throw new JobException("INVALID_INPUT", 400, "Invalid job request or idempotency key.");
        var upload = await store.GetUploadAsync(request.Input.UploadId, token).ConfigureAwait(false);
        if (upload is null || upload.OwnerId != owner || upload.Object is null)
            throw new JobException("INVALID_INPUT", 400, "Upload is not ready.");
        request = request with { ProjectName = request.ProjectName.Trim() };
        string hash = Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(
            new { request, upload.Object }, CloudJson.Options)));
        var replay = await store.FindSubmissionAsync(owner, key, hash, token).ConfigureAwait(false);
        if (replay is not null) return replay;
        await artifacts.VerifyAsync(upload.Object, token).ConfigureAwait(false);
        var now = clock.GetUtcNow();
        var job = new JobRecord
        {
            Id = Guid.NewGuid(),
            OwnerId = owner,
            Request = request,
            Input = upload.Object,
            RequestHash = hash,
            CreatedAt = now,
            UpdatedAt = now,
            QueueDeadlineAt = now.AddHours(1),
        };
        if (await store.AcceptAsync(job, key, upload.Id, token).ConfigureAwait(false)) return job;
        return await store.FindSubmissionAsync(owner, key, hash, token).ConfigureAwait(false)
            ?? throw new JobException("STORE_UNAVAILABLE", 503, "Job was not accepted. Retry with the same idempotency key.");
    }

    public async Task<JobRecord> GetOwnedAsync(string owner, Guid id, CancellationToken token)
    {
        var job = await store.GetAsync(id, token).ConfigureAwait(false);
        return job?.OwnerId == owner ? job : throw new JobException("NOT_FOUND", 404, "Job not found.");
    }

    public async Task<JobRecord> CancelAsync(string owner, Guid id, CancellationToken token)
    {
        await GetOwnedAsync(owner, id, token).ConfigureAwait(false);
        return await MutateAsync(id, job => job.OwnerId == owner ? JobRules.Cancel(job, clock.GetUtcNow())
            : throw new JobException("NOT_FOUND", 404, "Job not found."), token).ConfigureAwait(false);
    }

    public async Task<ExecutionClaim?> ClaimAsync(ExecutionClaim identity, CancellationToken token)
    {
        var job = await store.GetAsync(identity.JobId, token).ConfigureAwait(false);
        if (job is null) return null;
        var next = JobRules.Claim(job, identity, clock.GetUtcNow());
        if (next is null) return null;
        next = next with { Version = job.Version + 1 };
        return await store.CompareExchangeAsync(next, job.Version, token).ConfigureAwait(false)
            ? identity with { Generation = next.Generation } : null;
    }

    public Task<JobRecord> HeartbeatAsync(ExecutionClaim claim, CancellationToken token) => MutateAsync(claim.JobId, job =>
    {
        EnsureLease(job, claim);
        return job with { LeaseExpiresAt = clock.GetUtcNow().AddSeconds(90), UpdatedAt = clock.GetUtcNow() };
    }, token, claim);

    public Task<JobRecord> ReportProgressAsync(ExecutionClaim claim, PipelineProgressEvent progress, CancellationToken token) =>
        MutateAsync(claim.JobId, job =>
        {
            EnsureLease(job, claim);
            if (job.Progress is not null && progress.SequenceNumber <= job.Progress.Sequence) return job;
            return job with
            {
                Progress = new(job.AttemptId, progress.RunId?.ToString(), progress.SequenceNumber, progress.StageKey,
                progress.Phase, progress.PercentComplete, clock.GetUtcNow()),
                Phase = "pipeline"
            };
        }, token, claim);

    public Task<JobRecord> ReportCandidateAsync(ExecutionClaim claim, AttemptOutcome candidate, CancellationToken token) =>
        MutateAsync(claim.JobId, job =>
        {
            EnsureLease(job, claim);
            string prefix = AttemptPrefix(job);
            if (candidate.Claim != claim || candidate.Artifacts.Count > 16 || !candidate.ResultObject.Key.StartsWith(prefix, StringComparison.Ordinal)
                || candidate.Artifacts.Any(a => !a.Object.Key.StartsWith(prefix, StringComparison.Ordinal)))
                throw new JobException("INVALID_INPUT", 400, "Invalid candidate artifact inventory.");
            if (job.Candidate is not null && job.Candidate.ResultObject != candidate.ResultObject)
                throw new JobException("CANDIDATE_CONFLICT", 409, "Attempt candidate is immutable.");
            return job with { Candidate = candidate, Phase = "publishing" };
        }, token, claim);

    public async Task<JobRecord> MutateAsync(Guid id, Func<JobRecord, JobRecord> update, CancellationToken token, ExecutionClaim? claim = null)
    {
        for (int retry = 0; retry < 8; retry++)
        {
            var current = await store.GetAsync(id, token).ConfigureAwait(false)
                ?? throw new JobException("NOT_FOUND", 404, "Job not found.");
            var next = update(current);
            if (ReferenceEquals(next, current) || next == current) return current;
            next = next with { Version = current.Version + 1, UpdatedAt = clock.GetUtcNow() };
            if (await store.CompareExchangeAsync(next, current.Version, token, claim).ConfigureAwait(false)) return next;
        }
        throw new JobException("CONCURRENT_UPDATE", 503, "Job changed concurrently. Retry.");
    }

    private void EnsureLease(JobRecord job, ExecutionClaim claim)
    {
        if (!JobRules.OwnsLease(job, claim, clock.GetUtcNow()))
            throw new JobException("LEASE_LOST", 409, "Execution no longer owns the attempt.");
    }

    public static string OwnerKey(string owner) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(owner)));
    public static string AttemptPrefix(JobRecord job) => $"jobs/{OwnerKey(job.OwnerId)}/{job.Id}/attempts/{job.AttemptId}/";
}
