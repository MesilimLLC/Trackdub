namespace Trackdub.Cloud.Jobs;

public sealed record UploadRequest(long SizeBytes, string Sha256, string ContentType);
public sealed record UploadReservation(Guid Id, string Url, DateTimeOffset ExpiresAt, IReadOnlyDictionary<string, string> Headers);

public sealed class UploadService(IJobStore store, IArtifactStore artifacts, TimeProvider clock)
{
    public async Task<UploadReservation> CreateAsync(string owner, UploadRequest request, CancellationToken token)
    {
        if (request.SizeBytes is <= 0 or > CloudLimits.MaxUploadBytes || request.Sha256 is null || request.Sha256.Length != 64
            || !request.Sha256.All(Uri.IsHexDigit) || request.ContentType != "video/mp4")
            throw new JobException("INVALID_INPUT", 400, "Expected an MP4 up to 1 GiB and a SHA256 checksum.");
        var id = Guid.NewGuid();
        var upload = new UploadRecord(id, owner, null, $"uploads/{JobService.OwnerKey(owner)}/{id}/input.mp4",
            request.SizeBytes, request.Sha256.ToLowerInvariant(), request.ContentType, clock.GetUtcNow().AddMinutes(15));
        if (!await store.PutUploadAsync(upload, false, token).ConfigureAwait(false))
            throw new JobException("STORE_UNAVAILABLE", 503, "Upload reservation could not be saved.");
        string url = await artifacts.SignUploadAsync(upload, token).ConfigureAwait(false);
        return new(id, url, upload.ExpiresAt, new Dictionary<string, string>
        {
            ["Content-Type"] = upload.ContentType,
            ["Content-Length"] = upload.SizeBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
            ["x-amz-checksum-sha256"] = Convert.ToBase64String(Convert.FromHexString(upload.Sha256)),
        });
    }

    public async Task<UploadRecord> CompleteAsync(string owner, Guid id, string versionId, CancellationToken token)
    {
        var upload = await store.GetUploadAsync(id, token).ConfigureAwait(false);
        if (upload is null || upload.OwnerId != owner) throw new JobException("NOT_FOUND", 404, "Upload not found.");
        if (upload.Object is not null)
        {
            if (upload.Object.VersionId != versionId) throw new JobException("INPUT_CHANGED", 409, "Upload is already bound to another version.");
            return upload;
        }
        if (upload.ExpiresAt <= clock.GetUtcNow()) throw new JobException("UPLOAD_EXPIRED", 409, "Upload reservation expired.");
        var reference = await artifacts.CompleteUploadAsync(upload, versionId, token).ConfigureAwait(false);
        var ready = upload with { Object = reference };
        if (await store.PutUploadAsync(ready, true, token).ConfigureAwait(false)) return ready;
        var current = await store.GetUploadAsync(id, token).ConfigureAwait(false);
        if (current is not null && current.OwnerId == owner && current.Object == reference) return current;
        throw new JobException("INPUT_CHANGED", 409, "Upload completion changed concurrently.");
    }
}
