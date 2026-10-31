# SnapSoft application contract and functional acceptance

Application implementation was approved on 2026-10-21. AWS provisioning, deployment,
paid GPU runs and production cutover were not authorized. This document supplies the
application integration contract; infrastructure allocations remain proposed until
SnapSoft confirms them. No AWS readiness or dubbing quality is claimed from local tests.

## Application topology

`Trackdub.Api -> Trackdub.Cloud.Aws -> Trackdub.Cloud -> Trackdub.Contracts -> Domain`.
The CPU API contains no SDK, Composition, ONNX, CUDA or inference project references.
It needs .NET 10 and CPU `ffprobe` for uploaded-media validation. `Dockerfile.api`
supplies these. HTTP requests never wait for inference. Dispatch/reconciliation is one
bounded background loop in the API deployment, not another service.

`Trackdub.CloudJobAdapter` wraps the existing CLI as a separate process. It never runs
a second pipeline implementation. `Dockerfile.cloud-worker` requires `SERVER_IMAGE`
to name an approved **digest built with the new CLI protocol**. Historical server images
do not have `--result-file` or `--progress-file`; copying the adapter into one is insufficient.
Existing `server` and `rtx` stages and entrypoints remain intact.
Container CI enforces the server image contract before publication: the
`TRACKDUB_GPU_PROFILE=server` marker, the Ubuntu 24.04 base, `setsid`, the classic
TensorRT/cuDNN/CUDA sonames, the ONNX Runtime native libraries, and the
`--result-file`/`--progress-file` machine protocol.
`poc-server-v1` remains the accepted baseline. `poc-server-quality-v1` is an evaluation-only
CLI preset and is not part of this acceptance contract until its quality, latency, VRAM and
cost evidence is reviewed.

There is deliberately no production fake scheduler. `IJobScheduler` is the integration
seam. SnapSoft must select Batch or ECS before one concrete adapter is implemented.
Until that adapter is registered, `/health` reports `processingEnabled: false`, intake
returns 503, and no accepted jobs are placed in an in-memory queue. Merely setting
`Cloud__ProcessingEnabled=true` cannot enable a missing scheduler.

## API and authentication

- `POST /uploads`: reserve a private S3 key; return upload ID, presigned PUT URL, expiry
  and required content-type/checksum headers. Limit: 100 MiB, MP4 audio/video, five
  minutes, maximum 1920x1080. Declared metadata is not proof of valid media.
- Client PUTs the bytes directly to S3, retaining `x-amz-version-id`. S3 CORS must
  permit the configured portal origin and expose that header. Presigned URLs expire
  after 15 minutes and are bearer capabilities; never log or persist them as identity.
- `POST /uploads/{id}/complete` with `{ "versionId": "..." }`: verify the exact version,
  length, SHA256 and actual ffprobe metadata before setting the owned upload ready.
  The signed PUT binds exact Content-Length as well as checksum. Browsers set Content-Length
  automatically from the blob; do not manually override their forbidden header. A different
  actual length invalidates the signature. Real S3 rejection still needs AWS acceptance.
  Replacing the key later cannot change that reference. No arbitrary URLs/buckets/keys.
- `POST /jobs`: schema v1 request plus one `Idempotency-Key`. Languages `en`/`es`,
  preset `poc-server-v1`; project name is display text. Return 202, Location and Retry-After
  only after atomic job/idempotency/outbox creation and upload-condition validation.
  Same owner/key/body returns the original job after API restart; changed body returns 409.
- `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `GET /jobs/{id}/artifacts`: JWT ownership
  enforced; foreign IDs return 404. Artifact listing is empty before final publication.
  Download URLs expire after five minutes and can be refreshed by listing again.
- JWT signature, issuer, audience, expiry, subject and `token_use=access` are checked.
  Algorithms are restricted to RS256. `Cloud__JwtAudience` validates audience-bearing
  access tokens. Cognito access tokens without `aud` require `Cloud__JwtClientId` and
  matching `client_id`. ID tokens, Better Auth cookies and unsigned tokens are rejected.
  Actual Cognito resource/client registration, PKCE and web flow remain SnapSoft alignment work.
- Configure exact CORS origins; allow Authorization, Content-Type, Idempotency-Key.
  No browser/desktop receives AWS credentials. Error envelopes omit paths, secrets and stacks.

The participating portal currently uses `/api/dubs`. No compatibility route or portal
cutover was added without that participation decision. The AWS records remain the only
new-job authority. Do not migrate/delete Cloudflare services or legacy media. Before
cutover, decide retained history/migration separately and disable old intake explicitly.

## Durable state and scheduler contract

Create one DynamoDB table with string partition key `pk` and no sort key. Create a
`pending-work` GSI: string partition key `work`, numeric sort key `created`, projection
KEYS_ONLY. Active job items set `work=active`; terminal items omit it. Discovery is
eventually consistent and paginated. Claims, versions and idempotency use base-table
conditional writes and strongly consistent reads. No uniqueness decision uses the GSI.

Item keys: `job#{uuid}`, `upload#{uuid}`, `idempotency#{owner/key hash}`,
`outbox#{dispatch uuid}`. Acceptance uses one TransactWriteItems containing three
conditional creates and an exact ready-upload/owner/version-reference condition. Metadata
is limited to 128 KiB per serialized item; full strict run results live in S3. No TTL is
used for uniqueness. Retain idempotency mappings and dispatch intents at least as long
as the associated job; lifecycle deletion is a separately deployed operational policy.

The API loop discovers at most 25 jobs per pass, runs once per minute, and has a 50-second
pass bound. Multiple replicas claim dispatch with a 90-second CAS lease. A failed job
operation is logged and does not block other jobs in that production pass.

`IJobScheduler.SubmitAsync` consumes only job/attempt/dispatch IDs. The selected adapter
must supply a fixed approved image/definition and trusted execution identity, never client
command/environment overrides. `DescribeAsync(dispatchId)` must discover **all** matching
executions, including lost-submit-response duplicates and terminal jobs. `Stopped` means
actual cessation, not an accepted termination request. Preserve exit codes. Discovery
lag is not proof that an ambiguous submission never ran. Retain dispatch correlation long
enough for reconciliation. Batch attempts must be 1; ECS uses a stable client token and
pinned task definition. No automatic logical pipeline retry is currently enabled.

Only one execution can conditionally claim an attempt before input download. Claim identity
includes job, attempt, dispatch, execution and fencing generation. Workers renew every
15 seconds, lease 90 seconds; failed renewal cancels work conservatively before expiry.
Lease expiry alone never grants replacement authority. No stale generation may publish.
Enforce one GPU task per host in the selected scheduler and bind all containers on that
host to the same local `/data/jobs` root; the supervisor also holds an exclusive host lock.
Do not share this lock, WAL projects or writable model indices across a fleet filesystem.

Queue deadline is one hour; execution staging through publication is two hours. Running
cancel remains cancelRequested until scheduler cessation. Cancellation committed before
final CAS blocks success. Completion committed first stays succeeded. Exit 137/143 alone
does not prove cancellation. No replacement while an earlier execution could still live.

## S3 and publication

One configured private versioned encrypted bucket can hold operational objects. Block
public access. Keys use hashed owner IDs: `uploads/{ownerHash}/{uploadId}/input.mp4` and
`jobs/{ownerHash}/{jobId}/attempts/{attemptId}/...`. Authoritative references contain key,
version ID, size and SHA256; signed URLs never serve as durable identity.

The supervisor rechecks input bytes and ffprobe limits, launches fixed argv without a
shell, uses private attempt/project/tmp paths, and verifies artifact bytes and path
containment. Symbolic-link/reparse escapes are rejected. Per-stream logs are bounded to
4 MiB. Linux uses a separate process group; SIGTERM is forwarded, then the group is killed
after a 20-second grace and the direct child is reaped, within a 30-second container stop window.

Strict result is a new atomic file; progress is separate JSONL, never parsed from diagnostic
logs. A stdout summary followed by exit 139 fails. Exit 0 without a valid result fails.
ASR, translation, TTS and export must succeed and a nonempty verified video must exist.
The generic CLI can support partial goals; the cloud POC applies the stronger stage contract.

Upload artifacts and `result.json`, then a bounded `candidate.json` before the fenced
DynamoDB candidate write. Retry publication in the same attempt without rerunning CLI.
If the candidate write is lost, reconciliation recovers the deterministic candidate object,
checks claim identity, and verifies its referenced versions/hashes. Final success requires
CLI exit 0, adapter/scheduler exit 0, typed result, verified inventory and final DynamoDB CAS.
Upload failures never become success. Keep unreferenced objects for a separately configured
retention window; cleanup must not delete active attempt evidence.

## Configuration and least-privilege permission requirements

CPU API: `Cloud__TableName`, `Cloud__BucketName`, `Cloud__JwtIssuer`,
`Cloud__JwtAudience`, optional `Cloud__JwtClientId`, `Cloud__AllowedOrigins__0`,
`Cloud__ProcessingEnabled` (default false), and AWS region through the SDK's normal
configuration. Default task-role credential chain only; do not persist access keys.

Supervisor: `TRACKDUB_CLOUD_TABLE`, `TRACKDUB_CLOUD_BUCKET`, `TRACKDUB_JOB_ID`,
`TRACKDUB_ATTEMPT_ID`, `TRACKDUB_DISPATCH_ID`, `TRACKDUB_EXECUTION_ID`, AWS region.
The concrete scheduler adapter must populate execution ID from trusted task/job identity;
clients cannot choose it. CLI path is `/app/Trackdub.Cli.dll`; model root `/data/models`;
jobs root `/data/jobs`. Prepare writable UID/GID 10001 ownership before scheduling.

| Role | Required application permissions | Scope |
|---|---|---|
| API task role | `dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:ConditionCheckItem`, `dynamodb:Query` | Agreed table; Query on pending-work index |
| API task role | S3 PutObject, GetObject, GetObjectVersion | Owned upload/artifact prefixes; signed transfers remain ownership-checked in API |
| API task role | Selected scheduler submit, describe/discover, stop | Fixed approved queue/definition or cluster/task definition; enumerate exact actions after service selection |
| Supervisor task role | DynamoDB GetItem, PutItem | Table; conditional claim/heartbeat/progress/candidate logic shared with API |
| Supervisor task role | S3 GetObjectVersion, PutObject | Inputs and attempt outputs; no DeleteObject or public ACL changes |
| Execution/host role | Registry image pulls, logs, bootstrap mounts | Separate from API/supervisor application roles |

The acceptance transaction authorizes its underlying Put and ConditionCheck operations;
`TransactWriteItems` is the SDK API operation, not the item-level permission grant. See
[AWS transaction IAM guidance](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis-iam.html).

Add KMS Decrypt/GenerateDataKey permissions as needed for the selected SSE-KMS bucket/key,
with an agreed key policy and S3 service scope. Confirm multipart permissions if multipart uploads are later added;
this implementation uses bounded single PUT. Workers are trusted POC workloads: table IAM
is not per-job tenant isolation. The application fences concurrency; stronger job-specific
IAM is deferred. Cognito provisioning, task definitions, IAM deployment, registry access,
GPU AMI/driver, EBS lifecycle and AWS IaC/deployment remain the agreed SnapSoft workstream.

## Preset and prepared storage

`poc-server-v1` is a built-in CLI preset, not an arbitrary user preset file. It selects
Parakeet ASR, MADLAD quantized translation and Kokoro default TTS; VAD on CPU and selected
inference stages pinned to CUDA. Voice `ef_dora` is supplied for `SPEAKER_00` in the one-speaker
fixture. `docker/poc-server-preset.json` pins model IDs, variants and manifest hash.
`docker/poc-server-models.txt` is its seed list. Existing historical TRT warm/seed lists
remain available but do not substitute for this CUDA baseline's quantized MADLAD bundle.

Target GPU/VRAM, provider qualification and acceptable optional skips remain acceptance
decisions. Do not enable intake simply because packages/models/providers are registered.
Persist model/cache preparation separately from job projects. With `/data/models`, headless
settings/index/log roots derive there; `/logs` is not automatically the application log root.
`TRACKDUB_ENGINE_CACHE_ROOT` is supported. Project SQLite stays inside each project and is
not a cloud job database. No project/SQLite snapshots are published in this baseline.
An EBS bind mount establishes no replacement-host recovery policy by itself.

With only `--model-directory /data/models` and no separate log/cache overrides, the
source resolves these exact paths:

| Purpose | Path |
|---|---|
| Model cache and index | `/data/models`, `/data/models/model-cache-records.json` |
| Persisted settings | `/data/models/settings.json` |
| Application log | `/data/models/trackdub.log` |
| Default engine cache | `/data/models/EngineCache` |
| Tool and component caches | `/data/models/tools`, `/data/models/components` |
| Supervisor project database | `/data/jobs/{jobId}/{attemptId}/{generation}/project/trackdub.db` |

An explicit `TRACKDUB_ENGINE_CACHE_ROOT=/data/engine-cache` replaces the default engine
cache path. Supervisor stdout/stderr logs live in its private attempt workspace, separately
from the application's persisted log. Model/cache seeding must finish before job scheduling.

## Functional acceptance checklist

- [ ] SnapSoft selects scheduler and confirms host/region/quota, CPU hosting, auth flow,
  registry pull mechanism, storage durability/retention and funded responsibility boundary.
- [ ] Implement and locally test only the selected scheduler adapter; confirm stable dispatch
  discovery, fixed image/definition, duplicate execution control and actual stop semantics.
- [ ] Record source commits, server/adapter digests, task-definition revision, fixture hash,
  manifest hash, GPU/AMI/driver/ORT identities. Build base server with the new machine protocol.
- [ ] Validate IAM denies unauthorized access; validate real DynamoDB transactions, S3
  versioning/checksum/CORS/presigning, Cognito JWT flow and exact portal origins in AWS.
- [ ] Seed the pinned baseline and verify provider/model/variant/voice selections and GPU
  admission on the target host. No unexplained CPU fallback for a required CUDA stage.
- [ ] Submit the licensed `tests/Fixtures/cloud-poc/en-speech.mp4`, run EN to ES, verify
  nonempty ASR/translation/TTS, audible Spanish, subtitles, stream metadata and checksum downloads.
- [ ] Exercise restart after acceptance, duplicate submission, lost scheduler response,
  foreign-owner denial, queued/running cancellation, native crash, timeout and publication interruption.
- [ ] Replace the container on the prepared host; confirm model/cache reuse and artifact retrieval.
  Test replacement EC2 host only if that durability scope is separately agreed.
- [ ] Record actual normal process exits plus durable final status; stdout/result alone is insufficient.
- [ ] Settle portal compatibility/history policy before cutover. No automatic legacy migration/deletion.
- [ ] Sign off functional evidence and handoff. No formal throughput benchmark or cost report is included.

Local tests use fakes, actual subprocess fixtures, a loopback DynamoDB Local container and
CPU-only Linux containers. They do not establish real IAM, Cognito, S3, scheduler or GPU readiness.

Native local artifact check after downloading a result and preserving its relative project paths:

```sh
dotnet Trackdub.CloudJobAdapter.dll verify-artifacts result.json ./project 0 0
```

The last two arguments must be the captured CLI and adapter exit codes, not assumed zeros.
This checks the strict outcome, required stages and actual local artifact hashes/paths.
It needs no AWS credentials and does not establish durable job status or spoken-language quality.
