# SnapSoft onboarding: Trackdub cloud render POC

Audience: the SnapSoft delivery team (Solution Architect, Cloud Engineer). The SOW ("Infrastructure Setup POC",
signed 2026-10-30) is the contract. This page maps each SOW prerequisite to the file that satisfies it, states
where the repository differs from the SOW, and lists the decisions left for the Design phase. Where the repository
and the SOW disagree, the SOW is the POC target unless both parties agree a change order. Nothing here claims AWS
readiness: no AWS resource has been created and no GPU run on G6/G6e has happened.

## 1. SOW prerequisites and where to find them

| SOW prerequisite | Location |
|---|---|
| Technical Overview documentation | [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md) (section 2.6 covers headless and cloud execution), [docs/index.md](../index.md) |
| Premium model manifest (license + hash status) | Section 4 below; source `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json` |
| Provider allow-list for Linux NVIDIA workers | Section 5 below |
| `trackdub` CLI and SDK source | `src/Trackdub.Cli`, `src/Trackdub.Sdk` |
| Container build scripts / CI | `Dockerfile` (`server` target), `scripts/docker/`, `.github/workflows/container-image.yml` |
| Container build, seed and warm runbook | [docs/operations/cloud-gpu-container.md](../operations/cloud-gpu-container.md) |
| Fresh EC2 GPU host to first pipeline run | Section 6, "From a fresh EC2 GPU host to a first job" |
| What was verified locally and what was not | [docs/operations/snapsoft-poc-validation.md](../operations/snapsoft-poc-validation.md) |
| Smoke fixture | `tests/Fixtures/cloud-poc/en-speech.mp4` (single speaker, EN) |
| Kickoff item "containerization of application code" | The `Dockerfile` `server` target, configured to publish as `ghcr.io/mesilimllc/trackdub:<tag>-server` (section 6) |

The 10-clip test set (English-to-Spanish and English-to-Chinese) with reference output from a local RTX 5070 run
is Mesilim's deliverable, due by the end of Day 5 (SOW, Customer Responsibilities). It is not in the repository.

## 2. POC target architecture (SOW)

The SOW job path is:

```text
Cloudflare API (existing; auth and job record)
  -> SQS (job message)
  -> EC2 GPU worker (container: trackdub run pipeline)
  -> S3 (source media in, per-stage artifacts out; KMS SSE; presigned URLs; 7-day lifecycle)
```

- SnapSoft writes IaC for SQS, S3, KMS and the EC2 GPU worker, plus the chosen orchestrator (AWS Batch or ECS).
- SnapSoft implements the worker job loop: read the SQS message, fetch source media from S3, run
  `trackdub run pipeline`, parse the structured result (`--result-file`, `--progress-file`), and write artifacts to S3.
- SnapSoft defines the Cloudflare-to-SQS/S3 interface contract. Mesilim makes the Cloudflare-side changes.
- Status wiring back into the Cloudflare job API, Spot capacity and autoscaling are deferred to the follow-on phase.
- The SOW's initial architecture diagram labels the account region `us-east-1`; the target US region is still
  confirmed in the Design phase.

### Success criteria (SOW)

The POC is complete when:

1. The worker image is deployed into Mesilim's AWS account, passes the `trackdub doctor` health check, and runs
   `trackdub run pipeline` end to end against the 10-clip test set on at least one EC2 GPU instance type.
2. All POC AWS resources are provisioned by infrastructure-as-code in Mesilim's account, with no dependency on
   SnapSoft tooling.
3. A readout covers the monthly run-rate estimate, the orchestrator and model-cache decisions, and the follow-on plan.

The SOW also names these metrics for the pass/fail record: job success rate of at least 95%, cold start under two
minutes, and output parity with the local RTX 5070 reference (see section 3 item 6 for who measures parity).

### Out of scope (SOW)

- Replacing any pipeline stage with Amazon Transcribe, Amazon Polly or another AWS-managed AI service. Trackdub's
  own models do all transcription, translation and synthesis.
- Multi-region deployment and EU data residency. The POC is one US region.
- Billing, subscription metering and cloud-tier pricing.
- Status wiring into the Cloudflare job API, Spot capacity and autoscaling.
- Production deployment and go-live cutover.
- Later use cases, which reuse the POC's GPU workers and are scoped separately.

### AWS job code already in the repository (outside POC scope)

The repository also contains an AWS-only job stack built before the SOW: a CPU ASP.NET API with Cognito JWT
(`src/Trackdub.Api`), DynamoDB job, idempotency and outbox state (`src/Trackdub.Cloud.Aws`), a scheduler
interface with no concrete adapter (`IJobScheduler` in `src/Trackdub.Cloud`), and a one-shot worker wrapper
(`src/Trackdub.CloudJobAdapter`, image `Dockerfile.cloud-worker`) that reads its job identity from environment
variables and its state from DynamoDB. `AWSSDK.SQS` is pinned in `Directory.Packages.props` but unused.

This stack is not part of the POC. Do not build IaC for DynamoDB, Cognito or the CPU API, and do not write an
`IJobScheduler` adapter, unless a change order adds them. Pieces that can be reused in the SQS job loop if
useful: the CLI process supervision in `src/Trackdub.CloudJobAdapter/CliProcessRunner.cs`, the fixed-preset
argument building in `src/Trackdub.CloudJobAdapter/JobAdapter.cs` (which never accepts client-supplied commands,
paths or model IDs), and the cloud preset list in `src/Trackdub.Cloud/Jobs/CloudPresets.cs`.
Reference contract for that stack: [docs/operations/snapsoft-poc-acceptance.md](../operations/snapsoft-poc-acceptance.md).

## 3. Where the repository differs from the SOW text

1. **Classic TensorRT + CUDA, not TensorRT-RTX, on AWS.** The `server` image is based on
   `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04` and carries TensorRT 10.13.3.9. The TensorRT-RTX pin in
   `runtime/trt-rtx-ep.manifest.json` is for the desktop `rtx` profile and does not apply to G6/G6e.
2. **Engine caches cannot be built in CI.** CI runners have no GPU. Classic TensorRT caches live under
   `$TRACKDUB_ENGINE_CACHE_ROOT/classic-trt/<fingerprint>`, where the fingerprint covers the GPU, driver,
   TensorRT binary and ONNX Runtime identity. Build them on the target instance type with `scripts/docker/seed-and-warm-gpu-worker.sh`
   using the explicit server provisioning environment in section 6, then bake them into the image or an EBS
   snapshot as the SOW describes. Never copy a cache between GPU models or drivers.
3. **Model weights at build time vs a seeded volume.** The SOW has models downloaded and hash-verified at image
   build time. The runbook instead seeds a persistent model volume (`scripts/docker/seed-model-volume.sh`). Both
   use `trackdub models download` and `trackdub models verify`; which one the POC uses is a Design decision.
4. **Languages and limits.** Cloud presets (`src/Trackdub.Cloud/Jobs/CloudPresets.cs`): `poc-server-v1`
   (en->es, Kokoro, voice `ef_dora`) and `poc-server-zh-v1` (en->zh, Qwen3-TTS `qwen3:vivian`). Both are in POC
   scope. `poc-server-v1` is the qualified baseline; `poc-server-zh-v1` has not yet passed an end-to-end EN->ZH
   run, and Mesilim is qualifying it before the Build phase. Media limits enforced by the repository's validator:
   MP4, up to 1 GiB, up to 600 s, up to 1920x1080.
5. **S3 versioning.** If the artifact bucket is versioned, the 7-day lifecycle rule must also expire noncurrent
   versions.
6. **Benchmarks and parity.** The SOW says the final instance type is selected from self-guided benchmark results
   completed by Mesilim, and the readout includes a recommended instance type and a monthly run-rate estimate.
   The SOW lists parity test results and parity pass/fail under Deliverables, but its Out of Scope section says
   SnapSoft will not produce a per-stage benchmark report or measure parity against the RTX 5070 run. Agree at
   kickoff who runs the parity comparison. Local timings in the runbook come from an RTX 5070 and are not
   throughput or cost evidence for L4/L40S.

## 4. Premium model pack for the POC

For this POC, "Premium model pack" means the model sets of the two cloud presets: the union of
`docker/poc-server-models.txt` (es) and `docker/poc-server-zh-models.txt` (zh). The desktop Premium starter pack
(`src/Trackdub.Composition/StarterPacks/premium.json`: Whisper or Qwen3-ASR, Chatterbox voice-cloning TTS) is a
different model set and is not part of the POC image. The RTX 5070 reference output is produced with the same
presets, so the parity comparison runs the same pipeline on both sides.

Values are copied from `bundled-models.manifest.json`; `commercial_use_verified` is the manifest's own flag, not a
legal review. Hashes are the manifest `sha256` (first 12 hex digits shown; the CLI verifies the full value at
download).

| Model | Task | License | Commercial | Verified | Attribution | Revision | sha256 |
|---|---|---|---|---|---|---|---|
| onnx-community/silero-vad | vad (CPU) | MIT | yes | yes | no | e71cae966052 | a4a068cd6cf1 |
| Rikorose/DeepFilterNet3 | speech-enhancement | MIT | yes | yes | yes | dcbbe520263d | 7c5399d3da8a |
| cgus/diar_streaming_sortformer_4spk-v2.1-onnx | diarization | NVIDIA-Open-Model-License | yes | yes | yes | `cache-installed` | 82b9c735e1cf |
| tonythethompson/parakeet-tdt-0.6b-v3-onnx | asr | CC-BY-4.0 | yes | yes | yes | 107353b64181 | 98a74b21b4cc |
| google/madlad400-3b-mt (`quantized`) | translation | Apache-2.0 | yes | yes | yes | 67037ad42f58 | 3ab4690aab21 |
| csukuangfj/sherpa-onnx-spleeter-2stems | separation | MIT | yes | yes | yes | 3e5a4dd68259 | 932fff35af1d |
| onnx-community/Kokoro-82M-v1.0-ONNX (es preset) | tts | Apache-2.0 | yes | yes | yes | 1939ad2a8e41 | 8fbea51ea711 |
| tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX (zh preset) | tts | Apache-2.0 | yes | yes | yes | a2b73f56ffec | f55e99902cb2 |

Notes:

- Sortformer has a manifest download URL and sha256, but its revision is `cache-installed`, meaning no upstream
  revision is pinned. Mesilim will pin it so every model in the pack is downloadable and hash-verified at build
  time, as the SOW assumes.
- Seven of the eight entries require attribution. Keep each model's license and attribution notice next to the
  seeded weights (runbook: "Curated POC model volume").
- Preserve the full preset pin files: `docker/poc-server-preset.json`, `docker/poc-server-quality-preset.json`,
  `docker/poc-server-zh-preset.json`.

## 5. Provider allow-list for Linux NVIDIA workers (server profile)

There is no separate allow-list configuration file. The allow-list is the per-stage provider pin of each preset in
`src/Trackdub.Cli/PocServerPreset.cs`:

| Stage | Provider |
|---|---|
| vad | CPU (Silero is excluded from TensorRT; CPU cost is negligible) |
| asr, translation, tts, diarization, separation, speech-enhancement | CUDA |

TensorRT engines. The SOW includes building TensorRT engines once per instance type and baking them into the
image or an EBS snapshot. `docker/poc-warm-models.server.txt` lists the classic-TensorRT
graphs to warm (DeepFilterNet3, Sortformer, Spleeter, Parakeet), and `scripts/docker/seed-and-warm-gpu-worker.sh`
builds them. The presets currently pin those stages to CUDA, so preset runs do not yet use the warmed engines. On a
local RTX 5070, diarization (Sortformer) and ASR (Parakeet) have run end-to-end on classic TensorRT. Whether the
presets move those stages to TensorRT is decided in Design after measuring on L4/L40S.

Constraints:

- MADLAD `quantized` (the preset variant) is not classic-TensorRT qualified and runs on CUDA. The `trt-fp16`
  export is listed for classic TensorRT in the manifest and passed an isolated smoke test, but a full pipeline run
  with it has not completed. It needs a 24 GiB-class GPU (about 12.7 GB session bundle) and is not in either preset.
- Qwen3-TTS and Kokoro are excluded from TensorRT by planner policy and run on CUDA.
- Required stages must not fall back to CPU: acceptance fails on unexplained CPU fallback.
- VRAM admission auto-scales (75% up to 16 GiB, 85% above, floor 4096 MB). Check the `accelerator-vram-budget`
  row of `trackdub doctor`.

## 6. Commands you will use

After provisioning, place the source video at `docker/input/in.mp4` on the host (mounted at
`/work/input/in.mp4` in the container), then open a shell from the repository root on the GPU host:

```bash
docker compose -f docker-compose.yml -f docker-compose.aws.yml run --rm --entrypoint bash trackdub
```

Run the following in that container shell. The image entrypoint normally invokes the DLL through `dotnet`;
it does not install a `trackdub` executable on `PATH`, so define this wrapper for the interactive examples:

```bash
trackdub() { dotnet /app/Trackdub.Cli.dll --model-directory /data/models "$@"; }

trackdub doctor                       # health check; checks manifest, ffmpeg/ffprobe, model cache, VRAM budget
trackdub providers list               # CUDA and TensorRT must both report available on the GPU host
trackdub models download <model-id>   # hash-verified download (use scripts/docker/seed-model-volume.sh for the set)
trackdub models verify --model-id <model-id>  # verify one seeded model; repeat for each POC model

# Allocate a fresh directory for each job attempt, including retries.
job_dir="$(mktemp -d /data/poc-job.XXXXXXXX)"
trackdub run pipeline --media /work/input/in.mp4 --output "$job_dir/project" --source-language en --target-language es \
  --preset poc-server-v1 --voice SPEAKER_00:ef_dora --subtitle-format srt \
  --result-file "$job_dir/result.json" --progress-file "$job_dir/progress.jsonl"
```

`--result-file` and `--progress-file` must name new files; they do not overwrite prior attempt evidence.
Without `--model-id`, `trackdub models verify` checks **all manifest models**, not just the seeded POC set.
It emits a JSON array, including missing models with `hashMatch: false`, and exits 2 if any model fails
verification (0 only when all match). That nonzero status is expected for a deliberately partial cache; do not
use this whole-manifest check as the POC-set readiness gate. Use the targeted check for each seeded model and
retain the hash-verified download evidence for the exact variants listed in the model-set files.

To run a specific model variant without editing a preset, add `--variant <stage>:<variant>` (for example
`--variant translation:trt-fp16`); see [gpu-execution-providers.md](../reference/gpu-execution-providers.md#choosing-a-variant-explicitly-run-pipeline---variant).

Run these provisioning scripts from the repository root on the **GPU host**, not inside the container shell
(see the runbook for the EBS mount and UID 10001 ownership):

```bash
TRACKDUB_GPU_PROFILE=server \
TRACKDUB_MODEL_SET=docker/poc-server-models.txt \
TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml \
  ./scripts/docker/seed-and-warm-gpu-worker.sh

# Add the Mandarin preset's Qwen3-TTS weights to the same volume.
TRACKDUB_GPU_TARGET=server \
TRACKDUB_MODEL_SET=docker/poc-server-zh-models.txt \
TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml \
  ./scripts/docker/seed-model-volume.sh
```

The combined script already defaults to the Spanish set for the server profile; it is explicit here for
reproducibility. The standalone `seed-model-volume.sh` instead defaults to `docker/poc-models.txt`, so always
supply `TRACKDUB_MODEL_SET` when using it for a server preset. Keep Kokoro seeded for `poc-server-v1`; the two
server lists share every other model.

Use the combined script for engine warming: it supplies explicit `--model` paths from the reviewed
`docker/poc-warm-models.server.txt` allow-list. **Do not run bare `trackdub cache warm --provider tensorrt`**
for POC provisioning. Without `--model`, it recursively scans the entire cache, including graphs that can fail
compilation or terminate the native process. Warm only on the target GPU/driver, and inspect failed or skipped
targets; completion does not prove that preset stages used TensorRT.

Build and test without AWS or a GPU (the Cloud, Api and CloudJobAdapter tests cover the out-of-scope job stack in
section 2, including the reusable CLI runner and preset code):

```bash
dotnet test tests/Trackdub.Cloud.Tests -c Release -m:1
dotnet test tests/Trackdub.Api.Tests -c Release -m:1
dotnet test tests/Trackdub.CloudJobAdapter.Tests -c Release -m:1
dotnet test tests/Trackdub.Architecture.Tests -m:1
```

Registry: the container workflow is configured to publish `ghcr.io/mesilimllc/trackdub:<tag>-server`; the AWS
Compose override defaults to `ghcr.io/mesilimllc/trackdub:main-server`. This configuration is not proof that a
particular tag is pullable by a worker. The successful [November 16, 2026 container publish run](https://github.com/MesilimLLC/Trackdub/actions/runs/43910597341)
recorded `main-server`, `latest-server`, and `sha-e349c4bfb44bbd02f4929f502a9a2bb5c13efbd7-server` at
`sha256:f446fe1213b53268998b0668d622dca5613f0632c39ac710e2fc6e8f7d6a7f19`. Anonymous manifest inspection
returned HTTP 401 on that date; the local GitHub credential also lacks `read:packages`. Publication is confirmed
by the push log, but worker pull access is not. Confirm access to the chosen tag/digest before provisioning;
decide GHCR pull credentials, a visibility change, or an ECR mirror in Design.
`Dockerfile.cloud-worker` belongs to the out-of-scope job stack in section 2 and is built, not published, by CI.

### From a fresh EC2 GPU host to a first job

These are the intended steps for the `server` image on one EC2 GPU host. They have not been run on G6/G6e.
No image build depends on the GPU: CI builds the `server` image on a GPU-less runner. Only the TensorRT engine
warm in step 3 is tied to the GPU model and driver. Model seeding is hash-verified downloads and can run anywhere.

`scripts/docker/prepare-gpu-worker-host.sh` runs steps 2 to 4 in one go once step 1 is done and the checkout is at
the image's commit: it checks the host, pulls `TRACKDUB_IMAGE` (an immutable `sha-<commit>-server` tag or digest),
seeds both POC model sets, runs the in-container GPU check, `providers list` and `trackdub doctor`, and
hash-verifies every POC model, stopping at the first failure. Snapshot the AMI or the EBS volume after it prints
`READY`. It skips the TensorRT warm unless `TRACKDUB_WARM_ENGINES=1`: the presets run every stage on CUDA, so
jobs do not use the engines yet, and a volume without warmed engines is not tied to one GPU model. This has not
been run on G6/G6e; a stubbed test (`scripts/docker/tests/prepare-gpu-worker-host.test.sh`) covers its control flow.

**Step 1: host prerequisites (once per instance or AMI).**

- Instance: G6 (L4) or G6e (L40S).
- NVIDIA driver for CUDA 13.0 (R580 or newer per NVIDIA's CUDA 13.0 requirement; confirm when the AMI is chosen).
- Docker Engine with the Compose v2 plugin, and the NVIDIA Container Toolkit registered with Docker. The AWS Deep
  Learning Base GPU AMI may ship these; that is unverified for this image.
- A dedicated EBS volume, formatted and persistently mounted at `/mnt/trackdub-data`.

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
nvidia-smi -L                          # must list the L4 / L40S
sudo mkdir -p /mnt/trackdub-data /mnt/trackdub-data/jobs
sudo chown -R 10001:10001 /mnt/trackdub-data
```

**Step 2: repository and image.** The provisioning scripts, Compose files and model lists live in the repository,
so the host needs a checkout. Until the registry decision above is made, log in to GHCR with a token that has
`read:packages`. Pin an immutable `sha-<commit>-server` tag or digest instead of the moving `main-server`;
`docker-compose.aws.yml` reads `TRACKDUB_IMAGE` for the service image and falls back to `main-server` only when it
is unset. Check out the same commit the image tag names, so the seed lists, warm targets and scripts match the
image's bundled manifest. Create `docker/input` yourself before the first Compose run; otherwise Docker creates it
as root.

```bash
COMMIT=<full commit sha>
git init Trackdub && cd Trackdub
git remote add origin https://github.com/MesilimLLC/Trackdub.git
git fetch --depth 1 origin "$COMMIT" && git checkout --detach FETCH_HEAD
mkdir -p docker/input
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
export TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml
export TRACKDUB_IMAGE="ghcr.io/mesilimllc/trackdub:sha-$COMMIT-server"
docker compose -f docker-compose.yml -f docker-compose.aws.yml pull trackdub
```

**Step 3: seed models and warm engines (once per GPU model and driver).** Use the two provisioning commands shown
earlier in this section (`seed-and-warm-gpu-worker.sh` for the Spanish set and the warm, then
`seed-model-volume.sh` with `docker/poc-server-zh-models.txt` for Qwen3-TTS). Both write to `/mnt/trackdub-data`.
Repeat this step after any driver, TensorRT, ONNX Runtime or model change. An EBS snapshot can replace it only if
it was seeded on the same GPU model and driver with an image that has the same model set, TensorRT and ONNX
Runtime versions as the selected image; do not attach an older snapshot after a model change. The presets currently pin every warmed stage to CUDA, so a `poc-server-v1` run
does not yet depend on the warmed engines (section 5).

**Step 4: verify the worker before scheduling jobs.**

```bash
dc() { docker compose -f docker-compose.yml -f docker-compose.aws.yml "$@"; }
dc run --rm --entrypoint sh trackdub -c 'ls /proc/driver/nvidia/gpus'   # GPU visible inside the container
dc run --rm trackdub providers list    # CUDA and TensorRT available
dc run --rm trackdub doctor            # check the accelerator-vram-budget row
status=0
for m in $(cat docker/poc-server-models.txt docker/poc-server-zh-models.txt | grep -v '^#' | cut -d'|' -f1 | sort -u); do
  dc run --rm trackdub models verify --model-id "$m" || status=1
done
[ "$status" -eq 0 ] && echo "all POC models verified" || echo "model verification FAILED"
```

`doctor` exits nonzero only when a row is `fail`. In the `server` image (`TRACKDUB_GPU_PROFILE=server`) the
rows for components that profile never uses (desktop playback natives, the TensorRT-RTX plugin, MIGraphX and the
OpenVINO catalog EP) report `pass` with a "Not used by the server GPU profile" note instead of `warn`, so any
remaining `warn` row is worth reading.

Do not run `models verify` without `--model-id` here; on a partial cache it exits 2 by design. The loop checks every
model and reports a failure from any of them.

**Step 5: run one job.** The image entrypoint already passes `--model-directory /data/models`, so pipeline
arguments follow the service name directly. Use a new job directory for every attempt, including retries. Create
it on the host under the EBS mount, owned by the container user; inside the container the same directory is under
`/data/jobs`. The input file must be readable by UID 10001, so make it world-readable after copying.

```bash
cp /path/to/source.mp4 docker/input/in.mp4
chmod 0644 docker/input/in.mp4
JOB_ID=$(date +%Y%m%d-%H%M%S)
sudo install -d -o 10001 -g 10001 "/mnt/trackdub-data/jobs/$JOB_ID"
JOB=/data/jobs/$JOB_ID
dc run --rm trackdub run pipeline --media /work/input/in.mp4 --output "$JOB/project" \
  --source-language en --target-language es --preset poc-server-v1 --voice SPEAKER_00:ef_dora \
  --subtitle-format srt --result-file "$JOB/result.json" --progress-file "$JOB/progress.jsonl"
```

For Mandarin, use `--target-language zh --preset poc-server-zh-v1 --voice SPEAKER_00:qwen3:vivian` (not yet
qualified end to end, section 3 item 4). Results land on the host under `/mnt/trackdub-data/jobs/$JOB_ID/`.
Input limits are MP4, up to 1 GiB, 600 s and 1920x1080.

Steps 1 and 2 happen once per instance or AMI, step 3 once per GPU model and driver, step 4 after each
provisioning, and step 5 per job. The SQS job loop replaces step 5: it fetches the source from S3, runs the same
command, parses `result.json` and uploads the artifacts.

### Local command checks (November 16, 2026)

Runtime checks used the **pre-existing** `trackdub-gpu:server-snapsoft` image (ID `3969d1d88223`, built
November 12), not the newly published GHCR digest above. Cached weights were mounted read-only in an isolated
container with writable scratch metadata. These are local RTX 5070 / Docker Desktop checks, not AWS evidence.

| Check | Observed result |
|---|---|
| `doctor` | First attempt timed out at 180 s; retry exited 0 with warnings (including absent desktop playback natives and TRT-RTX plugin) |
| `providers list` | Exit 0; CUDA and classic TensorRT available |
| Targeted Silero verification | Exit 0, `hashMatch: true` |
| Silero download command against the existing cache | Exit 0, Installed; not a fresh network-download test |
| Whole-manifest verification against a separate Silero-only cache | Exit 2; one matching model and 47 missing entries |
| Whole-manifest verification against the larger mounted cache | Timed out at 120 s; integrity check did not complete |
| Interactive wrapper and pipeline example with `--help` | Exit 0; argument/help check only, no inference |
| Fresh `/data/poc-job.XXXXXXXX` directories | Distinct directories created; result/progress files did not exist |
| Current provisioning scripts | Bash syntax and stubbed dry-run passed: 14 download invocations across the two lists, six explicit classic-TensorRT warm paths, AWS override retained throughout |
| AWS Compose configuration | Rendered server target, expected GHCR image, and `/mnt/trackdub-data` bind to `/data` |
| Four documented offline suites | 76 passed using existing Release binaries with `-f net11.0 --no-build --no-restore`; no fresh build was performed |

The full provisioning download, engine compilation and real pipeline execution were **not** run in this check.
Another Python workload occupied about 11 GiB of the 12 GiB GPU at 99% utilization; it was left untouched.
An authenticated GHCR pull and an idle GPU are still required to validate the published image end to end.

## 7. Decisions for the Design phase

- Orchestrator: AWS Batch or ECS on EC2 GPU, driven by the SQS job loop.
- Instance: G6 (L4, 24 GB) or G6e (L40S, 48 GB); AMI and NVIDIA driver level for CUDA 13.
- Model storage: download at image build time (SOW) or a seeded volume (runbook), section 3 item 3.
- Engine-cache persistence: baked into the image or an EBS snapshot, rebuilt on driver, TensorRT or model change.
- Provider pins: keep CUDA for diarization and ASR, or move them to TensorRT after L4/L40S measurement (section 5).
- Capacity policy for the under-2-minute cold start: scale to zero or keep one host warm. Container exit alone
  does not stop instance billing.
- Registry path and pull credentials.
- S3 bucket layout, versioning and the 7-day lifecycle rule; target US region.

To agree at kickoff: who runs the parity comparison against the RTX 5070 reference (section 3 item 6).

## 8. Not verified

No G6/G6e execution, no real IAM/S3/SQS/KMS behavior, no native-Linux datacenter-GPU visibility in the container,
no full-pipeline quality or timing on L4/L40S, no end-to-end EN->ZH run with `poc-server-zh-v1`, no Mandarin audio
quality review. GPU evidence in the runbook is from an RTX 5070 (WSL2).
