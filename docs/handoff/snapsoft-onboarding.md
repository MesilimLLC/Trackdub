# SnapSoft onboarding: Trackdub cloud render POC

Audience: the SnapSoft delivery team (Solution Architect, Cloud Engineer). The SOW is signed. This page maps each
SOW prerequisite to the file that satisfies it, states where the repository already differs from the SOW's
assumptions, and lists the decisions left for the Design phase. Nothing here claims AWS readiness: no AWS resource
has been created and no GPU run on G6/G6e has happened.

## 1. SOW prerequisites and where to find them

| SOW prerequisite | Location |
|---|---|
| Technical Overview documentation | [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md), [docs/index.md](../index.md) |
| Application integration contract (API, DynamoDB, S3, scheduler, IAM) | [docs/operations/snapsoft-poc-acceptance.md](../operations/snapsoft-poc-acceptance.md) |
| What was verified locally and what was not | [docs/operations/snapsoft-poc-validation.md](../operations/snapsoft-poc-validation.md) |
| Container build, seed and warm runbook | [docs/operations/cloud-gpu-container.md](../operations/cloud-gpu-container.md) |
| Premium model manifest (license + hash status) | Section 3 below; source `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json` |
| Provider allow-list for Linux NVIDIA workers | Section 4 below |
| `trackdub` CLI and SDK source | `src/Trackdub.Cli`, `src/Trackdub.Sdk` |
| Container build scripts / CI | `Dockerfile`, `Dockerfile.cloud-worker`, `Dockerfile.api`, `docker-compose.aws.yml`, `scripts/docker/`, `.github/workflows/container-image.yml` |
| Test fixture | `tests/Fixtures/cloud-poc/en-speech.mp4` (single speaker, EN) and `create-job-v1.json` |

The 10-clip EN->ES / EN->ZH test set with RTX 5070 reference output is Trackdub's deliverable (SOW: end of Day 5)
and is not in the repository.

## 2. Where the repository differs from the SOW text

Read these before Design. They change what "containerize and wire up" means.

1. **No SQS, no Cloudflare in the job path.** SOW p4 describes Cloudflare API -> SQS -> EC2 worker. The repository
   was built to an AWS-only requirement: a CPU ASP.NET API (`src/Trackdub.Api`, Cognito JWT), DynamoDB for job,
   idempotency and outbox state (`src/Trackdub.Cloud.Aws`), S3 for versioned artifacts, and a scheduler seam
   (`IJobScheduler`). `AWSSDK.SQS` is pinned in `Directory.Packages.props` but unused. Align on this at kickoff.
2. **The scheduler adapter does not exist.** Intake returns 503 until one is registered. Someone has to write
   the concrete Batch or ECS adapter in C# against `IJobScheduler` (contract: acceptance doc, "Durable state and
   scheduler contract"). The SOW lists IaC and a worker job loop, not this adapter. Decide the owner.
3. **The worker is already a container.** `Dockerfile` has `rtx` and `server` targets; `Dockerfile.cloud-worker`
   wraps the one-shot CLI with `Trackdub.CloudJobAdapter` (claim, lease heartbeat, upload, fenced outcome write)
   and takes the server image as `SERVER_IMAGE`. CI now builds both the worker image (on the server image) and the
   CPU API image. What is missing is a registry path from your account (see section 5).
4. **Classic TensorRT + CUDA, not TensorRT-RTX, on AWS.** The `server` image carries CUDA 13, cuDNN and
   TensorRT 10.13.3.9. The TensorRT-RTX pin in `runtime/trt-rtx-ep.manifest.json` is for the desktop `rtx`
   profile and does not apply to G6/G6e.
5. **Engine caches cannot be built in CI.** CI runners have no GPU. Classic TensorRT caches are keyed to
   GPU, driver and TensorRT binary hash, so they are created on the target host with `trackdub cache warm`
   and then persisted (EBS volume or snapshot). Never copy a cache between GPU models or drivers.
6. **Languages and limits.** Cloud presets (`src/Trackdub.Cloud/Jobs/CloudPresets.cs`):
   `poc-server-v1` (en->es, Kokoro, voice `ef_dora`) is the accepted baseline; `poc-server-zh-v1` (en->zh,
   Qwen3-TTS `qwen3:vivian`) is accepted by the API but not qualified. Uploads: MP4, up to 1 GiB, up to 600 s,
   up to 1920x1080. Both presets run on CUDA with only the engines in `docker/poc-warm-models.server.txt`
   pre-compiled.
7. **S3 lifecycle vs versioning.** The bucket is versioned and DynamoDB stores S3 version IDs. A 7-day expiry must
   cover noncurrent versions, and artifact listing after expiry must be defined. The acceptance doc leaves
   retention to a separately deployed policy.
8. **Benchmark and cost-per-minute** are not delivered by anyone under the signed scope. Local diagnostic
   timings in the runbook come from an RTX 5070 and are not throughput or cost evidence for L4/L40S.

## 3. Premium model manifest (cloud presets)

Union of `docker/poc-server-models.txt` (es) and `docker/poc-server-zh-models.txt` (zh). Values are copied from
`bundled-models.manifest.json`; `commercial_use_verified` is the manifest's own flag, not a legal review.
Hashes are the manifest `sha256` for the pinned model files (first 12 hex digits shown; the CLI verifies the full
value at download).

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

- Sortformer's revision is `cache-installed`, meaning no upstream revision is pinned in the manifest. The
  SOW assumes every model is downloadable and hash-verified at image build time; confirm Sortformer's download
  path before relying on it.
- Seven of the eight entries require attribution. Keep each model's license and attribution notice next to the
  seeded weights (runbook: "Curated POC model volume").
- Preserve the full preset pin files: `docker/poc-server-preset.json`, `docker/poc-server-quality-preset.json`,
  `docker/poc-server-zh-preset.json`.

## 4. Provider allow-list for Linux NVIDIA workers (server profile)

There is no separate allow-list configuration file. The allow-list is the provider pin of each preset in
`src/Trackdub.Cli/PocServerPreset.cs`:

| Stage | Provider |
|---|---|
| vad | CPU (Silero fails TensorRT import; CPU costs nothing) |
| asr, translation, tts, diarization, separation, speech-enhancement | CUDA |

Classic TensorRT is available in the image, and `seed-and-warm-gpu-worker.sh` compiles engines for the graphs in
`docker/poc-warm-models.server.txt` (DeepFilterNet3, SortFormer, Spleeter, Parakeet). Both cloud presets pin
those stages to CUDA, so preset runs do not use the warmed engines; the warm step adds build time and EBS storage
without speeding up `poc-server-v1` or `poc-server-zh-v1`. Decide whether to skip it or to change the presets'
provider pins after measuring on L4/L40S. Constraints from the runbook:

- MADLAD `quantized` is not classic-TensorRT qualified; the `trt-fp16` export is, but needs a 24 GiB-class GPU
  (about 12.7 GB session bundle) and is not in the accepted preset.
- Qwen3-TTS is excluded from TensorRT by planner policy and runs on CUDA. CosyVoice `int8` runs on CUDA only.
- Required stages must not fall back to CPU: acceptance fails on unexplained CPU fallback.
- VRAM admission auto-scales (75% up to 16 GiB, 85% above, floor 4096 MB). Check the `accelerator-vram-budget`
  row of `trackdub doctor`.

## 5. Commands you will use

```bash
trackdub doctor                       # health check; checks manifest, ffmpeg/ffprobe, model cache, VRAM budget
trackdub providers list               # CUDA and TensorRT must both report available on the GPU host
trackdub models download <model-id>   # hash-verified download (use scripts/docker/seed-model-volume.sh for the set)
trackdub models verify                # integrity check of the seeded volume
trackdub cache warm --provider tensorrt   # per-GPU engine build; see docker/poc-warm-models.server.txt
trackdub run pipeline --media in.mp4 --output ./project --source-language en --target-language es \
  --preset poc-server-v1 --voice SPEAKER_00:ef_dora --subtitle-format srt \
  --result-file result.json --progress-file progress.jsonl
```

To run a specific model variant without editing a preset, add `--variant <stage>:<variant>` (for example
`--variant translation:trt-fp16`); see [gpu-execution-providers.md](../reference/gpu-execution-providers.md#choosing-a-variant-explicitly-run-pipeline---variant).

Provisioning a GPU host: `TRACKDUB_GPU_PROFILE=server ./scripts/docker/seed-and-warm-gpu-worker.sh` with
`TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml` (see the runbook for the EBS mount and UID 10001 ownership).
The script seeds the Spanish set (`docker/poc-server-models.txt`) by default. To accept `poc-server-zh-v1` jobs,
the host also needs the Qwen3-TTS weights; seed them before scheduling Chinese jobs, for example
`TRACKDUB_MODEL_SET=docker/poc-server-zh-models.txt ./scripts/docker/seed-model-volume.sh` (same compose override).
Keep Kokoro seeded for `poc-server-v1`; the two lists share every other model.

Build and test without AWS or a GPU:

```bash
dotnet test tests/Trackdub.Cloud.Tests -c Release -m:1
dotnet test tests/Trackdub.Api.Tests -c Release -m:1
dotnet test tests/Trackdub.CloudJobAdapter.Tests -c Release -m:1
dotnet test tests/Trackdub.Architecture.Tests -m:1
docker build -f Dockerfile.api -t trackdub-cloud-api:local .
```

Registry: images publish to `ghcr.io/mesilimllc/trackdub:<tag>-server` from `main`; the package may be private
until its visibility is changed. The worker image is not published by CI; build it from an approved server digest
with `docker build -f Dockerfile.cloud-worker --build-arg SERVER_IMAGE=<digest> .`. Decide GHCR pull credentials or
an ECR mirror in Design.

## 6. Decisions for the Design phase

- Orchestrator: AWS Batch or ECS on EC2 GPU, and who writes the `IJobScheduler` adapter. Batch `SubmitJob` has no
  idempotency token; ECS `RunTask` has `clientToken`. Dispatch discovery must find all executions for a dispatch ID.
- Instance: G6 (L4, 24 GB) or G6e (L40S, 48 GB); AMI and NVIDIA driver level for CUDA 13 (the server image is based
  on `nvidia/cuda:13.0.0`).
- Model and engine-cache persistence: per-host EBS, snapshot or AMI; rebuild on driver, TensorRT or model change.
- Capacity policy: scale to zero or keep one host warm; container exit alone does not stop instance billing, and the
  SOW's under-2-minute cold-start target depends on this choice.
- Registry path and pull credentials; CPU API hosting; Cognito app client and exact portal origins.
- S3 retention (section 2, item 7); region (single US region per SOW).
- Whether to qualify `poc-server-zh-v1` in this engagement, using the Trackdub-supplied EN->ZH clips.

## 7. Not verified

No G6/G6e execution, no real IAM/S3/DynamoDB/Cognito behavior, no native-Linux datacenter-GPU visibility in the
container, no full-pipeline quality or timing on L4/L40S, no Mandarin audio quality. GPU evidence in the runbook is
from an RTX 5070 (WSL2).
