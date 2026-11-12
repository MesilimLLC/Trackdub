# SnapSoft onboarding: Trackdub cloud render POC

Audience: the SnapSoft delivery team (Solution Architect, Cloud Engineer). The SOW ("Infrastructure Setup POC",
signed 2026-10-30) is the contract. This page maps each SOW prerequisite to the file that satisfies it, states
where the repository differs from the SOW, and lists the decisions left for the Design phase. Where the repository
and the SOW disagree, the SOW is the POC target unless both parties agree a change order. Nothing here claims AWS
readiness: no AWS resource has been created and no GPU run on G6/G6e has happened.

## 1. SOW prerequisites and where to find them

| SOW prerequisite | Location |
|---|---|
| Technical Overview documentation | [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md) (section 3.6 covers headless and cloud execution), [docs/index.md](../index.md) |
| Premium model manifest (license + hash status) | Section 4 below; source `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json` |
| Provider allow-list for Linux NVIDIA workers | Section 5 below |
| `trackdub` CLI and SDK source | `src/Trackdub.Cli`, `src/Trackdub.Sdk` |
| Container build scripts / CI | `Dockerfile` (`server` target), `scripts/docker/`, `.github/workflows/container-image.yml` |
| Container build, seed and warm runbook | [docs/operations/cloud-gpu-container.md](../operations/cloud-gpu-container.md) |
| What was verified locally and what was not | [docs/operations/snapsoft-poc-validation.md](../operations/snapsoft-poc-validation.md) |
| Smoke fixture | `tests/Fixtures/cloud-poc/en-speech.mp4` (single speaker, EN) |

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
   TensorRT binary and ONNX Runtime identity. Build them on the target instance type with `trackdub cache warm --provider tensorrt`,
   then bake them into the image or an EBS snapshot as the SOW describes. Never copy a cache between GPU models or
   drivers.
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

TensorRT engines. The SOW includes building TensorRT engines once per instance type with `trackdub cache warm --provider tensorrt` and
baking them into the image or an EBS snapshot. `docker/poc-warm-models.server.txt` lists the classic-TensorRT
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

```bash
trackdub doctor                       # health check; checks manifest, ffmpeg/ffprobe, model cache, VRAM budget
trackdub providers list               # CUDA and TensorRT must both report available on the GPU host
trackdub models download <model-id>   # hash-verified download (use scripts/docker/seed-model-volume.sh for the set)
trackdub models verify                # integrity check of the seeded models
trackdub cache warm --provider tensorrt   # per-GPU engine build; see docker/poc-warm-models.server.txt
trackdub run pipeline --media in.mp4 --output ./project --source-language en --target-language es \
  --preset poc-server-v1 --voice SPEAKER_00:ef_dora --subtitle-format srt \
  --result-file result.json --progress-file progress.jsonl
```

To run a specific model variant without editing a preset, add `--variant <stage>:<variant>` (for example
`--variant translation:trt-fp16`); see [gpu-execution-providers.md](../reference/gpu-execution-providers.md#choosing-a-variant-explicitly-run-pipeline---variant).

Provisioning a GPU host with the runbook scripts: `TRACKDUB_GPU_PROFILE=server ./scripts/docker/seed-and-warm-gpu-worker.sh`
with `TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml` (see the runbook for the EBS mount and UID 10001
ownership). The script seeds the Spanish set (`docker/poc-server-models.txt`) by default. For `poc-server-zh-v1`
the host also needs the Qwen3-TTS weights, for example
`TRACKDUB_MODEL_SET=docker/poc-server-zh-models.txt ./scripts/docker/seed-model-volume.sh` (same compose override).
Keep Kokoro seeded for `poc-server-v1`; the two lists share every other model.

Build and test without AWS or a GPU (the Cloud, Api and CloudJobAdapter tests cover the out-of-scope job stack in
section 2, including the reusable CLI runner and preset code):

```bash
dotnet test tests/Trackdub.Cloud.Tests -c Release -m:1
dotnet test tests/Trackdub.Api.Tests -c Release -m:1
dotnet test tests/Trackdub.CloudJobAdapter.Tests -c Release -m:1
dotnet test tests/Trackdub.Architecture.Tests -m:1
```

Registry: the server image publishes to `ghcr.io/mesilimllc/trackdub:<tag>-server` from `main`; the package may be
private until its visibility is changed. Decide GHCR pull credentials or an ECR mirror in Design.
`Dockerfile.cloud-worker` belongs to the out-of-scope job stack in section 2 and is built, not published, by CI.

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
