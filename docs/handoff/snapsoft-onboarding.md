# SnapSoft POC: Trackdub GPU batch runner

Contract: SOW "Infrastructure Setup POC," signed 2026-10-30. Where this page and the SOW differ, the SOW governs
unless both parties sign a change order.

Command runbook: [snapsoft-quickstart.md](snapsoft-quickstart.md).

## Status (2026-11-21)

- Not yet passing. Two EN→ZH runs on `g6e.xlarge` (us-east-1c, L40S 46 GiB) completed every stage through TTS,
  then failed at export. The dubbed take ran 6.8 s, then 4.9 s, past a 40.2 s source, and the exporter will not
  truncate it. No finished clip or subtitle file exists yet.
- The batch runner (`deploy/aws/`) is in PR #666 and not merged.
- EN→ES (`poc-server-v1`) has not been run on AWS.
- No run has used the IAM, S3 and SSM setup in a customer account. The POC account's stack has not been deployed
  from the template.

## Repository map

| Path | Purpose |
|---|---|
| `Dockerfile` (`server` target) | Image `ghcr.io/mesilimllc/trackdub:sha-<commit>-server`, CUDA 13.0 with classic TensorRT 10.13 |
| `deploy/aws/snapsoft-poc.yaml` | VPC, subnets, security group, run bucket, instance role. No compute |
| `deploy/aws/deploy-stack.sh` | Creates or updates the stack |
| `deploy/aws/launch-run.sh` | Uploads clips, renders instance config, launches one instance |
| `deploy/aws/run-batch.sh` | Runs on the instance: host prep, then one job per clip |
| `deploy/aws/summarize_result.py` | Turns `result.json` into one summary line |
| `scripts/docker/prepare-gpu-worker-host.sh` | Host checks, image pull, model seeding, providers, doctor, model verification |
| `docker/poc-server-models.txt`, `docker/poc-server-zh-models.txt` | Model sets for the two presets |
| `docker/poc-warm-models.server.txt` | Reviewed allow-list of TensorRT graphs to warm |

## Scope

SnapSoft owns: IaC for SQS, S3, KMS, the EC2 GPU worker and the orchestrator (AWS Batch or ECS); the worker job loop
(read SQS message, fetch from S3, run `trackdub run pipeline`, parse `result.json`, upload artifacts); and the
Cloudflare-to-SQS/S3 interface contract.

Mesilim owns: the image, the model sets and presets, and the Cloudflare-side changes.

Success criteria (SOW):

1. The image is deployed in Mesilim's account, passes `trackdub doctor`, and runs the pipeline end to end on the
   10-clip set (EN→ES and EN→ZH) on at least one GPU instance type.
2. All POC resources are provisioned by IaC in Mesilim's account, with no SnapSoft tooling dependency.
3. A readout with monthly run-rate, orchestrator and model-cache decisions, and the follow-on plan.

Metrics: job success ≥95%, cold start under 2 minutes, output parity with the local RTX 5070 reference.

Out of scope: Amazon Transcribe, Polly or other AWS AI services in the pipeline; multi-region or EU residency;
billing and metering; Cloudflare job-API status wiring, Spot and autoscaling; production cutover.

The repository also contains a DynamoDB/Cognito job stack (`src/Trackdub.Api`, `src/Trackdub.Cloud.Aws`,
`src/Trackdub.CloudJobAdapter`). It is not part of the POC. Do not build IaC for it unless a change order adds it.
Reusable parts for the SQS loop: `CliProcessRunner.cs`, the fixed-preset argument building in `JobAdapter.cs`
(it never accepts client-supplied commands, paths or model IDs), and `src/Trackdub.Cloud/Jobs/CloudPresets.cs`.

## Deviations from the SOW

1. **Classic TensorRT + CUDA, not TensorRT-RTX.** The `rtx` profile and `runtime/trt-rtx-ep.manifest.json` do not
   apply to G6/G6e.
2. **Engine caches are GPU- and driver-specific.** Build them on the target instance type only. CI has no GPU. Never
   copy a cache between GPU models or drivers. The presets pin every stage to CUDA, so jobs do not use warmed
   engines yet.
3. **Models come from a seeded volume, not at image build.** The SOW has build-time download; the runbook seeds
   `/mnt/trackdub-data` with hash-verified downloads. Both use `trackdub models download` and `verify`. Decide at
   Design.
4. **zh preset is not qualified.** `poc-server-v1` (en→es) is the baseline. `poc-server-zh-v1` has not passed an
   end-to-end run. See Status.
5. **S3 versioning.** If the artifact bucket is versioned, the 7-day lifecycle must also expire noncurrent versions.
6. **Parity and benchmarks.** The SOW lists parity results under Deliverables but excludes parity measurement from
   SnapSoft's scope. Agree at kickoff who runs the comparison. Local timings come from an RTX 5070 and say nothing
   about L4/L40S throughput or cost.

## Presets, voices and runtime rules

| Preset | Direction | Voice | Status |
|---|---|---|---|
| `poc-server-v1` | en→es | `SPEAKER_00:ef_dora` (Kokoro-82M) | Baseline |
| `poc-server-zh-v1` | en→zh | `SPEAKER_00:qwen3:vivian` (Qwen3-TTS 1.7B CustomVoice) | Failing at export; see Status |

- Input limits: MP4, ≤1 GiB, ≤600 s, ≤1920×1080.
- Provider pins (`src/Trackdub.Cli/PocServerPreset.cs`): VAD on CPU; ASR, translation, TTS, diarization, separation
  and speech enhancement on CUDA. Required stages must not fall back to CPU. Acceptance fails on unexplained
  fallback.
- Qwen3-TTS and Kokoro are excluded from TensorRT by planner policy.
- VRAM admission auto-scales: 75% up to 16 GiB, 85% above, 4096 MB floor. Check the `accelerator-vram-budget` row
  of `trackdub doctor`.
- TTS timing: `--tts-auto-stretch-max-overrun` (0–1, default 0.20) and `--tts-rubberband-stretch` (off by default).
  The export overrun failure is open. Raising the limit to 0.35 with rubberband on still left a 4.9 s overrun.

## Models

Values are from `bundled-models.manifest.json`. `commercial_use_verified` is the manifest's flag, not legal review.

| Model | Task | License | Attribution | Revision | sha256 (prefix) |
|---|---|---|---|---|---|
| onnx-community/silero-vad | vad (CPU) | MIT | no | e71cae966052 | a4a068cd6cf1 |
| Rikorose/DeepFilterNet3 | speech-enhancement | MIT | yes | dcbbe520263d | 7c5399d3da8a |
| cgus/diar_streaming_sortformer_4spk-v2.1-onnx | diarization | NVIDIA Open Model | yes | `cache-installed` (unpinned) | 82b9c735e1cf |
| tonythethompson/parakeet-tdt-0.6b-v3-onnx | asr | CC-BY-4.0 | yes | 107353b64181 | 98a74b21b4cc |
| google/madlad400-3b-mt (`quantized`) | translation | Apache-2.0 | yes | 67037ad42f58 | 3ab4690aab21 |
| csukuangfj/sherpa-onnx-spleeter-2stems | separation | MIT | yes | 3e5a4dd68259 | 932fff35af1d |
| onnx-community/Kokoro-82M-v1.0-ONNX (es) | tts | Apache-2.0 | yes | 1939ad2a8e41 | 8fbea51ea711 |
| tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX (zh) | tts | Apache-2.0 | yes | a2b73f56ffec | f55e99902cb2 |

- Sortformer has no pinned upstream revision. Pin it before the build-time-download decision is final.
- Keep each model's attribution notice next to the seeded weights.
- Preset pin files: `docker/poc-server-preset.json`, `docker/poc-server-quality-preset.json`,
  `docker/poc-server-zh-preset.json`.
- MADLAD `quantized` runs on CUDA. The `trt-fp16` variant passed an isolated smoke test but has not completed a full
  pipeline run. It needs a 24 GiB-class GPU and is in neither preset.

## Host provisioning (once per AMI and driver)

The batch runner calls `scripts/docker/prepare-gpu-worker-host.sh`, which runs these checks in order and stops at the
first failure:

1. `nvidia-smi -L`, Docker with Compose v2, python3, the `nvidia` runtime, and `/mnt/trackdub-data` owned by UID 10001.
2. Pull `TRACKDUB_IMAGE` (digest or pinned tag; `main-server` is rejected).
3. Seed the EN→ES set and the EN→ZH set into the data volume.
4. In-container checks: GPU visible, `providers list` reports CUDA, `trackdub doctor` does not report a failing check.
5. `trackdub models verify --model-id <id>` for each POC model.

Engine warming is off by default. Enable it with `TRACKDUB_WARM_ENGINES=1`. It uses the reviewed allow-list and
explicit `--model` paths. Never run bare `trackdub cache warm --provider tensorrt`: without `--model` it scans the
whole cache and can crash the native process.

Do not use whole-manifest `models verify` (no `--model-id`) as a readiness gate. It checks every manifest model
and exits 2 on a partial cache by design.

Interactive debugging: the image entrypoint runs the CLI through `dotnet`, so define the wrapper in the container:

```bash
trackdub() { dotnet /app/Trackdub.Cli.dll --model-directory /data/models "$@"; }
```

## Job invocation

`run-batch.sh` makes each call. The equivalent by hand:

```bash
job_dir="$(mktemp -d /data/poc-job.XXXXXXXX)"   # new directory per attempt, including retries
trackdub run pipeline --media /work/input/in.mp4 --output "$job_dir/project" \
  --source-language en --target-language es --preset poc-server-v1 --voice SPEAKER_00:ef_dora \
  --subtitle-format srt --result-file "$job_dir/result.json" --progress-file "$job_dir/progress.jsonl"
```

`--result-file` and `--progress-file` must be new files. The runner removes the previous clip's input before each
download, so a failed download cannot run the pipeline on the wrong video.

## Verified and not verified

Verified:
- The batch runner's summary logic (unit tests), the launcher's argument handling and subnet fallback (stub `aws`),
  CloudFormation lint, and shell syntax.
- Two EN→ZH runs on `g6e.xlarge` through TTS, failing at export. Logs, timings and per-stage results are in S3.
- Local RTX 5070 (WSL2) runs of Sortformer and Parakeet on classic TensorRT.

Not verified:
- The CloudFormation stack in a real account, and the instance role's SSM agent behaviour under the inline policy.
- A finished EN→ZH clip, or any EN→ES run on AWS.
- Warmed engines on G6/G6e, or any measured L4/L40S throughput.
- Mandarin audio quality. Parity against the RTX 5070 reference.
- Cold start under two minutes. The current prep step takes about 4.5 minutes, mostly model download and verification.

## Decisions for Design

- Orchestrator: AWS Batch or ECS on EC2 GPU, driven by the SQS loop.
- Instance type and AMI: G6 (L4, 24 GB) or G6e (L40S, 48 GB); driver for CUDA 13.
- Model storage: build-time download (SOW) or seeded volume (current runbook).
- Engine persistence: baked into the image or an EBS snapshot. Rebuild on any driver, TensorRT or model change.
- Provider pins: keep CUDA for diarization and ASR, or move them to TensorRT after L4/L40S measurement.
- Capacity under the 2-minute cold-start target: scale to zero, or keep one host warm. Container exit does not stop
  instance billing.
- Image registry visibility and pull credentials.
- S3 layout, versioning and lifecycle; target US region.
- Who runs the parity comparison against the RTX 5070 reference.
