# Cloud GPU container: publish, seed, and warm

For the asynchronous AWS API/supervisor, use the [SnapSoft application acceptance contract](snapsoft-poc-acceptance.md).
That path adds a CPU API and derived supervisor image; the CLI image below remains one-shot.
`poc-server-v1` uses `docker/poc-server-models.txt` (MADLAD quantized/CUDA), distinct from
the historical TRT warm list. `seed-and-warm-gpu-worker.sh` seeds that list by default for
`TRACKDUB_GPU_PROFILE=server`; `TRACKDUB_MODEL_SET` still overrides it.
No target AWS GPU readiness is established by packaging or local protocol tests.

Trackdub's Linux GPU worker is a one-shot CLI container. The image contains the application/runtime; model weights, the engine cache, and LinML provider downloads live in persistent Docker volumes. Do not copy TensorRT engine cache files between GPU families, GPU models, or driver versions.

## Published images

One `Dockerfile` publishes two GPU profiles from a shared `runtime-base` stage: `--target rtx` (`TRACKDUB_GPU_PROFILE=rtx`, the TensorRT-RTX route) and `--target server` (`TRACKDUB_GPU_PROFILE=server`, the AWS datacenter-GPU route: classic TensorRT plus CUDA, no TensorRT-RTX plugin or license). Neither profile bakes a VRAM budget; the session pool auto-scales its admission budget from detected GPU VRAM. The `server` image carries the CUDA-pinned `poc-server-v1` baseline; classic TensorRT is available and warmed only for the seeded, TRT-qualified graphs.

`.github/workflows/container-image.yml` builds `linux/amd64` for both profiles on container-related pull requests, checks the runtime, and publishes on `main`, `container-v*` tags, or manual dispatch. Every published tag carries the profile suffix:

```text
ghcr.io/mesililm-llc/trackdub:main-rtx
ghcr.io/mesililm-llc/trackdub:main-server
ghcr.io/mesililm-llc/trackdub:sha-<commit>-rtx
ghcr.io/mesililm-llc/trackdub:sha-<commit>-server
ghcr.io/mesililm-llc/trackdub:container-vX.Y.Z-rtx
ghcr.io/mesililm-llc/trackdub:container-vX.Y.Z-server
ghcr.io/mesililm-llc/trackdub:pr-N-rtx
ghcr.io/mesililm-llc/trackdub:pr-N-server
ghcr.io/mesililm-llc/trackdub:latest-rtx
ghcr.io/mesililm-llc/trackdub:latest-server
```

The earlier bare tags (`main`, `sha-<commit>`, `container-vX.Y.Z`, `latest`) are retired deliberately: with two profiles, an unsuffixed tag would not say which GPU route the image provisions for. Local Compose builds pick a target with `TRACKDUB_GPU_TARGET` (default `rtx`); `docker-compose.aws.yml` always selects the `server` target and defaults `TRACKDUB_IMAGE` to `ghcr.io/mesililm-llc/trackdub:main-server` so AWS workers pull the published server image.

The first GHCR package may default to private. After the first successful publish, set `ghcr.io/mesililm-llc/trackdub` visibility to public in the package settings if AWS workers should pull it anonymously. ECR mirroring can be added later without changing the model-volume process.

## Curated POC model volume

The initial seed list is `docker/poc-models.txt`, one pinned manifest model per standard pipeline stage for the proposal's EN→ES and EN→ZH test matrix:

- Silero VAD
- DeepFilterNet3 speech enhancement
- SortFormer 4-speaker diarization
- NVIDIA Parakeet TDT 0.6B v3 ASR (`tonythethompson/parakeet-tdt-0.6b-v3-onnx`)
- MADLAD-400 3B translation, explicitly seeded as the manifest variant `trt-fp16` (the supported TRT-RTX export, rather than the default quantized files)
- Spleeter source separation
- Kokoro 82M TTS (`onnx-community/Kokoro-82M-v1.0-ONNX`), the planner's first TTS preference
- CosyVoice 300M voice-cloning TTS, seeded as the `int8` variant (covers English, Spanish, and Chinese)

The set is intentionally curated, not every one of the 47 model entries or experimental variant in the developer cache. `scripts/docker/seed-model-volume.sh` downloads each exact model through Trackdub's CLI; the list supports `model-id|variant-alias` for pinned variants such as the `trt-fp16` translation export. The CLI resolves pinned revisions and SHA-256 hashes in `bundled-models.manifest.json`; an integrity mismatch fails instead of silently accepting different weights. The manifest marks all eight selected models as commercially usable and redistributable; seven require attribution. Preserve each model's license/attribution notice alongside the seeded weights. Silero VAD is intentionally CPU-side as a small model.

The runtime planner now places `parakeet-tdt-0.6b-v3` first in the ASR preference list and admits its `parakeet-tdt` engine family; the planner regression test confirms the EN POC resolves to this model when it is not yet cached. Parakeet passed `providers trt-rtx smoke --model parakeet-tdt-0.6b-v3` on an RTX 5070: 1 attempted, 1 passed, 0 failed, 0 skipped. It verified the selected provider and ran the encoder/decoder probe. That historical run then exited 139 because the CLI left the session in the process-wide pool until native process shutdown. Fixed on 2026-10-20: provider smoke and verify handlers evict idle sessions before returning. Rebuilt current-source container runs passed both DeepFilterNet3 and Parakeet TRT-RTX smokes with exit code 0.

MADLAD-400's manifest-pinned `trt-fp16` variant passed the same provider smoke (including its production translation probe) with `TRACKDUB_SESSION_VRAM_BUDGET_MB=9000` on the local 12 GiB RTX 5070: 1 passed, 0 failed, process exit 0. Without the explicit budget it was rejected by the 4096 MB floor before session creation (that was before VRAM auto-scaling landed; see the per-worker section below).

The previous `Xenova/whisper-medium` choice is deliberately not seeded: production policy excludes `whisper-onnx` from TensorRT-RTX. Qwen3-ASR is manifest-marked for TRT-RTX but its production planner explicitly prefers DML/CUDA because the TRT graph smoke falls back on unsupported ops. Nemotron is not in the auto-planning shipping lane. Parakeet was chosen because this actual container smoke passed on TRT-RTX.

Native CUDA in the container was initially broken by packaging, not by the GPU: NuGet publish asset selection placed the CPU `Microsoft.ML.OnnxRuntime` core at `/app/libonnxruntime.so`, shadowing the CUDA-enabled core from the transitive `Microsoft.ML.OnnxRuntime.Gpu.Linux` package, so `CUDAExecutionProvider` could not register even though `libonnxruntime_providers_cuda.so` was present. The Dockerfile now overwrites the published core with the GPU Linux core in the build stage (container publish only; desktop publishes are unaffected). After the fix, `providers list` reports CUDA available, and `check --execution-provider cuda --require-execution-provider` planned every stage ready on the RTX 5070 (ASR Parakeet, translation MADLAD, TTS Kokoro) with real CUDA sessions created during planner smoke.

TTS routing follows from that fix. The planner restricts Kokoro to CUDA/CPU (TensorRT-RTX cannot infer its duration-dependent shapes), so Kokoro is the CUDA-verified TTS tentpole and the planner's first TTS preference. CosyVoice remains the voice-cloning dubbing model: its `int8` variant declares CUDA, but its quantized graphs are not TRT-RTX compilable (the Myelin builder rejects their int8 dtype, verified live), and its vocoder graph falls back to CPU, so CosyVoice runs on the CUDA route rather than TRT-RTX.

On an AWS Linux GPU worker with Docker Compose v2:

```bash
git clone --depth 1 https://github.com/Mesililm-LLC/Trackdub.git
cd Trackdub

docker volume create trackdub_trackdub-data
./scripts/docker/seed-model-volume.sh
```

The Compose project name controls volume names. Use the same project name on every invocation (or set `COMPOSE_PROJECT_NAME`) so the `/data` volume is reused. The model files live at `/data/models` inside the container. Expected local cache sizes are only rough planning figures; allow extra volume space for downloaded artifacts, generated engine contexts, logs, and future revisions.

## CUDA and native TensorRT smoke matrix (2026-10-17)

On 2026-10-17, the full CUDA and native TensorRT smoke matrix ran against image `container-pipeline-trackdub:latest` on a local RTX 5070 (12 GiB). The image pins TensorRT 10.13.3.9. This pass tested the native `Cuda` and `TensorRt` execution providers in the image; it did not retest TensorRT-RTX beyond the earlier evidence above. A smoke pass means the production smoke tester created a session on the requested provider and ran its smoke inference. It does not mean a full pipeline end-to-end run.

To reproduce:

```bash
docker compose build
docker run --rm --gpus all container-pipeline-trackdub:latest providers list
GPU_SMOKE_OUTPUT_DIR=/path/to/gpu-smoke-evidence scripts/docker/smoke-gpu-matrix.sh
```

`providers list` must report both `Cuda` and `TensorRt` as available before running the matrix. The script runs one `docker run` per model so a native crash in one run cannot abort or contaminate the rest of the matrix, and it writes per-run stdout, stderr, and exit code files under `GPU_SMOKE_OUTPUT_DIR`. The JSON summary records inference results. The container exit code is also part of the verdict: any nonzero exit fails that row, even if inference printed a passing summary. The script completes the remaining runs, writes `matrix-summary.json`, and exits nonzero when any row fails.

| Provider | Model | Result | Note |
|---|---|---|---|
| CUDA | diarization | Passed | SortFormer diarization session and smoke inference passed. |
| CUDA | enhancement-deepfilternet3 | Passed | DeepFilterNet3 session and smoke inference passed. |
| CUDA | asr-parakeet-tdt-0.6b-v3 | Passed | Parakeet TDT session and smoke inference passed. |
| CUDA | separation-spleeter | Failed | Smoke dummy-input dimension mismatch: input `x` dimension index 1 got 1, expected 2. Identical on both execution providers, so this is a smoke harness limitation, not a provider failure. The production Spleeter entry path is unaffected; its stems were cache-warmed successfully during provisioning. |
| CUDA | translation-madlad-trt-fp16 | Failed | VRAM admission rejected the bundle before session creation: the session bundle needs about 12758 MB across its graphs, which exceeds the 9000 MB local admission budget. This route needs a 24 GiB-class GPU, where the auto-scaled budget (85% of detected VRAM) admits it. TensorRT-RTX remains the local MADLAD route, with a bundle estimate of about 8069 MB. |
| CUDA | tts-kokoro | Passed | Kokoro session and smoke inference passed. |
| CUDA | tts-cosyvoice-int8 | Passed | CosyVoice `int8` session and smoke inference passed. |
| TensorRT | diarization | Passed | SortFormer diarization session and smoke inference passed on the native TensorRT execution provider. |
| TensorRT | enhancement-deepfilternet3 | Passed | DeepFilterNet3 session and smoke inference passed on the native TensorRT execution provider. |
| TensorRT | asr-parakeet-tdt-0.6b-v3 | Passed | Parakeet TDT session and smoke inference passed on the native TensorRT execution provider. |
| TensorRT | separation-spleeter | Failed | Same smoke dummy-input dimension mismatch as CUDA (input `x` dimension index 1 got 1, expected 2). This is the same harness limitation, not a TensorRT-specific failure, and the production Spleeter entry path is unaffected. |
| TensorRT | translation-madlad-trt-fp16 | Passed (after policy correction) | The matrix run itself was refused by a production policy guard, not by TensorRT. An isolated-process retest on 2026-10-18 (ORT 1.31, TensorRT 10.13, RTX 5070) then passed the same smoke with exit 0 after a roughly 9 minute engine build. The guard had codified an unproven assumption: TensorRT support is an export property, and the pinned `trt-fp16` export is TensorRT-qualified on both TensorRT providers. The policy now says so; the other MADLAD exports (quantized, merged-decoder, int4-kv) were never built for TensorRT and stay unqualified until a TensorRT-built export passes the same isolated retest. Locally this route also needs the VRAM admission budget raised (bundle estimate about 12758 MB, as on CUDA); TensorRT-RTX remains the preferred MADLAD route. |
| TensorRT | tts-cosyvoice-int8 | Failed | The TensorRT execution provider failed to create an engine from the CosyVoice `int8` network. The int8 graphs are not TensorRT-compilable, consistent with the earlier TensorRT-RTX Myelin rejection of their int8 dtype. CUDA is the CosyVoice route. |

Conclusions from this pass:

- The tentpole routes are unchanged: Parakeet works on any GPU execution provider tested, MADLAD `trt-fp16` runs via TensorRT-RTX (preferred) and now classic TensorRT, Kokoro runs on CUDA, and CosyVoice `int8` runs on CUDA.
- Native TensorRT in the image is a working fallback execution provider for Parakeet, SortFormer, and DeepFilterNet3. That duplicates the existing TensorRT-RTX coverage for those models, so there is no production need to re-rank provider preferences for them.
- One planner and policy change did follow from this pass, after the isolated MADLAD retest: the planner now ranks classic TensorRT second for MADLAD (after TensorRT-RTX, before CPU), and the smoke guard treats TensorRT qualification as an export property instead of a model-family exclusion. The Spleeter outcome is a smoke harness limitation and the CUDA MADLAD outcome is VRAM admission on a 12 GiB card, not planner defects.
- The classic TensorRT engine cache is configured under the engine-cache root at `classic-trt/<hardware-and-runtime-fingerprint>`. The current fingerprint includes the loaded TensorRT core binary's SHA256 identity; unresolved identities use a process-specific namespace. The model warm-up measurements below verified persistence and cross-process engine reuse with the earlier version-named scheme. The separate first-append probe below verifies native load ordering and namespace stability for the current SHA256 scheme.

## Two-profile and engine-cache live verification (2026-10-18)

Live verification ran on the local Windows Docker Desktop host with an NVIDIA GeForce RTX 5070 (12,227 MiB, driver 617.42), TensorRT 10.13.3, and ORT 1.31.0. At initial `nvidia-smi` inspection the GPU was at 0% utilization with 2,944 MiB in use; unrelated containers were running, so CPU contention may have affected wall-clock measurements. The profiles were rebuilt locally during diagnosis; the final diagnostic image tags were `trackdub-gpu:rtx` and `trackdub-gpu:server`. The working-tree diagnostics were discarded after the acceptance still failed; the image tags used for the measurements include those temporary diagnostic changes and are not a clean build of the final committed source.

Both image tags reported `Cuda available=true` and `TensorRt available=true` with `--gpus all`. `TensorRTRtx available=false` on both bare `docker run` probes because the LinML plugin volume is not attached; this is expected. The requested DeepFilterNet3 and Parakeet model files were present in `trackdub_trackdub-data`. Before warming, `/data/engine-cache` did not exist; existing TensorRT-RTX engine files were instead under `/data/models/EngineCache`, outside the classic cache root. No classic cache directory was present to delete.

Classic TensorRT cache warm was attempted against DeepFilterNet3 encoder, Parakeet encoder, and Parakeet joint decoder, with one container per pass. The first pass took 53.036 s and returned `compiled=0, reused=0, failed=3`; each item reported `status=failed` because its session created no files in `/data/engine-cache/classic-trt/Blackwell-617.42-trt-10.13.3-ort-1.31.0`. Item warm-load times were 10,880.8 ms, 6,250.3 ms, and 2,264.0 ms. The identical second pass took 20.533 s and again returned `compiled=0, reused=0, failed=3` (4,575.8 ms, 1,618.4 ms, and 1,336.1 ms per item). A post-pass file listing found zero files under `/data/engine-cache/classic-trt`; thus this was not compiled-then-reused acceptance. The initial volume had no classic cache artifacts, so the failure was not caused by a stale classic cache.

MADLAD `translation-madlad-trt-fp16` was smoked twice on the server image with `TRACKDUB_SESSION_VRAM_BUDGET_MB=14000`. Both JSON verdicts were `passed=1, failed=0`, exit 0. The first wall-clock was 1,034.925 s (17m 14.925s); the second was 1,112.843 s (18m 32.843s), 7.53% slower rather than faster. No classic TensorRT engine files appeared in the cache root after the first smoke. The successful inference verdict therefore does not establish cache reuse. The 14,000 MB override is only a local 12 GiB admission workaround; auto-scaled admission for MADLAD on 24/48 GiB G6/G6e cards remains unverified.

The requested matrix rerun used `GPU_SMOKE_IMAGE=trackdub-gpu:rtx` and completed in 696.746 s with exit 0. The current script emitted **12 cells** (7 CUDA and 5 TensorRT), not 13: 8 passed and 4 failed. Parakeet, SortFormer diarization, DeepFilterNet3, Kokoro CUDA, and CosyVoice CUDA passed. Both Spleeter cells failed with the known dummy-input dimension mismatch (`x`, index 1: got 1, expected 2); both MADLAD cells were rejected before session creation because the bundle estimate was 12,758 MB versus the auto-scaled 9,170 MB admission budget. This matrix run did not include a CosyVoice TensorRT cell. Full machine-readable results and per-cell stdout/stderr/exit codes are in `matrix/matrix-summary.json` and its sibling files in the evidence directory noted below.

Raw command output, stderr, exit codes, wall-clock timings, image/provider probes, and cache file listings from that first pass were saved in a local evidence directory (not committed). That pass established the blocker; the fix verification below supersedes its cache verdict.

## Classic TensorRT engine-cache fix verification (2026-10-18)

Root cause, confirmed by a probe run inside the server image (ORT 1.31): `OrtEnv.GetAvailableProviders()` reports `TensorrtExecutionProvider`, but `OrtEnv.GetEpDevices()` enumerates only the CPU device — the built-in classic TensorRT EP is not an EP-device, so the generic EP-device append in `AppendTensorRtOrFallbackProvider` was unreachable and every session silently fell back to the legacy `AppendExecutionProvider_Tensorrt(deviceId: 0)` append, which cannot carry provider options. The `trt_engine_cache_enable`/`trt_engine_cache_path` options from `BuildTensorRtOptions` never reached ORT. Two fixes landed: commit `472529f4` appends the built-in EP via `OrtTensorRTProviderOptions` (`UpdateOptions` with the built options; the cache directory is created first; the generic EP-device branch and the legacy/DirectML/CPU fallbacks are kept after it), and commit `53daf07a` makes `cache warm` run one zero-input inference per graph — a minimal repro proved engine files are written only after the first `Run()`, because dynamic-shape graphs defer engine building to first inference (session creation alone writes nothing).

Both profile images were rebuilt from the committed tree (`trackdub-gpu:server` `c9f19e27133b`, `trackdub-gpu:rtx` `be806bae6089`); both report `Cuda available=true` and `TensorRt available=true` with `--gpus all` (TensorRT-RTX false on bare runs without the LinML volume, as expected). With only the `classic-trt` subdirectories deleted from `trackdub_trackdub-data` first, a fresh `cache warm --provider tensorrt` on DeepFilterNet3 `enc.onnx`, Parakeet `encoder-model.onnx`, and Parakeet `decoder_joint-model.onnx` (server image) returned `compiled=3, reused=0, failed=0` in 96 s (warm-load 31,441 ms / 44,732 ms / 9,673 ms) and wrote six engine/profile files (2.4 GB). The identical second warm in a fresh container returned `compiled=0, reused=3, failed=0` in 16 s (3,343 ms / 4,656 ms / 1,071 ms). MADLAD `translation-madlad-trt-fp16` smoked twice on the server image with `TRACKDUB_SESSION_VRAM_BUDGET_MB=14000`: both JSON verdicts `passed=1, failed=0`, exit 0; the cold run took 234 s (versus 1,035 s in the blocked pass) and persisted two more engines (about 7.5 GB and 5.3 GB), and the second run took 57 s (versus 1,113 s previously) on the cached engines. In the `docker run` form used here the resolved cache root is `/data/models/EngineCache/classic-trt/Unknown-unknown-trt-10.13.3-ort-1.31.0`: the CLI derives the cache root from the model directory, and at that time the GPU/driver fingerprint was `Unknown` because `/proc/driver/nvidia` is not visible under `docker run --gpus all`; the key was consistent across every run above, which is what reuse requires. That `Unknown` fingerprint is superseded by the NVML fix below. The 14,000 MB override remains a local 12 GiB admission workaround; auto-scaled admission for MADLAD on 24/48 GiB G6/G6e cards remains unverified.

NVML GPU identity detection (2026-10-18, commit `dd00337f`): `MachineHardwareProfileProvider`'s Linux branch now fills the GPU name and driver version from NVML (`AcceleratorVramProbe.QueryNvmlGpuIdentity`: `nvmlDeviceGetName` + `nvmlSystemGetDriverVersion` via `libnvidia-ml.so.1`, which the NVIDIA container toolkit injects whenever `--gpus all` is used — exactly where sysfs/proc give no usable name or driver). Rebuilt `trackdub-gpu:server` (`93b1336a37d4`) live-proved it: `cache warm --provider tensorrt` on DeepFilterNet3 `enc.onnx` returned `engineCacheDirectory` `/data/models/EngineCache/classic-trt/Blackwell-617.42-trt-10.13.3-ort-1.31.0` with `environmentFingerprint` `Blackwell|617.42|0.4.0+cu13+trt-rtx-1.6.1|1.31.0`, `compiled=1` on the first warm (49 s) and `reused=1, failed=0` on the identical second warm in a fresh container (13 s); `providers list` still reports Cuda and TensorRt available. The old `Unknown-unknown-trt-10.13.3-ort-1.31.0` cache directory remains orphaned in the volume (not deleted). Evidence is kept in a local directory (not committed).

First-append cache identity probe (2026-10-18): two disposable `trackdub-gpu:server` containers ran with the repaired portable inference assembly mounted read-only, private `/tmp/trt-load-sequence` cache roots, and no model-cache mounts. Both started without a mapped TensorRT core and reported different `unknown-process-<guid>` identities. Their first classic TensorRT append selected `TensorRt` and configured the same directory: `classic-trt/Blackwell-NVIDIA-GeForce-RTX-5070-617.42-trt-sha256-1692EAC238E7B0627B1D3B8A80214AEFCBFBA50B700FE76CCBDD3906C3B956FA-ort-1.31.0`. In both processes, that configured path equaled a fresh post-append cache-path resolution. The probe created session options without loading a model or running inference. This verifies first-append load ordering and namespace persistence; full-model engine reuse under the SHA256 scheme remains unverified by this probe.

Fix-verification evidence (diagnosis probe, RED/GREEN test logs, full-suite logs, warm/MADLAD JSON, wall clocks, and file listings) is kept in a local directory (not committed).

## First-session TensorRT cache identity verification (review follow-up)

The cache fingerprint is now resolved before provider append by preloading the TensorRT core library; if any hardware, driver, TensorRT binary, or ORT identity is unavailable, cache reuse is explicitly disabled. A clean `server` image built from the reviewed worktree (RTX 5070, driver 617.42, TensorRT 10.13.3, ORT 1.31.0) reported `Blackwell|NVIDIA GeForce RTX 5070|617.42|trt-sha256-1692EAC238E7B0627B1D3B8A80214AEFCBFBA50B700FE76CCBDD3906C3B956FA|ort-1.31.0` on the first classic-TRT warm. With an explicit `TRACKDUB_ENGINE_CACHE_ROOT=/cache` mounted as a fresh isolated volume, the first container compiled one DeepFilterNet3 encoder engine in 42.5 s; a second container using the same volume reported `compiled=0, reused=1` in 6.1 s. Both used `/cache/classic-trt/Blackwell-NVIDIA-GeForce-RTX-5070-617.42-trt-sha256-1692EAC238E7B0627B1D3B8A80214AEFCBFBA50B700FE76CCBDD3906C3B956FA-ort-1.31.0`. This revalidates the first-session load sequence and confirms an explicit cache root survives headless CLI model-directory configuration; the earlier version-named cache measurements above are historical evidence from before this identity scheme.

## TensorRT qualification retests (2026-10-18)

Several provider exclusions in the planner and the smoke guard dated from an ORT 1.24.5 / Windows-era run and had never been retested on the current stack. On 2026-10-18 each unproven exclusion was retested in an isolated process (one container per model and provider, so a host-killing failure could not hide as a catchable error) on ORT 1.31, TensorRT 10.13, RTX 5070. Policy now follows the evidence: TensorRT support is a property of a specific export, qualified when a TensorRT-built export passes this isolated smoke.

| Export | Classic TensorRT | TensorRT-RTX | Policy now |
|---|---|---|---|
| OPUS-MT (merged-decoder) | Passed | Session construction aborted the host process (exit 134) | TensorRT-RTX excluded only; classic TensorRT allowed |
| Silero VAD | Failed catchably: shape inference rejected the `If` subgraph ("TensorRT input ... has no shape specified") | Importer rejects the `If`/`Squeeze` graph (issue #329) | Both TensorRT providers excluded until a shape-inferred export ships; the model is tiny, so CPU/DirectML costs nothing |
| Whisper medium (`Xenova/whisper-medium`, whisper-onnx) | Passed | Passed | TensorRT denial removed; whisper-onnx uses the default provider list and variants are smoke-gated individually |
| Qwen3-ASR (`tonythethompson/qwen3-asr-0.6b-onnx`) | Isolated process terminated during import (plugin ops the importer cannot resolve) | Historically fell back catchably after a long compile | Classic TensorRT excluded (the planner smokes in-process and must never attempt a host-killing provider); TensorRT-RTX stays reachable behind DirectML/CUDA |
| GenAI text refinement (`tonythethompson/Qwen2.5-1.5B-Instruct`) | ORT GenAI has no classic TensorRT device ("NvTensorRtRtx execution provider is not supported in this build") | NvTensorRtRtx device aborted the host process (exit 134) | Both TensorRT kinds refused for bundled GenAI exports; CPU/CUDA remain the GenAI routes |

The GenAI retest also exposed a container packaging defect: publish output shipped only `libonnxruntime-genai-cuda.so`, not the base `libonnxruntime-genai.so` the loader probes for, so no GenAI model could load in the image at all. The Dockerfile now copies the base library from the GenAI NuGet package, the same pattern as the GPU ONNX Runtime core overwrite.

One related question was evaluated and rejected: serving CosyVoice through TensorRT-LLM or vLLM. CosyVoice `int8` already runs on the CUDA execution provider in Trackdub; what fails is TensorRT engine building from the int8 ModelOpt graphs, which is an export problem. TensorRT-LLM and vLLM host PyTorch LLM weights in a separate Python runtime, which would break the single self-contained binary and container deployment model, and neither would accelerate the HiFT vocoder that stays on CPU either way. If TTS latency ever measures as the bottleneck, the in-architecture path is an fp16 TensorRT-qualified CosyVoice export, qualified by the same isolated retest.

## Server image contract

The `server` stage is the deployable AWS worker base, not just a profile label:

- `TRACKDUB_GPU_PROFILE=server`, on an Ubuntu 24.04 runtime base with CUDA 13, cuDNN, and TensorRT 10.13.3.9 runtime libraries.
- CUDA and TensorRT ONNX Runtime providers plus the base GenAI native library staged in `/app`; `libonnxruntime.so` is the CUDA-enabled core.
- `util-linux` (`/usr/bin/setsid`) for the cloud job adapter's process-group supervision.
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` so NVENC export works on G6/G6e; Compose sets the same value.
- `/data/engine-cache` created under the persistent `/data` mount, matching `TRACKDUB_ENGINE_CACHE_ROOT`.
- The `run pipeline` machine protocol (`--result-file`, `--progress-file`) required by `Dockerfile.cloud-worker`.

Container CI enforces the profile marker, Ubuntu 24.04 base, `setsid`, classic-TensorRT/cuDNN/CUDA sonames, the ONNX Runtime native libraries, and the machine-protocol flags. These are packaging checks only: the CI runner has no GPU, so provider readiness and engine-cache behavior still require the per-worker verification below.

## Quality TTS preset (evaluation)

`poc-server-quality-v1` is an opt-in evaluation preset for the server profile. It keeps the `poc-server-v1` pipeline (Parakeet ASR, MADLAD quantized translation, same CUDA pins) and replaces Kokoro with Qwen3-TTS-1.7B CustomVoice (`qwen3-tts-1.7b-customvoice`). Seed it with `TRACKDUB_MODEL_SET=docker/poc-server-quality-models.txt`; the accepted baseline and its evidence are unchanged.

- Nine preset voices (`qwen3:vivian`, `qwen3:serena`, `qwen3:ryan`, and others); `qwen3:ryan` is the default for Spanish. Every preset speaks all ten supported languages (`zh,en,ja,ko,de,fr,ru,pt,es,it`).
- No voice cloning and no cloning consent. CosyVoice and Chatterbox are cloning-only in this codebase and require a reference clip, so they are not drop-in alternatives.
- `qwen3-tts` is excluded from TensorRT families by planner policy; it runs on the CUDA route.
- Measure quality (round-trip ASR WER plus a Spanish spot check), latency, peak VRAM, and cost per audio minute on the target L4/L40S before promotion. This preset is outside the accepted `poc-server-v1` contract until that evidence is reviewed.

## Per-worker provider install and engine warm

Provisioning branches on `TRACKDUB_GPU_PROFILE` (default `rtx`), which each published image carries as an environment variable; the same script provisions either profile, selecting that profile's default seed and warm lists (`TRACKDUB_MODEL_SET` and `TRACKDUB_WARM_MODEL_SET` override them).

RTX profile (local / TensorRT-RTX): install the NVIDIA driver and NVIDIA Container Toolkit on the host first. Review the NVIDIA TensorRT-RTX and CUDA license terms, then accept them explicitly for this worker:

```bash
export TRACKDUB_ACCEPT_NVIDIA_TENSORRT_RTX_LICENSE=1
./scripts/docker/seed-and-warm-gpu-worker.sh
```

The script checks for a host-visible NVIDIA GPU, seeds the curated model volume, installs the LinML TensorRT-RTX plugin into the dedicated Compose volume, and runs `trackdub cache warm` against only `docker/poc-warm-models.txt`. This explicit list avoids recursively warming every `.onnx` in the volume: a full scan hit an OPUS-MT decoder that can terminate the host with a native stack overflow, and the Silero VAD graph fails TensorRT import on its `If`/`Squeeze` branches. A failed target stops provisioning; unsupported models are deliberately excluded and listed in comments. Parakeet TDT is in the TRT warm allowlist; the small Silero VAD model intentionally runs on CPU. The NVIDIA acceptance variable is deliberately not defaulted in scripts or CI.

Server profile (AWS datacenter GPUs / classic TensorRT + CUDA): no TensorRT-RTX license acceptance and no plugin install. The server image and the AWS Compose override select this profile; the script seeds `docker/poc-server-models.txt` by default (the CUDA-pinned `poc-server-v1` bundle; `TRACKDUB_MODEL_SET` overrides it), then warms classic TensorRT engines with `trackdub cache warm --provider tensorrt` against `docker/poc-warm-models.server.txt`. That warm list covers only server-seeded, classic-TensorRT-qualified graphs (DeepFilterNet3, SortFormer, Spleeter, Parakeet); MADLAD is absent because the server seed uses the quantized export, which is not classic-TensorRT qualified. A warm-list override applies to either profile via `TRACKDUB_WARM_MODEL_SET`.

Before scheduling jobs, verify on the actual G6/G6e host that `/proc/driver/nvidia/gpus` is visible inside the container (the CUDA/TensorRT discovery gate), then run `trackdub providers list` and `trackdub doctor`. The recorded container GPU evidence in this runbook is from WSL2 on an RTX 5070; native-Linux datacenter-GPU visibility is still an acceptance prerequisite.

For AWS, use a dedicated EBS volume mounted on the host at `/mnt/trackdub-data` so weights and GPU-specific engine artifacts survive container replacement. Format/attach/mount it through your normal EC2 provisioning, make the mount persistent, and grant the container's UID/GID 10001 write access:

```bash
sudo mkdir -p /mnt/trackdub-data
sudo chown 10001:10001 /mnt/trackdub-data
export TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml
# Server profile: no TRACKDUB_ACCEPT_NVIDIA_TENSORRT_RTX_LICENSE is needed.
TRACKDUB_GPU_PROFILE=server ./scripts/docker/seed-and-warm-gpu-worker.sh
```

`docker-compose.aws.yml` defaults to the published `ghcr.io/mesililm-llc/trackdub:main-server` image (override with `TRACKDUB_IMAGE`) and builds the `server` target when Compose must build instead of pull. The AWS override binds that EBS mount to `/data`; the seed script detects the override and uses it for every CLI invocation. No VRAM budget is pinned anymore: the session pool auto-scales its admission budget from detected GPU VRAM (75% on cards up to 16 GiB, 85% above that, floor 4096 MB), detecting VRAM via /proc and sysfs with an NVML fallback for containers where /proc/driver/nvidia is hidden. Run `trackdub doctor` and check the `accelerator-vram-budget` row to see the detected VRAM, the resolved budget, and its source; it warns when detection failed and the floor fallback is in effect. Set `TRACKDUB_SESSION_VRAM_BUDGET_MB` only as an explicit operator override. MADLAD's TRT-RTX bundle is estimated at ~8069 MB for its encoder/decoder sessions. Treat the resulting engine cache as specific to that instance's GPU and driver.

Run the provisioning script once on each newly provisioned GPU/driver combination, and again after a driver or provider update. It writes the cache under `/data` (the persistent `trackdub-data` volume). The separate `trackdub-linml` volume persists plugin downloads. If changing the Compose project name, carry both volumes forward or the worker will appear unseeded.

For CPU-only local use, run only `seed-model-volume.sh`; do not run the GPU warm script. The CI workflow intentionally does not download weights or accept NVIDIA licenses. Its `doctor` check allows the empty model-volume readiness failure but requires the manifest, ffmpeg/ffprobe, espeak-ng, and writable model cache checks to pass.
