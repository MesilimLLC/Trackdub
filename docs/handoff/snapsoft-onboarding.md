# SnapSoft POC: Trackdub GPU batch runner

Contract: SOW "Infrastructure Setup POC," signed 2026-10-30. Where this page and the SOW differ, the SOW governs
unless both parties sign a change order.

To run a batch, use [snapsoft-quickstart.md](snapsoft-quickstart.md).

## Status (2026-11-21)

- Not yet passing. Two EN→ZH runs on `g6e.xlarge` completed every stage through TTS, then failed at export: the
  dubbed take ran past the 40.2 s source (6.8 s, then 4.9 s after raising the stretch limit), and the exporter will not
  truncate it. No finished clip exists.
- The batch runner (`deploy/aws/`) is in PR #666, not merged.
- EN→ES (`poc-server-v1`) has not run on AWS.
- The CloudFormation stack has not been deployed from the template.

## Scope (SOW)

SnapSoft: IaC for SQS, S3, KMS, the EC2 GPU worker and the orchestrator (AWS Batch or ECS); the worker job loop;
the Cloudflare-to-SQS/S3 interface contract.

Mesilim: the image, model sets and presets, and the Cloudflare-side changes.

Success criteria: the 10-clip set (EN→ES and EN→ZH) runs end to end on at least one GPU instance type; all POC
resources are IaC in Mesilim's account; job success ≥95%; cold start under 2 minutes; output parity with the RTX 5070
reference; a readout with run-rate and decisions.

Out of scope: AWS AI services in the pipeline, multi-region, billing, Cloudflare job-API status wiring, Spot and
autoscaling, production cutover. The repository's DynamoDB/Cognito job stack is not part of the POC.

## Deviations from the SOW

1. **Classic TensorRT + CUDA**, not TensorRT-RTX.
2. **Engine caches are GPU- and driver-specific.** Build on the target instance type only. The presets pin all stages to
   CUDA, so jobs do not use warmed engines yet.
3. **Models come from a seeded volume**, not at image build. The SOW has build-time download. Decide at Design.
4. **zh preset is not qualified.** `poc-server-zh-v1` has not passed end to end. `poc-server-v1` is the baseline.
5. **S3 versioning.** If the bucket is versioned, the 7-day lifecycle must also expire noncurrent versions.
6. **Parity.** The SOW lists parity under Deliverables but excludes parity measurement from SnapSoft's scope. Agree
   who runs the comparison at kickoff.

## Models

The sets are `docker/poc-server-models.txt` (en→es) and `docker/poc-server-zh-models.txt` (en→zh). Source of truth for
licenses and hashes: `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json`.

- All POC models are permissively licensed. Seven of eight require attribution, and attribution notices must stay with
  the seeded weights.
- The Sortformer diarization model has no pinned upstream revision. Pin it before the build-time-download decision.
- MADLAD `quantized` runs on CUDA. The `trt-fp16` variant is not in either preset.

## Decisions for Design

- Orchestrator: AWS Batch or ECS on EC2 GPU, driven by the SQS loop.
- Instance type and AMI: G6 (L4, 24 GB) or G6e (L40S, 48 GB); driver for CUDA 13.
- Model storage: build-time download (SOW) or seeded volume (current).
- Engine persistence: baked into the image or an EBS snapshot. Rebuild on any driver, TensorRT or model change.
- Provider pins: keep CUDA for diarization and ASR, or move them to TensorRT after L4/L40S measurement.
- Capacity for the 2-minute cold start: scale to zero, or keep one host warm. Stopping the container does not stop
  instance billing.
- Image registry visibility and pull credentials.
- S3 layout, versioning, lifecycle, and target US region.
- Who runs the parity comparison.

## Verified and not verified

Verified:
- Batch summary logic and launcher argument handling (unit and stub tests), CloudFormation lint, shell syntax.
- Two EN→ZH runs on `g6e.xlarge` through TTS, with logs and timings in S3.
- Local RTX 5070 (WSL2) runs of Sortformer and Parakeet on classic TensorRT.

Not verified:
- The CloudFormation stack and the instance role's SSM agent under the inline policy, in a real account.
- A finished EN→ZH clip, or any EN→ES run on AWS.
- Warmed engines on G6/G6e, or L4/L40S throughput.
- Mandarin audio quality, parity against the RTX 5070 reference, and cold start under two minutes.
