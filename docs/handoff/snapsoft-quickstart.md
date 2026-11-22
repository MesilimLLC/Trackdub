# Trackdub POC quickstart: batch runner in your AWS account

Runs the published `server` image on one GPU instance and processes a batch of MP4 clips. Each clip gets a job
directory, a result and one summary line. A failed clip does not stop the batch. The instance terminates when the
batch finishes or after `--max-minutes` (default 180), whichever comes first.

Scope and status: [snapsoft-onboarding.md](snapsoft-onboarding.md).

## Prerequisites

- IAM: the `SnapSoft-POC-Engineer` permission set, in `us-east-1`. It covers CloudFormation, EC2 (g6/g6e only), S3,
  SSM, and IAM roles named `trackdub-poc-*`.
- Tooling: AWS CLI v2, Python 3, and a checkout of this repository at the commit the image was built from.
- Image: pinned. Use a digest (`ghcr.io/mesilimllc/trackdub@sha256:…`) or `sha-<commit>-server`. Never `main-server`.
- GitHub: a classic PAT with `read:packages` only, SSO-authorized for MesilimLLC.
- Clips: MP4, at most 1 GiB, 600 s, 1920×1080.

## 1. Store the GHCR token

```bash
read -rs GHCR_TOKEN && echo
aws ssm put-parameter --region us-east-1 --name /trackdub/ghcr-read-token \
  --type SecureString --value "$GHCR_TOKEN" --overwrite
unset GHCR_TOKEN
```

The instance reads this parameter at boot. The token never goes into user data or the repository.

## 2. Deploy the stack

```bash
deploy/aws/deploy-stack.sh --stack snapsoft-poc            # add --bucket-name if the default is taken
```

Creates: a VPC with three public subnets, a security group with no inbound rules, a private bucket (`runs/`
expires after 30 days), and the instance role and profile. No compute. Re-running is safe.

## 3. Launch a batch

```bash
deploy/aws/launch-run.sh \
  --stack snapsoft-poc \
  --image "ghcr.io/mesilimllc/trackdub@sha256:<digest>" \
  --repo-commit <full-commit-sha> \
  --ghcr-user <github-user> \
  --clips-dir ./clips \
  --run-id 2026-11-26-batch1
```

Defaults are en→zh: `--target-language zh --preset poc-server-zh-v1`. For en→es: `--target-language es --preset poc-server-es-qwen-v1` (`poc-server-v1` needs Kokoro, which the Linux
server image does not ship; on images older than this preset, `poc-server-quality-v1` is the same pipeline).
With no `--voice`, the pipeline estimates each speaker's gender from pitch and gives each speaker a distinct Qwen3 preset
of that gender, preferring presets native to the target language (male zh → `uncle_fu`, male es → `ryan`, female zh →
`vivian`). The chosen voice is printed as a progress line per speaker. Pass `--voice SPEAKER_00:qwen3:<preset>` to force one.

Pipeline flags go through `--pipeline-args`, for example `--pipeline-args "--tts-rubberband-stretch --tts-auto-stretch-max-overrun 0.35"`.

Options: `--instance-type` (`g6.xlarge`, `g6.2xlarge`, `g6e.xlarge`, `g6e.2xlarge`; default `g6e.xlarge`),
`--max-minutes`, `--vram-budget-mb` (session pool VRAM admission budget; optional, the pool detects it from the GPU
by default, see Known failures), `--ami`, `--dry-run` (renders and prints the plan, uploads and launches nothing).

The launcher uploads clips to `s3://<bucket>/runs/<run-id>/input/`, then tries each subnet until one has capacity.
Clip filenames must be unique.

## 4. Watch and collect

```bash
BUCKET=$(aws cloudformation describe-stacks --stack-name snapsoft-poc --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)

aws s3 cp "s3://$BUCKET/runs/2026-11-26-batch1/summary.txt" -          # one line per clip, updated after each clip
aws s3 sync "s3://$BUCKET/runs/2026-11-26-batch1/" "./results/2026-11-26-batch1/"
```

Per clip: `jobs/<clip>/result.json`, `jobs/<clip>/progress.jsonl`, `jobs/<clip>/project/`. Per run: `run.log`,
`timings.json`, `summary.txt`.

Summary line format: `<clip> <ok|failed> <duration> <verdict>`. A failure names the stage and reason, and a
suggested fix where one is known.

## 5. Known failures

| Verdict contains | Cause | Fix |
|---|---|---|
| `FAILED at export: Dubbed take extends … s past the end` | TTS overran the source by more than the exporter fits on its own: it compresses a take that runs past the end of the clip by exactly its overhang, up to 1.25x, and won't truncate beyond that | Shorten the translation for that clip, or `--pipeline-args "--tts-rubberband-stretch --tts-auto-stretch-max-overrun 0.35"` (default limit 0.20) to compress earlier |
| `FAILED at input: could not download` | S3 read failed | Check the object exists under `input/`, rerun the launch for that clip |
| `UPLOAD FAILED` | Results could not reach S3 after 3 tries | Results are only on the instance; the data volume is deleted at termination. Open an incident before relaunching |
| `NO RESULT` | The pipeline exited before writing `result.json` | Read `run.log` and the clip's `progress.jsonl` |
| `STALLED in stage 'tts'` or `exceeds its admission budget` with GPU at 0% | The session pool's admission budget is below the reservation estimate for the Qwen3-TTS graphs held together: about 18.0 GB on CUDA, against the L4's detected 19.6 GB, so a stock L4 fits. A server image built before the CUDA 1.5x reservation factor (merge commit `ae5cb0d`) reserves about 23.8 GB for the same set and does not fit | Use an image built from `main` at or after `ae5cb0d`, or pass `--vram-budget-mb` with a value above the estimate (30000 admitted the measured run on an older image) |
| Launcher: `No subnet had capacity` | No g6/g6e capacity in the three zones | Retry later, or use another instance type |

The export overrun is still open. The 0.35 setting in the first retry still left a 4.9 s overrun on the test clip.

## 6. Teardown

The instance terminates itself. Nothing else runs.

When the POC ends, download what you need, then:

```bash
aws cloudformation delete-stack --stack-name snapsoft-poc --region us-east-1
aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET" --region us-east-1
aws ssm delete-parameter --region us-east-1 --name /trackdub/ghcr-read-token
```

The bucket is retained when the stack is deleted, so results are not lost by accident. Empty and delete it
explicitly.

## Cold start

Measured on the image built from `main` at commit `1ed89aa`, pulled as
`ghcr.io/mesilimllc/trackdub:sha-1ed89aa4a6a2923871c56ad27ba1bc834b75aece-server`. The clip was 40 seconds of English, with
the default Spleeter separator and the stock AMI:

| Instance | Host ready after boot | Clip total | Translation | TTS |
|---|---|---|---|---|
| `g6.xlarge` (L4, 23 GB) | 384 s | 361 s | 34 s | 166 s |
| `g6e.xlarge` (L40S, 46 GB) | 261 s | 361 s | 95 s | 149 s |

Both runs reported 361 s in their summaries, and the time from host ready to batch end was 367 s on the L4 and 366 s on the L40S. We have not checked why the summaries match.

Host preparation covers the data volume mount, the repo checkout, the GHCR login, the image pull, and seeding the
model volume. It runs once per instance. The launcher pins no VRAM budget; the session pool detects it from the GPU. An explicit `--vram-budget-mb` is passed through on any instance type. The translation time
differs by GPU in this single run; we have not looked into why.

Reusing a data-volume snapshot to skip host preparation is not available yet. `deploy/aws/run-batch.sh` formats
the unmounted data disk on every run (`mkfs.xfs -f`), which would erase a restored model cache. The launcher also
always creates a new data volume. Making this work needs launcher and bootstrap changes.

To record the image digest on an instance, after the pull, run, with `IMAGE` set to the same image you passed to the launcher:

```bash
docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE"
```

This command has not been run on an instance yet.

## Cost

`g6e.xlarge` is about $1.86/hour on demand in us-east-1, plus storage. Plan with the cold-start measurements above:
on `g6e.xlarge`, host ready took 261 s, and the 40-second clip batch ran 366 s from host ready to batch end. An earlier
first test reported about 4.5 minutes of host preparation and about 4 minutes per clip; the cold-start numbers
supersede it. Use a new `--run-id` for every batch.
