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
`--max-minutes`, `--vram-budget-mb` (session pool VRAM admission budget; defaults to 30000 on `g6.*`, see Known
failures), `--ami`, `--dry-run` (renders and prints the plan, uploads and launches nothing).

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
| `FAILED at export: Dubbed take extends … s past the end` | TTS overran the source; the exporter won't truncate | `--pipeline-args "--tts-rubberband-stretch --tts-auto-stretch-max-overrun 0.35"` (default limit 0.20), or shorten the translation for that clip |
| `FAILED at input: could not download` | S3 read failed | Check the object exists under `input/`, rerun the launch for that clip |
| `UPLOAD FAILED` | Results could not reach S3 after 3 tries | Results are only on the instance; the data volume is deleted at termination. Open an incident before relaunching |
| `NO RESULT` | The pipeline exited before writing `result.json` | Read `run.log` and the clip's `progress.jsonl` |
| `STALLED in stage 'tts'` on `g6.*` with GPU at 0% | Session pool admission budget (85% of the L4's 23 GB) is below the Qwen3-TTS reservation estimate; the wait never ends | The launcher sets `--vram-budget-mb 30000` on `g6.*` by default. If you overrode it, raise it again |
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

## Cost

`g6e.xlarge` is about $1.86/hour on demand in us-east-1, plus storage. In the first test, host preparation took
about 4.5 minutes, and a 40-second clip took about 4 minutes to process. Use a new `--run-id` for every batch.
