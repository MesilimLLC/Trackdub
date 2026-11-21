# SnapSoft quickstart: run a batch of clips in your AWS account

This runs the published `server` image on one GPU instance in your AWS account and processes a batch of clips.
Each clip gets its own result line, and a failed clip does not stop the rest. The instance powers itself off
when the batch finishes or after 180 minutes, whichever is first.

For the background, SOW criteria and the manual steps behind each script, see
[snapsoft-onboarding.md](snapsoft-onboarding.md).

## Before you start

- **Permissions:** the `SnapSoft-POC-Engineer` permission set, in `us-east-1`. It can deploy the stack, launch
  `g6` and `g6e` instances, and create IAM roles named `trackdub-poc-*`.
- **Tools:** AWS CLI v2 signed in to the account, Python 3, and a checkout of this repository at the commit
  the image was built from.
- **Image:** a pinned server image. Use a digest (`ghcr.io/mesilimllc/trackdub@sha256:…`) or a
  `sha-<commit>-server` tag. Do not use `main-server`: it moves.
- **Clips:** MP4, up to 600 s, up to 1920×1080, up to 1 GiB each.

## 1. Create a GitHub token for the image pull

The image is private. Create a classic personal access token with only the `read:packages` scope:

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token.
2. Select only `read:packages`, and set a short expiry.
3. If GitHub offers **Configure SSO**, authorize it for MesilimLLC.

Each engineer uses their own token, so access can be revoked per person. Fine-grained tokens do not cover
packages yet.

## 2. Store the token in SSM Parameter Store

Run this in the repository checkout. The token is read without echoing it to the terminal:

```bash
read -rs GHCR_TOKEN && echo
aws ssm put-parameter --region us-east-1 --name /trackdub/ghcr-read-token \
  --type SecureString --value "$GHCR_TOKEN" --overwrite
unset GHCR_TOKEN
```

The parameter is encrypted with the account's default SSM key. The instance reads it at boot; it is never
written to user data or to the repository.

## 3. Deploy the network, bucket and instance role

```bash
deploy/aws/deploy-stack.sh --stack snapsoft-poc
```

This creates a VPC with three public subnets, a security group with no inbound rules, a private S3 bucket
(outputs under `runs/` expire after 30 days), and the instance role. It creates no compute, so it costs nothing
while idle. Re-running it is safe. If the default bucket name is taken, add `--bucket-name <name>`.

## 4. Launch a batch

Put the clips in one folder, then:

```bash
deploy/aws/launch-run.sh \
  --stack snapsoft-poc \
  --image "ghcr.io/mesilimllc/trackdub@sha256:<digest>" \
  --repo-commit <full-commit-sha> \
  --ghcr-user <your-github-user> \
  --clips-dir ./clips \
  --run-id 2026-11-26-batch1
```

The defaults are English→Mandarin (`--target-language zh --preset poc-server-zh-v1 --voice SPEAKER_00:qwen3:vivian`).
For Spanish, add `--target-language es --preset poc-server-v1 --voice SPEAKER_00:ef_dora`.

To add a fix for a known failure, pass it with `--pipeline-args`, for example
`--pipeline-args "--tts-rubberband-stretch --tts-auto-stretch-max-overrun 0.35"`.

Add `--dry-run` to check the plan and the rendered configuration without uploading or launching anything.

The launcher uploads the clips to `s3://<bucket>/runs/<run-id>/input/`, then tries each subnet until one has
capacity. If none does, pick another instance type with `--instance-type g6e.2xlarge` or run again later.

## 5. Check the results

The batch writes one line per clip to `summary.txt`, after each clip finishes:

```bash
BUCKET=$(aws cloudformation describe-stacks --stack-name snapsoft-poc --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)
aws s3 cp "s3://$BUCKET/runs/2026-11-26-batch1/summary.txt" -
```

Each line is `clip  ok|failed  duration  verdict`. A failure names the stage and the reason, with a suggested fix.
This is an example of a failed line, shortened:

```
clip_03  failed  412s  FAILED at export: Dubbed take extends 6.826 s past the end of the source audio (40.171 s)…
                       Fix: TTS overran the source; rerun with --tts-rubberband-stretch and a higher
                       --tts-auto-stretch-max-overrun (e.g. 0.35), or shorten the translation (overrun 6.826 s)
```

Everything else is under `runs/<run-id>/`: `jobs/<clip>/project/` (artifacts and the database), `jobs/<clip>/result.json`,
`run.log` and `timings.json`. Download it all:

```bash
aws s3 sync "s3://$BUCKET/runs/2026-11-26-batch1/" ./results/2026-11-26-batch1/
```

## Teardown

The instance terminates itself. There is nothing else to stop.

To remove the network, role and bucket when the POC is done, download what you need first, then:

```bash
aws cloudformation delete-stack --stack-name snapsoft-poc --region us-east-1
aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET" --region us-east-1
```

The bucket is retained when the stack is deleted, so results are not lost by accident. Empty it and delete it
with the last two commands when you no longer need the results. Also delete the SSM parameter when the POC ends.

## Notes

- Cost: about $1.86 per hour for `g6e.xlarge` in us-east-1 (on-demand), plus storage. In the first test, host
  preparation took about 4.5 minutes, and a 40-second clip took about 4 minutes to process end to end.
- Do not reuse a `runs/` prefix. Use a new `--run-id` for each batch.
- Pre-seeding a data volume snapshot to cut host preparation is not built yet.
