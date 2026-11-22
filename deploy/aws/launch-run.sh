#!/usr/bin/env bash
# Launches one GPU instance that runs a batch of clips through the published server image.
#
# Prerequisites: deploy/aws/deploy-stack.sh has created the stack, and the GHCR token is in SSM.
# The instance terminates itself when the batch finishes or after --max-minutes.
#
#   deploy/aws/launch-run.sh --image ghcr.io/mesilimllc/trackdub@sha256:<digest> \
#     --ghcr-user <github-user> --clips-dir ./clips --target-language zh --preset poc-server-zh-v1 \
#     --voice SPEAKER_00:qwen3:vivian --pipeline-args "--tts-rubberband-stretch"
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: launch-run.sh --image REF --ghcr-user USER (--clips-dir DIR | --clip FILE [--clip FILE ...]) [options]

Required:
  --image REF            Pinned server image: @sha256 digest, :sha-<commit>-server or :container-vX.Y.Z-server.
  --ghcr-user USER       GitHub user the GHCR token belongs to.
  --clips-dir DIR        Upload every .mp4 in DIR.
  --clip FILE            Upload one .mp4 (repeatable).

Options:
  --stack NAME           CloudFormation stack from deploy-stack.sh (default: snapsoft-poc).
  --region REGION        Default: us-east-1.
  --run-id ID            Default: UTC timestamp.
  --instance-type TYPE   g6.xlarge, g6.2xlarge, g6e.xlarge or g6e.2xlarge (default: g6e.xlarge).
  --source-language CODE Default: en.
  --target-language CODE Default: zh.
  --preset NAME          Default: poc-server-zh-v1 (use poc-server-es-qwen-v1 for Spanish).
  --voice SPEAKER:VOICE  Default: none; the pipeline picks a Qwen3 preset per speaker by estimated gender.
  --pipeline-args "ARGS" Extra arguments for run pipeline, such as "--tts-rubberband-stretch".
  --repo-commit SHA      Commit whose scripts the instance uses. Default: the commit in the image tag, or HEAD.
  --ami ID               Override the Deep Learning Base GPU AMI lookup.
  --max-minutes N        Hard cost cap for the instance (default: 180).
  --vram-budget-mb N     Session pool VRAM admission budget for the container. Default: none, so the
                         pool uses the budget it detects from the GPU.
  --dry-run              Render the configuration and print the plan; upload and launch nothing.
USAGE
}

STACK=snapsoft-poc
REGION=us-east-1
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
INSTANCE_TYPE=g6e.xlarge
SOURCE_LANGUAGE=en
TARGET_LANG=zh
PRESET=poc-server-zh-v1
VOICE=
PIPELINE_ARGS=""
REPO_COMMIT=""
AMI=""
MAX_MINUTES=180
VRAM_BUDGET_MB=""
IMAGE=""
GHCR_USER=""
CLIPS=()
DRY_RUN=0
GHCR_TOKEN_PARAM=""
DLAMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
RUN_PREFIX=runs

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack) STACK="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --ghcr-user) GHCR_USER="$2"; shift 2 ;;
    --clips-dir)
      while IFS= read -r -d '' f; do CLIPS+=("$f"); done < <(find "$2" -maxdepth 1 -type f -iname '*.mp4' -print0 | sort -z)
      shift 2 ;;
    --clip) CLIPS+=("$2"); shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --source-language) SOURCE_LANGUAGE="$2"; shift 2 ;;
    --target-language) TARGET_LANG="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --voice) VOICE="$2"; shift 2 ;;
    --pipeline-args) PIPELINE_ARGS="$2"; shift 2 ;;
    --repo-commit) REPO_COMMIT="$2"; shift 2 ;;
    --ami) AMI="$2"; shift 2 ;;
    --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
    --vram-budget-mb) VRAM_BUDGET_MB="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

fail() { echo "error: $*" >&2; exit 1; }

[[ -n "$IMAGE" ]] || fail "--image is required."
[[ -n "$GHCR_USER" ]] || fail "--ghcr-user is required."
[[ "${#CLIPS[@]}" -gt 0 ]] || fail "Give --clips-dir or at least one --clip."
declare -A SEEN_NAMES=()
for clip in "${CLIPS[@]}"; do
  [[ -f "$clip" ]] || fail "Clip not found: $clip"
  name="$(basename "$clip")"
  [[ -z "${SEEN_NAMES[$name]:-}" ]] || fail "Two clips share the name $name; they would overwrite each other in S3. Rename one."
  SEEN_NAMES[$name]=1
done
case "$INSTANCE_TYPE" in
  g6.xlarge|g6.2xlarge|g6e.xlarge|g6e.2xlarge) ;;
  *) fail "Instance type must be g6.* or g6e.* (the POC permission set blocks other types): $INSTANCE_TYPE" ;;
esac
[[ -z "$VRAM_BUDGET_MB" || "$VRAM_BUDGET_MB" =~ ^[1-9][0-9]*$ ]] || fail "--vram-budget-mb must be a positive integer: $VRAM_BUDGET_MB"
if [[ "$IMAGE" != *@sha256:* && "$IMAGE" != *:sha-*-server && "$IMAGE" != *:container-v*-server ]]; then
  fail "Image must be pinned (@sha256 digest, :sha-<commit>-server or :container-vX.Y.Z-server): $IMAGE"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--run-id may contain only letters, digits, dot, underscore and dash."

if [[ -z "$REPO_COMMIT" ]]; then
  if [[ "$IMAGE" =~ :sha-([0-9a-f]{40})- ]]; then
    REPO_COMMIT="${BASH_REMATCH[1]}"
  else
    fail "Pass --repo-commit: the image is not tagged sha-<commit>-server, so the commit cannot be read from it."
  fi
fi

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

BUCKET="$(stack_output BucketName)"
SUBNETS="$(stack_output SubnetIds)"
SECURITY_GROUP="$(stack_output SecurityGroupId)"
PROFILE="$(stack_output InstanceProfileName)"
GHCR_TOKEN_PARAM="$(stack_output TokenParameterName)"
[[ -n "$BUCKET" && "$BUCKET" != None ]] || fail "Stack $STACK has no outputs. Run deploy/aws/deploy-stack.sh first."

if [[ -z "$AMI" ]]; then
  AMI="$(aws ssm get-parameter --name "$DLAMI_PARAM" --region "$REGION" --query Parameter.Value --output text)"
fi
ROOT_DEVICE="$(aws ec2 describe-images --image-ids "$AMI" --region "$REGION" --query 'Images[0].RootDeviceName' --output text)"

RUN_DIR="$RUN_PREFIX/$RUN_ID"
DEST="s3://$BUCKET/$RUN_DIR"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
USER_DATA="$WORK/user-data.sh"

{
  echo '#!/bin/bash'
  echo "# Generated by launch-run.sh for run $RUN_ID. Configuration below, then the batch body."
  printf 'export RUN_BUCKET=%q RUN_PREFIX=%q RUN_ID=%q AWS_REGION=%q\n' "$BUCKET" "$RUN_PREFIX" "$RUN_ID" "$REGION"
  printf 'export IMAGE=%q REPO_COMMIT=%q GHCR_USER=%q GHCR_TOKEN_PARAM=%q\n' "$IMAGE" "$REPO_COMMIT" "$GHCR_USER" "$GHCR_TOKEN_PARAM"
  printf 'export SOURCE_LANGUAGE=%q TARGET_LANG=%q PRESET=%q VOICE=%q\n' "$SOURCE_LANGUAGE" "$TARGET_LANG" "$PRESET" "$VOICE"
  printf 'export EXTRA_PIPELINE_ARGS=%q MAX_MINUTES=%q SESSION_VRAM_BUDGET_MB=%q\n' "$PIPELINE_ARGS" "$MAX_MINUTES" "$VRAM_BUDGET_MB"
  sed '1d' "$SCRIPT_DIR/run-batch.sh"
} > "$USER_DATA"

USER_DATA_BYTES="$(wc -c < "$USER_DATA")"
[[ "$USER_DATA_BYTES" -lt 16000 ]] || fail "Rendered user data is $USER_DATA_BYTES bytes; EC2 allows 16384."

echo "Stack:       $STACK ($REGION)"
echo "Bucket:      s3://$BUCKET/$RUN_DIR/"
echo "Image:       $IMAGE"
echo "Repo commit: $REPO_COMMIT"
echo "Instance:    $INSTANCE_TYPE from AMI $AMI"
echo "Clips:       ${#CLIPS[@]}"
echo "Job:         $SOURCE_LANGUAGE -> $TARGET_LANG, preset $PRESET, voice ${VOICE:-auto}"
[[ -n "$PIPELINE_ARGS" ]] && echo "Pipeline:    $PIPELINE_ARGS"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: nothing uploaded or launched. User data rendered to $USER_DATA_BYTES bytes."
  exit 0
fi

for clip in "${CLIPS[@]}"; do
  aws s3 cp "$clip" "$DEST/input/$(basename "$clip")" --region "$REGION" >/dev/null
done
aws s3 cp "$SCRIPT_DIR/summarize_result.py" "$DEST/tools/summarize_result.py" --region "$REGION" >/dev/null
echo "Uploaded ${#CLIPS[@]} clip(s) and tools."

IFS=',' read -r -a SUBNET_LIST <<< "$SUBNETS"
LAUNCHED=""
for subnet in "${SUBNET_LIST[@]}"; do
  set +e
  result="$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --subnet-id "$subnet" --security-group-ids "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$PROFILE" \
    --user-data "file://$USER_DATA" \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled" \
    --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEVICE\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}},{\"DeviceName\":\"/dev/sdf\",\"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=trackdub-poc-e2e},{Key=Name,Value=trackdub-poc-$RUN_ID}]" "ResourceType=volume,Tags=[{Key=Project,Value=trackdub-poc-e2e},{Key=Name,Value=trackdub-poc-$RUN_ID}]" \
    --query 'Instances[0].InstanceId' --output text 2>"$WORK/err")"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    LAUNCHED="$result"
    echo "Launched $LAUNCHED in $subnet."
    break
  fi
  echo "Subnet $subnet: $(tr '\n' ' ' < "$WORK/err" | cut -c1-200)" >&2
done

[[ -n "$LAUNCHED" ]] || fail "No subnet had capacity for $INSTANCE_TYPE. Try another instance type or run again later."

cat <<EOF

Watch progress:  s3://$BUCKET/$RUN_DIR/summary.txt   (one line per clip, written after each clip)
Results:         aws s3 sync s3://$BUCKET/$RUN_DIR/ ./results/$RUN_ID/
The instance terminates itself when the batch finishes or after $MAX_MINUTES minutes.
EOF
