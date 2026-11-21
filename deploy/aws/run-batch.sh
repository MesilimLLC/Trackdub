#!/bin/bash
# Runs as root on a GPU instance (EC2 user data). The launcher writes the configuration above this
# body as exported variables, uploads this script's helper to s3://$RUN_BUCKET/$RUN_PREFIX/$RUN_ID/tools/,
# and the instance copies its input clips from s3://$RUN_BUCKET/$RUN_PREFIX/$RUN_ID/input/.
#
# Every clip gets its own job directory, result and one summary line. A failed clip does not stop the
# batch. Host setup failures do, because nothing can run without them. Outputs upload after every
# clip, and the instance powers off when the batch ends or after MAX_MINUTES, whichever is first.
set -euo pipefail
# The cost cap must be set before anything else can fail, so it has a default.
: "${MAX_MINUTES:=180}"

WORK=/opt/trackdub-e2e
REPO_DIR="$WORK/Trackdub"
JOBS_ROOT=/mnt/trackdub-data/jobs
DEST="s3://$RUN_BUCKET/$RUN_PREFIX/$RUN_ID"
SUMMARY="$WORK/summary.txt"
mkdir -p "$WORK"
LOG="$WORK/run.log"
exec > >(tee -a "$LOG") 2>&1

shutdown -h "+$MAX_MINUTES" "trackdub poc batch cost cap"

declare -A T
mark() { T[$1]=$(date +%s); echo "== [$(date -u +%H:%M:%S)] $1"; }

finish() {
  local status=$?
  mark end
  {
    echo "{"
    echo "  \"exit_status\": $status,"
    for k in "${!T[@]}"; do echo "  \"t_$k\": ${T[$k]},"; done
    echo "  \"instance_type\": \"$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type)\","
    echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)\""
    echo "}"
  } > "$WORK/timings.json" || true
  aws s3 cp "$WORK/timings.json" "$DEST/timings.json" --region "$AWS_REGION" || true
  aws s3 cp "$LOG" "$DEST/run.log" --region "$AWS_REGION" || true
  [ -f "$SUMMARY" ] && aws s3 cp "$SUMMARY" "$DEST/summary.txt" --region "$AWS_REGION" || true
  shutdown -h now "trackdub poc batch finished (status $status)"
}
trap finish EXIT

mark boot
nvidia-smi -L

# The data volume is the one unmounted EBS disk attached besides the root disk.
DEV=""
for d in $(lsblk -dpno NAME,MODEL | awk '/Elastic Block Store/ {print $1}'); do
  if ! lsblk -no MOUNTPOINT "$d" | grep -q .; then DEV="$d"; fi
done
[ -n "$DEV" ] || { echo "No unmounted EBS data volume found"; exit 1; }
mkfs.xfs -f "$DEV"
mkdir -p /mnt/trackdub-data
mount "$DEV" /mnt/trackdub-data
mkdir -p "$JOBS_ROOT"
chown -R 10001:10001 /mnt/trackdub-data
mark data_volume_ready

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker compose version >/dev/null 2>&1 || { apt-get update -y && apt-get install -y docker-compose-plugin; }

aws ssm get-parameter --name "$GHCR_TOKEN_PARAM" --with-decryption --region "$AWS_REGION" \
  --query Parameter.Value --output text | docker login ghcr.io -u "$GHCR_USER" --password-stdin

git init -q "$REPO_DIR"
cd "$REPO_DIR"
git remote add origin https://github.com/MesilimLLC/Trackdub.git
git fetch -q --depth 1 origin "$REPO_COMMIT"
git checkout -q --detach FETCH_HEAD
mark repo_ready

TRACKDUB_IMAGE="$IMAGE" ./scripts/docker/prepare-gpu-worker-host.sh
mark host_ready

aws s3 cp "$DEST/tools/summarize_result.py" "$WORK/summarize_result.py" --region "$AWS_REGION"

mkdir -p docker/input
export TRACKDUB_IMAGE="$IMAGE"
dc() { docker compose -f docker-compose.yml -f docker-compose.aws.yml "$@"; }

# Clip names become job directory names, so keep only safe characters.
mapfile -t KEYS < <(aws s3api list-objects-v2 --bucket "$RUN_BUCKET" --prefix "$RUN_PREFIX/$RUN_ID/input/" \
  --query 'Contents[].Key' --output text --region "$AWS_REGION" | tr '\t' '\n' | grep -i '\.mp4$' | sort || true)
[ "${#KEYS[@]}" -gt 0 ] || { echo "No .mp4 clips found under input/"; exit 1; }
echo "Clips: ${#KEYS[@]}"

run_clip() {
  local key="$1"
  local stem name job start code verdict
  stem="$(basename "$key" .mp4)"
  name="$(printf '%s' "$stem" | tr -c 'A-Za-z0-9_-' '_')"
  # Two clip names can sanitize to the same string; give the later one its own job directory.
  local base_name="$name" suffix=2
  while [[ -e "$JOBS_ROOT/$name" ]]; do
    name="${base_name}_${suffix}"
    suffix=$((suffix + 1))
  done
  job="/data/jobs/$name"
  start=$(date +%s)
  install -d -o 10001 -g 10001 "$JOBS_ROOT/$name"

  # Remove the previous clip's input first: a failed download must not let the pipeline run on it.
  rm -f docker/input/in.mp4
  if ! aws s3 cp "s3://$RUN_BUCKET/$key" docker/input/in.mp4 --region "$AWS_REGION"; then
    printf '%s\tfailed\t0s\tFAILED at input: could not download %s. Fix: check the upload and rerun this clip.\n' \
      "$name" "$key" | tee -a "$SUMMARY"
    return 1
  fi
  chmod 0644 docker/input/in.mp4

  echo "== [$(date -u +%H:%M:%S)] job $name"
  set +e
  # shellcheck disable=SC2086  # EXTRA_PIPELINE_ARGS is a deliberate word list from the launcher.
  dc run --rm -T trackdub run pipeline --media /work/input/in.mp4 --output "$job/project" \
    --source-language "$SOURCE_LANGUAGE" --target-language "$TARGET_LANG" \
    --preset "$PRESET" --voice "$VOICE" $EXTRA_PIPELINE_ARGS \
    --subtitle-format srt --result-file "$job/result.json" --progress-file "$job/progress.jsonl" </dev/null
  code=$?
  set -e

  verdict="$(python3 "$WORK/summarize_result.py" "$JOBS_ROOT/$name/result.json" --exit-code "$code" 2>&1 || true)"
  printf '%s\t%s\t%ss\t%s\n' "$name" "$([ "$code" -eq 0 ] && echo ok || echo failed)" \
    "$(( $(date +%s) - start ))" "$verdict" | tee -a "$SUMMARY"

  # The data volume is deleted when the instance terminates, so an upload that never succeeds loses the
  # clip. Retry, and count a failed upload as a failed clip.
  if ! upload_with_retry aws s3 sync "$JOBS_ROOT/$name" "$DEST/jobs/$name" --region "$AWS_REGION"; then
    printf '%s\tfailed\t0s\tUPLOAD FAILED: results for this clip are only on the instance.\n' "$name" | tee -a "$SUMMARY"
    code=1
  fi
  upload_with_retry aws s3 cp "$SUMMARY" "$DEST/summary.txt" --region "$AWS_REGION" || true
  return "$code"
}

upload_with_retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    sleep $((attempt * 10))
  done
  return 1
}

failures=0
for key in "${KEYS[@]}"; do
  run_clip "$key" || failures=$((failures + 1))
done

mark batch_end
echo "Batch finished: ${#KEYS[@]} clip(s), $failures failed."
[ "$failures" -eq 0 ]
