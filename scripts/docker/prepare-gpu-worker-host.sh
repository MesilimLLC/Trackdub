#!/usr/bin/env bash
# Prepares one EC2 GPU host (G6/L4 or G6e/L40S) to run POC jobs with nothing left to download, so the host's
# AMI or the /mnt/trackdub-data EBS volume can be snapshotted afterwards. Runs steps 2-4 of
# docs/handoff/snapsoft-onboarding.md ("From a fresh EC2 GPU host to a first job") in order and stops at the
# first failure. Step 1 (driver, Docker, NVIDIA Container Toolkit, EBS mount) is checked here, not performed.
#
# Run from the repository checkout on the GPU host, at the commit the image tag names:
#   TRACKDUB_IMAGE=ghcr.io/mesilimllc/trackdub:sha-<commit>-server ./scripts/docker/prepare-gpu-worker-host.sh
#
# Optional:
#   TRACKDUB_WARM_ENGINES=1      also build classic TensorRT engines for docker/poc-warm-models.server.txt.
#                                The poc-server presets run every stage on CUDA, so jobs do not use these
#                                engines yet. Engines only load on the GPU model and driver that built them,
#                                so a warmed volume must not be reused on another instance family.
#   TRACKDUB_DATA_ROOT           host data mount (default /mnt/trackdub-data, as in docker-compose.aws.yml).
#   TRACKDUB_DOCTOR_TIMEOUT      seconds before trackdub doctor is stopped (default 600).
#   TRACKDUB_ALLOW_MUTABLE_IMAGE=1  accept a moving tag such as main-server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${TRACKDUB_DATA_ROOT:-/mnt/trackdub-data}"
export TRACKDUB_DATA_ROOT="$DATA_ROOT"
DOCTOR_TIMEOUT="${TRACKDUB_DOCTOR_TIMEOUT:-600}"
CONTAINER_UID=10001

export TRACKDUB_GPU_PROFILE=server
export TRACKDUB_GPU_TARGET=server
export TRACKDUB_COMPOSE_OVERRIDE=docker-compose.aws.yml
COMPOSE=(docker compose --project-directory "$REPO_ROOT" -f "$REPO_ROOT/docker-compose.yml" -f "$REPO_ROOT/docker-compose.aws.yml")
ES_MODELS="$REPO_ROOT/docker/poc-server-models.txt"
ZH_MODELS="$REPO_ROOT/docker/poc-server-zh-models.txt"

step() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

step "1/5 Host prerequisites"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found. Install the NVIDIA driver (CUDA 13.0, R580 or newer) first."
gpus="$(nvidia-smi -L)" || fail "nvidia-smi -L failed; the driver is not loaded."
[[ -n "$gpus" ]] || fail "nvidia-smi -L listed no GPU."
printf '%s\n' "$gpus"
command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 \
  || fail "Docker Engine with the Compose v2 plugin is required."
command -v python3 >/dev/null 2>&1 || fail "python3 is required to read the CLI's JSON output."
docker info --format '{{json .Runtimes}}' | grep -q nvidia \
  || fail "Docker has no nvidia runtime. Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
[[ -d "$DATA_ROOT" ]] || fail "$DATA_ROOT does not exist. Mount the dedicated EBS volume there first."
owner="$(stat -c '%u' "$DATA_ROOT")"
[[ "$owner" == "$CONTAINER_UID" ]] \
  || fail "$DATA_ROOT is owned by uid $owner, not $CONTAINER_UID. Run: sudo chown -R $CONTAINER_UID:$CONTAINER_UID $DATA_ROOT"
[[ -f "$ES_MODELS" && -f "$ZH_MODELS" ]] || fail "POC model lists not found under docker/. Run from a repository checkout."

step "2/5 Worker image"
image="${TRACKDUB_IMAGE:-}"
[[ -n "$image" ]] || fail "Set TRACKDUB_IMAGE to the server image, for example ghcr.io/mesilimllc/trackdub:sha-<commit>-server."
if [[ "$image" != *@sha256:* && "$image" != *:sha-*-server && "${TRACKDUB_ALLOW_MUTABLE_IMAGE:-}" != 1 ]]; then
  fail "TRACKDUB_IMAGE=$image is a moving tag. Pin sha-<commit>-server or a digest so the snapshot matches one build, or set TRACKDUB_ALLOW_MUTABLE_IMAGE=1."
fi
export TRACKDUB_IMAGE="$image"
mkdir -p "$REPO_ROOT/docker/input"
"${COMPOSE[@]}" pull trackdub </dev/null \
  || fail "Could not pull $image. If GHCR returns 401, log in with a token that has read:packages: docker login ghcr.io"

step "3/5 Seed models"
if [[ "${TRACKDUB_WARM_ENGINES:-}" == 1 ]]; then
  echo "Seeding the English->Spanish set and warming classic TensorRT engines for this GPU and driver."
  TRACKDUB_MODEL_SET="$ES_MODELS" bash "$SCRIPT_DIR/seed-and-warm-gpu-worker.sh"
else
  echo "Seeding the English->Spanish set (TensorRT warm skipped; set TRACKDUB_WARM_ENGINES=1 to build engines)."
  TRACKDUB_MODEL_SET="$ES_MODELS" bash "$SCRIPT_DIR/seed-model-volume.sh"
fi
echo "Seeding the English->Chinese set (adds Qwen3-TTS)."
TRACKDUB_MODEL_SET="$ZH_MODELS" bash "$SCRIPT_DIR/seed-model-volume.sh"
mkdir -p "$DATA_ROOT/jobs" 2>/dev/null || true

step "4/5 Health checks"
"${COMPOSE[@]}" run --rm --entrypoint sh trackdub -c 'ls /proc/driver/nvidia/gpus' </dev/null \
  || fail "The container cannot see the GPU."
providers="$("${COMPOSE[@]}" run --rm trackdub providers list </dev/null)" || fail "trackdub providers list failed."
printf '%s\n' "$providers"
python3 -c '
import json, sys
providers = json.load(sys.stdin)["providers"]
sys.exit(0 if any(p.get("kind") == "Cuda" and p.get("available") for p in providers) else 1)
' <<<"$providers" || fail "CUDA is not reported as available in the container."

doctor_exit=0
doctor_json="$(timeout "$DOCTOR_TIMEOUT" "${COMPOSE[@]}" run --rm trackdub doctor </dev/null)" || doctor_exit=$?
printf '%s\n' "$doctor_json"
[[ "$doctor_exit" -ne 124 ]] || fail "trackdub doctor did not finish within ${DOCTOR_TIMEOUT}s."
# A nonzero exit can come from docker compose (no image, network) as well as from doctor itself, so tell them
# apart by doctor's own JSON verdict.
doctor_status="$(python3 -c 'import json, sys; print(json.loads(sys.stdin.read()).get("status", ""))' \
  <<<"$doctor_json" 2>/dev/null || true)"
[[ "$doctor_status" != fail ]] || fail "trackdub doctor reported a failing check (exit $doctor_exit); see the fail rows above."
[[ "$doctor_exit" -eq 0 ]] || fail "trackdub doctor did not run to completion (exit $doctor_exit); check the docker compose output above."

step "5/5 Verify seeded models"
verify_failed=0
while IFS= read -r model; do
  echo "Verifying $model"
  "${COMPOSE[@]}" run --rm trackdub models verify --model-id "$model" </dev/null || verify_failed=1
done < <(cat "$ES_MODELS" "$ZH_MODELS" | sed 's/#.*//' | cut -d'|' -f1 | xargs -n1 | sort -u)
[[ "$verify_failed" -eq 0 ]] || fail "One or more POC models failed hash verification."

printf '\nREADY: %s\n' "$image"
printf 'GPU: %s\n' "$(head -n1 <<<"$gpus")"
if [[ "${TRACKDUB_WARM_ENGINES:-}" == 1 ]]; then
  echo "Engines: warmed for this GPU and driver only; snapshot per instance family."
else
  echo "Engines: not warmed; the seeded volume is not tied to this GPU model."
fi
echo "Snapshot the AMI or the $DATA_ROOT EBS volume now. Re-run after any image, model list, driver or TensorRT change."
