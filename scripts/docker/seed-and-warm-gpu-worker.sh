#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
  DOCKER_REPO_ROOT="$(cygpath -m "$REPO_ROOT")"
else
  DOCKER_REPO_ROOT="$REPO_ROOT"
fi
COMPOSE=(docker compose --project-directory "$DOCKER_REPO_ROOT" -f "$DOCKER_REPO_ROOT/docker-compose.yml")
# The image's ENV TRACKDUB_GPU_PROFILE marker is not visible from the host, so the profile
# follows the compose target. A compose override can select either image, so require it explicitly.
if [[ -n "${TRACKDUB_GPU_PROFILE:-}" ]]; then
  GPU_PROFILE="$TRACKDUB_GPU_PROFILE"
  if [[ -n "${TRACKDUB_GPU_TARGET:-}" && "$TRACKDUB_GPU_TARGET" != "$GPU_PROFILE" ]]; then
    echo "TRACKDUB_GPU_PROFILE=$GPU_PROFILE does not match TRACKDUB_GPU_TARGET=$TRACKDUB_GPU_TARGET." >&2
    exit 2
  fi
elif [[ -n "${TRACKDUB_COMPOSE_OVERRIDE:-}" ]]; then
  echo "Set TRACKDUB_GPU_PROFILE=rtx or server when TRACKDUB_COMPOSE_OVERRIDE is set." >&2
  exit 2
else
  GPU_PROFILE="${TRACKDUB_GPU_TARGET:-rtx}"
fi
case "$GPU_PROFILE" in
  rtx)
    DEFAULT_WARM_LIST="$REPO_ROOT/docker/poc-warm-models.txt"
    DEFAULT_MODEL_SET="$REPO_ROOT/docker/poc-models.txt"
    ;;
  server)
    DEFAULT_WARM_LIST="$REPO_ROOT/docker/poc-warm-models.server.txt"
    DEFAULT_MODEL_SET="$REPO_ROOT/docker/poc-server-models.txt"
    ;;
  *)
    echo "TRACKDUB_GPU_PROFILE must be 'rtx' or 'server': $GPU_PROFILE" >&2
    exit 2
    ;;
esac
export TRACKDUB_GPU_TARGET="$GPU_PROFILE"
echo "GPU profile: $GPU_PROFILE"
WARM_LIST="${TRACKDUB_WARM_MODEL_SET:-$DEFAULT_WARM_LIST}"
MODEL_SET="${TRACKDUB_MODEL_SET:-$DEFAULT_MODEL_SET}"
echo "Model set: $MODEL_SET"
if [[ -n "${TRACKDUB_COMPOSE_OVERRIDE:-}" ]]; then
  compose_override="$TRACKDUB_COMPOSE_OVERRIDE"
  if [[ "$compose_override" != /* && ! "$compose_override" =~ ^[A-Za-z]: ]]; then
    compose_override="$REPO_ROOT/$compose_override"
  fi
  if command -v cygpath >/dev/null 2>&1; then
    compose_override="$(cygpath -m "$compose_override")"
  fi
  COMPOSE+=(-f "$compose_override")
fi

if [[ "$GPU_PROFILE" == "rtx" ]]; then
  case "${TRACKDUB_ACCEPT_NVIDIA_TENSORRT_RTX_LICENSE:-}" in
    1|true|TRUE|yes|YES) ;;
    *)
      echo "Set TRACKDUB_ACCEPT_NVIDIA_TENSORRT_RTX_LICENSE=1 only after reviewing the NVIDIA TensorRT-RTX and CUDA license terms." >&2
      exit 2
      ;;
  esac
fi

if [[ ! -f "$WARM_LIST" ]]; then
  echo "Warm target list not found: $WARM_LIST" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Run this on the target NVIDIA GPU worker/AMI after installing its driver and NVIDIA Container Toolkit." >&2
  exit 2
fi
nvidia-smi -L

# Explicit downloads are pinned by the bundled manifest and survive in the Compose /data volume.
export TRACKDUB_MODEL_SET="$MODEL_SET"
"$SCRIPT_DIR/seed-model-volume.sh"

# cache warm otherwise recursively scans the whole model volume, which includes families such as
# OPUS-MT that can terminate the native host process during TensorRT compilation. Use the reviewed
# per-model allowlist, stored as paths relative to /data/models.
warm_args=()
while IFS= read -r model_path || [[ -n "$model_path" ]]; do
  model_path="${model_path%%#*}"
  model_path="$(printf '%s' "$model_path" | xargs)"
  [[ -z "$model_path" ]] && continue
  if [[ "$model_path" == /* || "$model_path" == *'..'* ]]; then
    echo "Warm target must be a relative path without '..': $model_path" >&2
    exit 2
  fi
  warm_args+=(--model "/data/models/$model_path")
done < "$WARM_LIST"
if [[ "${#warm_args[@]}" -eq 0 ]]; then
  echo "Warm target list is empty: $WARM_LIST" >&2
  exit 2
fi

if [[ "$GPU_PROFILE" == "rtx" ]]; then
  # Install the plugin into its dedicated persistent linml volume, then precompile only the reviewed
  # TRT-RTX-compatible POC graphs on THIS GPU/driver.
  "${COMPOSE[@]}" run --rm trackdub providers trt-rtx install </dev/null
  "${COMPOSE[@]}" run --rm trackdub cache warm "${warm_args[@]}" </dev/null
else
  # Server profile is strict classic TensorRT + CUDA: no TRT-RTX license, no
  # plugin install; warm classic-TRT engines for the server allowlist instead.
  "${COMPOSE[@]}" run --rm trackdub cache warm --provider tensorrt "${warm_args[@]}" </dev/null
fi

# WarmAsync returns a failure exit code if any target graph failed; skipped/unsupported models are
# reported explicitly. Reaching this line means pinned downloads and plugin registration succeeded,
# and the warm operation completed without failed targets.
echo "Model volume seeded and approved engine-cache targets warmed for this worker's current GPU/driver."
