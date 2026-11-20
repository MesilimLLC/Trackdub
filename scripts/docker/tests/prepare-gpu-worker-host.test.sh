#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
cleanup() { rm -rf -- "$TEST_ROOT"; }
trap cleanup EXIT

# Run the real host preparation script with inert docker, nvidia-smi and stat. No Docker daemon or GPU is used.
mkdir -p "$TEST_ROOT/bin" "$TEST_ROOT/data"
cat > "$TEST_ROOT/bin/docker" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_LOG"
case "$*" in
  'info --format {{json .Runtimes}}') printf '%s\n' "${FAKE_RUNTIMES:-{\"nvidia\":{},\"runc\":{}}}" ;;
  *' providers list') printf '{"providers":[{"kind":"Cpu","available":true},{"kind":"Cuda","available":%s}]}\n' "${FAKE_CUDA:-true}" ;;
  *' doctor') printf '{"status":"warn","checks":[]}\n'; exit "${FAKE_DOCTOR_EXIT:-0}" ;;
esac
exit 0
MOCK
printf '#!/usr/bin/env bash\necho "GPU 0: NVIDIA L4 (UUID: GPU-test)"\n' > "$TEST_ROOT/bin/nvidia-smi"
printf '#!/usr/bin/env bash\necho "${FAKE_DATA_UID:-10001}"\n' > "$TEST_ROOT/bin/stat"
chmod +x "$TEST_ROOT/bin/"*
export PATH="$TEST_ROOT/bin:$PATH"
export TEST_LOG="$TEST_ROOT/calls"
export TRACKDUB_DATA_ROOT="$TEST_ROOT/data"
export TRACKDUB_IMAGE=ghcr.io/mesilimllc/trackdub:sha-0123456789abcdef-server
unset TRACKDUB_WARM_ENGINES TRACKDUB_MODEL_SET TRACKDUB_WARM_MODEL_SET TRACKDUB_ALLOW_MUTABLE_IMAGE

run() {
  : > "$TEST_LOG"
  local status=0
  bash "$REPO_ROOT/scripts/docker/prepare-gpu-worker-host.sh" > "$TEST_ROOT/output" 2>&1 || status=$?
  return "$status"
}

expect_failure() {
  local name="$1" message="$2"
  if run; then
    echo "Expected failure: $name" >&2; cat "$TEST_ROOT/output" >&2; exit 1
  fi
  grep -qF -- "$message" "$TEST_ROOT/output" || { echo "Missing message for $name: $message" >&2; cat "$TEST_ROOT/output" >&2; exit 1; }
}

unique_models() {
  cat "$REPO_ROOT/docker/poc-server-models.txt" "$REPO_ROOT/docker/poc-server-zh-models.txt" \
    | sed 's/#.*//' | cut -d'|' -f1 | xargs -n1 | sort -u
}

# Default run: pull, seed both language sets, no TensorRT warm, health checks, verify every POC model once.
run || { cat "$TEST_ROOT/output" >&2; exit 1; }
grep -q 'docker-compose.aws.yml pull trackdub$' "$TEST_LOG"
grep -q 'models download google/madlad400-3b-mt --variant quantized$' "$TEST_LOG"
grep -q 'models download tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX$' "$TEST_LOG"
! grep -q 'cache warm' "$TEST_LOG"
grep -q ' doctor$' "$TEST_LOG"
test "$(grep -c 'models verify --model-id' "$TEST_LOG")" -eq "$(unique_models | wc -l)"
while IFS= read -r model; do
  grep -qF -- "models verify --model-id $model" "$TEST_LOG"
done < <(unique_models)
grep -q '^READY: ghcr.io/mesilimllc/trackdub:sha-0123456789abcdef-server$' "$TEST_ROOT/output"
grep -q 'Engines: not warmed' "$TEST_ROOT/output"
test -d "$REPO_ROOT/docker/input"

# Opt-in warm builds classic TensorRT engines from the reviewed server allow-list only.
TRACKDUB_WARM_ENGINES=1 run || { cat "$TEST_ROOT/output" >&2; exit 1; }
grep -q 'cache warm --provider tensorrt --model /data/models/' "$TEST_LOG"
! grep -q 'providers trt-rtx install' "$TEST_LOG"
grep -q 'Engines: warmed for this GPU and driver only' "$TEST_ROOT/output"

# Guard rails stop before anything is pulled or downloaded.
TRACKDUB_IMAGE=ghcr.io/mesilimllc/trackdub:main-server expect_failure 'moving tag' 'is a moving tag'
! grep -q ' pull ' "$TEST_LOG"
TRACKDUB_IMAGE='' expect_failure 'missing image' 'Set TRACKDUB_IMAGE'
FAKE_DATA_UID=0 expect_failure 'data root ownership' 'not 10001'
! grep -q ' pull ' "$TEST_LOG"
FAKE_RUNTIMES='{"runc":{}}' expect_failure 'nvidia runtime' 'no nvidia runtime'
TRACKDUB_DATA_ROOT="$TEST_ROOT/missing" expect_failure 'missing data root' 'does not exist'

# Health check failures stop before model verification.
FAKE_CUDA=false expect_failure 'cuda unavailable' 'CUDA is not reported as available'
FAKE_DOCTOR_EXIT=3 expect_failure 'doctor fail' 'trackdub doctor reported a failing check (exit 3)'
! grep -q 'models verify' "$TEST_LOG"

TRACKDUB_IMAGE=ghcr.io/mesilimllc/trackdub:main-server TRACKDUB_ALLOW_MUTABLE_IMAGE=1 run \
  || { cat "$TEST_ROOT/output" >&2; exit 1; }

echo "prepare-gpu-worker-host tests passed"
