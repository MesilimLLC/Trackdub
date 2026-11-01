#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -f -- "$TEST_ROOT/docker" "$TEST_ROOT/nvidia-smi" "$TEST_ROOT/models.txt" "$TEST_ROOT/warm.txt" "$TEST_ROOT/calls" "$TEST_ROOT/output"; rmdir -- "$TEST_ROOT"' EXIT

# Run the real provisioning scripts with inert commands. No Docker daemon or GPU is used.
cat > "$TEST_ROOT/docker" <<'MOCK'
#!/usr/bin/env bash
printf '%s|%s\n' "${TRACKDUB_GPU_TARGET:-unset}" "$*" >> "$TEST_LOG"
MOCK
printf '#!/usr/bin/env bash\nexit 0\n' > "$TEST_ROOT/nvidia-smi"
chmod +x "$TEST_ROOT/docker" "$TEST_ROOT/nvidia-smi"
printf 'fixture-model\n' > "$TEST_ROOT/models.txt"
printf 'fixture.onnx\n' > "$TEST_ROOT/warm.txt"
export PATH="$TEST_ROOT:$PATH"
export TEST_LOG="$TEST_ROOT/calls"
export TRACKDUB_MODEL_SET="$TEST_ROOT/models.txt"
export TRACKDUB_WARM_MODEL_SET="$TEST_ROOT/warm.txt"
export TRACKDUB_ACCEPT_NVIDIA_TENSORRT_RTX_LICENSE=1
unset TRACKDUB_COMPOSE_OVERRIDE

run_case() {
  local profile="$1" target="$2" expected="$3"
  : > "$TEST_LOG"
  TRACKDUB_GPU_PROFILE="$profile" TRACKDUB_GPU_TARGET="$target" \
    bash "$REPO_ROOT/scripts/docker/seed-and-warm-gpu-worker.sh" > "$TEST_ROOT/output" 2>&1
  test -s "$TEST_LOG"
  if grep -v "^$expected|" "$TEST_LOG"; then
    echo "Compose target did not follow selected profile: $profile / $target" >&2
    exit 1
  fi
  grep -q 'models download fixture-model' "$TEST_LOG"
  if [[ "$expected" == "server" ]]; then
    grep -q 'cache warm --provider tensorrt' "$TEST_LOG"
    ! grep -q 'providers trt-rtx install' "$TEST_LOG"
  else
    grep -q 'providers trt-rtx install' "$TEST_LOG"
  fi
}

run_default_seed_case() {
  local profile="$1"
  : > "$TEST_LOG"
  env -u TRACKDUB_MODEL_SET -u TRACKDUB_WARM_MODEL_SET \
    TRACKDUB_GPU_PROFILE="$profile" TRACKDUB_GPU_TARGET="$profile" \
    bash "$REPO_ROOT/scripts/docker/seed-and-warm-gpu-worker.sh" > "$TEST_ROOT/output" 2>&1
  test -s "$TEST_LOG"
}

expect_download() {
  local escaped
  escaped="$(printf '%s' "$1" | sed 's/[][\\.*^$+?(){}|]/\\&/g')"
  grep -qE -- "${escaped}\$" "$TEST_LOG"
}

run_case server '' server
run_case '' server server
run_case rtx rtx rtx
run_case '' '' rtx

# Without an explicit TRACKDUB_MODEL_SET, each profile seeds its own pinned list.
# Assert the full server list from literals, not from the manifest the seed step reads.
run_default_seed_case server
expected_server_downloads=(
  'models download onnx-community/silero-vad'
  'models download Rikorose/DeepFilterNet3'
  'models download cgus/diar_streaming_sortformer_4spk-v2.1-onnx'
  'models download tonythethompson/parakeet-tdt-0.6b-v3-onnx'
  'models download google/madlad400-3b-mt --variant quantized'
  'models download csukuangfj/sherpa-onnx-spleeter-2stems'
  'models download onnx-community/Kokoro-82M-v1.0-ONNX --variant default'
)
for expected_download in "${expected_server_downloads[@]}"; do
  expect_download "$expected_download"
done
test "$(grep -c 'models download ' "$TEST_LOG")" -eq "${#expected_server_downloads[@]}"
# The default server warm list must be selected, not the fixture list.
grep -qF -- '--model /data/models/Rikorose/DeepFilterNet3/enc.onnx' "$TEST_LOG"
! grep -q 'trt-fp16' "$TEST_LOG"
! grep -qi 'cosyvoice' "$TEST_LOG"

run_default_seed_case rtx
expect_download 'models download google/madlad400-3b-mt --variant trt-fp16'
grep -qiE -- 'models download tonythethompson/CosyVoice-300M-ONNX --variant int8$' "$TEST_LOG"
expect_download 'models download tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX'
test "$(grep -c 'models download ' "$TEST_LOG")" -eq 9
grep -qF -- '--model /data/models/google/madlad400-3b-mt/trt_rtx_mixed_fp16_fp32/encoder_model.onnx' "$TEST_LOG"

# The classic-TRT warm list must not target an export the server seed does not provide.
if grep -v '^[[:space:]]*#' "$REPO_ROOT/docker/poc-warm-models.server.txt" | grep -qi 'madlad'; then
  echo 'Server warm list must not target the unseeded MADLAD export.' >&2
  exit 1
fi
if grep -v '^[[:space:]]*#' "$REPO_ROOT/docker/poc-warm-models.server.txt" | grep -qi 'opus'; then
  echo 'Server warm list must not target the unseeded OPUS-MT export.' >&2
  exit 1
fi

: > "$TEST_LOG"
if TRACKDUB_GPU_PROFILE=server TRACKDUB_GPU_TARGET=rtx \
  bash "$REPO_ROOT/scripts/docker/seed-and-warm-gpu-worker.sh" > "$TEST_ROOT/output" 2>&1; then
  echo 'Conflicting profile and target should fail before provisioning.' >&2
  exit 1
else
  test "$?" -eq 2
fi
test ! -s "$TEST_LOG"
grep -q 'does not match' "$TEST_ROOT/output"
echo 'Provisioning profile/target regression checks passed (7 cases).'
