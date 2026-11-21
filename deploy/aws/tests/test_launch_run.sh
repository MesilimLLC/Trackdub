#!/usr/bin/env bash
# Exercises launch-run.sh against a stub aws CLI: argument checks, dry run, upload calls, and the
# subnet fallback when the first subnet has no capacity. No AWS calls are made.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/../launch-run.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

mkdir -p "$TEST_ROOT/bin" "$TEST_ROOT/clips"
printf 'x' > "$TEST_ROOT/clips/b.mp4"
printf 'x' > "$TEST_ROOT/clips/a.MP4"
printf 'x' > "$TEST_ROOT/clips/notes.txt"

cat > "$TEST_ROOT/bin/aws" <<'STUB'
#!/usr/bin/env bash
echo "$*" >> "$AWS_LOG"
case "$1 $2" in
  "cloudformation describe-stacks")
    case "$*" in
      *BucketName*) echo "test-bucket" ;;
      *SubnetIds*) echo "subnet-a,subnet-b" ;;
      *SecurityGroupId*) echo "sg-test" ;;
      *InstanceProfileName*) echo "test-profile" ;;
      *TokenParameterName*) echo "/trackdub/ghcr-read-token" ;;
    esac ;;
  "ssm get-parameter") echo "ami-test" ;;
  "ec2 describe-images") echo "/dev/sda1" ;;
  "ec2 run-instances")
    if [[ "$*" == *"subnet-a"* ]]; then
      echo "InsufficientInstanceCapacity: no capacity" >&2
      exit 255
    fi
    echo "i-test123" ;;
  *) ;;
esac
STUB
chmod +x "$TEST_ROOT/bin/aws"
export PATH="$TEST_ROOT/bin:$PATH"
export AWS_LOG="$TEST_ROOT/aws.log"

expect_failure() {
  local name="$1" message="$2"; shift 2
  local out="$TEST_ROOT/out"
  if "$@" > "$out" 2>&1; then
    echo "expected failure: $name" >&2; cat "$out" >&2; exit 1
  fi
  grep -qF -- "$message" "$out" || { echo "missing message for $name: $message" >&2; cat "$out" >&2; exit 1; }
}

base=(--image "ghcr.io/mesilimllc/trackdub:sha-33f559de19f41a683a902775384afc17cef70df0-server" \
      --ghcr-user tester --clips-dir "$TEST_ROOT/clips")

# Guard rails stop before any AWS call that changes state.
expect_failure "moving tag" "must be pinned" "$LAUNCHER" --image ghcr.io/mesilimllc/trackdub:main-server --ghcr-user t --clip "$TEST_ROOT/clips/a.MP4"
expect_failure "missing user" "--ghcr-user is required" "$LAUNCHER" --image ghcr.io/mesilimllc/trackdub@sha256:abc --clip "$TEST_ROOT/clips/a.MP4"
expect_failure "bad instance type" "must be g6.* or g6e.*" "$LAUNCHER" "${base[@]}" --instance-type p5.48xlarge
expect_failure "no clips" "Give --clips-dir" "$LAUNCHER" --image ghcr.io/mesilimllc/trackdub@sha256:abc --ghcr-user t
expect_failure "unpinned tag without commit" "Pass --repo-commit" "$LAUNCHER" --image ghcr.io/mesilimllc/trackdub:container-v0.1.0-server --ghcr-user t --clip "$TEST_ROOT/clips/a.MP4"

# Dry run renders the config and uploads nothing.
: > "$AWS_LOG"
"$LAUNCHER" "${base[@]}" --dry-run > "$TEST_ROOT/dry" 2>&1
grep -q "Clips:       2" "$TEST_ROOT/dry"
grep -q "Dry run: nothing uploaded" "$TEST_ROOT/dry"
! grep -q "s3 cp" "$AWS_LOG"
! grep -q "run-instances" "$AWS_LOG"

# A real run uploads both .mp4 clips (not the .txt), then falls back from subnet-a to subnet-b.
: > "$AWS_LOG"
"$LAUNCHER" "${base[@]}" --run-id test-run > "$TEST_ROOT/live" 2>&1
grep -q "Subnet subnet-a" "$TEST_ROOT/live"
grep -q "Launched i-test123 in subnet-b" "$TEST_ROOT/live"
grep -q "runs/test-run/input/a.MP4" "$AWS_LOG"
grep -q "runs/test-run/input/b.mp4" "$AWS_LOG"
! grep -q "notes.txt" "$AWS_LOG"
grep -q "runs/test-run/tools/summarize_result.py" "$AWS_LOG"
grep -q -- "--instance-initiated-shutdown-behavior terminate" "$AWS_LOG"
grep -q -- "HttpTokens=required" "$AWS_LOG"

echo "launch-run tests passed"
