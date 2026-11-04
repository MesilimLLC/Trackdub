#!/usr/bin/env bash
set -Eeuo pipefail

# ==================================================
# Trackdub - Kilo Code Cloud Setup
# .NET 11 SDK, GitHub CLI, gh-stack, repo verification
# ==================================================

# The script may run outside the checkout (KILO_PROJECT_DIR not guaranteed),
# so try each known location to find the repo root.
find_repo_root() {
  local candidate top
  for candidate in \
    "${KILO_PROJECT_DIR:-}" \
    "${CLAUDE_PROJECT_DIR:-}" \
    "$PWD" \
    "$(dirname "${BASH_SOURCE[0]}")" \
    /workspace \
    /home/user \
    /home/user/Trackdub \
    /mnt/workspace \
    /tmp/workspace; do
    [[ -n "$candidate" && -e "$candidate/.git" ]] || continue
    top="$(git -C "$candidate" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$top" ]]; then
      printf '%s\n' "$top"
      return 0
    fi
    if [[ -f "$candidate/Trackdub.slnx" || -d "$candidate/src/Trackdub.Domain" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

REPO_ROOT="$(find_repo_root || true)"

if [[ -z "$REPO_ROOT" ]]; then
  echo "ERROR: No repository checkout available during setup." >&2
  exit 1
fi

SOURCE_ROOT="$REPO_ROOT"

echo "Repository detected: $REPO_ROOT"

# Run regular-user steps as the owner of the checkout unless overridden.
SETUP_USER="${SETUP_USER:-$(stat -c %U "$REPO_ROOT" 2>/dev/null || whoami)}"

GH_VERSION="2.103.2"
GH_SHA256="62ccf747115ddc61237efffdde66a3c8f70ba249433dc64f70801089b59c59c8"

GH_STACK_VERSION="v0.2.1"
GH_STACK_SHA256="ffe9d48b6342faffc3b0cb420202ff1df32693debd7423274bbdf859"
GH_STACK_DIR="/opt/gh-stack"

# Change to 1 to build Trackdub during setup.
# Disabled by default to keep startup fast.
RUN_BUILD=0

# --------------------------------------------------
# Preflight
# --------------------------------------------------

if [[ "$EUID" -ne 0 ]]; then
  echo "ERROR: Setup must run as root." >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: Requires Linux x86_64." >&2
  exit 1
fi

if ! id "$SETUP_USER" >/dev/null 2>&1; then
  echo "ERROR: Setup user does not exist: $SETUP_USER" >&2
  exit 1
fi

as_user() {
  runuser -u "$SETUP_USER" -- "$@"
}

TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT
trap 'echo "ERROR at line $LINENO: $BASH_COMMAND" >&2' ERR

echo "========================================"
echo " Trackdub Kilo Code Cloud Environment Setup"
echo "========================================"

# --------------------------------------------------
# 1. Install .NET 11 SDK globally
# --------------------------------------------------

echo
echo "=== Installing .NET 11 SDK ==="

if [[ ! -x /usr/local/share/dotnet/dotnet ]] || \
   ! /usr/local/share/dotnet/dotnet --list-sdks | grep -Fq '11.0.100-rc.3.26522.148 ['; then
  curl -fsSL --retry 3 \
    https://dot.net/v1/dotnet-install.sh \
    -o "$TMP_DIR/dotnet-install.sh"

  bash "$TMP_DIR/dotnet-install.sh" \
    --version 11.0.100-rc.3.26522.148 \
    --install-dir /usr/local/share/dotnet
fi

ln -sfn \
  /usr/local/share/dotnet/dotnet \
  /usr/local/bin/dotnet

/usr/local/bin/dotnet --version

# --------------------------------------------------
# 2. Install GitHub CLI
# --------------------------------------------------

echo
echo "=== Installing GitHub CLI ==="

curl -fsSL --retry 3 \
  "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
  -o "$TMP_DIR/gh.tar.gz"

printf '%s  %s\n' \
  "$GH_SHA256" \
  "$TMP_DIR/gh.tar.gz" \
  | sha256sum -c -

tar -xzf "$TMP_DIR/gh.tar.gz" -C "$TMP_DIR"

install -m 0755 \
  "$TMP_DIR/gh_${GH_VERSION}_linux_amd64/bin/gh" \
  /usr/local/bin/gh

/usr/local/bin/gh --version

# --------------------------------------------------
# 3. Ensure Node.js supports skills CLI
# --------------------------------------------------

echo
echo "=== Checking Node.js ==="

if ! as_user bash -c '
  command -v node >/dev/null &&
  command -v npx >/dev/null &&
  node -e "
    const [major, minor] =
      process.versions.node.split(".").map(Number);

    process.exit(
      major > 22 || (major === 22 && minor >= 20)
        ? 0
        : 1
    );
  "
'; then
  echo "Installing Node.js 22..."

  curl -fsSL --retry 3 \
    https://deb.nodesource.com/setup_22.x \
    -o "$TMP_DIR/setup-node.sh"

  bash "$TMP_DIR/setup-node.sh"

  DEBIAN_FRONTEND=noninteractive \
    apt-get install -y nodejs
fi

as_user node --version
as_user npx --version

# --------------------------------------------------
# 4. Install gh-stack
# --------------------------------------------------

echo
echo "=== Installing gh-stack ==="

install -d -m 0755 "$GH_STACK_DIR"

curl -fsSL --retry 3 \
  "https://github.com/github/gh-stack/releases/download/${GH_STACK_VERSION}/linux-amd64" \
  -o "$TMP_DIR/gh-stack"

printf '%s  %s\n' \
  "$GH_STACK_SHA256" \
  "$TMP_DIR/gh-stack" \
  | sha256sum -c -

install -m 0755 \
  "$TMP_DIR/gh-stack" \
  "$GH_STACK_DIR/gh-stack"

# Install for the regular user, not root.
# Local installation avoids GitHub API rate limits.

as_user gh extension remove stack \
  >/dev/null 2>&1 || true

as_user bash -c '
  set -euo pipefail

  cd /opt/gh-stack

  gh extension install . --force
'

echo "gh-stack installed."

# --------------------------------------------------
# 5. Install gh-stack Claude Code skill
# --------------------------------------------------

echo
echo "=== Installing gh-stack Claude skill ==="

as_user bash -c '
  set -euo pipefail

  cd "$1"

  npx -y skills add github/gh-stack \
    --skill gh-stack \
    --agent claude-code \
    --yes
' _ "$REPO_ROOT"

echo "Claude skill installed."

# --------------------------------------------------
# 6. Git configuration
# --------------------------------------------------

echo
echo "=== Configuring Git ==="

as_user git config --global rerere.enabled true

# Avoid ambiguous push remotes for stacked PRs.
as_user git config --global remote.pushDefault origin

# --------------------------------------------------
# 7. Optional Trackdub build
# --------------------------------------------------

if [[ "$RUN_BUILD" == "1" ]]; then
  echo
  echo "=== Building Trackdub ==="

  as_user bash -c '
    set -euo pipefail

    cd "$1"

    timeout 1200 dotnet build \
      Trackdub.slnx \
      -m:1
  ' _ "$SOURCE_ROOT"

  echo "Trackdub build succeeded."
else
  echo
  echo "=== Build skipped (RUN_BUILD=0) ==="
fi

# --------------------------------------------------
# 8. Final verification
# --------------------------------------------------

echo
echo "========================================"
echo " Verifying environment"
echo "========================================"

echo
echo "=== .NET ==="
as_user dotnet --version

echo
echo "=== GitHub CLI ==="
as_user gh --version

echo
echo "=== gh-stack ==="
as_user gh stack --help >/dev/null
echo "PASS: gh stack works"

echo
echo "=== Installed extensions ==="

as_user bash -c '
  set -euo pipefail
  ext_dir="${XDG_DATA_HOME:-$HOME/.local/share}/gh/extensions/gh-stack"
  test -x "$ext_dir/gh-stack" && echo "gh-stack installed at $ext_dir"
'

echo
echo "=== Claude skill ==="

SKILL_PATH="$REPO_ROOT/.claude/skills/gh-stack/SKILL.md"

if as_user test -f "$SKILL_PATH"; then
  echo "PASS: gh-stack Claude skill exists"
else
  echo "ERROR: Claude skill not found at:" >&2
  echo "$SKILL_PATH" >&2
  exit 1
fi

echo
echo "=== dotnet format check ==="
as_user bash -c '
  set -euo pipefail
  cd "$1"
  dotnet format Trackdub.slnx --verify-no-changes
' _ "$SOURCE_ROOT"

echo
echo "========================================"
echo " Trackdub setup completed successfully"
echo "========================================"
