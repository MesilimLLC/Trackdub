#!/usr/bin/env bash
set -Eeuo pipefail

# ==================================================
# Trackdub - Claude Code Cloud Setup
# .NET 11 + Git submodule + GitHub CLI + gh-stack
# ==================================================

# The setup script may run outside the checkout (CLAUDE_PROJECT_DIR and the
# working directory are not guaranteed), so try each known location in turn.
find_repo_root() {
  local candidate top
  for candidate in \
    "${CLAUDE_PROJECT_DIR:-}" \
    "$PWD" \
    "$(dirname "${BASH_SOURCE[0]}")" \
    /home/user/Trackdub; do
    [[ -n "$candidate" && -e "$candidate/.git" ]] || continue
    top="$(git -C "$candidate" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$top" ]]; then
      printf '%s\n' "$top"
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

SOURCE_ROOT="$REPO_ROOT/external/Trackdub"

echo "Repository detected: $REPO_ROOT"

# Run the regular-user steps as the owner of the checkout unless overridden.
SETUP_USER="${SETUP_USER:-$(stat -c %U "$REPO_ROOT")}"

GH_VERSION="2.103.2"
GH_SHA256="62ccf747115ddc61237efffdde66a3c8f70ba249433dc64f70801089b59c59c8"

GH_STACK_VERSION="v0.2.1"
GH_STACK_SHA256="ffe9d48b6342faffc3b0cb420202ff1df32e24faf32693debd7423274bbdf859"
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
echo " Trackdub Claude Code Environment Setup"
echo "========================================"

# --------------------------------------------------
# 1. Install .NET 11 globally
# --------------------------------------------------

echo
echo "=== Installing .NET 11 SDK ==="

if [[ ! -x /usr/local/share/dotnet/dotnet ]] || \
   ! /usr/local/share/dotnet/dotnet --list-sdks | grep -Fq '11.0.100 ['; then
  curl -fsSL --retry 3 \
    https://dot.net/v1/dotnet-install.sh \
    -o "$TMP_DIR/dotnet-install.sh"

  bash "$TMP_DIR/dotnet-install.sh" \
    --version 11.0.100 \
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
      process.versions.node.split(\".\").map(Number);

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
# 4. Initialize Trackdub Git submodule
# --------------------------------------------------

echo
echo "=== Initializing Trackdub submodule ==="

# Only this checkout's .gitmodules decides whether the submodule exists.
HAS_SUBMODULE=0

if as_user git -C "$REPO_ROOT" config -f .gitmodules \
  --get-regexp '^submodule\..*\.path$' 2>/dev/null \
  | grep -qx '.* external/Trackdub'; then
  HAS_SUBMODULE=1
fi

if [[ "$HAS_SUBMODULE" == "1" ]]; then
  as_user bash -c '
    set -euo pipefail

    cd "$1"

    timeout 600 git submodule update \
      --init \
      --depth 1 \
      -- external/Trackdub

    git submodule status -- external/Trackdub
  ' _ "$REPO_ROOT"

  APP_PROJECT="$SOURCE_ROOT/src/Trackdub.App.Avalonia/Trackdub.App.Avalonia.csproj"

  if [[ ! -f "$APP_PROJECT" ]]; then
    echo "ERROR: Avalonia project not found:" >&2
    echo "$APP_PROJECT" >&2
    exit 1
  fi

  echo "Trackdub submodule initialized."
else
  echo "No external/Trackdub submodule declared in this checkout; skipping."
fi

# --------------------------------------------------
# 5. Install gh-stack
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
# The local directory must be named gh-stack.
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
# 6. Install gh-stack Claude Code skill
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
# 7. Git configuration
# --------------------------------------------------

echo
echo "=== Configuring Git ==="

as_user git config --global rerere.enabled true

# Avoid ambiguous push remotes for stacked PRs.
as_user git config --global remote.pushDefault origin

# --------------------------------------------------
# 8. Optional Avalonia build
# --------------------------------------------------

if [[ "$RUN_BUILD" == "1" ]]; then
  echo
  echo "=== Building Trackdub Avalonia ==="

  as_user bash -c '
    set -euo pipefail

    cd "$1"

    timeout 1200 dotnet build \
      src/Trackdub.App.Avalonia \
      -f net11.0 \
      -m:1
  ' _ "$SOURCE_ROOT"

  echo "Avalonia build succeeded."
else
  echo
  echo "=== Build skipped (RUN_BUILD=0) ==="
fi

# --------------------------------------------------
# 9. Final verification
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

# `gh extension list` needs a GitHub login, so check the install on disk.
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

if [[ "$HAS_SUBMODULE" == "1" ]]; then
  echo
  echo "=== Trackdub submodule ==="

  as_user git -C "$REPO_ROOT" \
    submodule status -- external/Trackdub

  echo
  echo "=== Avalonia project ==="

  if as_user test -f "$APP_PROJECT"; then
    echo "PASS: Avalonia project exists"
  else
    echo "ERROR: Avalonia project missing" >&2
    exit 1
  fi
fi

echo
echo "========================================"
echo " Trackdub setup completed successfully"
echo "========================================"
