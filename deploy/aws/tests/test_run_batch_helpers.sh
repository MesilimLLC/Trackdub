#!/usr/bin/env bash
# run-batch.sh runs unattended on the instance, so a helper that is called but no longer defined only
# shows up there as "command not found". Every snake_case name in command position must be defined.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${1:-$HERE/../run-batch.sh}"

defined="$(grep -oE '^[a-z_][a-z0-9_]*\(\)' "$SCRIPT" | tr -d '()' | sort -u)"
called="$(grep -oE '(^|[;&|(!]|\$\(|if|then|else|do|until|while)[[:space:]]*[a-z][a-z0-9]*_[a-z0-9_]+([[:space:]]|$|\))' "$SCRIPT" \
  | grep -oE '[a-z][a-z0-9]*_[a-z0-9_]+' | sort -u)"

missing=()
for name in $called; do
  grep -qx "$name" <<< "$defined" || missing+=("$name")
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "run-batch.sh calls undefined helpers: ${missing[*]}" >&2
  exit 1
fi
echo "run-batch helper tests passed ($(wc -w <<< "$called") helpers called, all defined)"
