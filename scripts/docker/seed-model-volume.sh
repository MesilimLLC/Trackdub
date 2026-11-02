#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
  DOCKER_REPO_ROOT="$(cygpath -m "$REPO_ROOT")"
else
  DOCKER_REPO_ROOT="$REPO_ROOT"
fi
MODEL_SET="${TRACKDUB_MODEL_SET:-$REPO_ROOT/docker/poc-models.txt}"
COMPOSE=(docker compose --project-directory "$DOCKER_REPO_ROOT" -f "$DOCKER_REPO_ROOT/docker-compose.yml")
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

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Engine and Docker Compose v2 are required." >&2
  exit 2
fi
if [[ ! -f "$MODEL_SET" ]]; then
  echo "Model set not found: $MODEL_SET" >&2
  exit 2
fi

while IFS= read -r entry || [[ -n "$entry" ]]; do
  entry="${entry%%#*}"
  entry="$(printf '%s' "$entry" | xargs)"
  [[ -z "$entry" ]] && continue

  model="${entry%%|*}"
  variant=""
  if [[ "$entry" == *'|'* ]]; then
    variant="${entry#*|}"
    variant="$(printf '%s' "$variant" | xargs)"
  fi

  echo "Seeding pinned model into the persistent /data volume: $model${variant:+ (variant: $variant)}"
  if [[ -n "$variant" ]]; then
    "${COMPOSE[@]}" run --rm trackdub models download "$model" --variant "$variant" </dev/null
  else
    "${COMPOSE[@]}" run --rm trackdub models download "$model" </dev/null
  fi
done < "$MODEL_SET"

echo "Seed complete. The Docker volume remains attached to /data/models."
