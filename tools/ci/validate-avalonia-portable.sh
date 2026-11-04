#!/usr/bin/env bash
set -euo pipefail

solution="${1:-Trackdub.Avalonia.slnf}"
framework="${TRACKDUB_PORTABLE_FRAMEWORK:-net11.0}"
configuration="${TRACKDUB_CONFIGURATION:-Release}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
artifacts_root="$repo_root/.artifacts/avalonia-portable"

case "$artifacts_root" in
  "$repo_root"/.artifacts/avalonia-portable) ;;
  *)
    echo "Refusing to clean unexpected artifacts path: $artifacts_root" >&2
    exit 2
    ;;
esac

rm -rf "$artifacts_root"
mkdir -p "$artifacts_root"

msbuild_maxcpu="${MSBUILD_MAXCPU:-1}"
msbuild_args=(
  -p:RestoreFallbackFolders=
  -p:DisableImplicitNuGetFallbackFolder=true
)

dotnet --info
dotnet restore "$solution" --configfile NuGet.config --force-evaluate --artifacts-path "$artifacts_root" "${msbuild_args[@]}"
dotnet build "$solution" \
  --configuration "$configuration" \
  --no-restore \
  -m:"$msbuild_maxcpu" \
  -warnaserror \
  --framework "$framework" \
  --artifacts-path "$artifacts_root" \
  "${msbuild_args[@]}"
dotnet test "$solution" \
  --configuration "$configuration" \
  --no-build \
  --framework "$framework" \
  --logger "trx;LogFilePrefix=test-results" \
  --artifacts-path "$artifacts_root" \
  --results-directory "$artifacts_root/test-results" \
  "${msbuild_args[@]}"
