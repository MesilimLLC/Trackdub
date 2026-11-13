# Operational & Environment Reference

Detailed operational facts and environment-specific instructions for Trackdub public core (`MesilimLLC/Trackdub`).

## Cloud & Headless Environment
- The primary entrypoint is the headless CLI (`src/Trackdub.Cli`). The only long-running service in this repository is the CPU job API (`src/Trackdub.Api`), which belongs to the AWS-only job stack outside the SnapSoft POC scope.
- Always pass `-m:1` to `dotnet restore`, `dotnet build`, and `dotnet test`.
- NuGet feeds: `nuget.org` and Azure Artifacts `dotnet-libraries` feed (for `Microsoft.ML.Tokenizers` 3.x preview).
- Verification gate: `dotnet format Trackdub.slnx --verify-no-changes`.
- Media stages require FFmpeg/ffprobe; verify with `dotnet run --project src/Trackdub.Cli -- doctor`.
- Model cache downloads: `dotnet run --project src/Trackdub.Cli -- models bundle-needed`, then `dotnet run --project src/Trackdub.Cli -- models download <id>`. Tests skip cleanly when models are absent.
- Default data and cache root paths land under `~/.local/share/Trackdub` (override via `TRACKDUB_DATA_ROOT` / `TRACKDUB_CACHE_ROOT`).

## Linux GPU container and cloud render
- The `Dockerfile` publishes two targets: `rtx` (TensorRT-RTX, consumer RTX GPUs) and `server` (CUDA and classic TensorRT, datacenter GPUs such as AWS G6/G6e). Build, seed and warm: `docs/operations/cloud-gpu-container.md`.
- SnapSoft cloud render POC (Cloudflare API -> SQS -> EC2 GPU worker -> S3): `docs/handoff/snapsoft-onboarding.md`.

## Execution Providers
- Desktop GPU provider routing: TRT RTX -> DirectML -> CPU. See `docs/reference/gpu-execution-providers.md`.
- Linux `server` profile: cloud presets pin stages to CUDA (VAD on CPU); classic TensorRT engines are warmed per GPU and driver with `trackdub cache warm --provider tensorrt`.
- TensorRT RTX uses standalone ORT EP ABI plugin. See `docs/reference/tensorrt-rtx-ep-abi-plugin.md`.
- Windows ML Phase 5 catalog routes: see `docs/reference/windows-ml-phase-5-catalog-eps.md`.
