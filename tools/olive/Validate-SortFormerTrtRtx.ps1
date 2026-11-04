#Requires -Version 7.0
<#
.SYNOPSIS
    Validate TRT-RTX optimization of the SortFormer diarization model using Microsoft Olive.

.DESCRIPTION
    Runs the Olive TRT-RTX recipe for the SortFormer 4-speaker diarization encoder
    (fp16 conversion with float32 I/O kept + TRT-RTX session param tuning), then stages the result
    for the C# SortFormerDiarizationEngine to load.

    Requires:
      - NVIDIA GPU with TRT-RTX (NvTensorRTRTXExecutionProvider) support
      - Model files already downloaded to the model cache (TRACKDUB_MODEL_CACHE, or
        %LOCALAPPDATA%\Trackdub\model-cache by default):
          <model cache>/cgus/diar_streaming_sortformer_4spk-v2.1-onnx/onnx/model.onnx
      - olive-ai[nvmo] + nvidia-modelopt[onnx] installed (Bootstrap-TrtRtxOliveVenv.ps1 auto-runs)

    Records results (pass = staged model verified on TensorRT RTX via `trackdub providers trt-rtx verify`) to build/sortformer-4spk-trtrtx-validation.json.
    Run .\tools\olive\Flip-TrtRtxAsrDiarization.ps1 to apply manifest + test changes.

.EXAMPLE
    .\tools\olive\Validate-SortFormerTrtRtx.ps1
    .\tools\olive\Validate-SortFormerTrtRtx.ps1 -SkipLatency
#>

param(
    [switch] $SkipLatency
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot       = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$VenvPath       = Join-Path $env:LOCALAPPDATA 'Trackdub\tools\olive-env-tensorrtrtx'
$OliveExe       = Join-Path $VenvPath 'Scripts\olive.exe'
$BuildDir       = Join-Path $RepoRoot 'build'
$Precision      = 'fp16'
$ResultFile     = Join-Path $BuildDir "sortformer-4spk-trtrtx-validation.json"

# A previous pass cannot authorize a later failed or incomplete run.
Remove-Item -LiteralPath $ResultFile -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# Model layout (matches the download-cache layout, not bundled-models.manifest.json's
# root_path: model_id "cgus/diar_streaming_sortformer_4spk-v2.1-onnx" splits on "/" into
# the cache's owner/repo directory segments)
# ---------------------------------------------------------------------------
. (Join-Path $PSScriptRoot 'TrtRtxOliveCommon.ps1')
$ModelCacheRoot = Resolve-TrackdubModelCacheRoot

# TensorRT RTX EP ABI plugin DLL, resolved like TensorRtRtxPluginLocator: TRACKDUB_TRT_RTX_EP_DIR,
# then the default install. Olive registers it via the recipe accelerator's (name, path) pair.
$TrtRtxEpDir    = if ($env:TRACKDUB_TRT_RTX_EP_DIR) { $env:TRACKDUB_TRT_RTX_EP_DIR } else { Join-Path $env:LOCALAPPDATA 'Trackdub\Providers\trt-rtx\0.5.0\cu13\win-x64' }
$TrtRtxEpPath   = Join-Path $TrtRtxEpDir 'onnxruntime_providers_nv_tensorrt_rtx.dll'
# The plugin's companion DLLs (tensorrt_rtx_1_7.dll, tensorrt_onnxparser_rtx_1_7.dll) live beside it but are
# resolved through the normal DLL search path, so the bundle dir must be on PATH for olive.
$env:PATH = "$TrtRtxEpDir;$env:PATH"
$modelRoot      = Join-Path $ModelCacheRoot 'cgus\diar_streaming_sortformer_4spk-v2.1-onnx'
$modelSrc       = Join-Path $modelRoot 'onnx\model.onnx'
$recipeDir      = Join-Path $RepoRoot 'resources\olive-recipes\cgus-diar_streaming_sortformer_4spk-v2.1-onnx\NvTensorRtRtx'
$recipeSrc      = Join-Path $recipeDir "encoder_trtrtx_$Precision.json"
$encoderOutputDirName = "sortformer-4spk-onnx_encoder_trtrtx_$Precision"
$stagingDirName      = "sortformer-4spk-onnx-trtrtx-validated-$Precision"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
Ensure-TrtRtxOliveEnvironment -VenvPath $VenvPath -BootstrapScript (Join-Path $PSScriptRoot 'Bootstrap-TrtRtxOliveVenv.ps1')

if (-not (Test-Path $modelSrc)) {
    Write-Error "SortFormer encoder model not found: $modelSrc`nDownload with: dotnet run --project src/Trackdub.Tools -- ingest --model cgus/diar_streaming_sortformer_4spk-v2.1-onnx`nSearched model cache root: $ModelCacheRoot (override with `$env:TRACKDUB_MODEL_CACHE)"
    exit 1
}

if (-not (Test-Path $recipeSrc)) {
    Write-Error "Olive recipe not found: $recipeSrc"
    exit 1
}

# ---------------------------------------------------------------------------
# Patch recipe: substitute ${MODEL_ROOT} with the absolute model path
# ---------------------------------------------------------------------------
$TempDir = Join-Path $env:TEMP "trackdub-olive-trtrtx-sortformer-$([System.Diagnostics.Process]::GetCurrentProcess().Id)"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

function Resolve-Recipe {
    param([string] $SrcPath, [string] $DestPath)
    $content = Get-Content -Raw $SrcPath
    $content = $content -replace '\$\{MODEL_ROOT\}', ($modelRoot -replace '\\', '/')
    $content = $content -replace '\$\{TRT_RTX_EP_PATH\}', ($TrtRtxEpPath -replace '\\', '/')
    $content = $content -replace '\$\{ENCODER_OUTPUT_DIR\}', ("build/$encoderOutputDirName" -replace '\\', '/')
    Set-Content -Path $DestPath -Value $content -Encoding UTF8
}

$encoderRecipeDst = Join-Path $TempDir "encoder_trtrtx_$Precision.json"
$latencyRecipeDst = Join-Path $TempDir 'eval_latency.json'
Resolve-Recipe $recipeSrc $encoderRecipeDst
Resolve-Recipe (Join-Path $recipeDir 'eval_latency.json') $latencyRecipeDst

$origDir = Get-Location
Set-Location $RepoRoot

$results = [ordered]@{
    model_id       = 'cgus/diar_streaming_sortformer_4spk-v2.1-onnx'
    model_root     = $modelRoot
    precision      = $Precision
    timestamp_utc  = $null
    encoder        = $null
    latency        = $null
    staging_dir    = $null
    staged         = $false
    pass           = $false
}

try {
    # ---------------------------------------------------------------------------
    # Step 1: Optimize encoder (fp16 + TRT-RTX session params)
    # ---------------------------------------------------------------------------
    Write-Host ""
    Write-Host "=== SortFormer encoder optimization ($Precision + TRT-RTX session params) ===" -ForegroundColor Cyan
    & $OliveExe run --config $encoderRecipeDst
    if ($LASTEXITCODE -ne 0) { Write-Error "Encoder optimization failed (exit $LASTEXITCODE)."; exit 1 }
    $results.encoder = @{ status = 'ok'; output = "build/$encoderOutputDirName"; precision = $Precision }

    # ---------------------------------------------------------------------------
    # Step 2 (optional): encoder latency evaluation on synthetic input
    # ---------------------------------------------------------------------------
    if (-not $SkipLatency) {
        Write-Host ""
        Write-Host "=== Encoder latency evaluation on synthetic waveform ===" -ForegroundColor Cyan
        & $OliveExe run --config $latencyRecipeDst
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Latency evaluation failed (exit $LASTEXITCODE)."
            $results.latency = @{ status = 'failed' }
        } else {
            $results.latency = @{ status = 'ok' }
        }
    } else {
        Write-Host "Skipping encoder latency evaluation (-SkipLatency)." -ForegroundColor Yellow
        $results.latency = @{ status = 'skipped' }
    }

    # ---------------------------------------------------------------------------
    # Step 3: Stage combined output for the C# validation test
    # ---------------------------------------------------------------------------
    Write-Host ""
    Write-Host "=== Staging combined output for C# validation test ===" -ForegroundColor Cyan

    $StagingDir     = Join-Path $BuildDir $stagingDirName
    $StagingOnnxDir = Join-Path $StagingDir "onnx"
    New-Item -ItemType Directory -Force -Path $StagingOnnxDir | Out-Null

    $encoderOutputDir = Join-Path $BuildDir $encoderOutputDirName
    $encoderOnnxSrc   = Get-ChildItem $encoderOutputDir -Filter "*.onnx" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($encoderOnnxSrc) {
        Copy-Item $encoderOnnxSrc.FullName (Join-Path $StagingOnnxDir "model.onnx") -Force
        Get-ChildItem $encoderOnnxSrc.Directory -Filter "*.onnx.data" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $StagingOnnxDir "model.onnx.data") -Force }
    } else {
        Write-Warning "No encoder *.onnx found in $encoderOutputDir - staging incomplete."
    }

    $results.staging_dir = $StagingDir
    $results.staged = ($null -ne $encoderOnnxSrc)

    # ---------------------------------------------------------------------------
    # Step 4: Verify the staged model actually loads on TensorRT RTX, not a silent
    # CPU/DirectML fallback. Uses the real production smoke path (`trackdub providers
    # trt-rtx verify`), not a file-existence check.
    # ---------------------------------------------------------------------------
    $results.provider_check = $null
    if ($results.staged) {
        Write-Host ""
        Write-Host "=== Verifying staged model on TensorRT RTX (trackdub providers trt-rtx verify) ===" -ForegroundColor Cyan
        $stagedModelPath = Join-Path $StagingOnnxDir "model.onnx"
        $verifyOutput = & dotnet run --project (Join-Path $RepoRoot "src\Trackdub.Cli") -c Release -f net11.0-windows10.0.19041.0 -- `
            providers trt-rtx verify --model "cgus/diar_streaming_sortformer_4spk-v2.1-onnx" --entry $stagedModelPath 2>&1
        $verifyExitCode = $LASTEXITCODE
        $verifyJsonLine = $verifyOutput | Where-Object { $_ -match '^\s*\{.*"passed"\s*:' } | Select-Object -Last 1
        if ($verifyJsonLine) {
            $verifyResult = $verifyJsonLine | ConvertFrom-Json
            $results.provider_check = @{
                ready  = $verifyResult.ready
                passed = $verifyResult.passed
                detail = $verifyResult.detail
            }
            $results.pass = [bool]$verifyResult.passed
        } else {
            Write-Warning "Could not parse 'trackdub providers trt-rtx verify' output (exit $verifyExitCode); provider not confirmed."
            $results.provider_check = @{ ready = $null; passed = $false; detail = "verify command produced no parseable JSON (exit $verifyExitCode)" }
            $results.pass = $false
        }
    } else {
        $results.pass = $false
    }

} finally {
    Set-Location $origDir
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Write result file
# ---------------------------------------------------------------------------
$results.timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$results | ConvertTo-Json -Depth 4 | Set-Content -Path $ResultFile -Encoding UTF8

Write-Host ""
if ($results.pass) {
    Write-Host "PASS - TRT-RTX ($Precision) optimization output staged and verified on hardware for SortFormer 4-spk." -ForegroundColor Green
    Write-Host "Results written to: $ResultFile"
    Write-Host ""
    Write-Host "Staging directory: $($results.staging_dir)"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. dotnet test tests/Trackdub.Inference.Tests --filter 'FullyQualifiedName~SortFormer'"
    Write-Host "  2. If that passes: run .\tools\olive\Flip-TrtRtxAsrDiarization.ps1 to enable trt-rtx in the manifest and tests."
} elseif ($results.staged) {
    Write-Host "STAGED BUT NOT VERIFIED - TRT-RTX ($Precision) output staged for SortFormer 4-spk, but the real provider check failed." -ForegroundColor Yellow
    Write-Host "Detail: $($results.provider_check.detail)"
    Write-Host "Results written to: $ResultFile"
    exit 1
} else {
    Write-Host "FAIL - see errors above." -ForegroundColor Red
    exit 1
}
