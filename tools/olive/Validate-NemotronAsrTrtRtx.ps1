#Requires -Version 7.0
<#
.SYNOPSIS
    Validate TRT-RTX optimization of the Nemotron 3.5 ASR streaming model using Microsoft Olive.

.DESCRIPTION
    Runs the Olive TRT-RTX recipes for the Nemotron ASR encoder and decoder_joint
    (fp16 conversion with float32 I/O kept + TRT-RTX session param tuning), then stages the result
    for the C# NemotronAsrOnnxAudioTranscriptionEngine to load.

    Requires:
      - NVIDIA GPU with TRT-RTX (NvTensorRTRTXExecutionProvider) support
      - Model files already downloaded to the model cache (TRACKDUB_MODEL_CACHE, or
        %LOCALAPPDATA%\Trackdub\model-cache by default):
          <model cache>/tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx/encoder.onnx
          <model cache>/tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx/decoder_joint.onnx
      - olive-ai[nvmo] + nvidia-modelopt[onnx] installed (Bootstrap-TrtRtxOliveVenv.ps1 auto-runs)

    Records results (pass = staged model verified on TensorRT RTX via `trackdub providers trt-rtx verify`) to build/nemotron-3.5-asr-trtrtx-validation.json.
    Run .\tools\olive\Flip-TrtRtxAsrDiarization.ps1 to apply manifest + test changes.

.EXAMPLE
    .\tools\olive\Validate-NemotronAsrTrtRtx.ps1
    .\tools\olive\Validate-NemotronAsrTrtRtx.ps1 -SkipLatency
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
$ResultFile     = Join-Path $BuildDir "nemotron-3.5-asr-trtrtx-validation.json"

# A previous pass cannot authorize a later failed or incomplete run.
Remove-Item -LiteralPath $ResultFile -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# Model layout (matches the download-cache layout, not bundled-models.manifest.json's
# root_path): the model id "tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx" splits on
# "/" into the cache's owner/repo directory segments.
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
$modelRoot      = Join-Path $ModelCacheRoot 'tonythethompson\nemotron-3.5-asr-streaming-0.6b-onnx'
$encoderSrc     = Join-Path $modelRoot 'encoder.onnx'
$decoderSrc     = Join-Path $modelRoot 'decoder_joint.onnx'
$recipeDir      = Join-Path $RepoRoot 'resources\olive-recipes\nemotron-3.5-asr-streaming-0.6b-onnx\NvTensorRtRtx'
$encoderRecipe  = Join-Path $recipeDir "encoder_trtrtx_$Precision.json"
$decoderRecipe  = Join-Path $recipeDir "decoder_joint_trtrtx_$Precision.json"
$encoderOutputDirName = "nemotron-3.5-asr-onnx_encoder_trtrtx_$Precision"
$decoderOutputDirName = "nemotron-3.5-asr-onnx_decoder_joint_trtrtx_$Precision"
$stagingDirName       = "nemotron-3.5-asr-onnx-trtrtx-validated-$Precision"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
Ensure-TrtRtxOliveEnvironment -VenvPath $VenvPath -BootstrapScript (Join-Path $PSScriptRoot 'Bootstrap-TrtRtxOliveVenv.ps1')

if (-not (Test-Path $encoderSrc)) {
    Write-Error "Nemotron encoder model not found: $encoderSrc`nDownload with: dotnet run --project src/Trackdub.Tools -- ingest --model tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx`nSearched model cache root: $ModelCacheRoot (override with `$env:TRACKDUB_MODEL_CACHE)"
    exit 1
}

if (-not (Test-Path $decoderSrc)) {
    Write-Error "Nemotron decoder_joint model not found: $decoderSrc"
    exit 1
}

if (-not (Test-Path $encoderRecipe)) {
    Write-Error "Olive encoder recipe not found: $encoderRecipe"
    exit 1
}

# ---------------------------------------------------------------------------
# Patch recipes: substitute ${MODEL_ROOT} with the absolute model path
# ---------------------------------------------------------------------------
$TempDir = Join-Path $env:TEMP "trackdub-olive-trtrtx-nemotron-$([System.Diagnostics.Process]::GetCurrentProcess().Id)"
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
$decoderRecipeDst = Join-Path $TempDir "decoder_joint_trtrtx_$Precision.json"
$latencyRecipeDst = Join-Path $TempDir 'eval_latency.json'
Resolve-Recipe $encoderRecipe $encoderRecipeDst
Resolve-Recipe $decoderRecipe $decoderRecipeDst
Resolve-Recipe (Join-Path $recipeDir 'eval_latency.json') $latencyRecipeDst

$origDir = Get-Location
Set-Location $RepoRoot

$results = [ordered]@{
    model_id       = 'tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx'
    model_root     = $modelRoot
    precision      = $Precision
    timestamp_utc  = $null
    encoder        = $null
    decoder        = $null
    latency        = $null
    staging_dir    = $null
    staged         = $false
    pass           = $false
}

try {
    # ---------------------------------------------------------------------------
    # Step 1: Optimize encoder
    # ---------------------------------------------------------------------------
    Write-Host ""
    Write-Host "=== Nemotron encoder optimization ($Precision + TRT-RTX session params) ===" -ForegroundColor Cyan
    & $OliveExe run --config $encoderRecipeDst
    if ($LASTEXITCODE -ne 0) { Write-Error "Encoder optimization failed (exit $LASTEXITCODE)."; exit 1 }
    $results.encoder = @{ status = 'ok'; output = "build/$encoderOutputDirName"; precision = $Precision }

    # ---------------------------------------------------------------------------
    # Step 2: Optimize decoder_joint
    # ---------------------------------------------------------------------------
    Write-Host ""
    Write-Host "=== Nemotron decoder_joint optimization ($Precision + TRT-RTX session params) ===" -ForegroundColor Cyan
    & $OliveExe run --config $decoderRecipeDst
    if ($LASTEXITCODE -ne 0) { Write-Error "Decoder_joint optimization failed (exit $LASTEXITCODE)."; exit 1 }
    $results.decoder = @{ status = 'ok'; output = "build/$decoderOutputDirName"; precision = $Precision }

    # ---------------------------------------------------------------------------
    # Step 3 (optional): encoder latency evaluation on synthetic input
    # ---------------------------------------------------------------------------
    if (-not $SkipLatency) {
        Write-Host ""
        Write-Host "=== Encoder latency evaluation on synthetic mel input ===" -ForegroundColor Cyan
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
    # Step 4: Stage combined output for the C# validation test
    # ---------------------------------------------------------------------------
    Write-Host ""
    Write-Host "=== Staging combined output for C# validation test ===" -ForegroundColor Cyan

    $StagingDir     = Join-Path $BuildDir $stagingDirName
    New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

    # Carry over companion files (tokenizer, config, README, LICENSE).
    foreach ($asset in @("README.md", "NOTICE.md", "LICENSE.OpenMDW-1.1", "config.json",
                          "tokenizer.model")) {
        $assetSrc = Join-Path $modelRoot $asset
        if (Test-Path $assetSrc) {
            Copy-Item $assetSrc (Join-Path $StagingDir $asset) -Force
        }
    }

    # Olive writes each component as model.onnx plus external data that keeps the source's
    # data filename (the graph references it by that name), so copy the .onnx.data files
    # under their own names and rename only the graph.
    $encoderOutputDir = Join-Path $BuildDir $encoderOutputDirName
    $encoderOnnxSrc   = Get-ChildItem $encoderOutputDir -Filter "*.onnx" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($encoderOnnxSrc) {
        Copy-Item $encoderOnnxSrc.FullName (Join-Path $StagingDir "encoder.onnx") -Force
        Get-ChildItem $encoderOnnxSrc.Directory -Filter "*.onnx.data" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $StagingDir $_.Name) -Force }
    } else {
        Write-Warning "No *.onnx found in $encoderOutputDir - staging incomplete."
    }

    $decoderOutputDir = Join-Path $BuildDir $decoderOutputDirName
    $decoderOnnxSrc   = Get-ChildItem $decoderOutputDir -Filter "*.onnx" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($decoderOnnxSrc) {
        Copy-Item $decoderOnnxSrc.FullName (Join-Path $StagingDir "decoder_joint.onnx") -Force
        Get-ChildItem $decoderOnnxSrc.Directory -Filter "*.onnx.data" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $StagingDir $_.Name) -Force }
    } else {
        Write-Warning "No *.onnx found in $decoderOutputDir - staging incomplete."
    }

    $results.staging_dir = $StagingDir
    $results.staged = ($null -ne $encoderOnnxSrc) -and ($null -ne $decoderOnnxSrc)

    # ---------------------------------------------------------------------------
    # Step 5: Verify the staged encoder + decoder_joint pair actually loads and runs on
    # TensorRT RTX, not a silent CPU/DirectML fallback. Uses the real production smoke
    # path (`trackdub providers trt-rtx verify`), which exercises both components with
    # the pinned NemotronAsrEncoderTrtProfiles shape profile — not a file-existence check.
    # ---------------------------------------------------------------------------
    $results.provider_check = $null
    if ($results.staged) {
        Write-Host ""
        Write-Host "=== Verifying staged encoder + decoder_joint on TensorRT RTX (trackdub providers trt-rtx verify) ===" -ForegroundColor Cyan
        $stagedEncoderPath = Join-Path $StagingDir "encoder.onnx"
        $verifyOutput = & dotnet run --project (Join-Path $RepoRoot "src\Trackdub.Cli") -c Release -f net11.0-windows10.0.19041.0 -- `
            providers trt-rtx verify --model "tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx" --entry $stagedEncoderPath 2>&1
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
    Write-Host "PASS - TRT-RTX ($Precision) optimization output staged and verified on hardware for Nemotron 3.5 ASR." -ForegroundColor Green
    Write-Host "Results written to: $ResultFile"
    Write-Host ""
    Write-Host "Staging directory: $($results.staging_dir)"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. dotnet test tests/Trackdub.Inference.Onnx.Tests --filter 'FullyQualifiedName~NemotronAsrEncoderTrtRtx'"
    Write-Host "  2. If that passes: run .\tools\olive\Flip-TrtRtxAsrDiarization.ps1 to enable trt-rtx in the manifest and tests."
} elseif ($results.staged) {
    Write-Host "STAGED BUT NOT VERIFIED - TRT-RTX ($Precision) output staged for Nemotron 3.5 ASR, but the real provider check failed." -ForegroundColor Yellow
    Write-Host "Detail: $($results.provider_check.detail)"
    Write-Host "Results written to: $ResultFile"
    exit 1
} else {
    Write-Host "FAIL - see errors above." -ForegroundColor Red
    exit 1
}
