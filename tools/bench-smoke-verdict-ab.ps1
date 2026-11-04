$ErrorActionPreference = "Continue"
$env:ProgramFiles = "C:\Program Files"
${env:ProgramFiles(x86)} = "C:\Program Files (x86)"
$env:ProgramW6432 = "C:\Program Files"
Remove-Item env:MSBuildSDKsPath -ErrorAction SilentlyContinue

$store = "$env:LOCALAPPDATA\Trackdub\smoke-verdicts.json"
$reports = "$env:LOCALAPPDATA\Trackdub\benchmark-reports"
$fixture = "$env:LOCALAPPDATA\Trackdub\benchmark-fixtures\baseline-v1\short.mp4"
$outRoot = "$env:LOCALAPPDATA\Trackdub\benchmark-smoke-verdict\controlled"
$proj = "src/Trackdub.Benchmarks.DevHost"
$tfm = "net11.0-windows10.0.19041.0"

function Get-NewestReport([string]$label) {
    $files = Get-ChildItem $reports -Filter *.json | Sort-Object LastWriteTime -Descending
    foreach ($f in $files) {
        try {
            $r = Get-Content $f.FullName -Raw | ConvertFrom-Json
            if ($r.Kind -eq "Benchmark" -and $r.Scenario) {
                return $r
            }
        } catch {
            Write-Verbose "Skipping unparseable report $($f.FullName): $($_.Exception.Message)"
        }
    }
    return $null
}

function Invoke-Controlled([string]$stage, [string]$provider, [string]$model, [string]$label) {
    $out = Join-Path $outRoot $label
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $before = (Get-Date).AddSeconds(-1)
    $dotnetArgs = @("run","--project",$proj,"-f",$tfm,"--no-build","--","controlled",$fixture,"--output",$out,"--mode","fresh-process","--reuse-engine-cache")
    if ($stage) { $dotnetArgs += @("--stage",$stage) }
    if ($provider) { $dotnetArgs += @("--provider",$provider) }
    if ($model) { $dotnetArgs += @("--model",$model) }
    $null = & dotnet @dotnetArgs 2>&1
    $files = Get-ChildItem $reports -Filter *.json | Where-Object { $_.LastWriteTime -gt $before } | Sort-Object LastWriteTime -Descending
    foreach ($f in $files) {
        try {
            $r = Get-Content $f.FullName -Raw | ConvertFrom-Json
            if ($r.Kind -eq "Benchmark") { return $r }
        } catch {
            Write-Verbose "Skipping unparseable report $($f.FullName): $($_.Exception.Message)"
        }
    }
    return $null
}

function Row($r, [string]$condition) {
    if ($null -eq $r) {
        return [PSCustomObject]@{ Condition = $condition; Status = "NO-REPORT"; Preflight = $null; Pipeline = $null; Total = $null; Provider = $null; Model = $null }
    }
    $t = $r.TimingsMilliseconds
    $preflight = $null; $pipeline = $null; $total = $null
    if ($t.preflight) { $preflight = [Math]::Round($t.preflight, 0) }
    if ($t.pipeline)  { $pipeline  = [Math]::Round($t.pipeline, 0) }
    if ($t.total)     { $total     = [Math]::Round($t.total, 0) }
    return [PSCustomObject]@{
        Condition = $condition
        Status    = $r.Status
        Preflight = $preflight
        Pipeline  = $pipeline
        Total     = $total
        Provider  = $r.ActualProvider
        Model     = $r.ActualModel
    }
}

$allRows = @()
$targets = @(
    @{ Stage = "vad";         Provider = "DirectMl"; Model = $null },
    @{ Stage = "asr";         Provider = "DirectMl"; Model = $null },
    @{ Stage = "translation"; Provider = "DirectMl"; Model = "madlad400" },
    @{ Stage = "tts";         Provider = "DirectMl"; Model = "cosyvoice-300m" },
    @{ Stage = "tts";         Provider = "DirectMl"; Model = "chatterbox-turbo-onnx" }
)

foreach ($t in $targets) {
    $name = "$($t.Stage)$(if ($t.Model) { '-' + $t.Model })"
    Write-Host "=== $name ==="

    # BEFORE: verdict store cleared -> smoke must run on every fresh-process launch.
    if (Test-Path $store) { Remove-Item $store -Force }
    $beforeReport = Invoke-Controlled $t.Stage $t.Provider $t.Model "$name-before"
    $allRows += Row $beforeReport "$name BEFORE (smoke runs)"
    Write-Host ("  BEFORE preflight={0} ms status={1} provider={2}" -f $allRows[-1].Preflight, $allRows[-1].Status, $allRows[-1].Provider)

    # AFTER: verdict store populated by the BEFORE run -> smoke skipped on this launch.
    $afterReport = Invoke-Controlled $t.Stage $t.Provider $t.Model "$name-after"
    $allRows += Row $afterReport "$name AFTER  (smoke skipped)"
    Write-Host ("  AFTER  preflight={0} ms status={1} provider={2}" -f $allRows[-1].Preflight, $allRows[-1].Status, $allRows[-1].Provider)
    Write-Host ("  verdict store: {0}" -f (Test-Path $store))
}

Write-Host ""
Write-Host "=========== SUMMARY ==========="
$allRows | Format-Table -AutoSize
$allRows | Export-Csv (Join-Path $outRoot "summary.csv") -NoTypeInformation
Write-Host "Saved: $(Join-Path $outRoot 'summary.csv')"
