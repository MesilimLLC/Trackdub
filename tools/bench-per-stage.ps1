$ErrorActionPreference = "Continue"
$env:ProgramFiles = "C:\Program Files"
${env:ProgramFiles(x86)} = "C:\Program Files (x86)"
$env:ProgramW6432 = "C:\Program Files"
Remove-Item env:MSBuildSDKsPath -ErrorAction SilentlyContinue

$reports = "$env:LOCALAPPDATA\Trackdub\benchmark-reports"
$fixture = "$env:LOCALAPPDATA\Trackdub\benchmark-fixtures\baseline-v1\short.mp4"
$outRoot = "$env:LOCALAPPDATA\Trackdub\benchmark-smoke-verdict\per-stage"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$proj = "src/Trackdub.Benchmarks.DevHost"
$tfm = "net11.0-windows10.0.19041.0"

function Invoke-Run([string]$stage, [string]$provider, [string]$model, [string]$mode, [string]$label) {
    $out = Join-Path $outRoot $label
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $before = (Get-Date).AddSeconds(-1)
    $dotnetArgs = @("run","--project",$proj,"-f",$tfm,"--no-build","--","controlled",$fixture,"--output",$out,"--mode",$mode,"--reuse-engine-cache")
    if ($stage)   { $dotnetArgs += @("--stage",$stage) }
    if ($provider){ $dotnetArgs += @("--provider",$provider) }
    if ($model)   { $dotnetArgs += @("--model",$model) }
    $null = & dotnet @dotnetArgs 2>&1
    $files = Get-ChildItem $reports -Filter *.json | Where-Object { $_.LastWriteTime -gt $before } | Sort-Object LastWriteTime -Descending
    foreach ($f in $files) {
        try {
            $r = Get-Content $f.FullName -Raw | ConvertFrom-Json
            if ($r.kind -eq "Benchmark") { return $r }
        } catch {
            Write-Verbose "Skipping unparseable report $($f.FullName): $($_.Exception.Message)"
        }
    }
    return $null
}

function Summarize($r, [string]$stage, [string]$model, [string]$mode) {
    if ($null -eq $r) {
        return [PSCustomObject]@{ Stage=$stage; Model=$model; Mode=$mode; Status="NO-REPORT"; Preflight=$null; Pipeline=$null; Total=$null; StageMs=$null; Provider=$null; ActualModel=$null }
    }
    $t = $r.timingsMilliseconds
    $preflight = $null; $pipeline = $null; $total = $null; $stageMs = $null
    if ($t.preflight) { $preflight = [Math]::Round($t.preflight, 0) }
    if ($t.pipeline)  { $pipeline  = [Math]::Round($t.pipeline, 0) }
    if ($t.total)     { $total     = [Math]::Round($t.total, 0) }
    if ($r.stages -and $r.stages.Count -gt 0 -and $r.stages[0].durationMilliseconds) {
        $stageMs = [Math]::Round($r.stages[0].durationMilliseconds, 0)
    }
    $actualProvider = $null; $actualModel = $null
    if ($r.stages -and $r.stages.Count -gt 0) {
        $actualProvider = $r.stages[0].actualProvider
        $actualModel = $r.stages[0].actualModel
    }
    return [PSCustomObject]@{
        Stage = $stage; Model = $model; Mode = $mode; Status = $r.Status
        Preflight = $preflight; Pipeline = $pipeline; Total = $total; StageMs = $stageMs
        Provider = $actualProvider; ActualModel = $actualModel
    }
}

$targets = @(
    @{ Stage = "vad";         Model = $null },
    @{ Stage = "asr";         Model = $null },
    @{ Stage = "translation"; Model = $null },
    @{ Stage = "tts";         Model = "cosyvoice-300m" },
    @{ Stage = "tts";         Model = "chatterbox-turbo-onnx" }
)
$modes = @("fresh-process", "warm-host")

$rows = @()
foreach ($t in $targets) {
    $name = "$($t.Stage)$(if ($t.Model) { '-' + $t.Model })"
    foreach ($m in $modes) {
        Write-Host ">>> $name [$m]"
        $r = Invoke-Run $t.Stage "Cpu" $t.Model $m "$name-$m"
        $row = Summarize $r $t.Stage $t.Model $m
        $rows += $row
        Write-Host ("    status={0} preflight={1}ms stage={2}ms pipeline={3}ms total={4}ms provider={5}" -f $row.Status, $row.Preflight, $row.StageMs, $row.Pipeline, $row.Total, $row.Provider)
    }
}

Write-Host ""
Write-Host "=========== PER-STAGE MATRIX (Cpu pin, short.mp4) ==========="
$rows | Format-Table -AutoSize
$rows | Export-Csv (Join-Path $outRoot "per-stage-matrix.csv") -NoTypeInformation
Write-Host "Saved: $(Join-Path $outRoot 'per-stage-matrix.csv')"
