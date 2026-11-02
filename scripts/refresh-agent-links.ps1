<#
.SYNOPSIS
    Re-creates hardlinks for agent CLIs into a machine-PATH directory.

.DESCRIPTION
    Run this after any agent CLI self-updates. Updates that replace the
    binary break hardlinks, leaving the linked copy stale.

.PARAMETER Dest
    Destination directory for the hardlinked shims. Defaults to the
    per-user Python Scripts dir already on PATH.

.PARAMETER Pause
    Wait for Enter before exiting (double-click friendly).

.EXAMPLE
    .\scripts\refresh-agent-links.ps1
#>
[CmdletBinding()]
param(
    [string]$Dest = "$env:LOCALAPPDATA\Programs\Python\Python314\Scripts",
    [switch]$Pause
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$userBin  = "$env:USERPROFILE\.local\bin"
$cargoBin = "$env:USERPROFILE\.cargo\bin"
$hermesBin = "$env:LOCALAPPDATA\hermes\bin"
$agyBin   = "$env:LOCALAPPDATA\agy\bin"

$tools = @(
    [pscustomobject]@{ Name = 'opencode';              Dir = "$env:USERPROFILE\.opencode\bin" }
    [pscustomobject]@{ Name = 'claude';                Dir = $userBin }
    [pscustomobject]@{ Name = 'aider';                 Dir = $userBin }
    [pscustomobject]@{ Name = 'goose';                 Dir = $userBin }
    [pscustomobject]@{ Name = 'openhands';             Dir = $userBin }
    [pscustomobject]@{ Name = 'openhands-acp';         Dir = $userBin }
    [pscustomobject]@{ Name = 'agents-cli';            Dir = $userBin }
    [pscustomobject]@{ Name = 'agentpool';             Dir = $userBin }
    [pscustomobject]@{ Name = 'agentpool-db';          Dir = $userBin }
    [pscustomobject]@{ Name = 'acp-bridge';            Dir = $userBin }
    [pscustomobject]@{ Name = 'ori';                   Dir = $userBin }
    [pscustomobject]@{ Name = 'vibe';                  Dir = $userBin }
    [pscustomobject]@{ Name = 'vibe-acp';              Dir = $userBin }
    [pscustomobject]@{ Name = 'vibe-app-server';       Dir = $userBin }
    [pscustomobject]@{ Name = 'serena';                Dir = $userBin }
    [pscustomobject]@{ Name = 'serena-agent';          Dir = $userBin }
    [pscustomobject]@{ Name = 'serena-hooks';          Dir = $userBin }
    [pscustomobject]@{ Name = 'agent-browser';         Dir = $cargoBin }
    [pscustomobject]@{ Name = 'opencode-wrap';         Dir = $cargoBin }
    [pscustomobject]@{ Name = 'hermes';                Dir = $hermesBin }
    [pscustomobject]@{ Name = 'hermes-acp';            Dir = $hermesBin }
    [pscustomobject]@{ Name = 'codex';                 Dir = "$env:LOCALAPPDATA\Programs\OpenAI\Codex\bin" }
    [pscustomobject]@{ Name = 'qodo';                  Dir = "$env:USERPROFILE\.qodo\bin" }
    [pscustomobject]@{ Name = 'agy';                   Dir = $agyBin }
    [pscustomobject]@{ Name = 'agy_acp_server';        Dir = $agyBin }
    [pscustomobject]@{ Name = 'localharness_external'; Dir = $agyBin }
)

function Test-AlreadyLinked {
    param(
        [string]$Source,
        [string]$Target
    )
    $links = fsutil hardlink list $Source 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    # Hardlinks only exist on the source volume; a target on another drive cannot be linked.
    $srcRoot = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Source))
    $tgtRoot = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Target))
    if (-not [string]::Equals($srcRoot, $tgtRoot, [System.StringComparison]::OrdinalIgnoreCase)) { return $false }
    # fsutil may emit volume-relative paths (\Users\...); compare drive-less tails.
    $targetKey = [System.IO.Path]::GetFullPath($Target).Substring(2)
    foreach ($link in $links) {
        $candidate = $link.Trim()
        if ($candidate.Length -gt 2 -and $candidate[1] -eq ':') {
            $candidate = $candidate.Substring(2)
        }
        if ([string]::Equals($candidate, $targetKey, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Invoke-Link {
    param(
        [string]$Name,
        [string]$Dir
    )
    $src = Join-Path $Dir "$Name.exe"
    $dst = Join-Path $Dest "$Name.exe"

    if (-not (Test-Path -LiteralPath $src)) {
        return [pscustomobject]@{ Name = $Name; Status = 'SKIP'; Detail = "source missing: $Dir" }
    }

    fsutil hardlink create $dst $src >$null 2>$null
    if ($LASTEXITCODE -eq 0) {
        return [pscustomobject]@{ Name = $Name; Status = 'OK'; Detail = '' }
    }

    if (Test-AlreadyLinked -Source $src -Target $dst) {
        return [pscustomobject]@{ Name = $Name; Status = 'LINK'; Detail = 'already linked' }
    }

    # Build the replacement link beside the destination first, then swap it in.
    # This leaves the existing shim untouched when link creation fails (cross-volume, locked, access denied).
    $tmp = Join-Path $Dest ("{0}.{1}.tmp" -f $Name, [guid]::NewGuid().ToString('N'))
    fsutil hardlink create $tmp $src >$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        return [pscustomobject]@{ Name = $Name; Status = 'FAIL'; Detail = 'hardlink creation failed (locked, cross-volume, or access denied)' }
    }

    try {
        Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $tmp -Destination $dst -Force
        return [pscustomobject]@{ Name = $Name; Status = 'OK'; Detail = 'refreshed' }
    }
    catch {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        return [pscustomobject]@{ Name = $Name; Status = 'FAIL'; Detail = $_.Exception.Message }
    }
}

if (-not (Test-Path -LiteralPath $Dest)) {
    throw "Destination '$Dest' does not exist. Pass -Dest to point at your on-PATH shim directory."
}

Write-Host "Linking into $Dest`n"

$results = foreach ($tool in $tools) {
    Invoke-Link -Name $tool.Name -Dir $tool.Dir
}

foreach ($r in $results) {
    $detail = if ($r.Detail) { " ($($r.Detail))" } else { '' }
    Write-Host ("{0,-4} {1}{2}" -f $r.Status, $r.Name, $detail)
}

$ok   = @($results | Where-Object Status -eq 'OK').Count
$link = @($results | Where-Object Status -eq 'LINK').Count
$skip = @($results | Where-Object Status -eq 'SKIP').Count
$fail = @($results | Where-Object Status -eq 'FAIL').Count

Write-Host "`nDone: $ok linked, $link already linked, $skip skipped, $fail failed."
if ($fail -gt 0) {
    Write-Host "FAIL lines mean the hardlink could not be created. Check the Detail column for the cause."
}
if ($Pause) {
    Read-Host "Press Enter to exit"
}
if ($fail -gt 0) { exit 1 }
