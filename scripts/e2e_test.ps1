<#
.SYNOPSIS
    End-to-end test of the running stack: trigger all 4 scenarios on different cameras and check
    that each one produces a confirmed incident of the right type, with a summary, served by the
    API, that resolves. Prints a pass/fail table with timings; exits non-zero on failure.
.DESCRIPTION
    Needs the stack running and healthy (.\scripts\run_demo.ps1). The checker
    (scripts/e2e_check.py, standard library only) runs in the api image on the compose network,
    so no Python is needed on the host. Results are also written to data/e2e/<time>.json.
.EXAMPLE
    .\scripts\e2e_test.ps1
    .\scripts\e2e_test.ps1 -TimeScale 10 -TimeoutMinutes 20
    .\scripts\e2e_test.ps1 -Scenarios flooding,road_works
#>
[CmdletBinding()]
param(
    # Scenario speed-up (6 runs an 18-minute scenario in 3).
    [double]$TimeScale = 6,
    # Give up on runs that haven't passed after this long.
    [int]$TimeoutMinutes = 30,
    # Subset of scenarios; default all.
    [string[]]$Scenarios = @()
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

# Every long-running service must be healthy first: a failure should be about the pipeline,
# not a service that is still starting.
$services = Get-ComposeOutput @("ps", "--format", "{{.Service}} {{.Health}}")
$notHealthy = @($services | Where-Object { $_ -match '^(\S+)\s+(\S*)$' -and $Matches[2] -ne "healthy" } | ForEach-Object { ($_ -split ' ')[0] })
if (-not $services) { throw "The stack is not running. Start it with .\scripts\run_demo.ps1" }
if ($notHealthy) {
    throw "Not healthy yet: $($notHealthy -join ', '). Wait, or check .\scripts\logs.ps1 -Services $($notHealthy -join ',')"
}

$outDir = Join-Path $RepoRoot "data\e2e"
New-Item -ItemType Directory -Force $outDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

$checkerArgs = @("run", "--rm", "--no-deps", "-T",
    "-v", "$($RepoRoot -replace '\\', '/')/scripts:/e2e:ro",
    "-v", "$($outDir -replace '\\', '/'):/out",
    "api", "python", "/e2e/e2e_check.py",
    "--api", "http://api:8000",
    "--time-scale", "$TimeScale",
    "--timeout-minutes", "$TimeoutMinutes",
    "--json", "/out/$stamp.json")
if ($Scenarios) { $checkerArgs += @("--scenarios", ($Scenarios -join ",")) }

Write-Host "End-to-end test: all scenarios at x$TimeScale, timeout $TimeoutMinutes min" -ForegroundColor Cyan
$started = Get-Date
Push-Location $RepoRoot
$ErrorActionPreference = "Continue"
try {
    & docker compose @checkerArgs
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}
$minutes = ((Get-Date) - $started).TotalMinutes
Write-Host ""
Write-Host ("Results: data\e2e\$stamp.json ({0:N1} min)" -f $minutes)
if ($code -eq 0) {
    Write-Host "E2E PASSED" -ForegroundColor Green
}
else {
    Write-Host "E2E FAILED (exit $code)" -ForegroundColor Red
}
exit $code
