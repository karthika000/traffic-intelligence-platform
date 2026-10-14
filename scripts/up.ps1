<#
.SYNOPSIS
    Build and start the platform (equivalent of `make up`).
.EXAMPLE
    .\scripts\up.ps1
    .\scripts\up.ps1 -Services api,dashboard
    .\scripts\up.ps1 -NoBuild -Attach
#>
[CmdletBinding()]
param(
    # Subset of services to start; defaults to all.
    [string[]]$Services = @(),
    # Skip rebuilding images.
    [switch]$NoBuild,
    # Run in the foreground instead of detached.
    [switch]$Attach
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envFile
    Write-Host "Created .env from .env.example" -ForegroundColor Yellow
}

$composeArgs = @("up")
if (-not $NoBuild) { $composeArgs += "--build" }
if (-not $Attach) { $composeArgs += "-d" }
$composeArgs += $Services

Invoke-Compose $composeArgs

if (-not $Attach) {
    Invoke-Compose @("ps")
    $envVars = @{}
    Get-Content $envFile | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$' } | ForEach-Object { $envVars[$Matches[1]] = $Matches[2].Trim() }
    $apiPort = if ($envVars.API_PORT) { $envVars.API_PORT } else { "8001" }
    $dashPort = if ($envVars.DASHBOARD_PORT) { $envVars.DASHBOARD_PORT } else { "5173" }
    Write-Host "`nDashboard: http://localhost:$dashPort  |  API: http://localhost:$apiPort  |  Logs: .\scripts\logs.ps1" -ForegroundColor Green
}
