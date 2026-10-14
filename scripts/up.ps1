<#
.SYNOPSIS
    Build and start the platform (equivalent of `make up`), then create Kafka topics.
.EXAMPLE
    .\scripts\up.ps1
    .\scripts\up.ps1 -Services api,dashboard
    .\scripts\up.ps1 -NoBuild -Attach
#>
[CmdletBinding()]
param(
    # Subset of services to start; defaults to all. Dependencies (Kafka, Postgres) start automatically.
    [string[]]$Services = @(),
    # Skip rebuilding images.
    [switch]$NoBuild,
    # Stream logs in the foreground after startup; Ctrl+C stops the containers.
    [switch]$Attach
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envFile
    Write-Host "Created .env from .env.example" -ForegroundColor Yellow
}

# Always start detached first so topics can be created once Kafka is healthy.
$composeArgs = @("up", "-d")
if (-not $NoBuild) { $composeArgs += "--build" }
$composeArgs += $Services
Invoke-Compose $composeArgs

if (Get-ComposeOutput @("ps", "-q", "kafka")) {
    & "$PSScriptRoot\create_topics.ps1"
}
else {
    Write-Host "Kafka is not part of this run; skipping topic creation." -ForegroundColor Yellow
}

Invoke-Compose @("ps")

$envVars = @{}
Get-Content $envFile | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*([^#]*)' } | ForEach-Object { $envVars[$Matches[1]] = $Matches[2].Trim() }
function Get-Port([string]$Name, [string]$Default) { if ($envVars[$Name]) { $envVars[$Name] } else { $Default } }

Write-Host ""
Write-Host "Dashboard  http://localhost:$(Get-Port DASHBOARD_PORT 5173)" -ForegroundColor Green
Write-Host "API        http://localhost:$(Get-Port API_PORT 8001)" -ForegroundColor Green
Write-Host "Kafka UI   http://localhost:$(Get-Port KAFKA_UI_PORT 8080)" -ForegroundColor Green
Write-Host "MLflow     http://localhost:$(Get-Port MLFLOW_PORT 5000)" -ForegroundColor Green
Write-Host "Adminer    http://localhost:$(Get-Port ADMINER_PORT 8081)" -ForegroundColor Green
Write-Host "Kafka      localhost:$(Get-Port KAFKA_PORT 9092)   Postgres  localhost:$(Get-Port POSTGRES_PORT 5432)" -ForegroundColor Green
Write-Host "Logs       .\scripts\logs.ps1"

if ($Attach) {
    # Containers are already up; this only attaches to their output.
    Invoke-Compose (@("up") + $Services)
}
