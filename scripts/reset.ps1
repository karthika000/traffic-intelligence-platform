<#
.SYNOPSIS
    Tear the platform down to a clean state, so .\scripts\run_demo.ps1 starts from scratch:
    containers, ALL volumes (Kafka, Postgres, MLflow models and runs, frames, Hugging Face and
    YOLO caches, drift reports) and the generated data under data/ and models/.
.DESCRIPTION
    Asks for confirmation first (type "reset"), unless -Force. Images and Docker's build cache
    are kept unless -Images, so the next run doesn't rebuild everything. .env is kept unless
    -IncludeEnv (run_demo.ps1 recreates it from .env.example).
.EXAMPLE
    .\scripts\reset.ps1
    .\scripts\reset.ps1 -KeepVideos        # keep the downloaded sample videos
    .\scripts\reset.ps1 -Force -Images     # no prompt; also remove the built images
#>
[CmdletBinding()]
param(
    # Don't ask for confirmation.
    [switch]$Force,
    # Keep data/videos (the downloaded sample videos and their attribution).
    [switch]$KeepVideos,
    # Also remove the images built for this project.
    [switch]$Images,
    # Also delete .env.
    [switch]$IncludeEnv
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

$project = "traffic-intel"
$envFile = Join-Path $RepoRoot ".env"
if (Test-Path $envFile) {
    $line = Get-Content $envFile | Where-Object { $_ -match '^\s*COMPOSE_PROJECT_NAME\s*=\s*(\S+)' } | Select-Object -First 1
    if ($line -and $line -match '=\s*(\S+)') { $project = $Matches[1] }
}

# What will go: generated data (everything under data/ and models/ except the placeholders).
$keep = @(".gitkeep", "README.md")
$dataItems = @()
foreach ($dir in @("data", "models")) {
    $path = Join-Path $RepoRoot $dir
    if (-not (Test-Path $path)) { continue }
    $dataItems += Get-ChildItem -Force $path | Where-Object {
        $keep -notcontains $_.Name -and -not ($KeepVideos -and $dir -eq "data" -and $_.Name -eq "videos")
    }
}
$volumes = @(& docker volume ls -q --filter "label=com.docker.compose.project=$project" 2> $null)
$containers = @(& docker ps -aq --filter "label=com.docker.compose.project=$project" 2> $null)

Write-Host "This permanently deletes, for compose project '$project':" -ForegroundColor Yellow
Write-Host "  containers: $($containers.Count)"
Write-Host "  volumes:    $(if ($volumes) { $volumes -join ', ' } else { 'none' })"
Write-Host "              (Kafka topics, Postgres incl. MLflow runs and registered models, MLflow artifacts,"
Write-Host "               frames, Hugging Face / YOLO caches, drift reports)"
foreach ($item in $dataItems) {
    $size = if ($item.PSIsContainer) {
        (Get-ChildItem -Recurse -Force -File $item.FullName -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    }
    else { $item.Length }
    Write-Host ("  {0,-10}  {1} ({2:N0} MB)" -f "data:", $item.FullName.Substring($RepoRoot.Length + 1), ($size / 1MB))
}
if (-not $dataItems) { Write-Host "  data:       nothing generated yet" }
if ($Images) { Write-Host "  images:     the ones built for this project" }
if ($IncludeEnv -and (Test-Path $envFile)) { Write-Host "  .env" }
if ($KeepVideos) { Write-Host "  (keeping data\videos)" }

if (-not $Force) {
    $answer = Read-Host "Type 'reset' to continue"
    if ($answer -ne "reset") {
        Write-Host "Cancelled; nothing was deleted."
        exit 1
    }
}

$downArgs = @("--profile", "ml", "--profile", "test", "down", "--volumes", "--remove-orphans", "--timeout", "20")
if ($Images) { $downArgs += @("--rmi", "local") }
Invoke-Compose $downArgs

# Volumes compose no longer knows about (e.g. from a renamed service) carry the project label too.
$left = @(& docker volume ls -q --filter "label=com.docker.compose.project=$project" 2> $null)
foreach ($v in $left) { & docker volume rm $v | Out-Null }

foreach ($item in $dataItems) {
    Remove-Item -Recurse -Force $item.FullName
}
if ($IncludeEnv -and (Test-Path $envFile)) { Remove-Item -Force $envFile }

Write-Host "Reset complete. Start again with .\scripts\run_demo.ps1" -ForegroundColor Green
