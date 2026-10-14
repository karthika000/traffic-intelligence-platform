# Shared helpers dot-sourced by the other scripts.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Assert-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker was not found on PATH. Install Docker Desktop first."
    }
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker daemon is not running. Start Docker Desktop and try again."
    }
}

function Invoke-Compose {
    # Pass compose args as a single array; splatting flags like "-d" would bind to common parameters.
    param([string[]]$Arguments)
    Push-Location $RepoRoot
    # docker writes progress to stderr; under "Stop", PowerShell 5.1 treats redirected stderr as a terminating error.
    # Failures are detected via $LASTEXITCODE below instead.
    $ErrorActionPreference = "Continue"
    try {
        & docker compose @Arguments
        if ($LASTEXITCODE -ne 0) { throw "docker compose $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
    }
    finally {
        Pop-Location
    }
}

function Get-ComposeOutput {
    # Like Invoke-Compose, but returns stdout lines instead of printing them. stderr is discarded.
    param([string[]]$Arguments)
    Push-Location $RepoRoot
    $ErrorActionPreference = "Continue"
    try {
        $out = & docker compose @Arguments 2> $null
        if ($LASTEXITCODE -ne 0) { throw "docker compose $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
        return @($out | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    }
    finally {
        Pop-Location
    }
}

function Wait-ServiceHealthy {
    param([string]$Service, [int]$TimeoutSec = 180)
    # @() because PowerShell unrolls a single-item array returned from a function into a plain string.
    $id = @(Get-ComposeOutput @("ps", "-q", $Service))
    if (-not $id) { throw "Service '$Service' is not running. Start it with .\scripts\up.ps1 first." }

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    Write-Host "Waiting for $Service to become healthy" -NoNewline
    while ($true) {
        $status = (& docker inspect --format "{{.State.Health.Status}}" $id[0]).Trim()
        if ($status -eq "healthy") { Write-Host " ok" -ForegroundColor Green; return }
        if ((Get-Date) -gt $deadline) {
            Write-Host ""
            throw "$Service is still '$status' after ${TimeoutSec}s. Check: .\scripts\logs.ps1 -Services $Service"
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 3
    }
}
