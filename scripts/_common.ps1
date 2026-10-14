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
