<#
.SYNOPSIS
    Run the test suite. By default it runs in a Linux container on the compose network,
    against the running Kafka and Postgres (needed on Windows when Smart App Control
    blocks confluent-kafka). -Local runs it in .venv instead (Kafka tests may skip).
.EXAMPLE
    .\scripts\test.ps1
    .\scripts\test.ps1 tests/test_messaging.py -x -q
    .\scripts\test.ps1 -Local -k schemas
#>
# Deliberately not [CmdletBinding()]: unbound arguments (pytest flags such as -k, -x, -q)
# pass through in $args instead of being parsed as PowerShell parameters.
param([switch]$Local)

. "$PSScriptRoot\_common.ps1"

$pytestArgs = @($args)

if ($Local) {
    $python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { throw "No .venv found. See README > Local development." }
    Push-Location $RepoRoot
    try {
        & $python -m pytest -p no:cacheprovider @pytestArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

Assert-Docker
# Container paths use forward slashes.
$pytestArgs = $pytestArgs | ForEach-Object { "$_" -replace '\\', '/' }
Invoke-Compose (@("--profile", "test", "run", "--rm", "--build", "tests") + $pytestArgs)
