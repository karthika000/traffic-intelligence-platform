<#
.SYNOPSIS
    Show container logs (equivalent of `make logs`).
.EXAMPLE
    .\scripts\logs.ps1
    .\scripts\logs.ps1 -Services api,fusion_service
    .\scripts\logs.ps1 -NoFollow -Tail 500
#>
[CmdletBinding()]
param(
    # Subset of services; defaults to all.
    [string[]]$Services = @(),
    # Number of lines to show from the end of each log.
    [int]$Tail = 100,
    # Print and exit instead of streaming.
    [switch]$NoFollow
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

$composeArgs = @("logs", "--tail", "$Tail")
if (-not $NoFollow) { $composeArgs += "--follow" }
$composeArgs += $Services

Invoke-Compose $composeArgs
