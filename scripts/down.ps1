<#
.SYNOPSIS
    Stop and remove the platform's containers (equivalent of `make down`).
.EXAMPLE
    .\scripts\down.ps1
    .\scripts\down.ps1 -Volumes
#>
[CmdletBinding()]
param(
    # Also remove named and anonymous volumes.
    [switch]$Volumes
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

$composeArgs = @("down", "--remove-orphans")
if ($Volumes) { $composeArgs += "--volumes" }

Invoke-Compose $composeArgs
