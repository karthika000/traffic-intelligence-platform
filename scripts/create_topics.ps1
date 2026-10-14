<#
.SYNOPSIS
    Create the platform's Kafka topics if they don't already exist. Safe to re-run.
.EXAMPLE
    .\scripts\create_topics.ps1
    .\scripts\create_topics.ps1 -Partitions 6
#>
[CmdletBinding()]
param(
    # Partitions for newly created topics; existing topics are left untouched.
    [int]$Partitions = 3,
    # How long to wait for Kafka to report healthy.
    [int]$TimeoutSec = 180
)

. "$PSScriptRoot\_common.ps1"
Assert-Docker

$Topics = @(
    "video.frames",
    "events.raw",
    "reports.text",
    "detections",
    "traffic.metrics",
    "anomalies",
    "nlp.results",
    "incidents",
    "summaries"
)

$kafkaTopics = "/opt/kafka/bin/kafka-topics.sh"
$bootstrap = "kafka:29092"

Wait-ServiceHealthy -Service kafka -TimeoutSec $TimeoutSec

# One JVM start to list, then only create what's missing (each kafka-topics.sh call takes a few seconds).
$existing = Get-ComposeOutput @("exec", "-T", "kafka", $kafkaTopics, "--bootstrap-server", $bootstrap, "--list")
$missing = $Topics | Where-Object { $existing -notcontains $_ }

if (-not $missing) {
    Write-Host "All $($Topics.Count) topics already exist." -ForegroundColor Green
    return
}

foreach ($topic in $missing) {
    Write-Host "Creating topic $topic ($Partitions partitions)"
    Invoke-Compose @("exec", "-T", "kafka", $kafkaTopics, "--bootstrap-server", $bootstrap,
        "--create", "--if-not-exists", "--topic", $topic,
        "--partitions", "$Partitions", "--replication-factor", "1")
}
Write-Host "Created $($missing.Count) topic(s); $($Topics.Count - $missing.Count) already existed." -ForegroundColor Green
