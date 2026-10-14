<#
.SYNOPSIS
    Create the platform's Kafka topics and per-consumer-group dead-letter topics if they
    don't already exist. Safe to re-run.
.DESCRIPTION
    Keep these lists in sync with shared/src/traffic_common/topics.py
    (tests/test_schemas.py fails if they drift apart).
.EXAMPLE
    .\scripts\create_topics.ps1
    .\scripts\create_topics.ps1 -Partitions 6
#>
[CmdletBinding()]
param(
    # Partitions for newly created data topics; existing topics are left untouched.
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
    "summaries",
    "drift.alerts"
)

# One dead-letter topic per consumer group: dlq.<group>.
$ConsumerGroups = @(
    "cv-inference",
    "nlp-service",
    "anomaly-service",
    "fusion-service",
    "summarizer",
    "api",
    "ml-monitoring"
)
$DeadLetterTopics = $ConsumerGroups | ForEach-Object { "dlq.$_" }
# Low volume, inspected by humans: one partition, kept for 14 days.
$DeadLetterRetentionMs = 14 * 24 * 60 * 60 * 1000

$kafkaTopics = "/opt/kafka/bin/kafka-topics.sh"
$bootstrap = "kafka:29092"

Wait-ServiceHealthy -Service kafka -TimeoutSec $TimeoutSec

# One JVM start to list, then only create what's missing (each kafka-topics.sh call takes a few seconds).
$existing = Get-ComposeOutput @("exec", "-T", "kafka", $kafkaTopics, "--bootstrap-server", $bootstrap, "--list")

$plan = @()
$plan += $Topics | ForEach-Object { @{ Name = $_; Partitions = $Partitions; Config = @() } }
$plan += $DeadLetterTopics | ForEach-Object { @{ Name = $_; Partitions = 1; Config = @("--config", "retention.ms=$DeadLetterRetentionMs") } }
$missing = @($plan | Where-Object { $existing -notcontains $_.Name })

if (-not $missing) {
    Write-Host "All $($plan.Count) topics already exist ($($Topics.Count) data, $($DeadLetterTopics.Count) dead-letter)." -ForegroundColor Green
    return
}

foreach ($t in $missing) {
    Write-Host "Creating topic $($t.Name) ($($t.Partitions) partition(s))"
    Invoke-Compose (@("exec", "-T", "kafka", $kafkaTopics, "--bootstrap-server", $bootstrap,
            "--create", "--if-not-exists", "--topic", $t.Name,
            "--partitions", "$($t.Partitions)", "--replication-factor", "1") + $t.Config)
}
Write-Host "Created $($missing.Count) topic(s); $($plan.Count - $missing.Count) already existed." -ForegroundColor Green
