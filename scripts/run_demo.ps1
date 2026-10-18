<#
.SYNOPSIS
    Start the whole platform and its demo with one command, from a fresh clone: prerequisites,
    .env, images, infrastructure, topics, database, sample data, every model, drift references,
    all services (waiting until each is healthy), then the dashboard.
.DESCRIPTION
    Safe to re-run: every step skips what already exists (models with a champion, datasets,
    videos, references). Each stage prints a rough time estimate. From a clean state (reset.ps1)
    with the images already built it took 19.6 minutes on a Core Ultra 5 225H with 14 vCPUs:
    ~10 for the DistilBERT text classifier (-QuickModels trains the TF-IDF baseline instead)
    and 5 for the vehicle-detector drift reference, which needs a few minutes of live output.
    Building the images the first time adds to that. A re-run with everything present takes
    under a minute.
.EXAMPLE
    .\scripts\run_demo.ps1
    .\scripts\run_demo.ps1 -Scenario multi_vehicle_accident                 # trigger one after the warm-up
    .\scripts\run_demo.ps1 -Scenario flooding -ScenarioCamera cam-old-street -TimeScale 10
    .\scripts\run_demo.ps1 -Retrain                                          # retrain every model
    .\scripts\run_demo.ps1 -QuickModels -NoBrowser
#>
[CmdletBinding()]
param(
    # Retrain and re-register every trainable model even if it has a champion.
    [switch]$Retrain,
    # Faster first run: the TF-IDF text classifier instead of DistilBERT (seconds instead of ~10 min).
    [switch]$QuickModels,
    # Trigger this scenario (multi_vehicle_accident, broken_down_lorry, flooding, road_works) after the warm-up.
    [string]$Scenario = "",
    [string]$ScenarioCamera = "cam-hyde-park-corner",
    [double]$TimeScale = 6,
    # Minutes of running before -Scenario is triggered: the anomaly detector compares each camera
    # with its own last ~10 minutes, so earlier incidents are seen by the text reports only.
    [int]$WarmupMinutes = 10,
    # Minutes of clean detections the vehicle-detector drift reference is built from.
    [int]$ReferenceMinutes = 5,
    # How long to wait for all services to become healthy.
    [int]$ServiceTimeoutMinutes = 25,
    # Docker VM memory below this fails the prerequisite check.
    [double]$MinMemoryGB = 6,
    [switch]$NoBrowser,
    # Skip rebuilding images (they are rebuilt from cache by default, which takes seconds).
    [switch]$NoBuild
)

. "$PSScriptRoot\_common.ps1"
$Started = Get-Date
$Stages = New-Object System.Collections.ArrayList
$script:StageNote = ""

# ------------------------------------------------------------------------------- helpers

function Format-Duration([double]$Seconds) {
    $s = [int][math]::Round($Seconds)
    if ($s -lt 60) { return "${s}s" }
    if ($s -lt 3600) { return "{0}m{1:00}s" -f [math]::Floor($s / 60), ($s % 60) }
    return "{0}h{1:00}m" -f [math]::Floor($s / 3600), [math]::Floor(($s % 3600) / 60)
}

function Invoke-Stage {
    param([string]$Name, [string]$Estimate, [scriptblock]$Body)
    $n = $Stages.Count + 1
    Write-Host ""
    Write-Host ("==> [{0:00}] {1}" -f $n, $Name) -ForegroundColor Cyan -NoNewline
    if ($Estimate) { Write-Host "   ($Estimate)" -ForegroundColor DarkGray } else { Write-Host "" }
    $script:StageNote = ""
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        & $Body | Write-Lines
    }
    catch {
        $sw.Stop()
        [void]$Stages.Add([pscustomobject]@{ Stage = $Name; Time = Format-Duration $sw.Elapsed.TotalSeconds; Seconds = $sw.Elapsed.TotalSeconds; Result = "FAILED" })
        throw
    }
    $sw.Stop()
    $result = if ($script:StageNote) { $script:StageNote } else { "ok" }
    [void]$Stages.Add([pscustomobject]@{ Stage = $Name; Time = Format-Duration $sw.Elapsed.TotalSeconds; Seconds = $sw.Elapsed.TotalSeconds; Result = $result })
    Write-Host ("    {0} in {1}" -f $result, (Format-Duration $sw.Elapsed.TotalSeconds)) -ForegroundColor DarkGray
}

function Write-Lines {
    # Print pipeline output through Write-Host rather than Out-Host, so it also lands in a log
    # redirected with *> (Out-Host writes to the console host, bypassing the redirection).
    process { Write-Host ($_ | Out-String -Width 160).TrimEnd() }
}

function Invoke-Native {
    # Run a native command without PowerShell 5.1 turning its stderr into a terminating error.
    param([scriptblock]$Command)
    $ErrorActionPreference = "Continue"
    & $Command
}

function Read-EnvFile([string]$Path) {
    $map = @{}
    if (Test-Path $Path) {
        Get-Content $Path | Where-Object { $_ -match '^\s*([A-Z0-9_]+)\s*=\s*([^#]*)' } | ForEach-Object {
            $map[$Matches[1]] = $Matches[2].Trim()
        }
    }
    return $map
}

function Get-Setting([string]$Name, [string]$Default) {
    if ($EnvVars[$Name]) { return $EnvVars[$Name] }
    return $Default
}

function Invoke-Trainer {
    # A one-off ML command in the ml-trainer image (ml/ and data/ are mounted), output streamed.
    param([string[]]$Command)
    Invoke-Compose (@("run", "--rm", "-T", "ml-trainer") + $Command)
}

function Get-Champion([string]$Model) {
    # The registry version behind <model>@champion, or $null.
    $name = [uri]::EscapeDataString($Model)
    try {
        $r = Invoke-RestMethod -TimeoutSec 15 -Uri "$MlflowUrl/api/2.0/mlflow/registered-models/alias?name=$name&alias=champion"
        return "$($r.model_version.version)"
    }
    catch {
        return $null
    }
}

function Test-Reference([string]$Model, [string]$Version) {
    return Test-Path (Join-Path $RepoRoot "data\ml\reference\$Model\v$Version\meta.json")
}

function Get-ServiceStates {
    # Service name -> state, health and exit code of its container, from docker compose ps.
    Push-Location $RepoRoot
    try {
        $raw = Invoke-Native { docker compose ps -a --format json 2> $null }
    }
    finally {
        Pop-Location
    }
    $states = @{}
    foreach ($line in @($raw)) {
        $text = "$line".Trim()
        if (-not $text) { continue }
        $items = $text | ConvertFrom-Json  # one object per line (newer compose) or one array
        foreach ($c in @($items)) {
            $states[$c.Service] = [pscustomobject]@{ State = $c.State; Health = $c.Health; ExitCode = $c.ExitCode }
        }
    }
    return $states
}

function Show-Logs([string[]]$Names, [int]$Tail = 40) {
    foreach ($n in $Names) {
        Write-Host "---- last $Tail log lines of $n ----" -ForegroundColor Yellow
        Push-Location $RepoRoot
        try { Invoke-Native { docker compose logs --no-color --tail $Tail $n 2>&1 } | Write-Lines }
        finally { Pop-Location }
    }
}

function Wait-Infrastructure([string[]]$Names, [int]$TimeoutSec = 300) {
    foreach ($n in $Names) { Wait-ServiceHealthy -Service $n -TimeoutSec $TimeoutSec }
}

# ------------------------------------------------------------------------------- stages

$EnvPath = Join-Path $RepoRoot ".env"
$EnvVars = Read-EnvFile $(if (Test-Path $EnvPath) { $EnvPath } else { Join-Path $RepoRoot ".env.example" })
$Ports = [ordered]@{
    DASHBOARD_PORT = 5173; API_PORT = 8001; KAFKA_PORT = 9092; KAFKA_UI_PORT = 8080
    POSTGRES_PORT = 5432; MLFLOW_PORT = 5000; ADMINER_PORT = 8081
}
foreach ($k in @($Ports.Keys)) { $Ports[$k] = [int](Get-Setting $k $Ports[$k]) }
$Project = Get-Setting "COMPOSE_PROJECT_NAME" "traffic-intel"
$ApiUrl = "http://127.0.0.1:$($Ports.API_PORT)"
$MlflowUrl = "http://127.0.0.1:$($Ports.MLFLOW_PORT)"
$DashboardUrl = "http://localhost:$($Ports.DASHBOARD_PORT)"
$ScenarioNames = @(Get-ChildItem (Join-Path $RepoRoot "services\event_simulator\scenarios") -Filter *.yaml | ForEach-Object { $_.BaseName })
if ($Scenario -and $ScenarioNames -notcontains $Scenario) {
    throw "Unknown scenario '$Scenario'. Available: $($ScenarioNames -join ', ')"
}

try {
    Invoke-Stage "Check prerequisites" "a few seconds" {
        $problems = @()
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "docker was not found on PATH. Install Docker Desktop (WSL 2 backend): https://docs.docker.com/desktop/setup/install/windows-install/"
        }
        $info = Invoke-Native { docker info --format "{{.MemTotal}} {{.NCPU}} {{.ServerVersion}}" 2> $null }
        if ($LASTEXITCODE -ne 0 -or -not $info) {
            throw "Docker is installed but not running. Start Docker Desktop, wait until it says 'Engine running', and re-run this script."
        }
        $memBytes, $cpus, $server = "$info".Trim() -split '\s+'
        $memGB = [double]$memBytes / 1GB
        Write-Host ("    Docker {0}: {1:N1} GB memory, {2} CPUs" -f $server, $memGB, $cpus)
        if ($memGB -lt $MinMemoryGB) {
            $problems += ("Docker's VM has {0:N1} GB of memory; at least {1} GB is needed (8 recommended): the CV, NLP and summarizer services plus training use about 5 GB together.`n" +
                "      Fix (WSL 2 backend): put these lines in $env:USERPROFILE\.wslconfig, then run 'wsl --shutdown' and restart Docker Desktop:`n" +
                "        [wsl2]`n        memory=8GB`n" +
                "      Fix (Hyper-V backend): Docker Desktop > Settings > Resources > Memory.") -f $memGB, $MinMemoryGB
        }
        if ([int]$cpus -lt 4) { Write-Host "    warning: only $cpus CPUs for Docker; inference and training will be slow." -ForegroundColor Yellow }

        $composeVersion = "$(Invoke-Native { docker compose version --short 2> $null })".Trim().TrimStart('v')
        if (-not $composeVersion) {
            $problems += "'docker compose' (Compose v2) is not available. Update Docker Desktop."
        }
        elseif ([version]($composeVersion -replace '[^0-9.].*$', '') -lt [version]"2.24") {
            $problems += "Docker Compose $composeVersion is too old (2.24+ needed for optional env files). Update Docker Desktop."
        }

        $drive = Get-PSDrive -Name ($env:LOCALAPPDATA.Substring(0, 1)) -ErrorAction SilentlyContinue
        if ($drive -and $drive.Free -lt 20GB) {
            Write-Host ("    warning: {0:N0} GB free on {1}: where Docker keeps its disk image. A clean run needs about 20 GB (images ~10 GB, model downloads ~2 GB, data)." -f ($drive.Free / 1GB), $drive.Name) -ForegroundColor Yellow
        }

        # Ports: free, or already published by this project's own containers (a re-run).
        $ours = @{}
        foreach ($line in @(Invoke-Native { docker ps --filter "label=com.docker.compose.project=$Project" --format "{{.Ports}}" 2> $null })) {
            foreach ($m in [regex]::Matches("$line", ':(\d+)->')) { $ours[[int]$m.Groups[1].Value] = $true }
        }
        $reserved = @()
        foreach ($line in @(Invoke-Native { netsh interface ipv4 show excludedportrange protocol=tcp 2> $null })) {
            if ("$line" -match '^\s*(\d+)\s+(\d+)') { $reserved += , @([int]$Matches[1], [int]$Matches[2]) }
        }
        foreach ($name in $Ports.Keys) {
            $port = $Ports[$name]
            if ($ours[$port]) { continue }
            $range = $reserved | Where-Object { $port -ge $_[0] -and $port -le $_[1] } | Select-Object -First 1
            if ($range) {
                $problems += "Port $port ($name) is in a range Windows reserves ($($range[0])-$($range[1]); see 'netsh interface ipv4 show excludedportrange protocol=tcp'). Fix: set $name to another port in .env."
                continue
            }
            $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($listener) {
                $proc = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
                $who = if ($proc) { "$($proc.ProcessName) (pid $($proc.Id))" } else { "pid $($listener.OwningProcess)" }
                $hint = if ($proc -and $proc.ProcessName -match 'docker|wslrelay|vpnkit') { " - another container? check 'docker ps'" } else { "" }
                $problems += "Port $port ($name) is already in use by $who$hint. Fix: stop it, or set $name to a free port in .env."
            }
        }
        if ($problems) {
            Write-Host ""
            foreach ($p in $problems) { Write-Host "  x $p" -ForegroundColor Red }
            throw "$($problems.Count) prerequisite problem(s); fix them and re-run."
        }
        Write-Host "    ports free: $(($Ports.Values | ForEach-Object { $_ }) -join ', ')"
    }

    Invoke-Stage "Create .env" "instant" {
        if (Test-Path $EnvPath) { $script:StageNote = "exists"; return }
        Copy-Item (Join-Path $RepoRoot ".env.example") $EnvPath
        Write-Host "    created .env from .env.example"
        $script:StageNote = "created"
    }

    if (-not $NoBuild) {
        Invoke-Stage "Build images" "~10 s from cache; much longer the first time (PyTorch, OpenVINO, Evidently wheels)" {
            Invoke-Compose @("--profile", "ml", "build")
        }
    }

    Invoke-Stage "Start infrastructure (Kafka, Postgres, MLflow, Kafka UI, Adminer)" "~15 s" {
        Invoke-Compose @("up", "-d", "--no-build", "kafka", "postgres", "mlflow", "kafka-ui", "adminer")
        Wait-Infrastructure @("kafka", "postgres", "mlflow")
    }

    Invoke-Stage "Create Kafka topics" "~5 s, ~40 s on a fresh broker" {
        & "$PSScriptRoot\create_topics.ps1"
    }

    Invoke-Stage "Run database migrations and seed the cameras" "~5 s" {
        Invoke-Compose @("run", "--rm", "-T", "migrate")
    }

    Invoke-Stage "Sample videos and zone configs" "~15 s to download 4 videos (~28 MB); seconds if present" {
        # In the video_ingest image (OpenCV, the shared package) with the repo mounted, so the
        # host needs no Python. Both scripts are idempotent and point the cameras at the files.
        $repo = ($RepoRoot -replace '\\', '/')
        $base = @("run", "--rm", "--no-deps", "-T", "-v", "${repo}:/repo", "-w", "/repo", "video_ingest", "python")
        Invoke-Compose ($base + @("scripts/download_sample_videos.py"))
        Invoke-Compose ($base + @("scripts/write_zone_configs.py", "--only-missing"))
        # A download that still fails after retries falls back to a generated video, so the
        # pipeline runs; say so instead of reporting a plain "ok".
        $attribution = Join-Path $RepoRoot "data\videos\ATTRIBUTION.md"
        $synthetic = @(Get-Content $attribution -ErrorAction SilentlyContinue | Where-Object { $_ -match '^## (cam-\S+)-synthetic\.mp4' } | ForEach-Object { $Matches[1] })
        if ($synthetic) {
            Write-Host "    warning: no real video for $($synthetic -join ', ') (download failed); using a generated one. Re-run this script to retry." -ForegroundColor Yellow
            $script:StageNote = "$(4 - $synthetic.Count)/4 real videos, synthetic for $($synthetic -join ', ')"
        }
        else { $script:StageNote = "4/4 real videos" }
    }

    Invoke-Stage "Generate the NLP dataset" "~5 s" {
        if ((Test-Path (Join-Path $RepoRoot "data\nlp\reports_train.csv")) -and (Test-Path (Join-Path $RepoRoot "data\nlp\reports_test.csv"))) {
            $script:StageNote = "exists"; return
        }
        Invoke-Compose @("run", "--rm", "--no-deps", "-T", "event_simulator", "python", "-m", "event_simulator.dataset")
        $script:StageNote = "generated"
    }

    # ---------------------------------------------------------------- models
    $nlpCommand = @("python", "-m", "ml.training.train_nlp")
    $nlpEstimate = "~10 min on CPU (DistilBERT, 3 epochs); -QuickModels: under a minute"
    if ($QuickModels) {
        $nlpCommand += "--skip-transformer"
        $nlpEstimate = "under a minute (TF-IDF baseline, -QuickModels)"
    }
    $models = @(
        @{ Name = "vehicle-detector"; Estimate = "~10 s (registers pretrained YOLO26n)"; Trainable = $false
            Run = { Invoke-Compose @("run", "--rm", "--no-deps", "-T", "-e", "GIT_PYTHON_REFRESH=quiet", "cv_inference", "python", "-m", "cv_inference.registry") } },
        @{ Name = "congestion-classifier"; Estimate = "~30 s"; Trainable = $true
            Run = { Invoke-Trainer @("python", "-m", "ml.training.train_congestion") } },
        @{ Name = "traffic-anomaly-detector"; Estimate = "~30 s"; Trainable = $true
            Run = { Invoke-Trainer @("python", "-m", "ml.training.train_anomaly") } },
        @{ Name = "incident-text-classifier"; Estimate = $nlpEstimate; Trainable = $true
            Run = { Invoke-Trainer $nlpCommand } },
        # The template and flan-t5-small (the full comparison with flan-t5-base and Qwen takes
        # ~30 min more: python -m ml.evaluation.summarizer_eval). The template wins either way.
        @{ Name = "incident-summarizer"; Estimate = "~1 min (template vs flan-t5-small)"; Trainable = $true
            Run = { Invoke-Trainer @("python", "-m", "ml.evaluation.summarizer_eval", "--models", "google/flan-t5-small:v1") } }
    )
    foreach ($m in $models) {
        $existing = Get-Champion $m.Name
        $force = $Retrain -and $m.Trainable
        $label = if ($m.Trainable) { "Train and register $($m.Name)" } else { "Register $($m.Name)" }
        if ($existing -and -not $force) {
            Invoke-Stage "$label" "skipped" { $script:StageNote = "champion v$existing exists" }
            continue
        }
        # Not $name / $run: the body runs inside Invoke-Stage, whose $Name parameter would win
        # (PowerShell variable names are case-insensitive).
        $modelRun = $m.Run
        $modelName = $m.Name
        Invoke-Stage "$label" $m.Estimate {
            & $modelRun
            $v = Get-Champion $modelName
            if (-not $v) { throw "$modelName has no @champion after training; see the output above." }
            $script:StageNote = "champion v$v"
        }
    }

    Invoke-Stage "Drift references for the trained models" "seconds (training saves them); ~1 min each if missing" {
        $built = @()
        foreach ($name in @("congestion-classifier", "traffic-anomaly-detector", "incident-text-classifier")) {
            $v = Get-Champion $name
            if ($v -and -not (Test-Reference $name $v)) {
                Invoke-Trainer @("python", "-m", "ml.monitoring.reference", "build", $name)
                $built += $name
            }
        }
        $script:StageNote = if ($built) { "built $($built -join ', ')" } else { "all present" }
    }

    # ---------------------------------------------------------------- services
    Invoke-Stage "Start all services and wait until each is healthy" "~1 min (longer if the NER or flan-t5 weights still need downloading)" {
        $expected = @(Get-ComposeOutput @("config", "--services") | Where-Object { $_ -ne "migrate" })
        $job = Start-Job -ArgumentList $RepoRoot -ScriptBlock {
            param($root)
            Set-Location $root
            $ErrorActionPreference = "Continue"
            $out = & docker compose up -d --no-build 2>&1 | ForEach-Object { "$_" }
            [pscustomobject]@{ Code = $LASTEXITCODE; Output = $out }
        }
        $deadline = (Get-Date).AddMinutes($ServiceTimeoutMinutes)
        $t0 = Get-Date
        $healthyAt = @{}
        $unhealthySince = @{}
        $lastReport = Get-Date
        $upResult = $null
        while ($true) {
            Start-Sleep -Seconds 3
            if ($job.State -ne "Running" -and -not $upResult) {
                $upResult = Receive-Job $job
                Remove-Job $job
                if ($upResult.Code -ne 0) {
                    $upResult.Output | Select-Object -Last 30 | Write-Lines
                    $states = Get-ServiceStates
                    $bad = @($expected | Where-Object { $states[$_] -and $states[$_].Health -eq "unhealthy" })
                    if ($bad) { Show-Logs $bad }
                    throw "docker compose up failed (exit $($upResult.Code))."
                }
            }
            $states = Get-ServiceStates
            foreach ($s in $expected) {
                $st = $states[$s]
                if (-not $st) { continue }
                if ($st.Health -eq "healthy" -and -not $healthyAt[$s]) {
                    $healthyAt[$s] = (Get-Date)
                    Write-Host ("    ok  {0,-16} healthy after {1}" -f $s, (Format-Duration ((Get-Date) - $t0).TotalSeconds)) -ForegroundColor Green
                }
                if ($st.Health -eq "unhealthy") {
                    if (-not $unhealthySince[$s]) { $unhealthySince[$s] = Get-Date }
                    elseif (((Get-Date) - $unhealthySince[$s]).TotalMinutes -ge 2) {
                        Show-Logs @($s)
                        throw "$s has been unhealthy for 2 minutes; its logs are above (healthcheck: docker inspect --format '{{json .State.Health}}' <container>)."
                    }
                }
                else { $unhealthySince.Remove($s) }
                if ($st.State -eq "exited") {
                    Show-Logs @($s)
                    throw "$s exited (code $($st.ExitCode)); its logs are above."
                }
            }
            $waiting = @($expected | Where-Object { -not $healthyAt[$_] })
            if (-not $waiting) { break }
            if ((Get-Date) -gt $deadline) {
                Show-Logs $waiting 30
                throw "Still not healthy after $ServiceTimeoutMinutes min: $($waiting -join ', '). Logs above; re-run with -ServiceTimeoutMinutes to wait longer."
            }
            if (((Get-Date) - $lastReport).TotalSeconds -ge 30) {
                $lastReport = Get-Date
                $detail = $waiting | ForEach-Object {
                    $st = $states[$_]
                    if ($st) { "$_ ($($st.State)$(if ($st.Health) { ", $($st.Health)" }))" } else { "$_ (not created)" }
                }
                Write-Host ("    [{0}] waiting for: {1}" -f (Format-Duration ((Get-Date) - $t0).TotalSeconds), ($detail -join ', ')) -ForegroundColor DarkGray
            }
        }
        if ($job.State -eq "Running") { Wait-Job $job | Out-Null; Remove-Job $job }
        $script:ServicesHealthyAt = Get-Date
        $script:StageNote = "$($expected.Count) services healthy"
    }

    Invoke-Stage "Check the pipeline produces output" "~30 s" {
        # Healthy already means each service is doing its work; this looks at the result end to
        # end: metrics windows from the CV pipeline and analysed reports, through the API.
        $deadline = (Get-Date).AddMinutes(3)
        while ($true) {
            $now = (Get-Date).ToUniversalTime()
            $cams = (Invoke-RestMethod -TimeoutSec 15 "$ApiUrl/api/v1/cameras").items
            $fresh = @($cams | Where-Object { $_.latest_metrics -and ($now - [datetime]::Parse($_.latest_metrics.window_end).ToUniversalTime()).TotalSeconds -lt 60 })
            $since = $now.AddMinutes(-5).ToString("o")
            $reports = (Invoke-RestMethod -TimeoutSec 15 "$ApiUrl/api/v1/reports?since=$([uri]::EscapeDataString($since))&page_size=1").total
            if ($fresh.Count -eq $cams.Count -and $reports -gt 0) { break }
            if ((Get-Date) -gt $deadline) {
                throw "After 3 min: live metrics for $($fresh.Count)/$($cams.Count) cameras, $reports analysed reports. Check .\scripts\logs.ps1 -Services cv_inference,nlp_service"
            }
            Start-Sleep -Seconds 5
        }
        $script:StageNote = "metrics from $($fresh.Count)/$($cams.Count) cameras, $reports reports analysed"
    }

    Invoke-Stage "Open the dashboard" "instant" {
        if (-not $NoBrowser) { Start-Process $DashboardUrl }
        $script:StageNote = if ($NoBrowser) { "skipped (-NoBrowser)" } else { "opened $DashboardUrl" }
    }

    Invoke-Stage "Drift reference for vehicle-detector" "waits until $ReferenceMinutes min of clean detections exist" {
        $v = Get-Champion "vehicle-detector"
        if (Test-Reference "vehicle-detector" $v) { $script:StageNote = "present"; return }
        # The detector is pretrained: its reference is its own output on the clean feeds.
        $ready = $script:ServicesHealthyAt.AddMinutes($ReferenceMinutes)
        while ((Get-Date) -lt $ready) {
            if (Test-Reference "vehicle-detector" $v) { break }  # ml-monitor may build it first
            Write-Host ("    collecting detections: {0} to go" -f (Format-Duration ($ready - (Get-Date)).TotalSeconds))
            Start-Sleep -Seconds ([math]::Min(60, [math]::Max(1, ($ready - (Get-Date)).TotalSeconds)))
        }
        if (Test-Reference "vehicle-detector" $v) { $script:StageNote = "built by ml-monitor"; return }
        Invoke-Trainer @("python", "-m", "ml.monitoring.reference", "build", "vehicle-detector", "--minutes", "$ReferenceMinutes")
        $script:StageNote = "built from $ReferenceMinutes min"
    }

    if ($Scenario) {
        Invoke-Stage "Trigger $Scenario on $ScenarioCamera" "after a $WarmupMinutes min warm-up" {
            $ready = $script:ServicesHealthyAt.AddMinutes($WarmupMinutes)
            while ((Get-Date) -lt $ready) {
                Write-Host ("    warming up (anomaly baselines): {0} to go" -f (Format-Duration ($ready - (Get-Date)).TotalSeconds))
                Start-Sleep -Seconds ([math]::Min(60, [math]::Max(1, ($ready - (Get-Date)).TotalSeconds)))
            }
            $body = @{ camera_id = $ScenarioCamera; time_scale = $TimeScale } | ConvertTo-Json
            $r = Invoke-RestMethod -Method Post -TimeoutSec 15 -ContentType "application/json" -Body $body "$ApiUrl/api/v1/scenarios/$Scenario/trigger"
            Write-Host "    scenario_id $($r.scenario_id); follow it on $DashboardUrl/demo"
            $script:StageNote = "triggered at x$TimeScale"
        }
    }
}
catch {
    Write-Host ""
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    if ($Stages.Count) { Write-Host ($Stages | Format-Table Stage, Time, Result -AutoSize | Out-String -Width 160) }
    Write-Host "Fix the problem and re-run .\scripts\run_demo.ps1; finished steps are skipped." -ForegroundColor Yellow
    exit 1
}

$total = ((Get-Date) - $Started).TotalSeconds
Write-Host ""
Write-Host "Stage timings" -ForegroundColor Cyan
Write-Host ($Stages | Format-Table Stage, Time, Result -AutoSize | Out-String -Width 160)
Write-Host ("Total: {0}" -f (Format-Duration $total)) -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard  $DashboardUrl" -ForegroundColor Green
Write-Host "  Demo     $DashboardUrl/demo  (trigger scenarios)"
Write-Host "  Models   $DashboardUrl/models  (drift status, drift mode)"
Write-Host "API        http://localhost:$($Ports.API_PORT)  (docs: /docs)" -ForegroundColor Green
Write-Host "Kafka UI   http://localhost:$($Ports.KAFKA_UI_PORT)" -ForegroundColor Green
Write-Host "MLflow     http://localhost:$($Ports.MLFLOW_PORT)" -ForegroundColor Green
Write-Host "Adminer    http://localhost:$($Ports.ADMINER_PORT)  (PostgreSQL / postgres / traffic / traffic / traffic)" -ForegroundColor Green
Write-Host "Kafka      localhost:$($Ports.KAFKA_PORT)   Postgres  localhost:$($Ports.POSTGRES_PORT)"
Write-Host ""
Write-Host "Logs: .\scripts\logs.ps1   End-to-end test: .\scripts\e2e_test.ps1   Stop: .\scripts\down.ps1   Clean slate: .\scripts\reset.ps1"
