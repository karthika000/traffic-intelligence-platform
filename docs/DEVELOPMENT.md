# Development

Running, testing and configuring the platform. The [README](../README.md) has the overview and the quick start; [ARCHITECTURE.md](ARCHITECTURE.md) describes how each part works.

- [Prerequisites](#prerequisites-windows) · [run_demo.ps1](#what-run_demops1-does) · [Other scripts](#other-scripts)
- [Local development](#local-development) · [Tests and lint](#tests-and-lint) · [Continuous integration](#continuous-integration) · [End-to-end test](#end-to-end-test)
- [Configuration](#configuration) · [Ports](#ports)

## Prerequisites (Windows)

- Docker Desktop with the WSL 2 backend, with at least 6 GB of memory for its VM (8 GB recommended) and about 20 GB of free disk
- Git

That is all the demo needs; everything else runs in containers. For local development: Python 3.11 (containers use `python:3.11-slim`), and optionally Node.js 20+ to run the dashboard outside Docker.

If script execution is blocked, allow local scripts once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## What run_demo.ps1 does

Every step skips what already exists, so a re-run takes under a minute (53 s here). `-Retrain` retrains the models anyway.

| Stage | Measured from a clean state |
| --- | --- |
| Prerequisites: Docker running, Compose 2.24+, VM memory (`-MinMemoryGB`, 6), CPUs, disk, and every host port (including Prometheus and Grafana) free (or already ours) and outside the Windows-reserved ranges. Each problem is printed with its fix. | 7 s |
| `.env` from `.env.example` if missing | 0 s |
| Build images (from cache here; a first build on a new machine takes much longer) | 10 s |
| Infrastructure up and healthy; Kafka topics | 14 s + 37 s |
| Migrations and the demo cameras (the `migrate` service) | 4 s |
| Sample videos and zone configs (in the video_ingest image, no host Python). Rate limiting and network errors are retried; a camera still without a video gets a generated one, and the stage says so. | 18 s |
| NLP dataset, if missing | 2 s |
| Models without a `@champion`: register `vehicle-detector`; train `congestion-classifier`, `traffic-anomaly-detector`, `incident-text-classifier` (DistilBERT; `-QuickModels` for the TF-IDF baseline) and `incident-summarizer` (template against flan-t5-small) | 6 s, 34 s, 26 s, 9 min 46 s, 1 min 2 s |
| Drift references of the trained models, if missing (training already saves them) | 0 s |
| Every service up, each followed until healthy (`-ServiceTimeoutMinutes`, 25; prints the logs of a service that fails) | 58 s |
| Pipeline output: live metrics from every camera, analysed reports | 0 s |
| Open the dashboard (`-NoBrowser` skips it) | 0 s |
| `vehicle-detector` drift reference, after `-ReferenceMinutes` (5) of clean detections | 5 min 12 s |
| `-Scenario`: trigger it on `-ScenarioCamera` at `-TimeScale` (6) after `-WarmupMinutes` (10), and once Prometheus reports no camera still in anomaly_service's warm-up (up to 5 more minutes) | – |
| **Total** (Core Ultra 5 225H, 14 vCPUs, 7.5 GB VM, images cached) | **19 min 37 s** |

To start again from nothing, run `.\scripts\reset.ps1`. After you confirm, it deletes the containers, every volume (Kafka, Postgres with MLflow's runs and registry, MLflow artifacts, frames, model caches, drift reports) and the generated contents of `data/` and `models/`. Its options: `-KeepVideos` keeps the downloaded videos, `-IncludeEnv` also deletes `.env`, `-Images` also removes the built images, and `-Force` skips the prompt.

To check the whole pipeline, run `.\scripts\e2e_test.ps1` (see [End-to-end test](#end-to-end-test)).

## Other scripts

```powershell
.\scripts\up.ps1                         # creates .env if missing, builds, starts everything, creates topics
.\scripts\logs.ps1                       # follow all logs
.\scripts\logs.ps1 -Services api         # follow one service
.\scripts\create_topics.ps1              # (re)create any missing Kafka topics; safe to re-run
.\scripts\down.ps1                       # stop; data in Kafka, Postgres and MLflow volumes is kept
.\scripts\down.ps1 -Volumes              # stop AND delete all Kafka, Postgres and MLflow data
```

| Script | Options |
| --- | --- |
| `run_demo.ps1` | `-Scenario name` · `-ScenarioCamera id` · `-TimeScale N` · `-WarmupMinutes N` · `-Retrain` · `-QuickModels` · `-ReferenceMinutes N` · `-ServiceTimeoutMinutes N` · `-MinMemoryGB N` · `-NoBrowser` · `-NoBuild` |
| `reset.ps1` | `-Force` · `-KeepVideos` · `-IncludeEnv` · `-Images` |
| `e2e_test.ps1` | `-TimeScale N` · `-TimeoutMinutes N` · `-Scenarios a,b` |
| `up.ps1` | `-Services a,b` · `-NoBuild` · `-Attach` |
| `down.ps1` | `-Volumes` |
| `logs.ps1` | `-Services a,b` · `-Tail N` · `-NoFollow` |
| `create_topics.ps1` | `-Partitions N` (new topics only) · `-TimeoutSec N` |
| `test.ps1` | `-Local` · any pytest arguments (`-k`, `-x`, paths, ...) |

| Open | URL |
| --- | --- |
| Dashboard | <http://localhost:5173> |
| API | <http://localhost:8001> |
| Kafka UI | <http://localhost:8080> |
| MLflow | <http://localhost:5000> |
| Adminer | <http://localhost:8081>. Log in with System `PostgreSQL`, Server `postgres`, and the user, password and database from `.env` (`traffic` / `traffic` / `traffic` by default). |
| Grafana | <http://localhost:3000>: the Traffic Intelligence dashboard, read-only without a login; `admin` / `admin` (from `.env`) to edit |
| Prometheus | <http://localhost:9090>; scrape status at `/targets` |

## Local development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e "shared[dev]" -r services\video_ingest\requirements.txt -r services\event_simulator\requirements.txt
```

### Database migrations

The one-shot `migrate` service applies every migration and seeds the demo cameras on each `docker compose up`, before any app service starts. The commands below are for development. Run them with the stack up. They use `POSTGRES_URL`, which defaults to `localhost:5432`:

```powershell
python -m traffic_common.db upgrade                            # apply all migrations
python -m traffic_common.db seed                               # 4 demo cameras (safe to re-run)
python -m traffic_common.db current                            # show the applied revision
python -m traffic_common.db downgrade -1                       # roll back one revision
python -m traffic_common.db revision -m "add x" --autogenerate # new migration from model changes
```

The same commands work inside any service container, for example `docker compose exec api python -m traffic_common.db upgrade`. See [shared/README.md](../shared/README.md) for the models and the Kafka helpers.

### Tests and lint

```powershell
.\scripts\test.ps1                          # whole suite in a Linux container on the compose network
.\scripts\test.ps1 tests/test_schemas.py -x # any pytest arguments pass through
.\scripts\test.ps1 -Local                   # in .venv against localhost:9092 / localhost:5432
.\scripts\test.ps1 -m "not integration"    # only the tests that need neither Kafka nor Postgres
ruff check .
ruff format --check .
cd shared; mypy                             # the shared package (configured in shared/pyproject.toml)
```

A test that uses the Kafka or Postgres fixtures, directly or through another fixture, is marked `integration` automatically (`tests/conftest.py`), so `-m integration` and `-m "not integration"` split the suite the way CI does.

| File | Needs | What it covers |
| --- | --- | --- |
| `test_schemas.py` | nothing | Schema round-trips, validation rules, and the topic lists in `create_topics.ps1` |
| `test_messaging.py` | Kafka | Producer validation and keys, consumer commits, retries, dead-lettering, shutdown, the consumer heartbeat with its partition count |
| `test_health.py` | nothing | Heartbeat files (throttling, state changes), the healthcheck CLI: missing, stale, per-heartbeat max age, consumers without partitions |
| `test_migrations.py` | Postgres | Upgrade, downgrade and upgrade again; models match the migration; constraints; idempotent seed |
| `test_skeleton.py` | nothing | Settings, logging, lifecycle, service layout |
| `test_video_ingest.py` | nothing | Frame sampling, resizing, JPEG encoding, stream_url parsing, file naming, retention, health metrics |
| `test_video_ingest_integration.py` | Kafka | A worker on a short synthetic video publishes to `video.frames`, plus retry when a source is missing |
| `test_event_simulator.py` | nothing | Report variety and label correctness, train/test holdout separation, dataset files, daily traffic pattern, weather, scenario YAML parsing and timelines, scenario runs, control commands |
| `test_cv_inference.py` | nothing | Zones (point-in-polygon, scaling, defaults), speed and smoothing, stopped detection, metric windows, latest-frame scheduling, message building |
| `test_cv_inference_integration.py` | Ultralytics (+ Kafka) | YOLO26n + ByteTrack on 12 frames of a sample video produce valid, tracked `Detection`s and metrics; the messages round-trip through `detections` |
| `test_ml_core.py` | pandas | Feature engineering, relative baselines, weak labels, window building, scenario labels, synthetic data, threshold calibration, alarm start, end, cooldown and persistence, evaluation metrics, hints, rule-based models |
| `test_ml_models.py` | torch (Docker) | The autoencoder and both congestion models round-trip through MLflow pyfunc packaging and score raw rows; per-camera calibration of an unseen camera (video scales from its warm-up, loop scales kept, never below the global values, ignored signals never scoring, settings parsing) |
| `test_anomaly_service.py` | Postgres | Model reload when the champion alias moves, rule fallback, keeping the current model when the registry is down, the sensor join, start and end events, cooldown, stale metrics, sensor storage, frozen baselines and the hold rule, escalation updates, system-wide degradation, pipeline noise, no alerts during the warm-up (and a new warm-up after a gap), alerts ending when traffic recovers, ignored signals |
| `test_fusion_integration.py` | Kafka + Postgres | An anomaly plus matching reports published to Kafka come out of an in-process fusion service as one confirmed incident on `incidents` |
| `test_nlp.py` | (Postgres for batching) | Preprocessing, label maps, entity rules, camera matching, the keyword fallback, and nlp_service batching, without PyTorch or downloads |
| `test_nlp_models.py` | torch, transformers (Docker) | A tiny randomly initialised DistilBERT and the TF-IDF baseline trained, saved and reloaded through the pyfunc wrapper |
| `test_nlp_integration.py` | Kafka + Postgres | `TextReport`s on `reports.text` come back as `NLPResult`s with the right camera, and are stored |
| `test_summarizer.py` | (Postgres for the context) | Context building, the template, the faithfulness guard (invented numbers, places, vehicles, contradictions fail), the fallback paths, the debounce |
| `test_summarizer_integration.py` | Kafka + Postgres | An `Incident` on `incidents` produces a stored `Summary` on `summaries` |
| `test_api.py` | Postgres | Every REST endpoint against a scratch database seeded with a small known world (MLflow mocked) |
| `test_api_live.py` | (Kafka for 1 test) | WebSocket fan-out (filters, throttling, slow clients), and a real consumer delivering a published message over `/ws/live` |
| `test_sample_videos.py` | nothing | `download_sample_videos.py` retries transient network failures, not permanent ones |
| `test_fusion.py` | (Postgres for 2 tests) | Type mapping, agreement, severity, the confidence breakdown, the lifecycle (suspected, confirmed, resolved by quiet period, clearing report or recovered anomaly), start times, unlocated reports, system degradation, the vision warm-up; restart recovery and evidence retries against Postgres |
| `test_anomaly_integration.py` | Kafka + Postgres | A synthetic incident published to `traffic.metrics` produces an `Anomaly` on `anomalies` and a row in `anomalies` |
| `test_drift_monitoring.py` | Evidently | Drift scores on synthetic shifted and unshifted data (numbers, categories, embeddings), threshold overrides, the ok / warning / drifted / no_data verdict and HTML report, data quality (missing, out of range, stale cameras), time-of-day reference slicing, settings, and the scheduler (interval, skipped slots, failures, on-demand runs) |
| `test_ml_lifecycle.py` | Postgres | The promotion gate (candidate better, worse, better on the primary metric but regressing on a guard, tolerances, lower-is-better guards, no champion), the per-model policies, retraining recommendations from drift alerts (counted once, resolved by a decision) and promotion decisions in `model_versions` |
| `test_drift_mode.py` | Postgres | Drift mode settings and presets, switching it on and off (expiry, history, active periods), the services' watcher, video_ingest's night and fog degradation and its worker toggling, and event_simulator switching to (and back from) the slang style |
| `test_api_drift.py` | Postgres | The drift endpoints: status per model with features, quality, recommendation and promotion; history; alerts; HTML reports; switching drift mode on and off; validation |
| `test_event_simulator_integration.py` | Kafka + Postgres | A scenario triggered through `simulator.control` emits tagged messages on `reports.text` and `events.raw`, and its ground truth lands in `scenario_runs` and `simulated_reports` |

Tests whose infrastructure isn't reachable are skipped, with the reason shown by `-rs`. The infrastructure tests work in throwaway topics, consumer groups and databases, which they delete afterwards. They never touch the `traffic` database. Four integration tests do publish to real topics:

- **video_ingest:** a few dozen frames on `video.frames`, under a `test-cam-<id>` camera id. Their frame files no longer exist.
- **cv_inference:** 4 `Detection`s on `detections` under a `test-cam-<id>` camera id.
- **anomaly_service:** about 36 `TrafficMetrics` on `traffic.metrics` and one `Anomaly`, for a `test-cam-<id>` camera.
- **event_simulator:** a scenario's reports and events on `reports.text` and `events.raw`, tagged with a `test-cam-<id>` camera and that run's `scenario_id`. The live simulator also receives the test's control command and logs `rejected start_scenario … unknown camera`, as intended.

Those messages stay until Kafka retention removes them.

On Windows, Smart App Control can block confluent-kafka's native library, so `import confluent_kafka` fails and the Kafka tests skip under `-Local`. The cv_inference integration test also skips under `-Local`, because PyTorch and Ultralytics are only installed in the Docker images. The default `test.ps1` mode runs in Linux and runs everything.

Run one service outside Docker: `python services\api\main.py`. Outside Docker, `traffic_common.get_settings()` defaults to the host-side addresses (`localhost:9092`, Postgres on `localhost:5432`, MLflow on `http://localhost:5000`), so a local process talks to the infrastructure running in Docker.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push and pull request:

| Job | What it runs |
| --- | --- |
| Lint and type-check | `ruff check`, `ruff format --check` (ruff pinned to the version used here), `mypy` on the shared package |
| Dashboard | `npm ci` and `npm run check` (TypeScript, ESLint, Vitest) |
| Build images | `docker build` of every image (11, as a matrix), with the GitHub Actions layer cache |
| Unit tests | The test image (`tests/Dockerfile`, as `test.ps1` uses) with `-m "not integration"` |
| Integration tests | The same image against Postgres and Kafka service containers, after creating the topics, with `-m integration` |

The test jobs run in the same image as `.\scripts\test.ps1`, so CI and a local run use the same dependencies. In CI the cv_inference integration test skips, because the sample videos are downloaded, not committed.

## End-to-end test

```powershell
.\scripts\e2e_test.ps1                       # all 4 scenarios at x6, timeout 30 min
.\scripts\e2e_test.ps1 -Scenarios flooding -TimeScale 10
```

With the stack running and every service healthy, it triggers each scenario on a different camera at once. It then follows each run, through the API only, until the run's incident resolves. The run's incident is the one whose evidence includes reports carrying the run's `scenario_id`. That incident can have been opened before the trigger: fusion attaches a run's reports to an incident already open on the camera. Before triggering, the test waits up to 10 minutes (`--wait-free-minutes`) for cameras without an open incident.

A run passes when its incident was **confirmed**, has the scenario's **type**, has at least one **summary**, is listed and returned by the **API**, and **resolved**. The checker (`scripts/e2e_check.py`) uses only the standard library and runs in the api image, so the host needs no Python. It prints a table with timings from the trigger, writes `data/e2e/<time>.json`, and exits 1 on any failure.

Results on this machine, at ×6 (times from the trigger; the scenarios last 2 min 40 s to 3 min 40 s at that speed):

| Scenario | Camera | Type | Suspected | Confirmed | First summary | Resolved | Start vs run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| broken_down_lorry | Elephant & Castle | breakdown | 24 s | 35 s | 34 s | 2 min 19 s | −314 s |
| flooding | Hyde Park Corner | weather_hazard | 43 s | 45 s | 55 s | 3 min 25 s | +20 s |
| multi_vehicle_accident | Old Street | accident | 28 s | 28 s | 39 s | 2 min 30 s | +23 s |
| road_works | Trafalgar Square | road_works | 38 s | 40 s | 51 s | 3 min 35 s | +13 s |

That is the final run (2026-10-19). Each scenario's camera raised a traffic anomaly 60 to 77 s after the trigger, and each incident resolved on a clearing report. The lorry's start time is 5 minutes before the run: background breakdown reports on that camera from just before the trigger formed one run of supporting evidence with the scenario's own (gaps under `FUSION_START_GAP_SECONDS`).

Before the warm-up, calibration and recovery changes (2026-10-18, directly after a clean `run_demo.ps1`):

| Scenario | Camera | Type | Suspected | Confirmed | First summary | Resolved | Start vs run | First evidence vs run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broken_down_lorry | Elephant & Castle | breakdown | 22 s | 36 s | 47 s | 2 min 40 s | +1 s | −407 s |
| flooding | Hyde Park Corner | weather_hazard | 39 s | 39 s | 49 s | 14 min 39 s | +25 s | −822 s |
| multi_vehicle_accident | Old Street | accident | 10 s | 23 s | 20 s | 3 min 13 s | +3 s | −549 s |
| road_works | Trafalgar Square | road_works | 54 s | 54 s | 1 min 4 s | 2 min 42 s | +33 s | −28 s |

- **Flooding resolved late before the changes.** An anomaly on Hyde Park Corner that started during the run stayed active for 22 minutes, because its score never fell below the clear level: the model had only synthetic calibration, so that camera was scored against other cameras' noise. Fusion doesn't resolve while a traffic anomaly is active. Now each camera is calibrated from its own warm-up, an alert ends when its traffic recovers, and fusion can resolve on a clearing report once the anomaly reports its traffic recovered.
- **Start times:** the last two columns show why [the start time](ARCHITECTURE.md#start-time) is the first evidence supporting the incident's type. The earliest attached evidence came up to 14 minutes before the run, while `start_time` is within about 30 seconds of it.
- **The test waits before triggering:** until Prometheus reports no camera in anomaly_service's warm-up (`--prometheus`), and up to 10 minutes (`--wait-free-minutes`) for cameras without an open incident. Before the warm-up existed, an anomaly 50 s after startup and a single background report once confirmed a false incident on Trafalgar Square, and the road works run joined it.
- **If the host sleeps,** every camera's metrics stop. On waking, anomaly_service starts each camera's warm-up again and drops the windows from before the gap. The test checks again just before triggering, and a camera without recent metrics counts as not ready, so a run doesn't start before the pipeline has resumed.

## Configuration

All settings come from environment variables. [.env.example](../.env.example) lists the ones you are likely to change; the rest (paths inside containers, stats intervals, minor tunables) have defaults in each service's `config.py`, under the same naming scheme (for fusion, every `FusionConfig` field as `FUSION_<FIELD>`).
`.env` is git-ignored.

`DISPLAY_TIMEZONE` (an IANA zone, default `Europe/London` because the demo cameras are in London) sets the zone for clock times everywhere a person reads them:
- **Summaries:** incident summaries name the zone, as in "since 18:48 BST".
- **Video:** the timestamp cv_inference writes on annotated frames.
- **Dashboard:** compose passes it to the dashboard as `VITE_DISPLAY_TIMEZONE`.

Stored timestamps and the API stay in UTC. After changing it, recreate the containers with `.\scripts\up.ps1`.

### Connection settings

`.env` holds host-side URLs for local development. Inside Docker, `docker-compose.yml` overrides them with in-network addresses:

| Variable | In `.env` (host) | In containers |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:${KAFKA_PORT}` | `kafka:29092` |
| `POSTGRES_URL` | `postgresql://traffic:traffic@localhost:${POSTGRES_PORT}/traffic` | `...@postgres:5432/traffic` |
| `MLFLOW_TRACKING_URI` | `http://localhost:${MLFLOW_PORT}` | `http://mlflow:5000` |

The Postgres credentials in `.env.example` are for development only.

### Ports

Each variable sets only the host side of a port mapping. Container ports are fixed.

| Variable | Default | Service | Notes |
| --- | --- | --- | --- |
| `API_PORT` | `8001` | api | The api listens on 8000 inside Docker. The host default is 8001 because 8000 is often taken by WSL processes. |
| `DASHBOARD_PORT` | `5173` | dashboard | |
| `KAFKA_PORT` | `9092` | kafka | Also the advertised external listener, so host clients must use this port |
| `KAFKA_UI_PORT` | `8080` | kafka-ui | |
| `POSTGRES_PORT` | `5432` | postgres | Change this if a local PostgreSQL install already uses 5432 |
| `MLFLOW_PORT` | `5000` | mlflow | |
| `ADMINER_PORT` | `8081` | adminer | |
| `PROMETHEUS_PORT` | `9090` | prometheus | |
| `GRAFANA_PORT` | `3000` | grafana | Often taken by local web dev servers; change it here if so |
| `VITE_API_BASE_URL` | `http://localhost:${API_PORT}` | dashboard | Only set this if the dashboard reaches the API through a different host or proxy |

The Kafka controller listener (29093) is never published. On this machine Windows reserves ports 9093 and 9095 (see `netsh interface ipv4 show excludedportrange protocol=tcp`), so avoid them as host ports.

After changing a port, re-run `.\scripts\up.ps1` so the containers are recreated.

To check whether a port is free, look on the Windows side and inside WSL:

```powershell
Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
wsl -e ss -ltn "sport = :5432"
```
