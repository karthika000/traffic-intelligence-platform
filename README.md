# Real-Time Multimodal Traffic Intelligence Platform

Ingests live video, synthetic/real traffic events and text reports, runs CV, NLP and
anomaly-detection models over them, fuses the signals into a single traffic picture,
and serves it to a React dashboard.

> **Status:** the infrastructure (Kafka, PostgreSQL, MLflow) is real. The app services are placeholders that start, log, and idle.

## Architecture

```
video_ingest ──► cv_inference ──┐
event_simulator ───────────────┼──► fusion_service ──► summarizer
nlp_service ───────────────────┤            │
anomaly_service ───────────────┘            ▼
                                            api ──► dashboard
```

| Service | Responsibility |
| --- | --- |
| `video_ingest` | Read camera / RTSP / file streams and publish frames |
| `event_simulator` | Generate synthetic traffic events for dev and testing |
| `cv_inference` | Vehicle/pedestrian detection, tracking, counting |
| `nlp_service` | Entity and intent extraction from incident reports and feeds |
| `anomaly_service` | Anomaly detection over traffic metrics and events |
| `fusion_service` | Combine vision, text and sensor signals into a unified state |
| `summarizer` | Natural-language summaries of current conditions |
| `api` | HTTP/WebSocket API for the dashboard (host port `API_PORT`, default 8001) |
| `dashboard` | React + Vite + TypeScript UI (port 5173) |

### Infrastructure

| Component | Image | Purpose |
| --- | --- | --- |
| `kafka` | `apache/kafka:3.8.0` | Event bus. Single node in KRaft mode (no ZooKeeper). |
| `kafka-ui` | `provectuslabs/kafka-ui:v0.7.2` | Browse topics, messages and consumer groups |
| `postgres` | `postgres:16-alpine` | Application database (`traffic`) and MLflow backend store (`mlflow`) |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.17.2` + psycopg2 | Experiment tracking and model registry. Artifacts are kept in the `mlflow_artifacts` volume. |
| `adminer` | `adminer:4.8.1` | Web SQL client for Postgres |

All five have healthchecks. The 8 app services wait for `kafka` and `postgres` to report healthy before they start.

Kafka has two listeners:

| Listener | Address | Used by |
| --- | --- | --- |
| Internal | `kafka:29092` | Containers on the compose network |
| External | `localhost:9092` (`KAFKA_PORT`) | Code running on your machine |

### Kafka topics

Created by `scripts/create_topics.ps1`, which `up.ps1` runs once Kafka is healthy. Each data topic gets 3 partitions and replication factor 1. Automatic topic creation is disabled, so a misspelled topic name fails instead of silently creating a new topic. Each topic has a Pydantic schema in `traffic_common.schemas` (see [shared/README.md](shared/README.md)).

| Topic | Carries |
| --- | --- |
| `video.frames` | Frames or frame references from `video_ingest` |
| `events.raw` | Raw traffic events from `event_simulator` and other feeds |
| `reports.text` | Free-text incident reports |
| `detections` | CV detections and tracks |
| `traffic.metrics` | Aggregated flow, speed and occupancy metrics |
| `anomalies` | Output of `anomaly_service` |
| `nlp.results` | Entities and intents extracted by `nlp_service` |
| `incidents` | Fused incidents from `fusion_service` |
| `summaries` | Text from `summarizer` |
| `drift.alerts` | Model drift alerts from `ml/monitoring` |

There is also one dead-letter topic per consumer group, `dlq.<group>`:

- **Topics:** `dlq.cv-inference`, `dlq.nlp-service`, `dlq.anomaly-service`, `dlq.fusion-service`, `dlq.summarizer`, `dlq.api`, `dlq.ml-monitoring`. Each has 1 partition and keeps messages for 14 days.
- **What lands there:** messages that fail validation, or that still fail after retries.
- **Inspecting them:** browse them in Kafka UI. Each record carries `dlq.*` headers explaining why it failed.

`create_topics.ps1` prints a Kafka warning for every topic with a `.` in its name. It's harmless: the warning is about topics that differ only by `.` versus `_`, and none of ours do.

## Repository layout

```
services/<name>/     main.py, Dockerfile, requirements.txt per service
shared/              traffic-common package: settings, Kafka schemas + clients, DB models + migrations
dashboard/           React + Vite + TypeScript frontend
ml/                  training/, evaluation/, monitoring/
data/                local datasets and sample videos (git-ignored)
models/              trained model artifacts (git-ignored, mounted read-only)
tests/               repository-level tests (+ Dockerfile for the in-Docker test runner)
infra/               MLflow image, Postgres init scripts
scripts/             PowerShell helpers (up / down / logs / create_topics / test)
ruff.toml            lint settings for the whole repo
```

## Prerequisites (Windows)

- Docker Desktop with the WSL 2 backend
- Python 3.11 (for local development and tests; containers use `python:3.11-slim`)
- Node.js 20+ (optional, only for running the dashboard outside Docker)

If script execution is blocked, allow local scripts once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Quick start

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

## Local development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e "shared[dev]"
```

### Database migrations

Run these with the stack up. They use `POSTGRES_URL`, which defaults to `localhost:5432`:

```powershell
python -m traffic_common.db upgrade                            # apply all migrations
python -m traffic_common.db seed                               # 4 demo cameras (safe to re-run)
python -m traffic_common.db current                            # show the applied revision
python -m traffic_common.db downgrade -1                       # roll back one revision
python -m traffic_common.db revision -m "add x" --autogenerate # new migration from model changes
```

The same commands work inside any service container, for example `docker compose exec api python -m traffic_common.db upgrade`. See [shared/README.md](shared/README.md) for the models and the Kafka helpers.

### Tests and lint

```powershell
.\scripts\test.ps1                          # whole suite in a Linux container on the compose network
.\scripts\test.ps1 tests/test_schemas.py -x # any pytest arguments pass through
.\scripts\test.ps1 -Local                   # in .venv against localhost:9092 / localhost:5432
ruff check .
```

| File | Needs | What it covers |
| --- | --- | --- |
| `test_schemas.py` | nothing | Schema round-trips, validation rules, and the topic lists in `create_topics.ps1` |
| `test_messaging.py` | Kafka | Producer validation and keys, consumer commits, retries, dead-lettering, shutdown |
| `test_migrations.py` | Postgres | Upgrade, downgrade and upgrade again; models match the migration; constraints; idempotent seed |
| `test_skeleton.py` | nothing | Settings, logging, lifecycle, service layout |

Tests whose infrastructure isn't reachable are skipped, with the reason shown by `-rs`. The infrastructure tests work in throwaway topics, consumer groups and databases, which they delete afterwards; they never touch `traffic` or the platform topics.

On Windows, Smart App Control can block confluent-kafka's native library, so `import confluent_kafka` fails and the Kafka tests skip under `-Local`. The default `test.ps1` mode runs in Linux and isn't affected.

Run one service outside Docker: `python services\api\main.py`. Outside Docker, `traffic_common.get_settings()` defaults to the host-side addresses (`localhost:9092`, Postgres on `localhost:5432`, MLflow on `http://localhost:5000`), so a local process talks to the infrastructure running in Docker.

## Configuration

All settings come from environment variables; see [.env.example](.env.example).
`.env` is git-ignored.

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
| `VITE_API_BASE_URL` | `http://localhost:${API_PORT}` | dashboard | Only set this if the dashboard reaches the API through a different host or proxy |

The Kafka controller listener (29093) is never published. On this machine Windows reserves ports 9093 and 9095 (see `netsh interface ipv4 show excludedportrange protocol=tcp`), so avoid them as host ports.

After changing a port, re-run `.\scripts\up.ps1` so the containers are recreated.

To check whether a port is free, look on the Windows side and inside WSL:

```powershell
Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
wsl -e ss -ltn "sport = :5432"
```
