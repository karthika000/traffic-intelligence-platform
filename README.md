# Real-Time Multimodal Traffic Intelligence Platform

Ingests live video, synthetic/real traffic events and text reports, runs CV, NLP and
anomaly-detection models over them, fuses the signals into a single traffic picture,
and serves it to a React dashboard.

> **Status:** the infrastructure (Kafka, PostgreSQL, MLflow), `video_ingest` and `event_simulator` are implemented. The other app services are placeholders that start, log, and idle.

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
| `simulator.control` | Commands for `event_simulator` (start/stop scenarios) |

There is also one dead-letter topic per consumer group, `dlq.<group>`:

- **Topics:** `dlq.cv-inference`, `dlq.nlp-service`, `dlq.anomaly-service`, `dlq.fusion-service`, `dlq.summarizer`, `dlq.api`, `dlq.ml-monitoring`, `dlq.event-simulator`. Each has 1 partition and keeps messages for 14 days.
- **What lands there:** messages that fail validation, or that still fail after retries.
- **Inspecting them:** browse them in Kafka UI. Each record carries `dlq.*` headers explaining why it failed.

`create_topics.ps1` prints a Kafka warning for every topic with a `.` in its name. It's harmless: the warning is about topics that differ only by `.` versus `_`, and none of ours do.

## Video ingest

`video_ingest` runs one worker thread per row in the `cameras` table, and re-reads the table every `CAMERA_REFRESH_SECONDS`. New cameras start, removed ones stop, and a camera whose `stream_url` changes is restarted.

For each camera it:

1. **Reads** the source. `stream_url` can be a file under `/data/videos` (which maps to `./data/videos`), an `rtsp://` URL, or an `http(s)://` URL. Files play in real time and loop forever, simulating a live feed.
2. **Samples** frames at `FRAME_SAMPLE_FPS`.
3. **Resizes** frames to at most `FRAME_MAX_WIDTH`, keeping the aspect ratio, and JPEG-encodes them at `FRAME_JPEG_QUALITY`.
4. **Writes** each frame to the shared `frames` volume as `/data/frames/<camera_id>/<timestamp>.jpg`. The volume is read-write for `video_ingest` and read-only for `cv_inference` and `api`.
5. **Publishes** a `FrameMessage` to `video.frames`, keyed by camera.

When a source fails, the worker logs it and retries with exponential backoff (`SOURCE_RETRY_*`); the other cameras keep running. A cleanup thread deletes frames older than `FRAME_RETENTION_MINUTES`, so consumers must handle a frame file that has already gone. Every `METRICS_LOG_INTERVAL_SECONDS`, each camera logs a health line:

```
health camera=cam-old-street state=streaming frames_read=897 frames_published=60 fps=2.00 errors=0
```

All settings are in `.env` (see `.env.example`).

### Sample videos

```powershell
python scripts\download_sample_videos.py
```

The script downloads one openly licensed traffic video per demo camera from Wikimedia Commons into `data/videos/`. It uses the 480p versions, a few MB each, and points each camera's `stream_url` at its file. Licence and author are read from the Commons API, printed, and saved to `data/videos/ATTRIBUTION.md`. If a download fails, the script generates a synthetic video instead, so the pipeline still runs.

Licences and authors are listed under [Video credits](#video-credits).

Options: `--force` (re-download), `--synthetic-only`, `--skip-db`.

### Watching frames

- **Kafka UI:** <http://localhost:8080/ui/clusters/traffic-intel/all-topics/video.frames/messages>
- **Console consumer:**

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:29092 --topic video.frames --max-messages 5 --property print.key=true --property print.headers=true
```

## Event simulator

`event_simulator` produces the non-video inputs for every camera in the `cameras` table, re-reading the table every `CAMERA_REFRESH_SECONDS`:

| Output | Topic | Rate |
| --- | --- | --- |
| Free-text reports (`TextReport`) | `reports.text` | One every `REPORT_INTERVAL_SECONDS` on average (default 10 s), across all cameras |
| Weather (`RawEvent`, `event_type=weather`) | `events.raw` | Every `SENSOR_INTERVAL_SECONDS` (15 s) per camera |
| Loop-detector readings (`RawEvent`, `event_type=loop_detector`) | `events.raw` | Every `SENSOR_INTERVAL_SECONDS` (15 s) per camera |

How the inputs are generated:

- **Reports** come from four source styles: formal road-operator bulletins, police updates, drivers (typos, lowercase, missing punctuation, abbreviations) and social media (emojis, hashtags, shouting). They cover 8 incident types (`accident`, `breakdown`, `road_works`, `congestion`, `weather_hazard`, `debris`, `police_activity`, and `none` for chatter) at 4 severities. The severity is written into the text, for example "minor bump, no injuries" versus "all lanes closed, casualties".
- **Location mentions** vary: official names, nearby streets, landmarks, abbreviations, misspellings, and "street near landmark" combinations. Operators and police never misspell.
- **Weather** changes condition rarely and moves gradually.
- **Traffic** follows a daily pattern with weekday peaks around 08:00 and 17:30, flatter at weekends. Speed falls as flow rises.

### Scenarios

Scenarios are YAML files in [services/event_simulator/scenarios/](services/event_simulator/scenarios/): `multi_vehicle_accident`, `broken_down_lorry`, `flooding` and `road_works`. Each is a timeline of phases. A phase sets:

- how many reports it emits, from which sources, with which incident types and severity;
- the traffic effect (speed and count factors, ramped in);
- optional weather.

Every scenario ends with a phase of "cleared" reports. Every message a run emits carries its `scenario_id`.

```powershell
docker compose exec event_simulator python -m event_simulator.trigger --list
docker compose exec event_simulator python -m event_simulator.trigger multi_vehicle_accident cam-hyde-park-corner
docker compose exec event_simulator python -m event_simulator.trigger flooding cam-old-street --time-scale 10   # 10x faster
docker compose exec event_simulator python -m event_simulator.trigger --stop <scenario_id>
```

The trigger CLI publishes a `SimulatorCommand` to `simulator.control`, and any other producer can do the same. Two more things to know:

- **Scheduled scenarios:** set `SCENARIO_SCHEDULE_MINUTES` to start a random scenario on a random idle camera every N minutes. It is off by default.
- **Limits:** each camera runs one scenario at a time, and commands older than 10 minutes are ignored.

### Ground truth

Every simulated report is stored with its labels in `simulated_reports`: incident type, severity, source style, `location_mention`, stage, scenario and phase. Every run is stored in `scenario_runs`: scenario, camera, status, trigger, and the timeline with actual start and end times. Both tables are kept separate from `reports` and `incidents`, which hold what the platform itself ingests and infers. Evaluation joins the two worlds on `report_id` and `scenario_id`.

```sql
SELECT scenario_name, camera_id, status, started_at, ended_at, reports_emitted FROM scenario_runs ORDER BY started_at DESC;
SELECT phase, incident_type, severity, style, text FROM simulated_reports WHERE scenario_run_id = '<scenario_id>';
```

### NLP training data

```powershell
docker compose exec event_simulator python -m event_simulator.dataset            # writes ./data/nlp/
```

The command writes `data/nlp/reports_train.csv` (5,000 rows) and `data/nlp/reports_test.csv` (1,000 rows), with columns `text, incident_type, severity, camera_id, location_mention, source`. It then prints the class distribution. The two splits differ in three ways:

- **Seeds:** each split uses a different random seed.
- **Phrasings:** every phrase pool (templates, incident and severity phrases, location forms) has held-out entries that only the test split uses, about half the time. A good test score therefore means the model generalises beyond memorised templates.
- **Duplicates:** texts are unique within each split, and no test text appears in train.

Options: `--train N`, `--test N`, `--train-seed`, `--test-seed`, `--out DIR`.

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
pip install -e "shared[dev]" -r services\video_ingest\requirements.txt -r services\event_simulator\requirements.txt
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
| `test_video_ingest.py` | nothing | Frame sampling, resizing, JPEG encoding, stream_url parsing, file naming, retention, health metrics |
| `test_video_ingest_integration.py` | Kafka | A worker on a short synthetic video publishes to `video.frames`, plus retry when a source is missing |
| `test_event_simulator.py` | nothing | Report variety and label correctness, train/test holdout separation, dataset files, daily traffic pattern, weather, scenario YAML parsing and timelines, scenario runs, control commands |
| `test_event_simulator_integration.py` | Kafka + Postgres | A scenario triggered through `simulator.control` emits tagged messages on `reports.text` and `events.raw`, and its ground truth lands in `scenario_runs` and `simulated_reports` |

Tests whose infrastructure isn't reachable are skipped, with the reason shown by `-rs`. The infrastructure tests work in throwaway topics, consumer groups and databases, which they delete afterwards. They never touch the `traffic` database. Two integration tests do publish to real topics:

- **video_ingest:** a few dozen frames on `video.frames`, under a `test-cam-<id>` camera id. Their frame files no longer exist.
- **event_simulator:** a scenario's reports and events on `reports.text` and `events.raw`, tagged with a `test-cam-<id>` camera and that run's `scenario_id`. The live simulator also receives the test's control command and logs `rejected start_scenario … unknown camera`, as intended.

Those messages stay until Kafka retention removes them.

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

## Video credits

The demo camera videos are not stored in git (`data/` is ignored); `scripts/download_sample_videos.py` fetches them from Wikimedia Commons. This section records their attribution, copied from the `data/videos/ATTRIBUTION.md` the script generates. If you redistribute these videos or material derived from them (for example extracted frames or screenshots), credit the author and keep the licence. CC BY-SA additionally requires sharing derivatives under the same licence.

| Local file (camera) | Commons file | Author | Licence |
| --- | --- | --- | --- |
| `cam-hyde-park-corner.webm` | [London roundabout.ogg](https://commons.wikimedia.org/wiki/File:London_roundabout.ogg) | Vcohen | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `cam-trafalgar-square.webm` | [Intersection Pie-IX-Sherbrooke.webm](https://commons.wikimedia.org/wiki/File:Intersection_Pie-IX-Sherbrooke.webm) | Thomas1313 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `cam-elephant-castle.webm` | [Road traffic on Stritarjeva street.webm](https://commons.wikimedia.org/wiki/File:Road_traffic_on_Stritarjeva_street.webm) | Sounds of Changes | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| `cam-old-street.webm` | [A intersection with traffic in Chiang Mai, Thailand.ogv](https://commons.wikimedia.org/wiki/File:A_intersection_with_traffic_in_Chiang_Mai,_Thailand.ogv) | Amada44 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

The local files are Wikimedia's 480p VP9 transcodes of these originals. The camera names are fictional placements: the footage was not filmed at the London junctions the demo cameras are named after.
