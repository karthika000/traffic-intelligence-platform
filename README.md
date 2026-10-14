# Real-Time Multimodal Traffic Intelligence Platform

Ingests live video, synthetic/real traffic events and text reports, runs CV, NLP and
anomaly-detection models over them, fuses the signals into a single traffic picture,
and serves it to a React dashboard.

> **Status:** skeleton only. Every service is a placeholder that starts, logs, and idles.

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

## Repository layout

```
services/<name>/     main.py, Dockerfile, requirements.txt per service
shared/              traffic-common Python package used by all services
dashboard/           React + Vite + TypeScript frontend
ml/                  training/, evaluation/, monitoring/
data/                local datasets and sample videos (git-ignored)
models/              trained model artifacts (git-ignored, mounted read-only)
tests/               repository-level tests
scripts/             PowerShell helpers (up / down / logs)
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
.\scripts\up.ps1                         # creates .env from .env.example, builds, starts everything
.\scripts\logs.ps1                       # follow all logs
.\scripts\logs.ps1 -Services api         # follow one service
.\scripts\down.ps1                       # stop (add -Volumes to drop volumes)
```

| Script | Options |
| --- | --- |
| `up.ps1` | `-Services a,b` · `-NoBuild` · `-Attach` |
| `down.ps1` | `-Volumes` |
| `logs.ps1` | `-Services a,b` · `-Tail N` · `-NoFollow` |

Dashboard: <http://localhost:5173> · API: <http://localhost:8001>

## Local development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e "shared[dev]"
pytest
```

Run one service outside Docker: `python services\api\main.py`.

## Configuration

All settings come from environment variables; see [.env.example](.env.example).
`.env` is git-ignored.

### Ports

| Variable | Default | Notes |
| --- | --- | --- |
| `API_PORT` | `8001` | Host port for `api`. Inside the Docker network the api still listens on 8000. Defaults to 8001 because 8000 is often taken by WSL processes. |
| `DASHBOARD_PORT` | `5173` | Host port for the dashboard |
| `VITE_API_BASE_URL` | `http://localhost:${API_PORT}` | Only set this if the dashboard reaches the API through a different host or proxy |

After changing a port, re-run `.\scripts\up.ps1` so the containers are recreated.
