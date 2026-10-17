"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine

from api_service import live, observability
from api_service.config import ApiConfig
from api_service.db import make_engine
from api_service.routers import (
    analytics,
    cameras,
    frames,
    health,
    incidents,
    models,
    reports,
    scenarios,
)
from api_service.routers.models import MlflowReader
from traffic_common.messaging import TypedProducer

log = logging.getLogger(__name__)
DESCRIPTION = """Cameras, incidents, reports, analytics, models, scenarios, frames and live updates
for the traffic intelligence dashboard.

Lists are paginated (`page`, `page_size` up to 200) and return `{items, total, page, page_size}`.
Records use the platform's shared message schemas, the same as on Kafka and `/ws/live`.
Every error has the same shape: `{"error": {"status", "code", "message", "request_id",
"details"}}`. Every response carries `X-Request-ID` (send your own to correlate with the
server's logs) and `Server-Timing`.

Live updates: connect to `/ws/live` (optionally `?topics=incidents,summaries&cameras=cam-a`)
for JSON envelopes `{"type": topic, "camera_ids": [...], "ts": ..., "data": {...}}`."""
TAGS = [
    {"name": "cameras", "description": "Camera status, metrics history and recent anomalies."},
    {"name": "incidents", "description": "Fused incidents with evidence, versions and summaries."},
    {"name": "reports", "description": "Text reports with their NLP results."},
    {"name": "analytics", "description": "Aggregates for charts, computed in SQL."},
    {"name": "models", "description": "Models in service and their MLflow metrics."},
    {"name": "scenarios", "description": "Simulator scenarios: list, runs and triggering."},
    {"name": "frames", "description": "Annotated camera frames: latest JPEG and MJPEG stream."},
    {"name": "health", "description": "Liveness and readiness probes."},
]


def create_app(
    config: ApiConfig,
    postgres_url: str,
    bootstrap: str,
    mlflow_uri: str,
    *,
    engine: AsyncEngine | None = None,
    producer=None,
    live_updates: bool = True,
    mlflow_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.bootstrap = bootstrap
        app.state.engine = engine or make_engine(postgres_url)
        app.state.producer = producer or TypedProducer(
            bootstrap, client_id="api", register_shutdown=False
        )
        mlflow_client = httpx.AsyncClient(
            base_url=mlflow_uri, timeout=5.0, transport=mlflow_transport
        )
        app.state.mlflow = MlflowReader(mlflow_client, config.mlflow_cache_seconds)
        app.state.hub = None
        if live_updates:
            app.state.hub = live.LiveHub(
                bootstrap,
                queue_size=config.ws_queue_size,
                metrics_interval=config.ws_metrics_interval_seconds,
                max_drops=config.ws_max_drops,
            )
            app.state.hub.start(asyncio.get_running_loop())
        try:
            yield
        finally:
            if app.state.hub is not None:
                app.state.hub.stop()
            await mlflow_client.aclose()
            if producer is None:
                app.state.producer.flush(5)
            if engine is None:
                await app.state.engine.dispose()

    app = FastAPI(
        title="Traffic Intelligence API",
        version="1.0.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    observability.install(app)  # request logging and the error format (inside CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Frame-Time", "X-Request-ID", "Server-Timing"],
    )
    for module in (cameras, incidents, reports, analytics, models, scenarios, frames):
        app.include_router(module.router, responses=observability.ERROR_RESPONSES)
    app.include_router(health.router)
    app.include_router(live.router)
    return app
