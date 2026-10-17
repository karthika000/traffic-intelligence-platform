"""API service: REST endpoints and live WebSocket updates for the dashboard (FastAPI).

/api/v1/...   cameras, incidents, reports, analytics, models, scenarios, frames
/ws/live      live incidents, summaries, anomalies, metrics and NLP results
/health       process up;  /ready  Postgres + Kafka reachable, live consumer running
/docs         OpenAPI documentation
"""

import asyncio
import sys

import uvicorn

from api_service.app import create_app
from api_service.config import ApiConfig
from traffic_common import get_logger, get_settings

SERVICE_NAME = "api"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    get_logger("api_service", settings.log_level)
    get_logger("traffic_common", settings.log_level)
    if sys.platform == "win32":  # async psycopg needs a selector loop on Windows (local runs)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    config = ApiConfig.from_env()
    log.info("starting: %s", config)
    app = create_app(
        config,
        settings.postgres_url,
        settings.kafka_bootstrap_servers,
        settings.mlflow_tracking_uri,
    )
    uvicorn.run(
        app, host="0.0.0.0", port=8000, log_level=settings.log_level.lower(), proxy_headers=True
    )


if __name__ == "__main__":
    main()
