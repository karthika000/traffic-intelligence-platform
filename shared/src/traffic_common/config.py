"""Environment-driven settings shared by all services."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    service_name: str
    environment: str
    log_level: str
    kafka_bootstrap_servers: str
    postgres_url: str
    mlflow_tracking_uri: str


# Defaults are host-side addresses for running a service outside Docker;
# docker-compose.yml overrides them with in-network addresses.
def get_settings(service_name: str) -> Settings:
    return Settings(
        service_name=service_name,
        environment=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        postgres_url=os.getenv(
            "POSTGRES_URL", "postgresql://traffic:traffic@localhost:5432/traffic"
        ),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
