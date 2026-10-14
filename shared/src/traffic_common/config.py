"""Environment-driven settings shared by all services."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    service_name: str
    environment: str
    log_level: str


def get_settings(service_name: str) -> Settings:
    return Settings(
        service_name=service_name,
        environment=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
