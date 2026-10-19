"""Smoke tests for the repository skeleton."""

import ast
from pathlib import Path

import pytest

from traffic_common import get_logger, get_settings, on_shutdown
from traffic_common.lifecycle import run_shutdown_callbacks

ROOT = Path(__file__).resolve().parents[1]
SERVICES = [
    "video_ingest",
    "event_simulator",
    "cv_inference",
    "nlp_service",
    "anomaly_service",
    "fusion_service",
    "summarizer",
    "api",
]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@postgres:5432/db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    settings = get_settings("api")
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.kafka_bootstrap_servers == "kafka:29092"
    assert settings.postgres_url == "postgresql://u:p@postgres:5432/db"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


def test_settings_defaults_point_at_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("KAFKA_BOOTSTRAP_SERVERS", "POSTGRES_URL", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(var, raising=False)
    settings = get_settings("api")
    assert settings.kafka_bootstrap_servers == "localhost:9092"
    assert settings.postgres_url.endswith("@localhost:5432/traffic")
    assert settings.mlflow_tracking_uri == "http://localhost:5000"


def test_shutdown_callbacks_run_newest_first_and_survive_failures() -> None:
    order: list[str] = []
    on_shutdown(lambda: order.append("first"))
    on_shutdown(lambda: 1 / 0)
    on_shutdown(lambda: order.append("last"))
    run_shutdown_callbacks()
    assert order == ["last", "first"]
    run_shutdown_callbacks()  # callbacks run once
    assert order == ["last", "first"]


def test_logger_is_not_duplicated() -> None:
    log = get_logger("dup-check")
    assert get_logger("dup-check") is log
    assert len(log.handlers) == 1


@pytest.mark.parametrize("service", SERVICES)
def test_service_skeleton(service: str) -> None:
    service_dir = ROOT / "services" / service
    for name in ("main.py", "Dockerfile", "requirements.txt"):
        assert (service_dir / name).is_file(), f"{service}/{name} missing"
    ast.parse((service_dir / "main.py").read_text(encoding="utf-8"))
