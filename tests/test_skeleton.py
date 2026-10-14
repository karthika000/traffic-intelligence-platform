"""Smoke tests for the repository skeleton."""

import ast
from pathlib import Path

import pytest
from traffic_common import get_logger, get_settings

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
    settings = get_settings("api")
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"


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
