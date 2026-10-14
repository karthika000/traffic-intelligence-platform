"""Shared fixtures. Infrastructure-backed tests skip (with a reason) when Kafka or Postgres
isn't reachable at the addresses from the settings (localhost by default; the `tests`
compose service points them at kafka:29092 / postgres:5432)."""

import signal
import socket
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy.engine import make_url

from traffic_common import lifecycle
from traffic_common.config import get_settings

SMART_APP_CONTROL_HINT = (
    "confluent-kafka's native library could not be loaded. On Windows this is usually "
    "Smart App Control blocking it; run the suite in Docker with .\\scripts\\test.ps1"
)


def _reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def _reset_lifecycle() -> Iterator[None]:
    """Keep process-wide shutdown state (event, callbacks, signal handlers) from leaking
    between tests, and keep Ctrl+C working for pytest itself."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    lifecycle._handlers_installed = False
    lifecycle._shutdown.clear()
    with lifecycle._lock:
        lifecycle._callbacks.clear()


@pytest.fixture(scope="session")
def kafka_bootstrap() -> str:
    servers = get_settings("tests").kafka_bootstrap_servers
    host, _, port = servers.split(",")[0].rpartition(":")
    if not _reachable(host, int(port)):
        pytest.skip(f"Kafka not reachable at {servers}")
    return servers


@pytest.fixture(scope="session")
def postgres_url() -> str:
    url = get_settings("tests").postgres_url
    parsed = make_url(url)
    if not _reachable(parsed.host or "localhost", parsed.port or 5432):
        pytest.skip(f"Postgres not reachable at {parsed.host}:{parsed.port}")
    return url


@pytest.fixture
def scratch_database(postgres_url: str) -> Iterator[str]:
    """A throwaway database on the same server, so tests never touch `traffic`."""
    import psycopg

    base = make_url(postgres_url)
    name = f"test_{uuid.uuid4().hex[:12]}"
    admin_dsn = base.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    try:
        yield base.set(database=name).render_as_string(hide_password=False)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
