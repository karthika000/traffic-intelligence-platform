"""/health: the process is up. /ready: Postgres and Kafka reachable and the live consumer
running (the container healthcheck)."""

import asyncio
import time

from confluent_kafka.admin import AdminClient
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter(tags=["health"])
KAFKA_CHECK_TTL = 5.0


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _kafka_ok(bootstrap: str) -> str | None:
    try:
        AdminClient({"bootstrap.servers": bootstrap}).list_topics(timeout=3)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    state = request.app.state
    checks: dict[str, str] = {}
    try:
        async with state.engine.connect() as conn:
            await asyncio.wait_for(conn.execute(text("SELECT 1")), timeout=3)
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {type(exc).__name__}: {exc}"[:300]
    cached = getattr(state, "kafka_check", None)
    if cached is None or time.monotonic() - cached[0] > KAFKA_CHECK_TTL:
        cached = (time.monotonic(), await run_in_threadpool(_kafka_ok, state.bootstrap))
        state.kafka_check = cached
    checks["kafka"] = "ok" if cached[1] is None else f"error: {cached[1]}"[:300]
    hub = getattr(state, "hub", None)
    if hub is None:
        checks["live_consumer"] = "disabled"
    else:
        checks["live_consumer"] = "ok" if hub.healthy else f"not running {hub.error or ''}".strip()
    ok = all(v in ("ok", "disabled") for v in checks.values())
    body = {"status": "ready" if ok else "not ready", "checks": checks}
    if hub is not None:
        body["live"] = {"clients": len(hub.clients), **hub.stats}
    return JSONResponse(body, status_code=200 if ok else 503)
