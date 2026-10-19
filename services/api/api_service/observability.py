"""Request logging and one error format.

Every HTTP request gets an id (the caller's ``X-Request-ID`` if it sent one) returned in the
``X-Request-ID`` response header, and is logged as one JSON line with method, path, status,
duration and client. Health probes log at DEBUG so the 15 s container healthcheck doesn't
drown everything else.

Every error, whatever raised it (a 404 from a route, a validation error, an unhandled
exception), comes back as

    {"error": {"status": 404, "code": "not_found", "message": "...", "request_id": "...",
               "details": [...]}}

``details`` is only present for validation errors (one entry per invalid field).

Prometheus metrics are served at ``/metrics``: requests by method, route template and status,
their latency, and requests in flight, plus the shared ``traffic_common.telemetry`` metrics
(the live consumer's messages) and the process collectors."""

import json
import logging
import re
import time
import uuid
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

access_log = logging.getLogger("api_service.access")
log = logging.getLogger(__name__)
QUIET_PATHS = {"/health", "/ready", "/metrics"}
REQUESTS = Counter(
    "traffic_api_requests_total", "HTTP requests by route template", ["method", "route", "status"]
)
REQUEST_SECONDS = Histogram(
    "traffic_api_request_duration_seconds",
    "Time to the response headers (for MJPEG streams: until streaming starts)",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
IN_FLIGHT = Gauge("traffic_api_requests_in_flight", "HTTP requests being handled")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ErrorDetail(BaseModel):
    status: int = Field(examples=[404])
    code: str = Field(examples=["not_found"], description="stable code for the status")
    message: str = Field(examples=["incident 0f8c… not found"])
    request_id: str | None = Field(default=None, examples=["9b2f4c1d0a6e4f57"])
    details: list[dict[str, Any]] | None = Field(
        default=None, description="validation errors: one entry per invalid field"
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail


ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Not found"},
    422: {"model": ErrorResponse, "description": "Invalid request"},
    500: {"model": ErrorResponse, "description": "Internal error"},
}


# Stable codes for the statuses the API returns (HTTPStatus phrases change between Python
# versions, e.g. 422 "Unprocessable Entity" became "Unprocessable Content").
CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    500: "internal_error",
    503: "service_unavailable",
}


def _code(status: int) -> str:
    if status in CODES:
        return CODES[status]
    try:
        return HTTPStatus(status).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "error"


def error_response(
    request: Request, status: int, message: str, details: list | None = None, headers=None
) -> JSONResponse:
    body = ErrorDetail(
        status=status,
        code=_code(status),
        message=message,
        request_id=getattr(request.state, "request_id", None),
        details=details,
    )
    return JSONResponse(
        {"error": body.model_dump(exclude_none=True)}, status_code=status, headers=headers
    )


def install(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        return error_response(request, exc.status_code, message, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(e.get("loc", ())), "message": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return error_response(request, 422, "invalid request", details)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(request, 500, "internal error")

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500
        IN_FLIGHT.inc()
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            IN_FLIGHT.dec()
            elapsed = time.perf_counter() - started
            ms = round(elapsed * 1000, 1)
            # The route template, not the path, so ids don't explode the label set.
            route = getattr(request.scope.get("route"), "path", None) or "unmatched"
            if route != "/metrics":
                REQUESTS.labels(request.method, route, str(status)).inc()
                REQUEST_SECONDS.labels(request.method, route).observe(elapsed)
            access_log.log(
                logging.DEBUG if request.url.path in QUIET_PATHS else logging.INFO,
                json.dumps(
                    {
                        "event": "request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "query": request.url.query or None,
                        "status": status,
                        "duration_ms": ms,
                        "client": request.client.host if request.client else None,
                    }
                ),
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={ms}"
        return response
