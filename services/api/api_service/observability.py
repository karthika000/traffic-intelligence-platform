"""Request logging and one error format.

Every HTTP request gets an id (the caller's ``X-Request-ID`` if it sent one) returned in the
``X-Request-ID`` response header, and is logged as one JSON line with method, path, status,
duration and client. Health probes log at DEBUG so the 15 s container healthcheck doesn't
drown everything else.

Every error, whatever raised it (a 404 from a route, a validation error, an unhandled
exception), comes back as

    {"error": {"status": 404, "code": "not_found", "message": "...", "request_id": "...",
               "details": [...]}}

``details`` is only present for validation errors (one entry per invalid field)."""

import json
import logging
import re
import time
import uuid
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

access_log = logging.getLogger("api_service.access")
log = logging.getLogger(__name__)
QUIET_PATHS = {"/health", "/ready"}
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

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            ms = round((time.perf_counter() - started) * 1000, 1)
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
