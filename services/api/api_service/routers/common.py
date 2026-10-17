"""Shared request helpers: pagination and time ranges."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Query, Request


@dataclass(frozen=True)
class Paging:
    page: int
    page_size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


def paging(
    request: Request,
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(50, ge=1, description="items per page (capped by the server)"),
) -> Paging:
    return Paging(page, min(page_size, request.app.state.config.max_page_size))


def time_range(
    since: datetime | None, until: datetime | None, default_hours: float | None = None
) -> tuple[datetime | None, datetime | None]:
    """Validated [since, until); naive datetimes are taken as UTC."""
    since = since.replace(tzinfo=UTC) if since and since.tzinfo is None else since
    until = until.replace(tzinfo=UTC) if until and until.tzinfo is None else until
    if since is None and default_hours is not None:
        since = (until or datetime.now(UTC)) - timedelta(hours=default_hours)
    if since and until and until <= since:
        raise HTTPException(422, "until must be after since")
    return since, until


def not_found(what: str) -> HTTPException:
    return HTTPException(404, f"{what} not found")
