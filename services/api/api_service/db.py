"""Async SQLAlchemy engine (psycopg 3) and the per-request connection dependency."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine


def async_url(url: str) -> str:
    """postgresql://... -> postgresql+psycopg://... (the async psycopg 3 driver)."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError(f"not a database URL: {url!r}")
    return f"postgresql+psycopg://{rest}" if scheme in ("postgresql", "postgres") else url


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(async_url(url), pool_size=10, max_overflow=10, pool_pre_ping=True)


async def connection(request: Request) -> AsyncIterator[AsyncConnection]:
    async with request.app.state.engine.connect() as conn:
        yield conn
