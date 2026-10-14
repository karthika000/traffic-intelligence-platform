"""Engine and session helpers."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from traffic_common.config import get_settings


def to_sqlalchemy_url(url: str) -> str:
    """Point plain ``postgresql://`` URLs (as in .env) at the psycopg 3 driver."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def create_db_engine(url: str | None = None, **kwargs: object) -> Engine:
    """Engine for ``url``, defaulting to the POSTGRES_URL setting."""
    url = url or get_settings("db").postgres_url
    return create_engine(to_sqlalchemy_url(url), pool_pre_ping=True, **kwargs)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
