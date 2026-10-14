"""Database access for the `traffic` PostgreSQL database.

    from traffic_common.db import models, create_db_engine, session_factory

Migrations: ``python -m traffic_common.db upgrade`` (see ``python -m traffic_common.db --help``).
"""

from traffic_common.db import models
from traffic_common.db.models import Base
from traffic_common.db.session import create_db_engine, session_factory, to_sqlalchemy_url

__all__ = ["Base", "create_db_engine", "models", "session_factory", "to_sqlalchemy_url"]
