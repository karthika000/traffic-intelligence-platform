"""Alembic environment. The URL comes from the Alembic config (set by traffic_common.db.migrate
or alembic.ini) and falls back to the POSTGRES_URL setting."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from traffic_common.config import get_settings
from traffic_common.db.models import Base
from traffic_common.db.session import to_sqlalchemy_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or to_sqlalchemy_url(
        get_settings("alembic").postgres_url
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
