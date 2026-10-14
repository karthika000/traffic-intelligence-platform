"""Programmatic Alembic access. The migration scripts ship inside this package, so this works
from any working directory and inside the service images, with no alembic.ini needed."""

from alembic import command
from alembic.config import Config

from traffic_common.config import get_settings
from traffic_common.db.session import to_sqlalchemy_url

SCRIPT_LOCATION = "traffic_common.db:migrations"


def alembic_config(url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", SCRIPT_LOCATION)
    url = to_sqlalchemy_url(url or get_settings("alembic").postgres_url)
    # ConfigParser interpolation treats '%' specially (e.g. URL-encoded passwords).
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "-1") -> None:
    command.downgrade(alembic_config(url), revision)
