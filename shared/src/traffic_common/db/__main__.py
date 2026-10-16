"""Database command line.

    python -m traffic_common.db upgrade [REVISION]       # default: head
    python -m traffic_common.db downgrade REVISION       # e.g. -1, base
    python -m traffic_common.db current
    python -m traffic_common.db history
    python -m traffic_common.db revision -m "add x" [--autogenerate]
    python -m traffic_common.db seed

Every command accepts --url; otherwise POSTGRES_URL (or its localhost default) is used.
"""

import argparse
import logging
import sys
from pathlib import Path

from alembic import command

from traffic_common.db.migrate import alembic_config
from traffic_common.db.seed import seed_cameras
from traffic_common.db.session import create_db_engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_common.db",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", help="database URL (default: POSTGRES_URL)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade").add_argument("revision", nargs="?", default="head")
    sub.add_parser("downgrade").add_argument("revision")
    sub.add_parser("current")
    sub.add_parser("history")
    rev = sub.add_parser("revision")
    rev.add_argument("-m", "--message", required=True)
    rev.add_argument("--autogenerate", action="store_true")
    sub.add_parser("seed")
    args = parser.parse_args(argv)

    # Without an alembic.ini nothing configures logging, and Alembic reports progress via logging.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = alembic_config(args.url)
    if args.command == "upgrade":
        command.upgrade(cfg, args.revision)
    elif args.command == "downgrade":
        command.downgrade(cfg, args.revision)
    elif args.command == "current":
        command.current(cfg, verbose=True)
    elif args.command == "history":
        command.history(cfg, verbose=True)
    elif args.command == "revision":
        # Write new revisions into the source tree, not an installed copy.
        versions = Path(__file__).parent / "migrations" / "versions"
        command.revision(
            cfg, message=args.message, autogenerate=args.autogenerate, version_path=str(versions)
        )
    elif args.command == "seed":
        count = seed_cameras(create_db_engine(args.url))
        print(f"seeded demo cameras: {count} present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
