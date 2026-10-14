"""Uniform logging setup so every service's output looks the same in `docker compose logs`."""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
