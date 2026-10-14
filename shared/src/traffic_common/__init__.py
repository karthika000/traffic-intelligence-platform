"""Shared utilities for the Traffic Intelligence Platform services."""

from traffic_common.config import Settings, get_settings
from traffic_common.lifecycle import run_forever
from traffic_common.logging import get_logger

__all__ = ["Settings", "get_logger", "get_settings", "run_forever"]
__version__ = "0.1.0"
