"""Shared utilities for the Traffic Intelligence Platform services.

Heavier modules are imported explicitly so this package stays cheap to import:
    traffic_common.schemas    Kafka message schemas and TOPIC_SCHEMAS
    traffic_common.messaging  TypedProducer / TypedConsumer (needs confluent-kafka)
    traffic_common.db         SQLAlchemy models, engine helpers, migrations, seed data
"""

from traffic_common.config import Settings, get_settings
from traffic_common.lifecycle import on_shutdown, request_shutdown, shutdown_event
from traffic_common.logging import get_logger

__all__ = [
    "Settings",
    "get_logger",
    "get_settings",
    "on_shutdown",
    "request_shutdown",
    "shutdown_event",
]
__version__ = "0.2.0"
