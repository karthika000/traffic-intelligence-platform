"""Kafka topic names, consumer groups and dead-letter topics.

scripts/create_topics.ps1 creates these; tests/test_schemas.py checks the two stay in sync.
"""

from enum import StrEnum


class Topic(StrEnum):
    VIDEO_FRAMES = "video.frames"
    EVENTS_RAW = "events.raw"
    REPORTS_TEXT = "reports.text"
    DETECTIONS = "detections"
    TRAFFIC_METRICS = "traffic.metrics"
    ANOMALIES = "anomalies"
    NLP_RESULTS = "nlp.results"
    INCIDENTS = "incidents"
    SUMMARIES = "summaries"
    DRIFT_ALERTS = "drift.alerts"
    SIMULATOR_CONTROL = "simulator.control"


class ConsumerGroup(StrEnum):
    """One group per consuming service. Hyphens, not underscores, so `dlq.<group>` topic
    names never mix '.' and '_' (Kafka warns that those can collide in metric names)."""

    CV_INFERENCE = "cv-inference"
    NLP_SERVICE = "nlp-service"
    ANOMALY_SERVICE = "anomaly-service"
    FUSION_SERVICE = "fusion-service"
    SUMMARIZER = "summarizer"
    API = "api"
    ML_MONITORING = "ml-monitoring"
    EVENT_SIMULATOR = "event-simulator"


DEAD_LETTER_PREFIX = "dlq."


def dead_letter_topic(group_id: str) -> str:
    return f"{DEAD_LETTER_PREFIX}{group_id}"


DEAD_LETTER_TOPICS: tuple[str, ...] = tuple(dead_letter_topic(g) for g in ConsumerGroup)
