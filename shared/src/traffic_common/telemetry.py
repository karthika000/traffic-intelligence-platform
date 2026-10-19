"""Prometheus metrics shared by every service, and the HTTP endpoint that exposes them.

Each process calls ``serve_metrics()`` once at startup; Prometheus scrapes
``http://<service>:<METRICS_PORT>/metrics`` (default 9100; ``METRICS_PORT=0`` turns it off).
The api serves the same registry on its own port at ``/metrics`` instead. Prometheus adds the
``job`` label (the service), so metrics here carry only what varies within one service.

Recorded for every service by ``traffic_common.messaging``: messages consumed and produced,
handler latency, message age on arrival (time since the producer stamped it, i.e. the latency
of everything upstream), errors by stage, and consumer lag per partition. Services add their
own: model inference latency (``INFERENCE_SECONDS``) and domain gauges.
"""

import json
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram, start_http_server

log = logging.getLogger(__name__)

DEFAULT_PORT = 9100
_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_AGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)

MESSAGES_CONSUMED = Counter(
    "traffic_messages_consumed_total", "Kafka messages handed to a handler", ["topic", "group"]
)
MESSAGES_PRODUCED = Counter("traffic_messages_produced_total", "Kafka messages sent", ["topic"])
MESSAGE_ERRORS = Counter(
    "traffic_message_errors_total",
    "Message failures by stage: validation, handler (each failed attempt), dead_letter, delivery",
    ["topic", "stage"],
)
PROCESSING_SECONDS = Histogram(
    "traffic_message_processing_seconds",
    "Handler time per message (a batch's time is split over its messages)",
    ["topic", "group"],
    buckets=_LATENCY_BUCKETS,
)
MESSAGE_AGE_SECONDS = Histogram(
    "traffic_message_age_seconds",
    "Time from the message's timestamp to its arrival at the consumer",
    ["topic", "group"],
    buckets=_AGE_BUCKETS,
)
CONSUMER_LAG = Gauge(
    "traffic_consumer_lag",
    "Messages behind the partition's end, from librdkafka statistics",
    ["topic", "partition", "group"],
)
INFERENCE_SECONDS = Histogram(
    "traffic_model_inference_seconds",
    "Model inference latency per call",
    ["model"],
    buckets=_LATENCY_BUCKETS,
)
SERVICE_ERRORS = Counter(
    "traffic_service_errors_total", "Errors outside message handling (loops, storage)", ["kind"]
)

_server_lock = threading.Lock()
_server_port: int | None = None


def serve_metrics(port: int | None = None) -> int | None:
    """Start the /metrics endpoint once per process (``METRICS_PORT``, default 9100; 0 = off).
    Returns the port, or None when disabled or the port is taken (logged, not fatal)."""
    global _server_port
    if port is None:
        port = int(os.getenv("METRICS_PORT", str(DEFAULT_PORT)))
    if port <= 0:
        return None
    with _server_lock:
        if _server_port is not None:
            return _server_port
        try:
            start_http_server(port)
        except OSError as exc:
            log.warning("metrics endpoint not started on port %d: %s", port, exc)
            return None
        _server_port = port
        log.info("metrics on :%d/metrics", port)
        return port


@contextmanager
def timed(histogram: Histogram, **labels: str) -> Iterator[None]:
    """Observe the duration of the block (also when it raises)."""
    start = perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(perf_counter() - start)


def record_kafka_stats(group: str, stats_json: str) -> None:
    """librdkafka ``stats_cb``: consumer lag of every assigned partition."""
    try:
        stats = json.loads(stats_json)
    except ValueError:
        return
    for topic, t in (stats.get("topics") or {}).items():
        for partition, p in (t.get("partitions") or {}).items():
            lag = p.get("consumer_lag")
            if partition == "-1" or lag is None or lag < 0:
                continue  # the internal UA partition, or no committed offset yet
            CONSUMER_LAG.labels(topic, partition, group).set(lag)
