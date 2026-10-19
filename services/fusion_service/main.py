"""Fusion service.

Consumes anomalies, nlp.results and traffic.metrics (group fusion-service) and turns them
into incidents: correlated per camera in a sliding window, typed, scored with a transparent
fused confidence, and moved through suspected -> confirmed -> resolved. Incidents and their
evidence are stored in Postgres; every meaningful change is published to `incidents` as a
new version. Open incidents are rebuilt from Postgres on startup.
"""

import threading

from fusion_service.config import FusionConfig
from fusion_service.service import FusionService
from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.health import heartbeat
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.telemetry import serve_metrics
from traffic_common.topics import ConsumerGroup, Topic

SERVICE_NAME = "fusion_service"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    get_logger("traffic_common", settings.log_level)
    install_signal_handlers(log)
    serve_metrics()
    config = FusionConfig.from_env()
    log.info("starting: %s", config)
    stop = shutdown_event()

    engine = create_db_engine(settings.postgres_url)
    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)
    service = FusionService(config, engine, producer.send)
    service.recover()

    consumer = TypedConsumer(
        [Topic.ANOMALIES, Topic.NLP_RESULTS, Topic.TRAFFIC_METRICS],
        ConsumerGroup.FUSION_SERVICE,
        service.handle,
        settings.kafka_bootstrap_servers,
        stop_event=stop,
        # latest: a new group starts from now (recent evidence comes from Postgres instead).
        auto_offset_reset="latest",
    )

    def every(interval: float, fn, beat: str | None = None) -> None:
        """Run ``fn`` every ``interval`` s; ``beat`` names a heartbeat for each success."""
        hb = heartbeat(beat) if beat else None
        while not stop.wait(interval):
            try:
                fn()
            except Exception:
                log.exception("%s failed", fn.__name__)
            else:
                if hb is not None:
                    hb.beat()

    threads = [
        threading.Thread(target=consumer.run, name="consumer", daemon=True),
        threading.Thread(
            target=every,
            args=(config.tick_seconds, service.tick, "tick"),
            name="tick",
            daemon=True,
        ),
        threading.Thread(
            target=every,
            args=(config.stats_interval_seconds, service.log_stats),
            name="stats",
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    try:
        stop.wait()
    finally:
        for t in threads:
            t.join(timeout=15)
        run_shutdown_callbacks(log)
        engine.dispose()
        log.info("stopped")


if __name__ == "__main__":
    main()
