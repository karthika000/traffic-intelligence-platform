"""Anomaly service.

Consumes traffic.metrics and events.raw (group anomaly-service). Loop-detector readings are
stored in sensor_readings; metric windows are joined with the latest reading per camera.
Every few seconds each camera's recent windows go through the congestion classifier and
the anomaly detector (MLflow champions, or rule-based fallbacks). A camera raises no alerts
during its warm-up (ANOMALY_WARMUP_SECONDS after its first metrics or a restart), at the end of
which it is calibrated if the model lacks a calibration for it. Anomaly episodes are
published to `anomalies` (start and end events) and stored in Postgres together with the
current congestion level per camera.
"""

import os
import threading

from anomaly_service.config import AnomalyConfig
from anomaly_service.models import ModelManager
from anomaly_service.service import AnomalyService
from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.health import heartbeat
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.telemetry import serve_metrics
from traffic_common.topics import ConsumerGroup, Topic

SERVICE_NAME = "anomaly_service"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    get_logger("traffic_common", settings.log_level)
    install_signal_handlers(log)
    serve_metrics()
    config = AnomalyConfig.from_env()
    log.info("starting: %s", config)
    import torch

    torch.set_num_threads(int(os.getenv("MODEL_THREADS", "2")))  # see Dockerfile
    stop = shutdown_event()

    engine = create_db_engine(settings.postgres_url)
    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)
    service_ref: dict[str, AnomalyService] = {}
    models = ModelManager(
        settings.mlflow_tracking_uri,
        {"congestion": config.congestion_model, "anomaly": config.anomaly_model},
        config.model_alias,
        on_change=lambda m: service_ref["svc"].record_model(m) if "svc" in service_ref else None,
    )
    service = AnomalyService(config, engine, models, producer.send)
    service_ref["svc"] = service
    service.close_orphaned_anomalies()
    service.refresh_settings()
    models.refresh()

    consumer = TypedConsumer(
        [Topic.TRAFFIC_METRICS, Topic.EVENTS_RAW],
        ConsumerGroup.ANOMALY_SERVICE,
        service.handle,
        settings.kafka_bootstrap_servers,
        stop_event=stop,
        # earliest: a new group backfills sensor_readings from events.raw history;
        # replayed traffic.metrics are too old to be scored and are skipped.
        auto_offset_reset="earliest",
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
            args=(config.score_interval_seconds, service.score_all, "scoring"),
            name="scoring",
            daemon=True,
        ),
        threading.Thread(
            target=every,
            args=(config.model_refresh_seconds, models.refresh),
            name="model-refresh",
            daemon=True,
        ),
        threading.Thread(
            target=every,
            args=(config.camera_refresh_seconds, service.refresh_settings),
            name="camera-settings",
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
