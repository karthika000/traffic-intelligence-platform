"""Summarizer service.

Consumes incidents (group summarizer). When an incident is created, confirmed, resolved or
changes type or severity, it waits for the updates to settle (debounce), builds a fact-only
context from the incident and Postgres, and writes a headline plus a 2-4 sentence summary with
a local model (flan-t5, from the Hugging Face cache) behind a faithfulness guard, falling back
to a deterministic template. Summaries are stored in `summaries` and published to `summaries`.
"""

import os
import threading

from ml.registry import ModelManager
from summarizer_service.config import SummarizerServiceConfig
from summarizer_service.models import default_factory, mlflow_loader, warm_up
from summarizer_service.service import SummarizerService
from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.topics import ConsumerGroup, Topic

SERVICE_NAME = "summarizer"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    get_logger("traffic_common", settings.log_level)
    get_logger("ml", settings.log_level)
    install_signal_handlers(log)
    config = SummarizerServiceConfig.from_env()
    log.info("starting: %s", config)
    import torch

    torch.set_num_threads(int(os.getenv("MODEL_THREADS", "2")))  # see Dockerfile
    stop = shutdown_event()

    engine = create_db_engine(settings.postgres_url)
    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)
    service_ref: dict[str, SummarizerService] = {}

    def on_change(model) -> None:
        warm_up(model)
        if "svc" in service_ref:
            service_ref["svc"].record_model(model)

    models = ModelManager(
        settings.mlflow_tracking_uri,
        {"summarizer": config.registered_model},
        config.model_alias,
        loader=mlflow_loader,
        on_change=on_change,
        fallbacks={"summarizer": default_factory(config)},
    )
    models.refresh()
    service = SummarizerService(config, engine, models, producer.send)
    service_ref["svc"] = service
    service.record_model(models.get("summarizer"))
    service.recover()

    consumer = TypedConsumer(
        [Topic.INCIDENTS],
        ConsumerGroup.SUMMARIZER,
        service.handle,
        settings.kafka_bootstrap_servers,
        stop_event=stop,
        auto_offset_reset="latest",  # history is covered by recover() for the last hour
    )

    def every(interval: float, fn) -> None:
        while not stop.wait(interval):
            try:
                fn()
            except Exception:
                log.exception("%s failed", fn.__name__)

    threads = [
        threading.Thread(target=consumer.run, name="consumer", daemon=True),
        threading.Thread(
            target=every, args=(config.tick_seconds, service.tick), name="worker", daemon=True
        ),
        threading.Thread(
            target=every,
            args=(config.model_refresh_seconds, models.refresh),
            name="model-refresh",
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
            t.join(timeout=30)
        run_shutdown_callbacks(log)
        engine.dispose()
        log.info("stopped")


if __name__ == "__main__":
    main()
