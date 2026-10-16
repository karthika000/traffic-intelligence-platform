"""NLP service.

Consumes reports.text (group nlp-service) in batches. Each report is stored in `reports`,
classified by the incident-text-classifier champion from MLflow (incident type, severity
and their probabilities; a keyword fallback until one is registered), run through NER and
the rule-based entity extractor, and matched to a camera from its text. The result is
stored in `nlp_results` and published to nlp.results.
"""

import os
import threading

from ml.nlp.ner import NerExtractor
from ml.nlp.rules import KeywordIncidentClassifier
from ml.registry import ModelManager
from nlp_service.config import NlpConfig
from nlp_service.service import NlpService
from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.db.registry import record_model_version
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.topics import ConsumerGroup, Topic

SERVICE_NAME = "nlp_service"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    get_logger("traffic_common", settings.log_level)
    get_logger("ml", settings.log_level)
    install_signal_handlers(log)
    config = NlpConfig.from_env()
    log.info("starting: %s", config)
    import torch

    torch.set_num_threads(int(os.getenv("MODEL_THREADS", "2")))  # see Dockerfile
    stop = shutdown_event()

    engine = create_db_engine(settings.postgres_url)
    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)
    ner = None
    if config.ner_enabled:
        ner = NerExtractor(config.ner_model, config.ner_min_score).load()
        try:
            record_model_version(
                engine,
                config.ner_model,
                ner.version,
                metrics={"role": "ner", "loaded_by": SERVICE_NAME},
            )
        except Exception as exc:
            log.warning("could not record the NER model in model_versions: %s", exc)
    service_ref: dict[str, NlpService] = {}
    models = ModelManager(
        settings.mlflow_tracking_uri,
        {"classifier": config.classifier_model},
        config.model_alias,
        on_change=lambda m: service_ref["svc"].record_model(m) if "svc" in service_ref else None,
        fallbacks={"classifier": KeywordIncidentClassifier},
    )
    models.refresh()
    service = NlpService(config, engine, models, producer.send, ner)
    service_ref["svc"] = service
    service.record_model(models.get("classifier"))

    consumer = TypedConsumer(
        [Topic.REPORTS_TEXT],
        ConsumerGroup.NLP_SERVICE,
        None,
        settings.kafka_bootstrap_servers,
        batch_handler=service.handle_batch,
        max_batch=config.max_batch,
        stop_event=stop,
        # earliest: a new group analyses the reports already on the topic.
        auto_offset_reset="earliest",
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
            target=every,
            args=(config.model_refresh_seconds, models.refresh),
            name="model-refresh",
            daemon=True,
        ),
        threading.Thread(
            target=every,
            args=(config.camera_refresh_seconds, service.refresh_cameras),
            name="camera-refresh",
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
