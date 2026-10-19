"""ml-monitor: scheduled drift checks for every monitored model, and an on-demand CLI.

    python -m ml.monitoring          # the service: every DRIFT_CHECK_INTERVAL_MINUTES
    python -m ml.monitoring check [--model M] [--window-minutes N] [--no-publish]
                                  [--no-mlflow] [--json]

In Docker: ``docker compose exec ml-monitor python -m ml.monitoring check``. The service also
consumes drift.alerts (group ml-monitoring) to record retraining recommendations.
"""

import argparse
import json
import logging
import os
import threading

from ml.monitoring.config import MONITORED_MODELS, MonitorConfig

SERVICE_NAME = "ml-monitor"


def _setup(config: MonitorConfig):
    import mlflow
    import torch

    from traffic_common.config import get_settings

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:  # one INFO line per Hugging Face request
        handler.addFilter(lambda record: not record.name.startswith("httpx"))
    torch.set_num_threads(config.threads)
    settings = get_settings(SERVICE_NAME)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return settings


def serve(config: MonitorConfig) -> None:
    from ml.monitoring.monitor import Monitor
    from ml.monitoring.scheduler import Scheduler
    from traffic_common import get_logger, shutdown_event
    from traffic_common.db import create_db_engine
    from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
    from traffic_common.messaging import TypedConsumer, TypedProducer
    from traffic_common.telemetry import serve_metrics
    from traffic_common.topics import ConsumerGroup, Topic

    log = get_logger(SERVICE_NAME, os.getenv("LOG_LEVEL", "INFO"))
    get_logger("ml", os.getenv("LOG_LEVEL", "INFO"))
    get_logger("traffic_common", os.getenv("LOG_LEVEL", "INFO"))
    settings = _setup(config)
    install_signal_handlers(log)
    serve_metrics()
    stop = shutdown_event()
    log.info("starting: %s", config)

    engine = create_db_engine(settings.postgres_url)
    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)
    monitor = Monitor(config, engine, bootstrap=settings.kafka_bootstrap_servers, producer=producer)
    alerts = TypedConsumer(
        [Topic.DRIFT_ALERTS],
        ConsumerGroup.ML_MONITORING,
        monitor.on_alert,
        settings.kafka_bootstrap_servers,
        stop_event=stop,
        auto_offset_reset="latest",
    )
    consumer_thread = threading.Thread(target=alerts.run, name="drift-alerts", daemon=True)
    consumer_thread.start()
    scheduler = Scheduler(
        config.interval_minutes * 60, monitor.run_all, stop, run_at_start=config.run_at_start
    )
    log.info(
        "checking %s every %.0f min over the last %.0f min (auto retrain %s)",
        ", ".join(config.models),
        config.interval_minutes,
        config.window_minutes,
        "on" if config.auto_retrain else "off: alerts record a recommendation",
    )
    try:
        scheduler.run_forever()
    finally:
        consumer_thread.join(timeout=15)
        run_shutdown_callbacks(log)
        engine.dispose()
        log.info("stopped (%s)", scheduler.stats)


def check(args: argparse.Namespace, config: MonitorConfig) -> None:
    from ml.monitoring.monitor import Monitor
    from traffic_common.db import create_db_engine
    from traffic_common.messaging import TypedProducer

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
    )
    settings = _setup(config)
    engine = create_db_engine(settings.postgres_url)
    producer = (
        None
        if args.no_publish or args.dry_run
        else TypedProducer(settings.kafka_bootstrap_servers, client_id=f"{SERVICE_NAME}-cli")
    )
    monitor = Monitor(config, engine, bootstrap=settings.kafka_bootstrap_servers, producer=producer)
    results = []
    for model in args.model or config.models:
        results.append(
            monitor.check(
                model,
                window_minutes=args.window_minutes,
                now=args.at,
                publish=not args.no_publish,
                log_mlflow=not args.no_mlflow,
                store=not args.dry_run,
            )
        )
    if producer is not None:
        producer.close()
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return
    for r in results:
        print(f"\n{r.model_name} v{r.model_version}: {r.status.upper()}  "
              f"drift share {r.drift_share:.2f} (threshold {r.share_threshold:.2f})  "
              f"rows current {r.current_rows} / reference {r.reference_rows}")  # fmt: skip
        if r.note:
            print(f"  reference: {r.note}")
        for f in r.features:
            score = "   n/a" if f.score is None else f"{f.score:6.3f}"
            flag = "DRIFT" if f.drifted else "     "
            print(f"  {flag} {f.feature:<24} {f.method:<14} {score}  (threshold {f.threshold:g})"
                  + (f"  {f.note}" if f.note else ""))  # fmt: skip
        for q in r.quality:
            print(f"  QUALITY {q.message}")
        if r.html_path:
            print(f"  report: {r.html_path}  (/api/v1/drift/reports/{r.check_id})")


def _utc(value: str):
    from datetime import UTC, datetime

    t = datetime.fromisoformat(value)
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    c = sub.add_parser("check", help="run drift checks now")
    c.add_argument("--model", action="append", choices=MONITORED_MODELS, help="repeatable")
    c.add_argument("--window-minutes", type=float, default=None)
    c.add_argument(
        "--at",
        type=_utc,
        default=None,
        help="end of the window (ISO time, UTC if no offset); default now. Frame files are "
        "only kept ~10 min, so older windows have no brightness/contrast",
    )
    c.add_argument("--no-publish", action="store_true", help="don't publish drift alerts")
    c.add_argument("--no-mlflow", action="store_true", help="don't log to MLflow")
    c.add_argument(
        "--dry-run", action="store_true", help="record nothing (no rows, MLflow run or alert)"
    )
    c.add_argument("--json", action="store_true", help="print the full results as JSON")
    args = parser.parse_args()
    config = MonitorConfig.from_env()
    if args.command == "check":
        check(args, config)
    else:
        serve(config)


if __name__ == "__main__":
    main()
