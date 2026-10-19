"""Event Simulator service.

Emits synthetic text reports (reports.text) and weather / loop-detector events
(events.raw) for every camera in the database, runs labelled scenarios on demand
(simulator.control, the trigger CLI, or a schedule) and records ground truth in
Postgres (simulated_reports, scenario_runs).
"""

import threading

from event_simulator.config import SimulatorConfig
from event_simulator.emitter import Emitter
from event_simulator.groundtruth import GroundTruthStore
from event_simulator.simulator import Simulator
from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.health import heartbeat
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
from traffic_common.messaging import TypedConsumer, TypedProducer
from traffic_common.telemetry import serve_metrics
from traffic_common.topics import ConsumerGroup, Topic

SERVICE_NAME = "event_simulator"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)  # module loggers are event_simulator.*
    get_logger("traffic_common", settings.log_level)
    install_signal_handlers(log)
    serve_metrics()

    config = SimulatorConfig.from_env()
    log.info("starting: %s", config)
    stop = shutdown_event()

    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)
    engine = create_db_engine(settings.postgres_url)
    store = GroundTruthStore(engine)
    emitted = heartbeat("emitter")  # healthcheck: reports and sensor events are going out

    def send(topic, message):
        sent = producer.send(topic, message)
        emitted.beat()
        return sent

    simulator = Simulator(config, engine, Emitter(send, store), store, stop)

    control = TypedConsumer(
        [Topic.SIMULATOR_CONTROL],
        ConsumerGroup.EVENT_SIMULATOR,
        simulator.handle_command,
        settings.kafka_bootstrap_servers,
        stop_event=stop,
        auto_offset_reset="latest",  # don't replay old commands when the group is new
    )
    control_thread = threading.Thread(target=control.run, name="control", daemon=True)
    control_thread.start()
    try:
        simulator.run()
    finally:
        control_thread.join(timeout=15)
        run_shutdown_callbacks(log)  # flushes the producer
        engine.dispose()
        log.info("stopped")


if __name__ == "__main__":
    main()
