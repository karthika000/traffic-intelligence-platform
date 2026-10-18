"""Video Ingest service.

Reads every camera in the `cameras` table (local files loop forever; RTSP and HTTP
streams reconnect with backoff), samples frames, writes them as JPEGs to the shared
frames volume and publishes a FrameMessage per frame to `video.frames`.
"""

import logging

from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.health import heartbeat
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks
from traffic_common.messaging import TypedProducer
from traffic_common.schemas import FrameMessage
from traffic_common.topics import Topic
from video_ingest.config import IngestConfig
from video_ingest.supervisor import Supervisor

SERVICE_NAME = "video_ingest"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)  # module loggers are video_ingest.*
    get_logger("traffic_common", settings.log_level)  # producer delivery errors
    install_signal_handlers(log)

    config = IngestConfig.from_env()
    log.info("starting: %s", config)
    config.frames_dir.mkdir(parents=True, exist_ok=True)

    producer = TypedProducer(settings.kafka_bootstrap_servers, client_id=SERVICE_NAME)

    frames = heartbeat("frames")  # healthcheck: some camera is publishing frames

    def publish(message: FrameMessage) -> None:
        producer.send(Topic.VIDEO_FRAMES, message)
        frames.beat()

    engine = create_db_engine(settings.postgres_url)
    try:
        Supervisor(config, engine, publish, shutdown_event()).run()
    finally:
        run_shutdown_callbacks(log)  # flushes the producer
        engine.dispose()
        log.info("stopped")


if __name__ == "__main__":
    logging.captureWarnings(True)
    main()
