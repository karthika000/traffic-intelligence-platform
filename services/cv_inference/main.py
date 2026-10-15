"""CV Inference service.

Consumes video.frames, runs YOLO + ByteTrack on CPU (OpenVINO by default), and publishes
a Detection per frame to `detections` and TrafficMetrics every few seconds per camera to
`traffic.metrics` (also stored in the traffic_metrics table). The latest annotated frame
per camera is written to /data/annotated/<camera_id>/latest.jpg.
"""

import cv2

from cv_inference.config import CvConfig
from cv_inference.service import CvService
from traffic_common import get_logger, get_settings, shutdown_event
from traffic_common.db import create_db_engine
from traffic_common.lifecycle import install_signal_handlers, run_shutdown_callbacks

SERVICE_NAME = "cv_inference"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)  # module loggers are cv_inference.*
    get_logger("traffic_common", settings.log_level)
    install_signal_handlers(log)

    config = CvConfig.from_env()
    log.info("starting: %s", config)
    # OpenCV's own thread pool (decode, resize, JPEG encode) defaults to every core and
    # competes with inference; these per-frame operations are small.
    cv2.setNumThreads(2)
    engine = create_db_engine(settings.postgres_url)
    service = CvService(config, settings, engine, shutdown_event())
    try:
        service.load_model()
        service.run()
    finally:
        run_shutdown_callbacks(log)  # flushes the producer
        engine.dispose()
        log.info("stopped")


if __name__ == "__main__":
    main()
