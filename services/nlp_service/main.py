"""NLP Service service (placeholder).

Extracts entities and intents from text sources such as incident reports and social feeds.
"""

from traffic_common import get_logger, get_settings, run_forever

SERVICE_NAME = "nlp_service"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    log.info("starting in %s mode (not implemented yet)", settings.environment)
    run_forever(log)


if __name__ == "__main__":
    main()
