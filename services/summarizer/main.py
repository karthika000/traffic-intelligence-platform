"""Summarizer service (placeholder).

Produces human-readable summaries of the current traffic situation.
"""

from traffic_common import get_logger, get_settings, run_forever

SERVICE_NAME = "summarizer"


def main() -> None:
    settings = get_settings(SERVICE_NAME)
    log = get_logger(SERVICE_NAME, settings.log_level)
    log.info("starting in %s mode (not implemented yet)", settings.environment)
    run_forever(log)


if __name__ == "__main__":
    main()
