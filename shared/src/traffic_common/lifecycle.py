"""Process lifecycle helpers."""

import logging
import signal
import threading

_HEARTBEAT_SECONDS = 60.0


def run_forever(log: logging.Logger, heartbeat: float = _HEARTBEAT_SECONDS) -> None:
    """Block until SIGINT/SIGTERM, logging a heartbeat. Keeps placeholder containers running."""
    stop = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        log.info("received %s, shutting down", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    while not stop.wait(heartbeat):
        log.debug("heartbeat")
