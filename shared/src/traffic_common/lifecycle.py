"""Process lifecycle: one process-wide shutdown event, signal handling and shutdown callbacks.

SIGINT/SIGTERM set the shutdown event. Long-running loops (``TypedConsumer.run`` and the
services' own loops) stop when it is set, then run the registered callbacks (for example
producer flushes) in reverse registration order.
"""

import logging
import os
import signal
import threading
from collections.abc import Callable

_shutdown = threading.Event()
_callbacks: list[Callable[[], None]] = []
_lock = threading.Lock()
_handlers_installed = False

_log = logging.getLogger(__name__)


def shutdown_event() -> threading.Event:
    return _shutdown


def request_shutdown() -> None:
    _shutdown.set()


def on_shutdown(callback: Callable[[], None]) -> None:
    """Register a callback to run once when the process shuts down."""
    with _lock:
        _callbacks.append(callback)


def run_shutdown_callbacks(log: logging.Logger | None = None) -> None:
    """Run and clear registered callbacks, newest first.

    A failing callback is logged and doesn't stop the rest."""
    log = log or _log
    while True:
        with _lock:
            if not _callbacks:
                return
            callback = _callbacks.pop()
        try:
            callback()
        except Exception:
            log.exception("shutdown callback %r failed", callback)


def install_signal_handlers(log: logging.Logger | None = None) -> None:
    """Route SIGINT/SIGTERM to the shutdown event. Idempotent; no-op outside the main thread."""
    global _handlers_installed
    if _handlers_installed or threading.current_thread() is not threading.main_thread():
        return
    log = log or _log
    owner_pid = os.getpid()

    def _handle(signum: int, _frame: object) -> None:
        if os.getpid() != owner_pid:
            # A forked child (e.g. a library's worker process) inherited this handler:
            # terminate it normally instead of logging a misleading shutdown.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        log.info("received %s, shutting down", signal.Signals(signum).name)
        _shutdown.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    _handlers_installed = True
