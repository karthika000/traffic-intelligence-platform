"""A fixed-interval scheduler for the drift checks.

Runs are due every ``interval`` seconds from the start (the first one at once with
``run_at_start``). A run that overruns doesn't cause a burst of catch-up runs: missed slots are
skipped and the next run is the next slot still in the future. A failing job is logged and
the schedule carries on. ``trigger()`` asks for a run as soon as possible (on demand)."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from traffic_common.health import heartbeat

log = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    runs: int = 0
    failures: int = 0
    skipped_slots: int = 0


class Scheduler:
    def __init__(
        self,
        interval: float,
        job: Callable[[], object],
        stop: threading.Event,
        *,
        run_at_start: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = interval
        self.job = job
        self._stop = stop
        self._clock = clock
        self._start = clock()
        self.next_due = self._start if run_at_start else self._start + interval
        self.stats = SchedulerStats()

    def trigger(self) -> None:
        """Run as soon as possible (the regular schedule is unchanged)."""
        self.next_due = min(self.next_due, self._clock())

    def due(self) -> bool:
        return self._clock() >= self.next_due

    def run_pending(self) -> bool:
        """Run the job if it is due; returns whether it ran."""
        if not self.due():
            return False
        try:
            self.job()
        except Exception:
            self.stats.failures += 1
            log.exception("scheduled job failed")
        finally:
            self.stats.runs += 1
            self._advance()
        return True

    def _advance(self) -> None:
        now = self._clock()
        slots = int((now - self._start) // self.interval) + 1
        following = self._start + slots * self.interval
        expected = max(self.next_due, self._start) + self.interval
        if following > expected:
            self.stats.skipped_slots += int(round((following - expected) / self.interval))
        self.next_due = following

    def seconds_until_due(self) -> float:
        return max(0.0, self.next_due - self._clock())

    def run_forever(self) -> None:
        hb = heartbeat("scheduler")  # healthcheck; silent while a check runs (minutes)
        while not self._stop.is_set():
            hb.beat()
            self.run_pending()
            # Short waits, so trigger() and shutdown take effect within a second.
            if self._stop.wait(min(self.seconds_until_due(), 1.0)):
                return
