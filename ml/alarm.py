"""Turning a stream of anomaly scores into alarm episodes (used by anomaly_service and by
the offline evaluation, so both apply exactly the same rules).

- start:    score >= threshold for min_consecutive checks in a row, while idle and not
            cooling down (one noisy window doesn't raise an alarm)
- end:      score < clear_ratio * threshold for clear_after consecutive checks, while the
            caller's ``hold`` condition is off (e.g. speeds still far below the baseline
            frozen when the alert started keep it open even if the score has come down)
- cooldown: no new start for cooldown_s after an end (stops one incident flapping into many)
"""

from dataclasses import dataclass, field
from enum import StrEnum


class AlarmEvent(StrEnum):
    START = "start"
    UPDATE = "update"  # raised by the service (escalation), not by the tracker
    END = "end"


@dataclass
class AlarmTracker:
    cooldown_s: float = 120.0
    clear_ratio: float = 0.8
    clear_after: int = 3
    min_consecutive: int = 2
    active: bool = False
    started_at: float | None = None
    peak_score: float = 0.0
    held_checks: int = 0  # checks the hold condition kept an otherwise-clearing alert open
    _below: int = 0
    _above: int = 0
    _first_above: float | None = None
    _cooldown_until: float = field(default=float("-inf"))

    def update(
        self, t: float, score: float, threshold: float, hold: bool = False
    ) -> AlarmEvent | None:
        if not self.active:
            if score >= threshold and t >= self._cooldown_until:
                self._above += 1
                if self._first_above is None:
                    self._first_above = t
                self.peak_score = max(self.peak_score, score) if self._above > 1 else score
                if self._above >= self.min_consecutive:
                    # The episode started when the score first crossed, not when it was confirmed.
                    self.active, self.started_at, self._below = True, self._first_above, 0
                    self.held_checks = 0
                    self._above, self._first_above = 0, None
                    return AlarmEvent.START
            else:
                self._above, self._first_above = 0, None
            return None
        self.peak_score = max(self.peak_score, score)
        if score < self.clear_ratio * threshold and hold:
            self.held_checks += 1
            self._below = 0
        elif score < self.clear_ratio * threshold:
            self._below += 1
            if self._below >= self.clear_after:
                self.active = False
                self._cooldown_until = t + self.cooldown_s
                return AlarmEvent.END
        else:
            self._below = 0
        return None

    def suppress(self) -> None:
        """Forget a start that is building up (e.g. while a system-wide degradation explains
        the high scores), so it can't fire the moment suppression lifts."""
        self._above, self._first_above = 0, None

    def close(self, t: float, cooldown: bool = True) -> None:
        """End the active episode from outside (no END event is returned)."""
        self.active, self._below = False, 0
        if cooldown:
            self._cooldown_until = t + self.cooldown_s


@dataclass(frozen=True)
class Episode:
    start: float
    end: float | None
    peak: float


def replay(times, scores, threshold: float, **tracker_kwargs) -> list[Episode]:
    """Alarm episodes for a score series (seconds, scores) with the service's rules."""
    tracker = AlarmTracker(**tracker_kwargs)
    episodes, start = [], None
    for t, s in zip(times, scores, strict=True):
        event = tracker.update(float(t), float(s), threshold)
        if event is AlarmEvent.START:
            start = tracker.started_at  # the first crossing, not the confirming check
        elif event is AlarmEvent.END:
            episodes.append(Episode(start, float(t), tracker.peak_score))
            start = None
    if start is not None:
        episodes.append(Episode(start, None, tracker.peak_score))
    return episodes
