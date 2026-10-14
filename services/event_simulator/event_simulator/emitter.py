"""Turns generated reports and readings into Kafka messages, recording ground truth first."""

import logging
import random
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from event_simulator.groundtruth import GroundTruthStore
from event_simulator.locations import CameraInfo
from event_simulator.reports import GeneratedReport, Style
from traffic_common.schemas import KafkaMessage, RawEvent, TextReport
from traffic_common.topics import Topic

log = logging.getLogger(__name__)

Send = Callable[[str, KafkaMessage], Any]
SOURCE_NAME = "event_simulator"

# Share of reports carrying device coordinates (phone apps geotag; official feeds don't).
_GEOTAG_PROBABILITY = {Style.DRIVER: 0.4, Style.SOCIAL_MEDIA: 0.3}
_GEOTAG_JITTER_DEG = 0.0012  # roughly 100 m


@dataclass
class EmitterStats:
    reports: int = 0
    events: int = 0
    db_errors: int = 0
    send_errors: int = 0


class Emitter:
    def __init__(
        self, send: Send, store: GroundTruthStore | None, rng: random.Random | None = None
    ):
        self._send = send
        self._store = store
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self.stats = EmitterStats()

    def report(
        self,
        camera: CameraInfo,
        report: GeneratedReport,
        scenario_id: uuid.UUID | None = None,
        phase: str | None = None,
    ) -> TextReport:
        latitude = longitude = None
        if self._rng.random() < _GEOTAG_PROBABILITY.get(report.style, 0.0):
            latitude = camera.latitude + self._rng.uniform(-_GEOTAG_JITTER_DEG, _GEOTAG_JITTER_DEG)
            longitude = camera.longitude + self._rng.uniform(
                -_GEOTAG_JITTER_DEG, _GEOTAG_JITTER_DEG
            )
        message = TextReport(
            source=report.source,
            text=report.text,
            camera_id=camera.id,
            latitude=latitude,
            longitude=longitude,
            scenario_id=scenario_id,
        )
        if self._store is not None:
            try:
                self._store.record_report(
                    message.report_id, report, message.timestamp, scenario_id, phase
                )
            except Exception as exc:  # ground truth is best effort; keep simulating
                self._count("db_errors")
                log.error("could not record report %s: %s", message.report_id, exc)
        self._publish(Topic.REPORTS_TEXT, message, "reports")
        return message

    def event(
        self,
        camera: CameraInfo,
        event_type: str,
        payload: dict[str, Any],
        scenario_id: uuid.UUID | None = None,
    ) -> RawEvent:
        message = RawEvent(
            event_type=event_type,
            source=SOURCE_NAME,
            camera_id=camera.id,
            payload=payload,
            scenario_id=scenario_id,
        )
        self._publish(Topic.EVENTS_RAW, message, "events")
        return message

    def _publish(self, topic: str, message: KafkaMessage, counter: str) -> None:
        try:
            self._send(topic, message)
        except Exception as exc:
            self._count("send_errors")
            log.error("could not publish to %s: %s", topic, exc)
            return
        self._count(counter)

    def _count(self, name: str) -> None:
        with self._lock:
            setattr(self.stats, name, getattr(self.stats, name) + 1)
