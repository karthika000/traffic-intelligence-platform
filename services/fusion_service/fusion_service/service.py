"""Wire the engine to Kafka and Postgres: messages in, persisted incidents and published
Incident versions out. One lock serialises the consumer thread and the tick thread."""

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine

from fusion_service.config import FusionConfig
from fusion_service.engine import FusionEngine
from fusion_service.model import Change, FusedIncident
from fusion_service.store import IncidentStore
from traffic_common.schemas import Anomaly, Incident, KafkaMessage, NLPResult, TrafficMetrics
from traffic_common.topics import Topic

log = logging.getLogger(__name__)


@dataclass
class Stats:
    anomalies: int = 0
    reports: int = 0
    metrics: int = 0
    created: int = 0
    confirmed: int = 0
    resolved: int = 0
    published: int = 0

    def reset(self) -> str:
        text = " ".join(f"{k}={v}" for k, v in vars(self).items())
        self.__init__()  # type: ignore[misc]
        return text


def title_of(inc: FusedIncident, camera_names: dict[str, str]) -> str:
    place = camera_names.get(inc.camera_id, inc.camera_id)
    return f"{inc.incident_type.replace('_', ' ').capitalize()} at {place}"[:200]


def description_of(inc: FusedIncident) -> str:
    c = inc.components
    v, t = c.get("vision", {}), c.get("text", {})
    parts = [
        f"{inc.status.capitalize()} {inc.incident_type.replace('_', ' ')}, "
        f"{inc.severity} severity, confidence {inc.fused:.2f}."
    ]
    if t.get("present"):
        parts.append(
            f"{t['supporting_reports']:g} supporting reports ({t['recent_supporting_reports']:g} "
            f"recent) from {t['n_sources']} sources "
            f"(classifier {t['classifier']:.2f}, camera match {t['camera_match']:.2f})."
        )
    if v.get("present"):
        parts.append(
            f"Traffic anomaly at {v['peak_ratio']:.1f}x its threshold for "
            f"{v['active_seconds']:.0f}s ({', '.join(v['hints'])})"
            + (", seen during a system-wide degradation." if v.get("degraded") else ".")
        )
    if inc.resolution:
        parts.append(f"Resolved: {inc.resolution}.")
    return " ".join(parts)


def to_message(inc: FusedIncident, reasons: list[str], camera_names: dict[str, str]) -> Incident:
    reports = [uuid.UUID(k) for k in inc.reports]
    return Incident(
        incident_id=inc.incident_id,
        camera_id=inc.camera_id,
        version=inc.version,
        status=inc.status,
        incident_type=inc.incident_type,
        severity=inc.severity,
        fused_confidence=inc.fused,
        confidence_components=inc.components,
        vision_degraded=inc.vision_degraded,
        start_time=inc.start_time,
        first_evidence_at=inc.first_evidence_at,
        suspected_at=inc.suspected_at,
        confirmed_at=inc.confirmed_at,
        end_time=inc.end_time,
        title=title_of(inc, camera_names),
        description=description_of(inc),
        change_reasons=reasons,
        metric_message_ids=[uuid.UUID(m) for m in inc.metric_ids],
        anomaly_ids=[uuid.UUID(a) for a in inc.anomalies],
        report_ids=reports,
    )


class FusionService:
    def __init__(
        self,
        config: FusionConfig,
        engine: Engine,
        send: Callable,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.cfg = config
        self.store = IncidentStore(engine)
        self.send = send  # (topic, message) -> None
        self.clock = clock
        self.stats = Stats()
        self._lock = threading.Lock()
        self._ticks = 0
        self.fusion = FusionEngine(config, self.store.camera_names())

    def recover(self) -> int:
        """Rebuild open incidents and recent evidence from Postgres (call before consuming)."""
        with self._lock:
            incidents = self.store.load_open()
            self.fusion.restore(
                incidents,
                self.store.recent_anomalies(self.cfg.window_seconds),
                self.store.recent_reports(self.cfg.window_seconds),
                self.store.system_state(),
            )
        if incidents:
            log.info(
                "recovered %d open incident(s): %s",
                len(incidents),
                [f"{i.camera_id}:{i.status}" for i in incidents],
            )
        return len(incidents)

    def handle(self, message: KafkaMessage) -> None:
        """Consumer handler for anomalies, nlp.results and traffic.metrics."""
        with self._lock:
            now = self.clock()
            if isinstance(message, Anomaly):
                self.stats.anomalies += 1
                self.fusion.on_anomaly(message, now)
            elif isinstance(message, NLPResult):
                self.stats.reports += 1
                self.fusion.on_nlp(message, now)
            elif isinstance(message, TrafficMetrics):
                self.stats.metrics += 1
                self.fusion.on_metrics(message)
            self._flush()

    def tick(self) -> None:
        with self._lock:
            self._ticks += 1
            if self._ticks % 60 == 0:  # ~5 minutes: cameras added since startup
                self.fusion.camera_names = self.store.camera_names()
            try:
                for camera_id, level in self.store.congestion_levels().items():
                    self.fusion.set_congestion(camera_id, level)
            except Exception as exc:
                log.warning("could not read congestion levels: %s", exc)
            self.fusion.tick(self.clock())
            self._flush()
            self.store.flush_evidence()

    def _flush(self) -> None:
        for change in self.fusion.drain():
            self._apply(change)

    def _apply(self, change: Change) -> None:
        inc = change.incident
        names = self.fusion.camera_names
        self.store.save(change, title_of(inc, names), description_of(inc))
        if not change.publish:
            return
        self.send(Topic.INCIDENTS, to_message(inc, change.reasons, names))
        self.stats.published += 1
        if inc.version == 1:
            self.stats.created += 1
        if any(r.startswith("confirmed") for r in change.reasons):
            self.stats.confirmed += 1
        if inc.status == "resolved":
            self.stats.resolved += 1
        log.warning(
            "INCIDENT v%d %s camera=%s type=%s severity=%s confidence=%.2f "
            "(vision %.2f, text %.2f, agreement %.2f) reasons=%s id=%s",
            inc.version,
            inc.status,
            inc.camera_id,
            inc.incident_type,
            inc.severity,
            inc.fused,
            inc.components["vision"]["score"],
            inc.components["text"]["score"],
            inc.components["agreement"]["score"],
            change.reasons,
            inc.incident_id,
        )

    def log_stats(self) -> None:
        with self._lock:
            open_ = [
                f"{i.camera_id}:{i.status}/{i.incident_type}"
                for i in self.fusion.incidents.values()
            ]
            pending = len(self.store.pending)
            system = sorted(self.fusion.system["cameras"]) if self.fusion.system else "ok"
        log.info(
            "stats %s open=%s pending_evidence=%d system=%s",
            self.stats.reset(),
            open_ or "none",
            pending,
            system,
        )
