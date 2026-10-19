"""Evidence and incident state. Everything round-trips through plain dicts (``to_dict`` /
``from_dict``) because an open incident's whole state is stored in incidents.details and
rebuilt from there after a restart."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

from fusion_service.types import is_traffic


def _ts(value) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass
class AnomalyEvidence:
    anomaly_id: str
    camera_id: str
    started_at: datetime
    last_seen: datetime
    peak_ratio: float  # peak score / threshold
    hints: list[str]
    severity: str | None = None
    congestion: str | None = None
    active: bool = True
    ended_at: datetime | None = None
    degraded: bool = False  # observed during a system-wide degradation
    # anomaly_service: since when the traffic metrics have been back near the frozen baseline
    # while the alert is still active (None while they aren't)
    recovered_since: datetime | None = None

    @property
    def traffic(self) -> bool:
        return is_traffic(self.hints)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("started_at", "last_seen", "ended_at", "recovered_since"):
            d[k] = _iso(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyEvidence":
        d = dict(d)
        for k in ("started_at", "last_seen", "ended_at", "recovered_since"):
            d[k] = _ts(d.get(k))
        return cls(**d)


@dataclass
class ReportEvidence:
    report_id: str
    camera_id: str | None
    ts: datetime
    incident_type: str
    confidence: float
    severity: str | None = None
    source: str | None = None
    camera_confidence: float = 0.0
    stage: str = "active"
    places: list[str] = field(default_factory=list)  # normalised LOCATION entity texts
    unlocated: bool = False  # attached through its place entities, not a camera match

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = _iso(self.ts)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReportEvidence":
        d = dict(d)
        d["ts"] = _ts(d["ts"])
        return cls(**d)


@dataclass
class MetricPoint:
    ts: datetime
    avg_speed: float | None
    stopped: int
    message_id: str


@dataclass
class FusedIncident:
    incident_id: uuid.UUID
    camera_id: str
    status: str  # suspected | confirmed | resolved
    incident_type: str
    severity: str
    # When the incident of its current type began: the first evidence supporting that type
    # (FusionEngine._start_time). first_evidence_at is the earliest evidence attached at all,
    # which can be a background report from before the event.
    start_time: datetime
    suspected_at: datetime
    last_evidence_at: datetime
    first_evidence_at: datetime | None = None
    confirmed_at: datetime | None = None
    end_time: datetime | None = None
    # Last time the incident was "firing" (anomaly active, or a burst of recent supporting
    # reports); quiet periods count from here, so stray background reports can't keep it open.
    last_active_at: datetime | None = None
    version: int = 0  # publications so far
    fused: float = 0.0
    components: dict = field(default_factory=dict)
    anomalies: dict[str, AnomalyEvidence] = field(default_factory=dict)
    reports: dict[str, ReportEvidence] = field(default_factory=dict)
    metric_ids: list[str] = field(default_factory=list)
    baseline: dict | None = None  # camera speed / stopped vehicles when it started
    cleared_at: datetime | None = None  # first report saying it's over
    resolution: str | None = None  # why it was resolved
    at_confirmation: dict | None = None  # confidence breakdown when confirmed
    published: dict = field(default_factory=dict)  # what the last published version said
    history: list[dict] = field(default_factory=list)  # one entry per published version

    @property
    def open(self) -> bool:
        return self.status != "resolved"

    @property
    def vision_degraded(self) -> bool:
        return any(a.degraded for a in self.anomalies.values() if a.traffic)

    @property
    def active_anomalies(self) -> list[AnomalyEvidence]:
        """Reliable traffic anomalies still active. Degraded ones (seen during a system
        degradation, or on a camera reporting itself degraded) can't hold an incident open."""
        return [a for a in self.anomalies.values() if a.active and a.traffic and not a.degraded]

    @property
    def active_anomaly(self) -> bool:
        return bool(self.active_anomalies)

    def places(self) -> set[str]:
        return {p for r in self.reports.values() for p in r.places}

    def to_dict(self) -> dict:
        return {
            "incident_id": str(self.incident_id),
            "camera_id": self.camera_id,
            "status": self.status,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "start_time": _iso(self.start_time),
            "suspected_at": _iso(self.suspected_at),
            "last_evidence_at": _iso(self.last_evidence_at),
            "first_evidence_at": _iso(self.first_evidence_at),
            "last_active_at": _iso(self.last_active_at),
            "confirmed_at": _iso(self.confirmed_at),
            "end_time": _iso(self.end_time),
            "version": self.version,
            "fused": self.fused,
            "components": self.components,
            "anomalies": {k: a.to_dict() for k, a in self.anomalies.items()},
            "reports": {k: r.to_dict() for k, r in self.reports.items()},
            "metric_ids": self.metric_ids,
            "baseline": self.baseline,
            "cleared_at": _iso(self.cleared_at),
            "resolution": self.resolution,
            "at_confirmation": self.at_confirmation,
            "published": self.published,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FusedIncident":
        return cls(
            incident_id=uuid.UUID(d["incident_id"]),
            camera_id=d["camera_id"],
            status=d["status"],
            incident_type=d["incident_type"],
            severity=d["severity"],
            start_time=_ts(d["start_time"]),
            suspected_at=_ts(d["suspected_at"]),
            last_evidence_at=_ts(d["last_evidence_at"]),
            # states saved before first_evidence_at existed kept the earliest one in start_time
            first_evidence_at=_ts(d.get("first_evidence_at") or d["start_time"]),
            last_active_at=_ts(d.get("last_active_at")),
            confirmed_at=_ts(d.get("confirmed_at")),
            end_time=_ts(d.get("end_time")),
            version=d.get("version", 0),
            fused=d.get("fused", 0.0),
            components=d.get("components", {}),
            anomalies={k: AnomalyEvidence.from_dict(a) for k, a in d.get("anomalies", {}).items()},
            reports={k: ReportEvidence.from_dict(r) for k, r in d.get("reports", {}).items()},
            metric_ids=d.get("metric_ids", []),
            baseline=d.get("baseline"),
            cleared_at=_ts(d.get("cleared_at")),
            resolution=d.get("resolution"),
            at_confirmation=d.get("at_confirmation"),
            published=d.get("published", {}),
            history=d.get("history", []),
        )


@dataclass
class Change:
    """What happened to an incident in one step: persist it always, publish if ``publish``."""

    incident: FusedIncident
    reasons: list[str]
    publish: bool
    new_evidence: list[dict] = field(default_factory=list)  # rows for incident_evidence
