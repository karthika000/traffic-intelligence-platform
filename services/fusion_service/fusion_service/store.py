"""Postgres: incident rows (with the full engine state in details, for restart recovery),
incident_evidence links, and the queries that rebuild recent evidence pools on startup.

Evidence can arrive here before its target row exists (anomaly_service publishes before it
stores; cv_inference may store metrics after publishing), so a link whose target is missing
is kept and retried on later ticks instead of failing the foreign key."""

import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, select, text
from sqlalchemy.dialects.postgresql import insert

from fusion_service.engine import places_from_entities
from fusion_service.model import AnomalyEvidence, Change, FusedIncident, ReportEvidence
from ml.nlp.stage import ACTIVE
from traffic_common.db.models import Camera
from traffic_common.db.models import Incident as IncidentRow

log = logging.getLogger(__name__)
MAX_EVIDENCE_ATTEMPTS = 60  # x tick (5 s) = 5 minutes

_TARGET_SQL = {
    "anomaly": (
        "INSERT INTO incident_evidence (incident_id, evidence_type, anomaly_id, weight, details) "
        "SELECT :incident_id, 'anomaly', a.id, :weight, CAST(:details AS jsonb) FROM anomalies a "
        "WHERE a.id = CAST(:target AS uuid) AND NOT EXISTS (SELECT 1 FROM incident_evidence e "
        "WHERE e.incident_id = :incident_id AND e.anomaly_id = a.id)"
    ),
    "report": (
        "INSERT INTO incident_evidence (incident_id, evidence_type, report_id, weight, details) "
        "SELECT :incident_id, 'report', r.id, :weight, CAST(:details AS jsonb) FROM reports r "
        "WHERE r.id = CAST(:target AS uuid) AND NOT EXISTS (SELECT 1 FROM incident_evidence e "
        "WHERE e.incident_id = :incident_id AND e.report_id = r.id)"
    ),
    "metric": (
        "INSERT INTO incident_evidence (incident_id, evidence_type, traffic_metric_id, weight, "
        "details) SELECT :incident_id, 'metric', m.id, :weight, CAST(:details AS jsonb) "
        "FROM traffic_metrics m WHERE m.message_id = CAST(:target AS uuid) AND NOT EXISTS "
        "(SELECT 1 FROM incident_evidence e WHERE e.incident_id = :incident_id "
        "AND e.traffic_metric_id = m.id)"
    ),
}
_TARGET_KEY = {"anomaly": "anomaly_id", "report": "report_id", "metric": "metric_message_id"}
# Whether the link exists (just inserted, or already there from before a restart).
_LINKED_SQL = {
    "anomaly": "SELECT count(*) FROM incident_evidence WHERE incident_id = :incident_id "
    "AND anomaly_id = CAST(:target AS uuid)",
    "report": "SELECT count(*) FROM incident_evidence WHERE incident_id = :incident_id "
    "AND report_id = CAST(:target AS uuid)",
    "metric": "SELECT count(*) FROM incident_evidence e JOIN traffic_metrics m "
    "ON m.id = e.traffic_metric_id WHERE e.incident_id = :incident_id "
    "AND m.message_id = CAST(:target AS uuid)",
}


class IncidentStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.pending: list[tuple[str, dict, int]] = []  # (incident id, row, attempts)
        self.evidence_dropped = 0

    # ------------------------------------------------------------------ writes

    def save(self, change: Change, title: str, description: str) -> None:
        inc = change.incident
        values = {
            "id": inc.incident_id,
            "camera_id": inc.camera_id,
            "status": inc.status,
            "incident_type": inc.incident_type,
            "version": max(inc.version, 1),
            "severity": inc.severity,
            "fused_confidence": inc.fused,
            "confidence_components": inc.components,
            "vision_degraded": inc.vision_degraded,
            "start_time": inc.start_time,
            "suspected_at": inc.suspected_at,
            "confirmed_at": inc.confirmed_at,
            "end_time": inc.end_time,
            "title": title,
            "description": description,
            "details": {
                "state": inc.to_dict(),
                "resolution": inc.resolution,
                "at_confirmation": inc.at_confirmation,
                "history": inc.history,
            },
        }
        stmt = insert(IncidentRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[IncidentRow.id],
            set_={k: stmt.excluded[k] for k in values if k != "id"}
            | {"updated_at": datetime.now(UTC)},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        for row in change.new_evidence:
            self.pending.append((str(inc.incident_id), row, 0))
        self.flush_evidence()

    def flush_evidence(self) -> None:
        """Insert evidence links whose targets exist; keep the rest for a later try."""
        keep = []
        for incident_id, row, attempts in self.pending:
            kind = row["evidence_type"]
            target = row[_TARGET_KEY[kind]]
            params = {
                "incident_id": incident_id,
                "target": target,
                "weight": row.get("weight"),
                "details": json.dumps(row.get("details") or {}, default=str),
            }
            with self.engine.begin() as conn:
                conn.execute(text(_TARGET_SQL[kind]), params)
                linked = conn.scalar(text(_LINKED_SQL[kind]), params)
            if linked:
                continue
            if attempts + 1 >= MAX_EVIDENCE_ATTEMPTS:
                self.evidence_dropped += 1
                log.warning(
                    "evidence %s %s for incident %s never appeared; not linked",
                    kind,
                    target,
                    incident_id,
                )
                continue
            keep.append((incident_id, row, attempts + 1))
        self.pending = keep

    # ------------------------------------------------------------------ recovery

    def load_open(self) -> list[FusedIncident]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(IncidentRow.details).where(
                    IncidentRow.status.in_(["suspected", "confirmed"])
                )
            ).scalars()
            out = []
            for details in rows:
                state = (details or {}).get("state")
                if state:
                    out.append(FusedIncident.from_dict(state))
        return out

    def recent_anomalies(self, window_seconds: float) -> list[AnomalyEvidence]:
        sql = text(
            "SELECT id, camera_id, window_start, window_end, score, threshold, hints, severity, "
            "congestion_level, status, ended_at FROM anomalies WHERE scope = 'camera' AND "
            "(status = 'active' OR ended_at > now() - make_interval(secs => :w))"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"w": window_seconds}).all()
        out = []
        for r in rows:
            started = r.window_start + timedelta(seconds=60)  # start events cover the last minute
            out.append(
                AnomalyEvidence(
                    anomaly_id=str(r.id),
                    camera_id=r.camera_id,
                    started_at=started,
                    last_seen=r.ended_at or r.window_end,
                    peak_ratio=r.score / r.threshold if r.threshold else 0.0,
                    hints=list(r.hints or []),
                    severity=r.severity,
                    congestion=r.congestion_level,
                    active=r.status == "active",
                    ended_at=r.ended_at,
                )
            )
        return out

    def system_state(self) -> dict | None:
        sql = text(
            "SELECT id, affected_cameras, window_start FROM anomalies WHERE scope = 'system' "
            "AND status = 'active' ORDER BY detected_at DESC LIMIT 1"
        )
        with self.engine.connect() as conn:
            row = conn.execute(sql).first()
        if row is None:
            return None
        return {
            "anomaly_id": str(row.id),
            "cameras": set(row.affected_cameras or []),
            "started_at": row.window_start,
        }

    def recent_reports(self, window_seconds: float) -> list[ReportEvidence]:
        sql = text(
            "SELECT DISTINCT ON (n.report_id) n.report_id, n.camera_id, n.processed_at, "
            "n.incident_type, n.incident_confidence, n.severity, n.camera_confidence, n.stage, "
            "n.entities, r.source FROM nlp_results n JOIN reports r ON r.id = n.report_id "
            "WHERE n.processed_at > now() - make_interval(secs => :w) "
            "AND n.incident_type <> 'none' ORDER BY n.report_id, n.processed_at DESC"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"w": window_seconds}).all()
        return [
            ReportEvidence(
                report_id=str(r.report_id),
                camera_id=r.camera_id,
                ts=r.processed_at,
                incident_type=r.incident_type,
                confidence=r.incident_confidence or 0.0,
                severity=r.severity,
                source=r.source,
                camera_confidence=r.camera_confidence or 0.0,
                stage=r.stage or ACTIVE,
                places=places_from_entities(r.entities or []),
            )
            for r in rows
        ]

    def congestion_levels(self) -> dict[str, str]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT camera_id, level FROM camera_congestion")).all()
        return {r.camera_id: r.level for r in rows}

    def camera_names(self) -> dict[str, str]:
        with self.engine.connect() as conn:
            return dict(conn.execute(select(Camera.id, Camera.name)).all())
