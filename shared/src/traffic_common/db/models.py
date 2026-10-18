"""SQLAlchemy 2.0 models for the `traffic` database.

Rows that originate from a Kafka message keep its UUID (``id`` or ``message_id``) so
ingestion can be made idempotent with ON CONFLICT DO NOTHING.
All timestamps are timezone-aware (TIMESTAMPTZ). Enums are stored as strings with CHECK
constraints rather than native PostgreSQL enums, so adding a value is a simple migration.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Double,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from traffic_common.enums import (
    CongestionLevel,
    DriftType,
    EvidenceType,
    IncidentStatus,
    IncidentType,
    ModelStage,
    ReportSource,
    ScenarioStatus,
    Severity,
    SummaryScope,
)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
        list[dict[str, Any]]: JSONB,
        str: String(200),
        float: Double,
    }


def _enum(enum_cls: type[StrEnum], name: str) -> Enum:
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


def _pk() -> Mapped[int]:
    return mapped_column(BigInteger, Identity(), primary_key=True)


def _created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


def _camera_fk(nullable: bool = False) -> Mapped[Any]:
    return mapped_column(String(64), ForeignKey("cameras.id"), nullable=nullable)


def _empty_json() -> Any:
    return text("'{}'::jsonb")


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str]
    location_name: Mapped[str]
    latitude: Mapped[float]
    longitude: Mapped[float]
    stream_url: Mapped[str] = mapped_column(String(500))
    zone_config: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("latitude BETWEEN -90 AND 90", name="latitude_range"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="longitude_range"),
    )


class TrafficMetric(Base):
    __tablename__ = "traffic_metrics"

    id: Mapped[int] = _pk()
    message_id: Mapped[uuid.UUID | None] = mapped_column(unique=True)
    camera_id: Mapped[str] = _camera_fk()
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    vehicle_count: Mapped[int] = mapped_column(Integer)
    avg_speed_kmh: Mapped[float | None]
    occupancy: Mapped[float | None]
    counts_by_class: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    p85_speed_kmh: Mapped[float | None]
    density: Mapped[float | None]
    stopped_vehicles: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    pedestrians_in_roadway: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    mean_confidence: Mapped[float | None]
    frames: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    model_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("window_end > window_start", name="window_ordered"),
        CheckConstraint("vehicle_count >= 0", name="vehicle_count_non_negative"),
        CheckConstraint("occupancy IS NULL OR occupancy BETWEEN 0 AND 1", name="occupancy_range"),
        Index("ix_traffic_metrics_camera_id_window_start", "camera_id", "window_start"),
        Index("ix_traffic_metrics_window_start", "window_start"),
    )


class DetectionAggregate(Base):
    """Per-camera, per-class object counts over a time window."""

    __tablename__ = "detection_aggregates"

    id: Mapped[int] = _pk()
    camera_id: Mapped[str] = _camera_fk()
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    object_class: Mapped[str] = mapped_column(String(50))
    object_count: Mapped[int] = mapped_column(Integer)
    avg_confidence: Mapped[float | None]
    model_name: Mapped[str]
    model_version: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("window_end > window_start", name="window_ordered"),
        CheckConstraint("object_count >= 0", name="object_count_non_negative"),
        # Explicit name: the generated one would exceed PostgreSQL's 63-character identifier limit.
        UniqueConstraint(
            "camera_id",
            "window_start",
            "object_class",
            "model_name",
            "model_version",
            name="uq_detection_aggregates_window_class_model",
        ),
        Index("ix_detection_aggregates_camera_id_window_start", "camera_id", "window_start"),
        Index("ix_detection_aggregates_window_start", "window_start"),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # TextReport.report_id
    source: Mapped[ReportSource] = mapped_column(_enum(ReportSource, "report_source"))
    text: Mapped[str] = mapped_column(Text)
    camera_id: Mapped[str | None] = _camera_fk(nullable=True)
    location_name: Mapped[str | None]
    latitude: Mapped[float | None]
    longitude: Mapped[float | None]
    reported_at: Mapped[datetime]
    received_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_reports_camera_id_reported_at", "camera_id", "reported_at"),
        Index("ix_reports_reported_at", "reported_at"),
    )


class NlpResult(Base):
    __tablename__ = "nlp_results"

    id: Mapped[int] = _pk()
    message_id: Mapped[uuid.UUID | None] = mapped_column(unique=True)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"))
    camera_id: Mapped[str | None] = _camera_fk(nullable=True)  # matched camera
    camera_confidence: Mapped[float | None]
    incident_type: Mapped[IncidentType] = mapped_column(_enum(IncidentType, "incident_type"))
    incident_confidence: Mapped[float | None]
    severity: Mapped[Severity | None] = mapped_column(_enum(Severity, "severity"))
    severity_confidence: Mapped[float | None]
    probabilities: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    stage: Mapped[str] = mapped_column(String(20), server_default=text("'active'"))
    sentiment: Mapped[float | None]
    location_name: Mapped[str | None]
    entities: Mapped[list[dict[str, Any]]] = mapped_column(server_default=text("'[]'::jsonb"))
    model_name: Mapped[str]
    model_version: Mapped[str] = mapped_column(String(100))
    model_versions: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    details: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    processed_at: Mapped[datetime]

    __table_args__ = (
        CheckConstraint("sentiment IS NULL OR sentiment BETWEEN -1 AND 1", name="sentiment_range"),
        Index("ix_nlp_results_report_id", "report_id"),
        Index("ix_nlp_results_camera_id_processed_at", "camera_id", "processed_at"),
        Index("ix_nlp_results_processed_at", "processed_at"),
    )


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # Anomaly.anomaly_id
    camera_id: Mapped[str | None] = _camera_fk(nullable=True)  # NULL for scope 'system'
    scope: Mapped[str] = mapped_column(String(20), server_default=text("'camera'"))
    affected_cameras: Mapped[list[Any]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    anomaly_type: Mapped[str] = mapped_column(String(50))
    metric_name: Mapped[str] = mapped_column(String(100))
    observed_value: Mapped[float]
    expected_value: Mapped[float | None]
    score: Mapped[float]
    threshold: Mapped[float]
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    detected_at: Mapped[datetime]
    model_name: Mapped[str]
    model_version: Mapped[str] = mapped_column(String(100))
    details: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    status: Mapped[str] = mapped_column(
        String(20), server_default=text("'active'")
    )  # active | ended
    ended_at: Mapped[datetime | None]
    hints: Mapped[list[Any]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    contributing_features: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    congestion_level: Mapped[CongestionLevel | None] = mapped_column(
        _enum(CongestionLevel, "congestion_level")
    )
    model_versions: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    severity: Mapped[Severity | None] = mapped_column(_enum(Severity, "severity"))
    update_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))

    __table_args__ = (
        CheckConstraint("window_end > window_start", name="window_ordered"),
        CheckConstraint("status IN ('active', 'ended')", name="status_valid"),
        CheckConstraint("scope IN ('camera', 'system')", name="scope_valid"),
        CheckConstraint(
            "scope = 'system' OR camera_id IS NOT NULL", name="camera_for_camera_scope"
        ),
        Index("ix_anomalies_camera_id_detected_at", "camera_id", "detected_at"),
        Index("ix_anomalies_detected_at", "detected_at"),
    )


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # Incident.incident_id
    camera_id: Mapped[str] = _camera_fk()
    status: Mapped[IncidentStatus] = mapped_column(
        _enum(IncidentStatus, "incident_status"), server_default=IncidentStatus.SUSPECTED.value
    )
    incident_type: Mapped[IncidentType] = mapped_column(
        _enum(IncidentType, "incident_type"), server_default=IncidentType.NONE.value
    )
    version: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    severity: Mapped[Severity] = mapped_column(_enum(Severity, "severity"))
    fused_confidence: Mapped[float]
    confidence_components: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    vision_degraded: Mapped[bool] = mapped_column(server_default=text("false"))
    start_time: Mapped[datetime]
    suspected_at: Mapped[datetime | None]
    confirmed_at: Mapped[datetime | None]
    end_time: Mapped[datetime | None]
    title: Mapped[str]
    description: Mapped[str | None] = mapped_column(Text)
    # fusion_service's full incident state (rebuilt from here on restart) and history.
    details: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    evidence: Mapped[list["IncidentEvidence"]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("fused_confidence BETWEEN 0 AND 1", name="fused_confidence_range"),
        CheckConstraint("end_time IS NULL OR end_time >= start_time", name="time_ordered"),
        CheckConstraint(
            "status <> 'resolved' OR end_time IS NOT NULL", name="resolved_has_end_time"
        ),
        Index("ix_incidents_camera_id_start_time", "camera_id", "start_time"),
        Index("ix_incidents_start_time", "start_time"),
        Index("ix_incidents_status", "status"),
    )


class IncidentEvidence(Base):
    """Links an incident to exactly one supporting traffic metric, anomaly or report."""

    __tablename__ = "incident_evidence"

    id: Mapped[int] = _pk()
    incident_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    evidence_type: Mapped[EvidenceType] = mapped_column(_enum(EvidenceType, "evidence_type"))
    traffic_metric_id: Mapped[int | None] = mapped_column(
        ForeignKey("traffic_metrics.id", ondelete="CASCADE")
    )
    anomaly_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("anomalies.id", ondelete="CASCADE")
    )
    report_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE")
    )
    weight: Mapped[float | None]  # contribution to the incident's fused confidence
    details: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    created_at: Mapped[datetime] = _created_at()

    incident: Mapped[Incident] = relationship(back_populates="evidence")

    __table_args__ = (
        CheckConstraint(
            "(evidence_type = 'metric' AND traffic_metric_id IS NOT NULL"
            " AND anomaly_id IS NULL AND report_id IS NULL)"
            " OR (evidence_type = 'anomaly' AND anomaly_id IS NOT NULL"
            " AND traffic_metric_id IS NULL AND report_id IS NULL)"
            " OR (evidence_type = 'report' AND report_id IS NOT NULL"
            " AND traffic_metric_id IS NULL AND anomaly_id IS NULL)",
            name="exactly_one_target",
        ),
        CheckConstraint("weight IS NULL OR weight BETWEEN 0 AND 1", name="weight_range"),
        Index("ix_incident_evidence_incident_id", "incident_id"),
        Index("ix_incident_evidence_traffic_metric_id", "traffic_metric_id"),
        Index("ix_incident_evidence_anomaly_id", "anomaly_id"),
        Index("ix_incident_evidence_report_id", "report_id"),
    )


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # Summary.summary_id
    scope: Mapped[SummaryScope] = mapped_column(_enum(SummaryScope, "summary_scope"))
    camera_id: Mapped[str | None] = _camera_fk(nullable=True)
    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("incidents.id", ondelete="SET NULL")
    )
    incident_version: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[datetime]
    period_end: Mapped[datetime]
    headline: Mapped[str | None]
    text: Mapped[str] = mapped_column(Text)
    generation: Mapped[str | None] = mapped_column(String(20))  # model | template
    model_name: Mapped[str]
    model_version: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    latency_ms: Mapped[float | None]
    faithfulness: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    context: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())  # the facts used
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("period_end > period_start", name="period_ordered"),
        CheckConstraint(
            "generation IS NULL OR generation IN ('model', 'template')", name="generation_valid"
        ),
        Index("ix_summaries_camera_id_created_at", "camera_id", "created_at"),
        Index("ix_summaries_incident_id", "incident_id"),
        Index("ix_summaries_created_at", "created_at"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = _pk()
    model_name: Mapped[str]
    version: Mapped[str] = mapped_column(String(100))
    stage: Mapped[ModelStage] = mapped_column(
        _enum(ModelStage, "model_stage"), server_default=ModelStage.NONE.value
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    # Evaluation-gated promotion (ml.lifecycle): decision, reasons and the champion/candidate
    # comparison. Written by the retrain CLI; the services' upserts never touch it.
    promotion: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    registered_at: Mapped[datetime] = _created_at()

    __table_args__ = (UniqueConstraint("model_name", "version"),)


class DriftReport(Base):
    """Drift check results (ml.monitoring): one row per monitored feature and one overall row
    (``feature`` NULL, ``metric`` "drift_share", ``status`` set) per check, grouped by
    ``check_id``. The overall row's ``report`` holds the full result."""

    __tablename__ = "drift_reports"

    id: Mapped[int] = _pk()
    check_id: Mapped[uuid.UUID | None]
    status: Mapped[str | None] = mapped_column(String(20))  # ok | warning | drifted | no_data
    model_name: Mapped[str]
    model_version: Mapped[str] = mapped_column(String(100))
    drift_type: Mapped[DriftType] = mapped_column(_enum(DriftType, "drift_type"))
    metric: Mapped[str] = mapped_column(String(50))
    feature: Mapped[str | None]
    drift_score: Mapped[float]
    threshold: Mapped[float]
    is_drift: Mapped[bool]
    camera_id: Mapped[str | None] = _camera_fk(nullable=True)
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    report: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("window_end > window_start", name="window_ordered"),
        Index("ix_drift_reports_model_name_created_at", "model_name", "created_at"),
        Index("ix_drift_reports_camera_id_created_at", "camera_id", "created_at"),
        Index("ix_drift_reports_created_at", "created_at"),
        Index("ix_drift_reports_check_id", "check_id"),
        CheckConstraint(
            "status IS NULL OR status IN ('ok', 'warning', 'drifted', 'no_data')",
            name="status_valid",
        ),
    )


class DriftMode(Base):
    """The demo drift mode (one row, id 1): visual degradation in video_ingest and a new report
    style in event_simulator. Both services poll it; ``expires_at`` switches it off by itself."""

    __tablename__ = "drift_mode"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    enabled: Mapped[bool] = mapped_column(server_default=text("false"))
    settings: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    expires_at: Mapped[datetime | None]
    updated_by: Mapped[str | None] = mapped_column(String(50))
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (CheckConstraint("id = 1", name="single_row"),)


class DriftModeEvent(Base):
    """Every drift mode change, so drift checks and reference windows can tell clean
    periods from degraded ones."""

    __tablename__ = "drift_mode_events"

    id: Mapped[int] = _pk()
    enabled: Mapped[bool]
    settings: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    expires_at: Mapped[datetime | None]
    changed_by: Mapped[str | None] = mapped_column(String(50))
    changed_at: Mapped[datetime] = _created_at()

    __table_args__ = (Index("ix_drift_mode_events_changed_at", "changed_at"),)


class RetrainingRecommendation(Base):
    """Retraining recommended for a model after drift alerts (automatic retraining is off by
    default). At most one open (recommended or retraining) recommendation per model; the
    retrain CLI resolves it, or someone dismisses it with a reason (in ``resolution``)."""

    __tablename__ = "retraining_recommendations"

    id: Mapped[int] = _pk()
    model_name: Mapped[str]
    model_version: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), server_default=text("'recommended'"))
    reason: Mapped[str] = mapped_column(Text)
    alert_count: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    drifted_features: Mapped[list[dict[str, Any]]] = mapped_column(
        server_default=text("'[]'::jsonb")
    )
    first_alert_at: Mapped[datetime]
    last_alert_at: Mapped[datetime]
    last_alert_id: Mapped[uuid.UUID | None]
    resolved_at: Mapped[datetime | None]
    resolution: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(
            "status IN ('recommended', 'retraining', 'resolved', 'dismissed')",
            name="status_valid",
        ),
        Index(
            "uq_retraining_recommendations_open",
            "model_name",
            unique=True,
            postgresql_where=text("status IN ('recommended', 'retraining')"),
        ),
        Index("ix_retraining_recommendations_created_at", "created_at"),
    )


class SensorReading(Base):
    """Loop-detector readings from events.raw, stored by anomaly_service so they can be joined
    with traffic_metrics (training data) by camera and time."""

    __tablename__ = "sensor_readings"

    id: Mapped[int] = _pk()
    event_id: Mapped[uuid.UUID] = mapped_column(unique=True)  # RawEvent.event_id
    camera_id: Mapped[str] = _camera_fk()
    detector_id: Mapped[str | None] = mapped_column(String(100))
    reading_at: Mapped[datetime]
    vehicle_count: Mapped[int | None] = mapped_column(Integer)
    flow_veh_per_hour: Mapped[float | None]
    avg_speed_kmh: Mapped[float | None]
    occupancy: Mapped[float | None]
    interval_seconds: Mapped[float | None]
    scenario_id: Mapped[uuid.UUID | None]  # set while a simulated scenario drove the reading
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_sensor_readings_camera_id_reading_at", "camera_id", "reading_at"),
        Index("ix_sensor_readings_reading_at", "reading_at"),
    )


class CameraCongestion(Base):
    """Current congestion level per camera (one row per camera, updated every few seconds)."""

    __tablename__ = "camera_congestion"

    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id"), primary_key=True)
    level: Mapped[CongestionLevel] = mapped_column(_enum(CongestionLevel, "congestion_level"))
    confidence: Mapped[float | None]
    anomaly_score: Mapped[float | None]
    anomaly_threshold: Mapped[float | None]
    model_versions: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    window_end: Mapped[datetime]
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class CongestionHistory(Base):
    """Every congestion classification (one row per camera per check), for analytics over time."""

    __tablename__ = "congestion_history"

    id: Mapped[int] = _pk()
    camera_id: Mapped[str] = _camera_fk()
    window_end: Mapped[datetime]
    level: Mapped[CongestionLevel] = mapped_column(_enum(CongestionLevel, "congestion_level"))
    confidence: Mapped[float | None]
    # The anomaly model's score and the camera's threshold at the same check (drift monitoring).
    anomaly_score: Mapped[float | None]
    anomaly_threshold: Mapped[float | None]

    __table_args__ = (
        Index("ix_congestion_history_camera_id_window_end", "camera_id", "window_end"),
        Index("ix_congestion_history_window_end", "window_end"),
    )


# ------------------------------------------------------------------ simulation ground truth
# Written only by event_simulator. Kept apart from the operational tables (reports, incidents, ...)
# so labels never mix with what the platform itself ingests and infers.


class ScenarioRun(Base):
    """One execution of a simulated scenario; every message it emits carries its id."""

    __tablename__ = "scenario_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # scenario_id in messages
    scenario_name: Mapped[str] = mapped_column(String(100))
    camera_id: Mapped[str] = _camera_fk()
    incident_type: Mapped[IncidentType] = mapped_column(_enum(IncidentType, "incident_type"))
    peak_severity: Mapped[Severity] = mapped_column(_enum(Severity, "severity"))
    status: Mapped[ScenarioStatus] = mapped_column(
        _enum(ScenarioStatus, "scenario_status"), server_default=ScenarioStatus.RUNNING.value
    )
    triggered_by: Mapped[str] = mapped_column(String(50))
    time_scale: Mapped[float] = mapped_column(server_default=text("1"))
    started_at: Mapped[datetime]
    expected_end_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]
    # Phases with their actual start/end times, as run (after time scaling).
    timeline: Mapped[list[dict[str, Any]]] = mapped_column(server_default=text("'[]'::jsonb"))
    definition: Mapped[dict[str, Any]] = mapped_column(server_default=_empty_json())
    reports_emitted: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    events_emitted: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("time_scale > 0", name="time_scale_positive"),
        Index("ix_scenario_runs_camera_id_started_at", "camera_id", "started_at"),
        Index("ix_scenario_runs_started_at", "started_at"),
    )


class SimulatedReport(Base):
    """Ground-truth labels for every TextReport the simulator emitted."""

    __tablename__ = "simulated_reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # TextReport.report_id
    scenario_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("scenario_runs.id", ondelete="CASCADE")
    )
    camera_id: Mapped[str] = _camera_fk()
    source: Mapped[ReportSource] = mapped_column(_enum(ReportSource, "report_source"))
    style: Mapped[str] = mapped_column(String(20))  # driver, police, operator, social_media
    text: Mapped[str] = mapped_column(Text)
    incident_type: Mapped[IncidentType] = mapped_column(_enum(IncidentType, "incident_type"))
    severity: Mapped[Severity] = mapped_column(_enum(Severity, "severity"))
    location_mention: Mapped[str | None]  # exact substring of text, if any
    stage: Mapped[str] = mapped_column(String(20))  # active | cleared
    # "slang" for reports written while drift mode was on; NULL for the standard style.
    text_style: Mapped[str | None] = mapped_column(String(20))
    phase: Mapped[str | None] = mapped_column(String(50))
    emitted_at: Mapped[datetime]

    __table_args__ = (
        Index("ix_simulated_reports_camera_id_emitted_at", "camera_id", "emitted_at"),
        Index("ix_simulated_reports_emitted_at", "emitted_at"),
        Index("ix_simulated_reports_scenario_run_id", "scenario_run_id"),
    )
