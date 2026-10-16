"""Pydantic v2 schemas for every Kafka message, plus the topic -> schema mapping.

Conventions for all messages:
- ``schema_version`` is a Literal, so a consumer rejects versions it doesn't understand instead
  of misreading them. Bump it (and the Literal) on breaking changes.
- ``timestamp`` is the event time in UTC. Naive datetimes are rejected; other offsets are
  converted to UTC. The same applies to every other datetime field.
- Unknown fields are rejected (``extra="forbid"``) so producer typos surface immediately.
- Messages are immutable.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from traffic_common.enums import (
    AnomalyHint,
    CongestionLevel,
    DriftType,
    IncidentStatus,
    IncidentType,
    ReportSource,
    Severity,
    SummaryScope,
)
from traffic_common.topics import Topic


def utcnow() -> datetime:
    return datetime.now(UTC)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
CameraId = Annotated[str, Field(min_length=1, max_length=64)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
ShortStr = Annotated[str, Field(min_length=1, max_length=200)]


class KafkaMessage(BaseModel):
    """Base for every message on the bus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    message_id: UUID = Field(default_factory=uuid4)
    timestamp: UtcDatetime = Field(default_factory=utcnow)


class _Window(BaseModel):
    """Mixin for messages that describe a time window."""

    window_start: UtcDatetime
    window_end: UtcDatetime

    @model_validator(mode="after")
    def _window_ordered(self) -> Self:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self


# --------------------------------------------------------------------------- video / CV


class FrameMessage(KafkaMessage):
    """A sampled frame, stored as an image file on the shared frames volume.

    ``timestamp`` is the capture time. ``frame_path`` is the path inside containers
    (``/data/frames/<camera_id>/<timestamp>.jpg``); files are deleted after the ingest
    service's retention window, so consumers must tolerate a missing file.
    ``sequence`` counts published frames per camera and restarts at 0 when video_ingest restarts.
    """

    camera_id: CameraId
    frame_id: UUID = Field(default_factory=uuid4)
    sequence: int = Field(ge=0)
    frame_path: str = Field(min_length=1, max_length=500)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_fps: float = Field(gt=0)
    encoding: Literal["jpeg", "png"] = "jpeg"


class BoundingBox(BaseModel):
    """Normalised [0, 1] image coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: Probability
    y_min: Probability
    x_max: Probability
    y_max: Probability

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box max must be greater than min")
        return self


class DetectedObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: ShortStr
    confidence: Probability
    bbox: BoundingBox
    track_id: int | None = Field(default=None, ge=0)
    # Tracking analytics (cv_inference); None when the object isn't tracked yet.
    speed_kmh: float | None = Field(default=None, ge=0)
    stopped: bool = False  # vehicle below the stopped-speed threshold for long enough
    in_roadway: bool | None = None  # ground point inside the camera's roadway polygon


class Detection(KafkaMessage):
    """All objects detected in one frame (``frame_id`` from the FrameMessage)."""

    camera_id: CameraId
    frame_id: UUID
    model_name: ShortStr
    model_version: ShortStr
    objects: list[DetectedObject] = Field(default_factory=list)
    inference_ms: float | None = Field(default=None, ge=0)


class TrafficMetrics(_Window, KafkaMessage):
    """Per-camera aggregates over a short window (cv_inference publishes one every 5 s).

    Counts are unique tracks seen in the window: ``vehicle_count`` covers cars, trucks,
    buses, motorcycles and bicycles; ``counts_by_class`` has every class including persons.
    Speeds use one value per vehicle track (its mean speed in the window).
    """

    camera_id: CameraId
    vehicle_count: int = Field(ge=0)
    avg_speed_kmh: float | None = Field(default=None, ge=0)
    p85_speed_kmh: float | None = Field(default=None, ge=0)
    occupancy: Probability | None = None
    density: float | None = Field(default=None, ge=0)  # mean vehicles inside the roadway per frame
    stopped_vehicles: int = Field(default=0, ge=0)
    pedestrians_in_roadway: int = Field(default=0, ge=0)
    mean_confidence: Probability | None = None
    frames: int = Field(default=0, ge=0)  # frames analysed in the window
    model_version: ShortStr | None = None
    counts_by_class: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)


# --------------------------------------------------------------------------- text / events


class TextReport(KafkaMessage):
    report_id: UUID = Field(default_factory=uuid4)
    source: ReportSource
    text: str = Field(min_length=1, max_length=10_000)
    camera_id: CameraId | None = None
    location_name: ShortStr | None = None
    latitude: Latitude | None = None
    longitude: Longitude | None = None
    # Set on messages emitted by a simulated scenario run (ground truth in scenario_runs).
    scenario_id: UUID | None = None

    @model_validator(mode="after")
    def _coordinates_paired(self) -> Self:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be set together")
        return self


class RawEvent(KafkaMessage):
    """An external or simulated event (signal failure, road closure, weather, ...)."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: ShortStr
    source: ShortStr
    camera_id: CameraId | None = None
    severity: Severity | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    scenario_id: UUID | None = None  # set by simulated scenario runs


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    label: ShortStr
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _span_ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("entity end must be after start")
        return self


class NLPResult(KafkaMessage):
    """What nlp_service made of one TextReport: incident type and severity (with confidences
    and the full ``probabilities`` per task: {"incident_type": {label: p}, "severity": ...}),
    extracted entities, and the camera it was matched to (``camera_id`` is None when no camera
    matched confidently; ``location_name`` is the text it matched on)."""

    report_id: UUID
    source: ReportSource | None = None  # copied from the report
    incident_type: IncidentType
    incident_confidence: Probability
    severity: Severity | None = None
    severity_confidence: Probability | None = None
    probabilities: dict[str, dict[str, Probability]] = Field(default_factory=dict)
    # "cleared" when the report says the incident is over (rule-based, ml.nlp.stage).
    stage: Literal["active", "cleared"] = "active"
    sentiment: float | None = Field(default=None, ge=-1.0, le=1.0)
    entities: list[Entity] = Field(default_factory=list)
    camera_id: CameraId | None = None
    camera_confidence: Probability | None = None
    location_name: ShortStr | None = None
    model_name: ShortStr
    model_version: ShortStr
    model_versions: dict[str, str] = Field(default_factory=dict)  # every model involved
    details: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- analytics


class Anomaly(_Window, KafkaMessage):
    """An anomaly episode for one camera, as a series of events with the same ``anomaly_id``:
    ``start`` when the score crosses the threshold, ``update`` when an active episode's
    severity rises or its hints change (escalation), ``end`` when it is back to normal.
    ``score`` on update/end events is the episode's peak so far. ``contributing_features``
    keys are "reconstruction:<feature>" or "level:<feature>" (distance from the baseline);
    ``details["fired_by"]`` says which part produced the score.

    ``scope="system"`` marks a pipeline-wide degradation (several cameras' frames or detection
    confidence dropping in the same check): it has no ``camera_id``, lists the cameras in
    ``affected_cameras``, and per-camera traffic alerts are suppressed while it is active."""

    anomaly_id: UUID = Field(default_factory=uuid4)
    scope: Literal["camera", "system"] = "camera"
    camera_id: CameraId | None = None  # required for scope "camera"
    affected_cameras: list[CameraId] = Field(default_factory=list)
    event: Literal["start", "update", "end"] = "start"
    severity: Severity | None = None  # from score / threshold, see anomaly_service config
    anomaly_type: ShortStr  # the leading hint, or "unknown"
    hints: list[AnomalyHint] = Field(default_factory=list)
    metric_name: ShortStr  # the most anomalous input
    observed_value: float
    expected_value: float | None = None
    score: float
    threshold: float
    contributing_features: dict[str, float] = Field(default_factory=dict)  # share of the score
    congestion_level: CongestionLevel | None = None
    model_name: ShortStr
    model_version: ShortStr
    model_versions: dict[str, str] = Field(default_factory=dict)  # every model involved
    started_at: UtcDatetime | None = None
    ended_at: UtcDatetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _camera_for_camera_scope(self) -> Self:
        if self.scope == "camera" and self.camera_id is None:
            raise ValueError("camera_id is required for a camera-scoped anomaly")
        if self.scope == "system" and not self.affected_cameras:
            raise ValueError("a system-scoped anomaly must list affected_cameras")
        return self


class Incident(KafkaMessage):
    """A fused incident, published by fusion_service on creation and on every meaningful
    change; ``version`` counts the publications of one ``incident_id`` (1 = created).
    ``confidence_components`` explains ``fused_confidence``: per-modality scores, their
    sub-scores and weights (see fusion_service). ``start_time`` is the earliest evidence,
    ``suspected_at`` / ``confirmed_at`` when fusion reached those states."""

    incident_id: UUID = Field(default_factory=uuid4)
    camera_id: CameraId
    version: int = Field(default=1, ge=1)
    status: IncidentStatus = IncidentStatus.SUSPECTED
    incident_type: IncidentType = IncidentType.NONE
    severity: Severity
    fused_confidence: Probability
    confidence_components: dict[str, Any] = Field(default_factory=dict)
    vision_degraded: bool = False  # vision evidence seen during a system-wide degradation
    start_time: UtcDatetime
    suspected_at: UtcDatetime | None = None
    confirmed_at: UtcDatetime | None = None
    end_time: UtcDatetime | None = None
    title: ShortStr
    description: str | None = Field(default=None, max_length=5_000)
    change_reasons: list[str] = Field(default_factory=list)  # why this version was published
    metric_message_ids: list[UUID] = Field(default_factory=list)
    anomaly_ids: list[UUID] = Field(default_factory=list)
    report_ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def _times_consistent(self) -> Self:
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not be before start_time")
        if self.status is IncidentStatus.RESOLVED and self.end_time is None:
            raise ValueError("resolved incidents need an end_time")
        return self


class Summary(KafkaMessage):
    summary_id: UUID = Field(default_factory=uuid4)
    scope: SummaryScope
    camera_id: CameraId | None = None
    incident_id: UUID | None = None
    period_start: UtcDatetime
    period_end: UtcDatetime
    text: str = Field(min_length=1, max_length=10_000)
    model_name: ShortStr
    model_version: ShortStr

    @model_validator(mode="after")
    def _scope_target(self) -> Self:
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")
        if self.scope is SummaryScope.CAMERA and self.camera_id is None:
            raise ValueError("camera summaries need a camera_id")
        if self.scope is SummaryScope.INCIDENT and self.incident_id is None:
            raise ValueError("incident summaries need an incident_id")
        return self


class DriftAlert(_Window, KafkaMessage):
    model_name: ShortStr
    model_version: ShortStr
    drift_type: DriftType
    metric: ShortStr  # e.g. "psi", "ks", "wasserstein"
    feature: ShortStr | None = None
    drift_score: float
    threshold: float
    camera_id: CameraId | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- control


class SimulatorCommand(KafkaMessage):
    """Control message for event_simulator on ``simulator.control``."""

    action: Literal["start_scenario", "stop_scenario"]
    scenario_id: UUID = Field(default_factory=uuid4)
    scenario: ShortStr | None = None  # YAML file name without extension
    camera_id: CameraId | None = None
    time_scale: float = Field(default=1.0, gt=0, le=1000)  # >1 runs the timeline faster
    requested_by: ShortStr = "cli"

    @model_validator(mode="after")
    def _start_has_target(self) -> Self:
        if self.action == "start_scenario" and (self.scenario is None or self.camera_id is None):
            raise ValueError("start_scenario needs scenario and camera_id")
        return self


TOPIC_SCHEMAS: Mapping[str, type[KafkaMessage]] = MappingProxyType(
    {
        Topic.VIDEO_FRAMES: FrameMessage,
        Topic.EVENTS_RAW: RawEvent,
        Topic.REPORTS_TEXT: TextReport,
        Topic.DETECTIONS: Detection,
        Topic.TRAFFIC_METRICS: TrafficMetrics,
        Topic.ANOMALIES: Anomaly,
        Topic.NLP_RESULTS: NLPResult,
        Topic.INCIDENTS: Incident,
        Topic.SUMMARIES: Summary,
        Topic.DRIFT_ALERTS: DriftAlert,
        Topic.SIMULATOR_CONTROL: SimulatorCommand,
    }
)


def schema_for(topic: str) -> type[KafkaMessage]:
    try:
        return TOPIC_SCHEMAS[topic]
    except KeyError:
        raise KeyError(f"no schema registered for topic {topic!r}") from None
