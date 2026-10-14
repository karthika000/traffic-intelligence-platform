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

from traffic_common.enums import DriftType, IncidentStatus, ReportSource, Severity, SummaryScope
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


class Detection(KafkaMessage):
    """All objects detected in one frame (``frame_id`` from the FrameMessage)."""

    camera_id: CameraId
    frame_id: UUID
    model_name: ShortStr
    model_version: ShortStr
    objects: list[DetectedObject] = Field(default_factory=list)
    inference_ms: float | None = Field(default=None, ge=0)


class TrafficMetrics(_Window, KafkaMessage):
    camera_id: CameraId
    vehicle_count: int = Field(ge=0)
    avg_speed_kmh: float | None = Field(default=None, ge=0)
    occupancy: Probability | None = None
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
    report_id: UUID
    camera_id: CameraId | None = None
    category: ShortStr
    severity: Severity | None = None
    sentiment: float | None = Field(default=None, ge=-1.0, le=1.0)
    location_name: ShortStr | None = None
    entities: list[Entity] = Field(default_factory=list)
    model_name: ShortStr
    model_version: ShortStr


# --------------------------------------------------------------------------- analytics


class Anomaly(_Window, KafkaMessage):
    anomaly_id: UUID = Field(default_factory=uuid4)
    camera_id: CameraId
    anomaly_type: ShortStr
    metric_name: ShortStr
    observed_value: float
    expected_value: float | None = None
    score: float
    threshold: float
    model_name: ShortStr
    model_version: ShortStr
    details: dict[str, Any] = Field(default_factory=dict)


class Incident(KafkaMessage):
    incident_id: UUID = Field(default_factory=uuid4)
    camera_id: CameraId
    status: IncidentStatus = IncidentStatus.SUSPECTED
    severity: Severity
    fused_confidence: Probability
    start_time: UtcDatetime
    end_time: UtcDatetime | None = None
    title: ShortStr
    description: str | None = Field(default=None, max_length=5_000)
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
    }
)


def schema_for(topic: str) -> type[KafkaMessage]:
    try:
        return TOPIC_SCHEMAS[topic]
    except KeyError:
        raise KeyError(f"no schema registered for topic {topic!r}") from None
