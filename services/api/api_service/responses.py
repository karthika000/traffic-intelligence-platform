"""Response models. Records the platform already has a schema for (incidents, summaries,
anomalies, NLP results, reports, metrics) are returned as the shared Kafka schemas from
traffic_common.schemas, so the dashboard sees the same shape over REST and the WebSocket;
these models add the API's aggregates and envelopes around them."""

from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from traffic_common.schemas import (
    Anomaly,
    Incident,
    NLPResult,
    Summary,
    TextReport,
    TrafficMetrics,
)

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int


class Congestion(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"level": "heavy", "confidence": 0.91, "window_end": "2026-10-17T08:15:05Z"}
            ]
        }
    )

    level: str
    confidence: float | None = None
    window_end: datetime | None = None


class CameraStatus(BaseModel):
    id: str
    name: str
    location_name: str
    latitude: float
    longitude: float
    congestion: Congestion | None = None
    latest_metrics: TrafficMetrics | None = None
    open_incidents: int = 0
    vision_degraded: bool = False
    vision_degraded_reasons: list[str] = Field(default_factory=list)
    frame_url: str


class MetricsBucket(BaseModel):
    bucket_start: datetime
    windows: int
    vehicle_count_mean: float | None
    avg_speed_kmh: float | None
    p85_speed_kmh: float | None
    density_mean: float | None
    stopped_vehicles_max: int | None
    pedestrians_in_roadway_max: int | None
    mean_confidence: float | None
    frames_mean: float | None


class AnomalyRecord(BaseModel):
    anomaly: Anomaly
    status: str
    detected_at: datetime
    update_count: int = 0


class CameraDetail(BaseModel):
    camera: CameraStatus
    zone_config: dict[str, Any]
    range_minutes: int
    resolution_seconds: int
    metrics: list[MetricsBucket]
    anomalies: list[AnomalyRecord]


class ReportRecord(BaseModel):
    report: TextReport
    nlp: NLPResult | None = None
    unlocated: bool  # analysed, but matched to no camera


class Evidence(BaseModel):
    anomalies: list[AnomalyRecord]
    reports: list[ReportRecord]
    metrics: list[TrafficMetrics]
    weights: dict[str, float | None] = Field(default_factory=dict)  # evidence id -> weight


class IncidentDetail(BaseModel):
    incident: Incident
    resolution: str | None = None
    at_confirmation: dict[str, Any] | None = None
    versions: list[dict[str, Any]]
    evidence: Evidence
    summaries: list[Summary]


class HourlyVolume(BaseModel):
    hour: datetime
    camera_id: str
    vehicles: int  # sum of per-window unique tracks (a vehicle in two windows counts twice)
    windows: int
    avg_speed_kmh: float | None


class CongestionShare(BaseModel):
    bucket_start: datetime
    camera_id: str
    counts: dict[str, int]  # level -> checks
    shares: dict[str, float]  # level -> share of the bucket's checks


class IncidentCount(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"incident_type": "accident", "severity": "high", "status": "resolved", "count": 5}
            ]
        }
    )

    incident_type: str
    severity: str
    status: str
    count: int


class ModelInfo(BaseModel):
    model_name: str
    version: str
    stage: str
    mlflow_run_id: str | None = None
    registered_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)  # as recorded in model_versions
    mlflow: dict[str, Any] | None = None  # champion alias, run metrics, ... (None: unreachable)


class ScenarioPhase(BaseModel):
    name: str
    duration_seconds: float
    severity: str


class ScenarioInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "broken_down_lorry",
                    "description": "Broken-down lorry blocking a lane.",
                    "incident_type": "breakdown",
                    "duration_seconds": 960,
                    "phases": [
                        {"name": "breakdown", "duration_seconds": 180, "severity": "medium"}
                    ],
                }
            ]
        }
    )

    name: str
    description: str
    incident_type: str
    duration_seconds: float
    phases: list[ScenarioPhase]


class ScenarioRunRecord(BaseModel):
    id: UUID
    scenario_name: str
    camera_id: str
    incident_type: str
    peak_severity: str
    status: str
    triggered_by: str
    time_scale: float
    started_at: datetime
    expected_end_at: datetime
    ended_at: datetime | None
    reports_emitted: int
    events_emitted: int
    timeline: list[dict[str, Any]]
    error: str | None = None


class TriggerRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"camera_id": "cam-hyde-park-corner", "time_scale": 6}]}
    )

    camera_id: str = Field(description="a camera from GET /api/v1/cameras")
    time_scale: float = Field(
        default=1.0, gt=0, le=1000, description="speed-up: 6 runs an 18-minute scenario in 3"
    )


class TriggerResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "scenario_id": "285b60b8-d0a3-4790-84d2-8e49208a21e7",
                    "scenario": "multi_vehicle_accident",
                    "camera_id": "cam-hyde-park-corner",
                    "time_scale": 6,
                    "status": "requested",
                }
            ]
        }
    )

    scenario_id: UUID
    scenario: str
    camera_id: str
    time_scale: float
    status: str = "requested"
