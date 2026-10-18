"""Response models. Records the platform already has a schema for (incidents, summaries,
anomalies, NLP results, reports, metrics) are returned as the shared Kafka schemas from
traffic_common.schemas, so the dashboard sees the same shape over REST and the WebSocket;
these models add the API's aggregates and envelopes around them."""

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar
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


# --------------------------------------------------------------------------- drift monitoring


class DriftFeature(BaseModel):
    feature: str
    kind: str  # num | cat | embedding
    drift_type: str  # data | prediction
    method: str  # wasserstein | jensenshannon | model_roc_auc
    score: float | None = None
    threshold: float
    drifted: bool
    reference_summary: dict[str, Any] = Field(default_factory=dict)
    current_summary: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class DataQualityIssue(BaseModel):
    kind: str  # missing_values | out_of_range | stale_camera
    target: str
    value: float
    limit: float
    message: str


class RetrainingRecommendation(BaseModel):
    status: str  # recommended | retraining | resolved | dismissed
    model_version: str
    reason: str
    alert_count: int
    drifted_features: list[dict[str, Any]] = Field(default_factory=list)
    first_alert_at: datetime
    last_alert_at: datetime
    resolved_at: datetime | None = None
    resolution: dict[str, Any] = Field(default_factory=dict)


class PromotionDecision(BaseModel):
    decision: str  # promoted | rejected
    candidate_version: str | None = None
    champion_version: str | None = None
    reasons: list[str] = Field(default_factory=list)
    comparison: list[dict[str, Any]] = Field(default_factory=list)
    primary_metric: str | None = None
    heldout: str | None = None
    decided_at: datetime | None = None
    gate_run_id: str | None = None


class DriftStatus(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_name": "vehicle-detector",
                    "model_version": "1",
                    "status": "drifted",
                    "drift_share": 0.67,
                    "share_threshold": 0.3,
                    "checked_at": "2026-10-17T15:00:00Z",
                    "drifted_features": ["frame_brightness", "confidence", "objects_per_frame"],
                    "report_url": "/api/v1/drift/reports/7d7f2b1e-0d0c-4b39-9a55-1f0f2f3c9a10",
                }
            ]
        }
    )

    model_name: str
    model_version: str | None = None
    status: str  # ok | warning | drifted | no_data | unknown (never checked)
    drift_share: float | None = None
    share_threshold: float | None = None
    check_id: UUID | None = None
    checked_at: datetime | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    current_rows: int | None = None
    reference_rows: int | None = None
    features: list[DriftFeature] = Field(default_factory=list)
    drifted_features: list[str] = Field(default_factory=list)
    quality: list[DataQualityIssue] = Field(default_factory=list)
    report_url: str | None = None
    mlflow_run_id: str | None = None
    mlflow_experiment_id: str | None = None
    reference: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None
    retraining: RetrainingRecommendation | None = None
    last_promotion: PromotionDecision | None = None


class DriftHistoryPoint(BaseModel):
    check_id: UUID
    model_name: str
    model_version: str
    checked_at: datetime
    window_end: datetime
    drift_share: float
    status: str
    drifted_features: list[str] = Field(default_factory=list)


class DriftModeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"enabled": True, "preset": "night", "duration_minutes": 60},
                {"enabled": False},
            ]
        },
    )

    enabled: bool
    preset: Literal["night", "fog", "custom"] = "night"
    visual: bool = Field(default=True, description="degrade frames in video_ingest")
    text: bool = Field(default=True, description="switch event_simulator to slang reports")
    brightness: float | None = Field(default=None, ge=0.05, le=1.0)
    noise_std: float | None = Field(default=None, ge=0.0, le=80.0)
    haze: float | None = Field(default=None, ge=0.0, le=0.95)
    blur: float | None = Field(default=None, ge=0.0, le=25.0)
    duration_minutes: float | None = Field(
        default=None, gt=0, le=24 * 60, description="switch off automatically after this long"
    )


class DriftModeInfo(BaseModel):
    active: bool  # enabled and not expired
    enabled: bool
    settings: dict[str, Any]
    visual_params: dict[str, float]
    expires_at: datetime | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None


class DismissRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"reason": "alerts came from the drift demo; the model is fine"}]
        },
    )

    reason: str = Field(min_length=3, max_length=500, description="why it is dismissed")
    dismissed_by: str = Field(default="api", min_length=1, max_length=50)
