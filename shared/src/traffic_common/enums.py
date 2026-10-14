"""Enumerations shared by the Kafka schemas and the database models."""

from enum import StrEnum


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(StrEnum):
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class ReportSource(StrEnum):
    OPERATOR = "operator"
    CITIZEN = "citizen"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    POLICE = "police"


class SummaryScope(StrEnum):
    CAMERA = "camera"
    INCIDENT = "incident"
    CITYWIDE = "citywide"


class EvidenceType(StrEnum):
    METRIC = "metric"
    ANOMALY = "anomaly"
    REPORT = "report"


class DriftType(StrEnum):
    DATA = "data"
    PREDICTION = "prediction"
    CONCEPT = "concept"


class ModelStage(StrEnum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class IncidentType(StrEnum):
    """Incident categories for text reports (labels for the NLP model). ``none`` is chatter."""

    ACCIDENT = "accident"
    BREAKDOWN = "breakdown"
    ROAD_WORKS = "road_works"
    CONGESTION = "congestion"
    WEATHER_HAZARD = "weather_hazard"
    DEBRIS = "debris"
    POLICE_ACTIVITY = "police_activity"
    NONE = "none"


class ScenarioStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
