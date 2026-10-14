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
