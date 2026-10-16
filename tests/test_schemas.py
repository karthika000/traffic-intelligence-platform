"""Schema round-trip and validation tests (no infrastructure needed)."""

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from traffic_common.enums import IncidentStatus, ReportSource, Severity, SummaryScope
from traffic_common.schemas import (
    TOPIC_SCHEMAS,
    Anomaly,
    Detection,
    DriftAlert,
    FrameMessage,
    Incident,
    KafkaMessage,
    NLPResult,
    RawEvent,
    SimulatorCommand,
    Summary,
    TextReport,
    TrafficMetrics,
    schema_for,
)
from traffic_common.topics import DEAD_LETTER_TOPICS, ConsumerGroup, Topic, dead_letter_topic

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 14, 8, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=1)
CAM = "cam-hyde-park-corner"

BOX = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}

VALID: dict[type[KafkaMessage], dict] = {
    FrameMessage: {
        "camera_id": CAM,
        "sequence": 42,
        "frame_path": f"/data/frames/{CAM}/20261014T080000.000000Z.jpg",
        "width": 1280,
        "height": 720,
        "source_fps": 29.97,
    },
    Detection: {
        "camera_id": CAM,
        "frame_id": uuid4(),
        "model_name": "yolo",
        "model_version": "11n",
        "objects": [{"label": "car", "confidence": 0.91, "bbox": BOX, "track_id": 7}],
    },
    TrafficMetrics: {
        "camera_id": CAM,
        "window_start": T0,
        "window_end": T1,
        "vehicle_count": 37,
        "avg_speed_kmh": 23.5,
        "occupancy": 0.42,
        "counts_by_class": {"car": 30, "bus": 7},
    },
    TextReport: {
        "source": ReportSource.CITIZEN,
        "text": "Lorry broken down on Piccadilly",
        "latitude": 51.5027,
        "longitude": -0.1528,
    },
    RawEvent: {
        "event_type": "signal_failure",
        "source": "simulator",
        "camera_id": CAM,
        "severity": Severity.HIGH,
        "payload": {"junction": "J12"},
    },
    NLPResult: {
        "report_id": uuid4(),
        "incident_type": "breakdown",
        "incident_confidence": 0.93,
        "severity": Severity.MEDIUM,
        "probabilities": {"incident_type": {"breakdown": 0.93, "accident": 0.07}},
        "camera_id": CAM,
        "camera_confidence": 0.88,
        "sentiment": -0.4,
        "model_name": "ner",
        "model_version": "1",
        "entities": [{"text": "Piccadilly", "label": "LOC", "start": 22, "end": 32}],
    },
    Anomaly: {
        "camera_id": CAM,
        "anomaly_type": "speed_drop",
        "metric_name": "avg_speed_kmh",
        "observed_value": 4.0,
        "expected_value": 25.0,
        "score": 4.2,
        "threshold": 3.0,
        "window_start": T0,
        "window_end": T1,
        "model_name": "iforest",
        "model_version": "3",
    },
    Incident: {
        "camera_id": CAM,
        "severity": Severity.HIGH,
        "fused_confidence": 0.87,
        "start_time": T0,
        "title": "Stalled vehicle",
        "anomaly_ids": [uuid4()],
    },
    Summary: {
        "scope": SummaryScope.CAMERA,
        "camera_id": CAM,
        "period_start": T0,
        "period_end": T1,
        "text": "Heavy congestion westbound.",
        "model_name": "llm",
        "model_version": "1",
    },
    DriftAlert: {
        "model_name": "yolo",
        "model_version": "11n",
        "drift_type": "data",
        "metric": "psi",
        "drift_score": 0.31,
        "threshold": 0.2,
        "window_start": T0,
        "window_end": T1,
    },
    SimulatorCommand: {
        "action": "start_scenario",
        "scenario": "multi_vehicle_accident",
        "camera_id": CAM,
        "time_scale": 10,
    },
}


def test_every_schema_has_a_valid_example() -> None:
    assert set(VALID) == set(TOPIC_SCHEMAS.values())


@pytest.mark.parametrize("schema", list(VALID), ids=lambda s: s.__name__)
def test_json_round_trip(schema: type[KafkaMessage]) -> None:
    original = schema.model_validate(VALID[schema])
    restored = schema.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.schema_version == 1
    assert restored.timestamp.tzinfo is UTC


@pytest.mark.parametrize("schema", list(VALID), ids=lambda s: s.__name__)
def test_rejects_unknown_fields(schema: type[KafkaMessage]) -> None:
    with pytest.raises(ValidationError, match="extra"):
        schema.model_validate({**VALID[schema], "unexpected": 1})


@pytest.mark.parametrize("schema", list(VALID), ids=lambda s: s.__name__)
def test_rejects_unsupported_schema_version(schema: type[KafkaMessage]) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        schema.model_validate({**VALID[schema], "schema_version": 2})


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        TrafficMetrics.model_validate({**VALID[TrafficMetrics], "timestamp": datetime(2026, 1, 1)})


def test_offset_timestamps_normalised_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    msg = TrafficMetrics.model_validate(
        {
            **VALID[TrafficMetrics],
            "timestamp": datetime(2026, 1, 1, 10, 30, tzinfo=ist),
            "window_start": "2026-01-01T10:00:00+05:30",
            "window_end": "2026-01-01T10:01:00+05:30",
        }
    )
    assert msg.timestamp == datetime(2026, 1, 1, 5, 0, tzinfo=UTC)
    assert msg.timestamp.tzinfo is UTC
    assert msg.window_start.tzinfo is UTC
    assert '"timestamp":"2026-01-01T05:00:00Z"' in msg.model_dump_json()


def test_messages_are_immutable() -> None:
    msg = TrafficMetrics.model_validate(VALID[TrafficMetrics])
    with pytest.raises(ValidationError):
        msg.vehicle_count = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("schema", "override", "error"),
    [
        (FrameMessage, {"frame_path": ""}, "at least 1"),
        (FrameMessage, {"sequence": -1}, "greater than or equal"),
        (FrameMessage, {"source_fps": 0}, "greater than 0"),
        (FrameMessage, {"width": 0}, "greater than 0"),
        (
            Detection,
            {"objects": [{"label": "car", "confidence": 1.2, "bbox": BOX}]},
            "less than or equal",
        ),
        (
            Detection,
            {"objects": [{"label": "car", "confidence": 0.5, "bbox": {**BOX, "x_max": 0.05}}]},
            "bounding box",
        ),
        (TrafficMetrics, {"window_end": T0}, "window_end must be after"),
        (TrafficMetrics, {"vehicle_count": -1}, "greater than or equal"),
        (TrafficMetrics, {"occupancy": 1.5}, "less than or equal"),
        (TrafficMetrics, {"camera_id": ""}, "at least 1"),
        (TextReport, {"longitude": None}, "set together"),
        (TextReport, {"source": "carrier_pigeon"}, "source"),
        (
            NLPResult,
            {"entities": [{"text": "x", "label": "LOC", "start": 5, "end": 5}]},
            "end must be",
        ),
        (NLPResult, {"sentiment": -2}, "greater than or equal"),
        (NLPResult, {"incident_type": "alien_landing"}, "incident_type"),
        (NLPResult, {"camera_confidence": 1.2}, "less than or equal"),
        (Incident, {"status": IncidentStatus.RESOLVED}, "end_time"),
        (Incident, {"end_time": T0 - timedelta(seconds=1)}, "before start_time"),
        (Incident, {"fused_confidence": 1.01}, "less than or equal"),
        (Summary, {"camera_id": None}, "camera_id"),
        (Summary, {"scope": SummaryScope.INCIDENT}, "incident_id"),
        (DriftAlert, {"drift_type": "vibes"}, "drift_type"),
        (SimulatorCommand, {"camera_id": None}, "needs scenario and camera_id"),
        (SimulatorCommand, {"time_scale": 0}, "greater than 0"),
        (SimulatorCommand, {"action": "explode"}, "action"),
    ],
)
def test_validation_errors(schema: type[KafkaMessage], override: dict, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        schema.model_validate({**VALID[schema], **override})


def test_schema_for_unknown_topic() -> None:
    assert schema_for(Topic.TRAFFIC_METRICS) is TrafficMetrics
    with pytest.raises(KeyError, match="no schema"):
        schema_for("not.a.topic")


def test_every_topic_has_a_schema() -> None:
    assert set(TOPIC_SCHEMAS) == {t.value for t in Topic}


def test_dead_letter_topics() -> None:
    assert dead_letter_topic("fusion-service") == "dlq.fusion-service"
    assert len(DEAD_LETTER_TOPICS) == len(ConsumerGroup)
    # Kafka warns that '.' and '_' can collide in metric names; keep DLQ names free of '_'.
    assert not any("_" in t for t in DEAD_LETTER_TOPICS)


def _ps_array(script: str, name: str) -> list[str]:
    block = re.search(rf"\${name}\s*=\s*@\((.*?)\)", script, re.S)
    assert block, f"${name} not found in create_topics.ps1"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_create_topics_script_matches_python() -> None:
    script = (ROOT / "scripts" / "create_topics.ps1").read_text(encoding="utf-8")
    assert sorted(_ps_array(script, "Topics")) == sorted(t.value for t in Topic)
    assert sorted(_ps_array(script, "ConsumerGroups")) == sorted(g.value for g in ConsumerGroup)
