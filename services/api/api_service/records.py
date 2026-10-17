"""Database rows -> the shared message schemas (traffic_common.schemas)."""

import logging
from datetime import timedelta
from typing import Any

from pydantic import ValidationError

from api_service.responses import AnomalyRecord
from traffic_common.schemas import (
    Anomaly,
    Incident,
    NLPResult,
    Summary,
    TextReport,
    TrafficMetrics,
)

log = logging.getLogger(__name__)
METRIC_COLUMNS = (
    "camera_id",
    "window_start",
    "window_end",
    "vehicle_count",
    "avg_speed_kmh",
    "p85_speed_kmh",
    "occupancy",
    "density",
    "stopped_vehicles",
    "pedestrians_in_roadway",
    "mean_confidence",
    "frames",
    "model_version",
    "counts_by_class",
)


def _ids(values) -> dict:
    return {k: v for k, v in values.items() if v is not None}


def metrics(row: Any) -> TrafficMetrics:
    m = row._mapping
    return TrafficMetrics(
        **{c: m[c] for c in METRIC_COLUMNS if m.get(c) is not None},
        **_ids({"message_id": m.get("message_id")}),
        timestamp=m["window_end"],
    )


def anomaly(row: Any) -> AnomalyRecord | None:
    m = row._mapping
    try:
        message = Anomaly(
            anomaly_id=m["id"],
            scope=m["scope"],
            camera_id=m["camera_id"],
            affected_cameras=m["affected_cameras"] or [],
            event="end" if m["status"] == "ended" else "start",
            severity=m["severity"],
            anomaly_type=m["anomaly_type"],
            hints=m["hints"] or [],
            metric_name=m["metric_name"],
            observed_value=m["observed_value"],
            expected_value=m["expected_value"],
            score=m["score"],
            threshold=m["threshold"],
            contributing_features=m["contributing_features"] or {},
            congestion_level=m["congestion_level"],
            model_name=m["model_name"],
            model_version=m["model_version"],
            model_versions=m["model_versions"] or {},
            # A start event's window covers the minute before the alert started.
            started_at=m["window_start"] + timedelta(seconds=60),
            ended_at=m["ended_at"],
            window_start=m["window_start"],
            window_end=m["window_end"],
            timestamp=m["detected_at"],
            details=m["details"] or {},
        )
    except ValidationError as exc:  # an old row from before a schema change
        log.debug("skipping anomaly %s: %s", m["id"], exc)
        return None
    return AnomalyRecord(
        anomaly=message,
        status=m["status"],
        detected_at=m["detected_at"],
        update_count=m["update_count"],
    )


def incident(row: Any) -> Incident:
    m = row._mapping
    details = m["details"] or {}
    state = details.get("state") or {}
    history = details.get("history") or []
    return Incident(
        incident_id=m["id"],
        camera_id=m["camera_id"],
        version=m["version"],
        status=m["status"],
        incident_type=m["incident_type"],
        severity=m["severity"],
        fused_confidence=m["fused_confidence"],
        confidence_components=m["confidence_components"] or {},
        vision_degraded=m["vision_degraded"],
        start_time=m["start_time"],
        suspected_at=m["suspected_at"],
        confirmed_at=m["confirmed_at"],
        end_time=m["end_time"],
        title=m["title"],
        description=m["description"],
        change_reasons=history[-1]["reasons"] if history else [],
        metric_message_ids=state.get("metric_ids", []),
        anomaly_ids=list(state.get("anomalies", {})),
        report_ids=list(state.get("reports", {})),
        timestamp=m["updated_at"],
    )


def summary(row: Any) -> Summary:
    m = row._mapping
    return Summary(
        summary_id=m["id"],
        scope=m["scope"],
        camera_id=m["camera_id"],
        incident_id=m["incident_id"],
        incident_version=m["incident_version"],
        period_start=m["period_start"],
        period_end=m["period_end"],
        headline=m["headline"],
        text=m["text"],
        generation=m["generation"],
        model_name=m["model_name"],
        model_version=m["model_version"],
        prompt_version=m["prompt_version"],
        latency_ms=m["latency_ms"],
        faithfulness=m["faithfulness"] or {},
        timestamp=m["created_at"],
    )


def report(row: Any, prefix: str = "r_") -> TextReport:
    m = row._mapping
    return TextReport(
        report_id=m[f"{prefix}id"],
        source=m[f"{prefix}source"],
        text=m[f"{prefix}text"],
        camera_id=m[f"{prefix}camera_id"],
        location_name=m[f"{prefix}location_name"],
        latitude=m[f"{prefix}latitude"],
        longitude=m[f"{prefix}longitude"],
        timestamp=m[f"{prefix}reported_at"],
        scenario_id=m.get(f"{prefix}scenario_id"),
    )


def nlp(row: Any, prefix: str = "n_") -> NLPResult | None:
    m = row._mapping
    if m.get(f"{prefix}report_id") is None:
        return None
    return NLPResult(
        **_ids({"message_id": m[f"{prefix}message_id"]}),
        report_id=m[f"{prefix}report_id"],
        source=m.get("r_source"),
        incident_type=m[f"{prefix}incident_type"],
        incident_confidence=m[f"{prefix}incident_confidence"] or 0.0,
        severity=m[f"{prefix}severity"],
        severity_confidence=m[f"{prefix}severity_confidence"],
        probabilities=m[f"{prefix}probabilities"] or {},
        stage=m[f"{prefix}stage"] or "active",
        sentiment=m[f"{prefix}sentiment"],
        entities=m[f"{prefix}entities"] or [],
        camera_id=m[f"{prefix}camera_id"],
        camera_confidence=m[f"{prefix}camera_confidence"],
        location_name=m[f"{prefix}location_name"],
        model_name=m[f"{prefix}model_name"],
        model_version=m[f"{prefix}model_version"],
        model_versions=m[f"{prefix}model_versions"] or {},
        details=m[f"{prefix}details"] or {},
        timestamp=m[f"{prefix}processed_at"],
    )


REPORT_SELECT = """
    r.id AS r_id, r.source AS r_source, r.text AS r_text, r.camera_id AS r_camera_id,
    r.location_name AS r_location_name, r.latitude AS r_latitude, r.longitude AS r_longitude,
    r.reported_at AS r_reported_at,
    (SELECT s.scenario_run_id FROM simulated_reports s WHERE s.id = r.id) AS r_scenario_id,
    n.message_id AS n_message_id, n.report_id AS n_report_id, n.incident_type AS n_incident_type,
    n.incident_confidence AS n_incident_confidence, n.severity AS n_severity,
    n.severity_confidence AS n_severity_confidence, n.probabilities AS n_probabilities,
    n.stage AS n_stage, n.sentiment AS n_sentiment, n.entities AS n_entities,
    n.camera_id AS n_camera_id, n.camera_confidence AS n_camera_confidence,
    n.location_name AS n_location_name, n.model_name AS n_model_name,
    n.model_version AS n_model_version, n.model_versions AS n_model_versions,
    n.details AS n_details, n.processed_at AS n_processed_at
"""
# The latest NLP result per report.
LATEST_NLP_JOIN = """
    LEFT JOIN LATERAL (
        SELECT * FROM nlp_results x WHERE x.report_id = r.id ORDER BY x.processed_at DESC LIMIT 1
    ) n ON true
"""
