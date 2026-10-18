"""Evaluation-gated promotion (ml.lifecycle): the gate's decision when the candidate is better,
worse, or better on the primary metric but regressing elsewhere; the per-model policies; and
the lifecycle records in Postgres (retraining recommendations from drift alerts, promotion
decisions in model_versions)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from ml.lifecycle.gate import GatePolicy, MetricSpec, decide
from ml.lifecycle.retrain import policy_for

POLICY = GatePolicy(
    MetricSpec("macro_f1"),
    (
        MetricSpec("accuracy"),
        MetricSpec("recall_standstill", tolerance=0.02),
        MetricSpec("false_alarms_per_hour", higher_is_better=False, tolerance=0.25),
    ),
    tolerance=0.01,
)
CHAMPION = {
    "macro_f1": 0.90,
    "accuracy": 0.92,
    "recall_standstill": 0.80,
    "false_alarms_per_hour": 1.0,
}


def test_candidate_better_everywhere_is_promoted() -> None:
    candidate = {"macro_f1": 0.93, "accuracy": 0.93, "recall_standstill": 0.85,
                 "false_alarms_per_hour": 0.8}  # fmt: skip
    d = decide(POLICY, CHAMPION, candidate)
    assert d.promoted and d.decision == "promoted"
    assert d.primary_delta == pytest.approx(0.03)
    assert "beats the champion on macro_f1" in d.reasons[0]
    assert all(c["ok"] for c in d.comparison)


def test_candidate_worse_on_primary_is_rejected() -> None:
    candidate = {**CHAMPION, "macro_f1": 0.88}
    d = decide(POLICY, CHAMPION, candidate)
    assert not d.promoted and d.decision == "rejected"
    assert "does not beat the champion on macro_f1" in d.reasons[0]


def test_equal_primary_is_not_an_improvement() -> None:
    assert not decide(POLICY, CHAMPION, dict(CHAMPION)).promoted


def test_better_on_primary_but_regressing_beyond_tolerance_is_rejected() -> None:
    candidate = {**CHAMPION, "macro_f1": 0.95, "recall_standstill": 0.70}  # -0.10 > 0.02
    d = decide(POLICY, CHAMPION, candidate)
    assert not d.promoted
    assert any("regresses on recall_standstill" in r for r in d.reasons)
    bad = [c for c in d.comparison if not c["ok"]]
    assert [c["metric"] for c in bad] == ["recall_standstill"]


def test_lower_is_better_guard_regression() -> None:
    candidate = {**CHAMPION, "macro_f1": 0.95, "false_alarms_per_hour": 1.5}  # +0.5 > 0.25
    d = decide(POLICY, CHAMPION, candidate)
    assert not d.promoted and any("false_alarms_per_hour" in r for r in d.reasons)
    within = {**CHAMPION, "macro_f1": 0.95, "false_alarms_per_hour": 1.2}
    assert decide(POLICY, CHAMPION, within).promoted


def test_regression_within_the_default_tolerance_is_allowed() -> None:
    candidate = {**CHAMPION, "macro_f1": 0.91, "accuracy": 0.912}  # -0.008 <= 0.01
    d = decide(POLICY, CHAMPION, candidate)
    assert d.promoted and "no guard metric regressed" in d.reasons[-1]


def test_min_improvement() -> None:
    strict = GatePolicy(MetricSpec("macro_f1"), (), min_improvement=0.02)
    assert not decide(strict, CHAMPION, {**CHAMPION, "macro_f1": 0.915}).promoted
    assert decide(strict, CHAMPION, {**CHAMPION, "macro_f1": 0.93}).promoted


def test_no_champion_promotes_and_missing_metrics() -> None:
    assert decide(POLICY, None, {"macro_f1": 0.5}).promoted
    assert not decide(POLICY, CHAMPION, {"accuracy": 0.99}).promoted  # no primary metric
    d = decide(POLICY, {"macro_f1": 0.9}, {"macro_f1": 0.95})  # guards not available
    assert d.promoted and all(c.get("note") for c in d.comparison if c["role"] == "guard")


def test_policies_per_model() -> None:
    text_policy = policy_for("incident-text-classifier", 0.01, 0.0)
    assert text_policy.primary.name == "heldout_mean_macro_f1"
    assert {g.name for g in text_policy.guards} >= {"test_incident_type_macro_f1",
                                                     "test_severity_macro_f1"}  # fmt: skip
    anomaly = policy_for("traffic-anomaly-detector", 0.01, 0.0)
    fa = next(g for g in anomaly.guards if g.name == "real_false_alarms_per_hour")
    assert not fa.higher_is_better
    assert policy_for("congestion-classifier", 0.05, 0.0).tolerance == 0.05


# ----------------------------------------------------------------------------- records


@pytest.fixture
def engine(scratch_database: str):
    from alembic import command

    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config

    command.upgrade(alembic_config(scratch_database), "head")
    eng = create_db_engine(scratch_database)
    yield eng
    eng.dispose()


def test_retraining_recommendation_counts_alerts_once_and_resolves(engine) -> None:
    from ml.lifecycle.records import recommend_retraining, resolve_recommendation

    features = [{"feature": "report_embedding", "score": 0.97, "threshold": 0.55}]
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    row = recommend_retraining(engine, "incident-text-classifier", "1", "drift", features, first)
    assert row["status"] == "recommended" and row["alert_count"] == 1
    assert recommend_retraining(engine, "incident-text-classifier", "1", "drift", [], first) is None
    row = recommend_retraining(engine, "incident-text-classifier", "1", "again", features, second)
    assert row["alert_count"] == 2
    assert resolve_recommendation(engine, "incident-text-classifier", {"decision": "promoted"})
    assert not resolve_recommendation(engine, "incident-text-classifier", {})
    # A new alert after resolution opens a new recommendation.
    row = recommend_retraining(
        engine, "incident-text-classifier", "2", "drift", features, str(uuid.uuid4())
    )
    assert row["alert_count"] == 1
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT status, resolution FROM retraining_recommendations ORDER BY id")
        ).all()
    assert [r.status for r in rows] == ["resolved", "recommended"]
    assert rows[0].resolution == {"decision": "promoted"}


def test_dismissed_recommendation_closes_and_a_new_alert_reopens(engine) -> None:
    from ml.lifecycle.records import recommend_retraining, resolve_recommendation

    recommend_retraining(engine, "vehicle-detector", "1", "drift", [], str(uuid.uuid4()))
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE retraining_recommendations SET status = 'dismissed', resolved_at = now(), "
                "resolution = '{\"decision\": \"dismissed\"}' WHERE status = 'recommended'"
            )
        )
    assert not resolve_recommendation(engine, "vehicle-detector", {})  # nothing open
    row = recommend_retraining(engine, "vehicle-detector", "1", "again", [], str(uuid.uuid4()))
    assert row["alert_count"] == 1  # a new record, not the dismissed one
    with engine.connect() as conn:
        statuses = conn.execute(
            text("SELECT status FROM retraining_recommendations ORDER BY id")
        ).scalars()
        assert list(statuses) == ["dismissed", "recommended"]


def test_promotion_decision_in_model_versions(engine) -> None:
    from ml.lifecycle.records import record_promotion
    from traffic_common.db.registry import record_model_version

    record_promotion(engine, "congestion-classifier", "2", "run-2",
                     {"promoted": False, "decision": "rejected", "reasons": ["worse"]})  # fmt: skip
    record_promotion(engine, "congestion-classifier", "3", "run-3",
                     {"promoted": True, "decision": "promoted", "reasons": ["better"]})  # fmt: skip
    # The service loads v3: production, and the decision is kept.
    record_model_version(engine, "congestion-classifier", "3", mlflow_run_id="run-3")
    with engine.connect() as conn:
        rows = {
            r.version: r
            for r in conn.execute(
                text("SELECT version, stage, promotion FROM model_versions ORDER BY version")
            )
        }
    assert rows["2"].stage == "archived" and rows["2"].promotion["decision"] == "rejected"
    assert rows["3"].stage == "production" and rows["3"].promotion["decision"] == "promoted"


def test_monitor_records_retraining_recommended_on_drift_alert(engine) -> None:
    from ml.monitoring.config import MonitorConfig
    from ml.monitoring.monitor import Monitor
    from traffic_common.schemas import DriftAlert

    monitor = Monitor(MonitorConfig(), engine, bootstrap="unused:9092")
    now = datetime.now(UTC)
    alert = DriftAlert(
        window_start=now - timedelta(minutes=15),
        window_end=now,
        model_name="vehicle-detector",
        model_version="1",
        drift_type="data",
        metric="drift_share",
        drift_score=0.67,
        threshold=0.3,
        details={"drifted_features": [{"feature": "frame_brightness", "score": 2.4}]},
    )
    monitor.on_alert(alert)
    monitor.on_alert(alert)  # redelivered: counted once
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM retraining_recommendations")).one()
    assert row.status == "recommended" and row.alert_count == 1
    assert "67% of features drifted" in row.reason and "frame_brightness" in row.reason
    assert monitor._retraining == {}  # automatic retraining is off by default
