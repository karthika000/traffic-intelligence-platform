"""Lifecycle records in Postgres: retraining recommendations (from drift alerts) and
promotion decisions (from the retrain CLI, in model_versions)."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, text

from traffic_common.enums import ModelStage

log = logging.getLogger(__name__)

_RECOMMEND_SQL = """
INSERT INTO retraining_recommendations
    (model_name, model_version, status, reason, alert_count, drifted_features,
     first_alert_at, last_alert_at, last_alert_id)
VALUES (:model_name, :model_version, :status, :reason, 1, CAST(:features AS jsonb),
        :at, :at, :alert_id)
ON CONFLICT (model_name) WHERE status IN ('recommended', 'retraining') DO UPDATE SET
    model_version = EXCLUDED.model_version,
    reason = EXCLUDED.reason,
    drifted_features = EXCLUDED.drifted_features,
    last_alert_at = EXCLUDED.last_alert_at,
    last_alert_id = EXCLUDED.last_alert_id,
    alert_count = retraining_recommendations.alert_count + 1
WHERE retraining_recommendations.last_alert_id IS DISTINCT FROM EXCLUDED.last_alert_id
RETURNING id, status, alert_count
"""
_SET_STATUS_SQL = """
UPDATE retraining_recommendations SET status = :status
WHERE model_name = :model_name AND status IN ('recommended', 'retraining')
"""
_RESOLVE_SQL = """
UPDATE retraining_recommendations
SET status = 'resolved', resolved_at = now(), resolution = CAST(:resolution AS jsonb)
WHERE model_name = :model_name AND status IN ('recommended', 'retraining')
RETURNING id
"""
_PROMOTION_SQL = """
INSERT INTO model_versions (model_name, version, stage, mlflow_run_id, metrics, promotion)
VALUES (:model_name, :version, :stage, :run_id, CAST(:metrics AS jsonb), CAST(:promotion AS jsonb))
ON CONFLICT (model_name, version) DO UPDATE SET promotion = EXCLUDED.promotion,
    stage = CASE WHEN model_versions.stage = 'production' THEN model_versions.stage
                 ELSE EXCLUDED.stage END
"""


def recommend_retraining(
    engine: Engine,
    model_name: str,
    model_version: str,
    reason: str,
    drifted_features: list[dict[str, Any]],
    alert_id: str | None,
    at: datetime | None = None,
    status: str = "recommended",
) -> dict[str, Any] | None:
    """Open (or update) the model's retraining recommendation; None if this alert was already
    counted (a redelivered message)."""
    with engine.begin() as conn:
        row = conn.execute(
            text(_RECOMMEND_SQL),
            {
                "model_name": model_name,
                "model_version": model_version,
                "status": status,
                "reason": reason,
                "features": json.dumps(drifted_features),
                "at": at or datetime.now(UTC),
                "alert_id": alert_id,
            },
        ).first()
    return dict(row._mapping) if row else None


def set_recommendation_status(engine: Engine, model_name: str, status: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(_SET_STATUS_SQL), {"model_name": model_name, "status": status})


def resolve_recommendation(engine: Engine, model_name: str, resolution: dict[str, Any]) -> bool:
    with engine.begin() as conn:
        row = conn.execute(
            text(_RESOLVE_SQL),
            {"model_name": model_name, "resolution": json.dumps(resolution, default=str)},
        ).first()
    return row is not None


def record_promotion(
    engine: Engine,
    model_name: str,
    version: str,
    run_id: str | None,
    promotion: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    """The candidate version's row in model_versions with the gate's decision. A promoted
    candidate is "staging" until a service loads it (the service marks it production);
    a rejected one is "archived"."""
    stage = ModelStage.STAGING if promotion.get("promoted") else ModelStage.ARCHIVED
    with engine.begin() as conn:
        conn.execute(
            text(_PROMOTION_SQL),
            {
                "model_name": model_name,
                "version": version,
                "stage": stage.value,
                "run_id": run_id,
                "metrics": json.dumps(metrics or {}, default=str),
                "promotion": json.dumps(promotion, default=str),
            },
        )
