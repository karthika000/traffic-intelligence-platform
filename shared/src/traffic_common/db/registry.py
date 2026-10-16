"""Recording which model versions are serving, in the model_versions table."""

from typing import Any

from sqlalchemy import Engine, update
from sqlalchemy.dialects.postgresql import insert

from traffic_common.db.models import ModelVersion
from traffic_common.enums import ModelStage


def record_model_version(
    engine: Engine,
    model_name: str,
    version: str,
    *,
    mlflow_run_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Upsert ``model_name``/``version`` as production and archive any other production
    version of the same model (one version serves at a time)."""
    stmt = insert(ModelVersion).values(
        model_name=model_name,
        version=version,
        stage=ModelStage.PRODUCTION,
        mlflow_run_id=mlflow_run_id,
        metrics=metrics or {},
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[ModelVersion.model_name, ModelVersion.version],
        set_={
            "stage": stmt.excluded.stage,
            "mlflow_run_id": stmt.excluded.mlflow_run_id,
            "metrics": stmt.excluded.metrics,
        },
    )
    with engine.begin() as conn:
        conn.execute(
            update(ModelVersion)
            .where(
                ModelVersion.model_name == model_name,
                ModelVersion.version != version,
                ModelVersion.stage == ModelStage.PRODUCTION,
            )
            .values(stage=ModelStage.ARCHIVED)
        )
        conn.execute(stmt)
