"""Resolve the detector weights through the MLflow model registry, caching them locally.

1. ``<name>@<alias>`` exists  -> download its artifacts once into the weights volume.
2. it doesn't                 -> download the pretrained Ultralytics weights, log them as a
                                 run, register a new version and point the alias at it.
3. MLflow unreachable         -> use the pretrained weights directly (version "local").
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, update
from sqlalchemy.dialects.postgresql import insert

from traffic_common.db.models import ModelVersion
from traffic_common.enums import ModelStage

log = logging.getLogger(__name__)

EXPERIMENT = "vehicle-detector"


@dataclass
class ResolvedModel:
    name: str
    version: str
    weights: Path
    source: str  # registry | registered | local-fallback
    run_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.name}/{self.version}"


def pretrained_weights(model: str, weights_dir: Path) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset

    target = weights_dir / "pretrained" / model
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(attempt_download_asset(str(target)))


def resolve_model(
    tracking_uri: str,
    name: str,
    alias: str,
    model: str,
    imgsz: int,
    classes: tuple[str, ...],
    weights_dir: Path,
) -> ResolvedModel:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    # Check the model and alias explicitly: a missing alias is reported as
    # INVALID_PARAMETER_VALUE by MLflow 2.x, not RESOURCE_DOES_NOT_EXIST.
    try:
        registered = client.get_registered_model(name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            return _local_fallback(model, name, weights_dir, exc)
        registered = None
    except Exception as exc:  # connection refused, DNS, timeouts
        return _local_fallback(model, name, weights_dir, exc)
    if registered is None or alias not in (registered.aliases or {}):
        log.info("no %s@%s in the MLflow registry; registering pretrained %s", name, alias, model)
        return _register_pretrained(client, name, alias, model, imgsz, classes, weights_dir)
    version = client.get_model_version_by_alias(name, alias)
    cache = weights_dir / "registry" / name / f"v{version.version}"
    weights = next(cache.rglob("*.pt"), None)
    if weights is None:
        log.info("downloading %s v%s from MLflow", name, version.version)
        uri = client.get_model_version_download_uri(name, version.version)
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(cache))
        weights = next(cache.rglob("*.pt"), None)
        if weights is None:
            raise RuntimeError(f"{name} v{version.version} has no .pt file in its artifacts")
    run_params = client.get_run(version.run_id).data.params if version.run_id else {}
    return ResolvedModel(
        name, str(version.version), weights, "registry", version.run_id, run_params
    )


def _register_pretrained(client, name, alias, model, imgsz, classes, weights_dir) -> ResolvedModel:
    import mlflow
    import ultralytics
    from mlflow.exceptions import MlflowException

    weights = pretrained_weights(model, weights_dir)
    params = {
        "model": model,
        "input_size": imgsz,
        "classes": ",".join(classes),
        "source": "Ultralytics pretrained (COCO)",
        "ultralytics_version": ultralytics.__version__,
    }
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"register-pretrained-{Path(model).stem}") as run:
        mlflow.log_params(params)
        mlflow.log_artifact(str(weights), artifact_path="model")
    try:
        client.create_registered_model(name, description="Vehicle and pedestrian detector (YOLO)")
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    mv = client.create_model_version(name, f"runs:/{run.info.run_id}/model", run.info.run_id)
    for _ in range(30):  # file-backed sources are READY almost immediately
        if client.get_model_version(name, mv.version).status == "READY":
            break
        time.sleep(1)
    client.set_registered_model_alias(name, alias, mv.version)
    log.info("registered %s v%s (run %s) as @%s", name, mv.version, run.info.run_id, alias)
    return ResolvedModel(name, str(mv.version), weights, "registered", run.info.run_id, params)


def _local_fallback(model: str, name: str, weights_dir: Path, exc: Exception) -> ResolvedModel:
    log.warning("MLflow registry unavailable (%s); using pretrained %s directly", exc, model)
    return ResolvedModel(
        name,
        "local",
        pretrained_weights(model, weights_dir),
        "local-fallback",
        params={"model": model},
    )


def record_model_version(engine: Engine, model: ResolvedModel, metrics: dict[str, Any]) -> None:
    """Upsert the model in use into model_versions as production; any other production
    version of the same model is archived (one version serves at a time)."""
    stmt = insert(ModelVersion).values(
        model_name=model.name,
        version=model.version,
        stage=ModelStage.PRODUCTION,
        mlflow_run_id=model.run_id,
        metrics=metrics,
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
                ModelVersion.model_name == model.name,
                ModelVersion.version != model.version,
                ModelVersion.stage == ModelStage.PRODUCTION,
            )
            .values(stage=ModelStage.ARCHIVED)
        )
        conn.execute(stmt)
