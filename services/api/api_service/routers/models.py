"""Models in service (model_versions) with their champion and evaluation metrics from MLflow.

MLflow is read over its REST API (no mlflow package in the API image) and cached; if it is
unreachable, the database part is still returned with ``mlflow: null``."""

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service.db import connection
from api_service.responses import ModelInfo

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


class MlflowReader:
    def __init__(self, client: httpx.AsyncClient, cache_seconds: float) -> None:
        self.client, self.cache_seconds = client, cache_seconds
        self._cache: dict[str, tuple[float, dict | None]] = {}

    async def _get(self, path: str, **params) -> dict | None:
        r = await self.client.get(f"/api/2.0/mlflow/{path}", params=params)
        if r.status_code == 404 or (r.status_code == 400 and "RESOURCE_DOES_NOT_EXIST" in r.text):
            return None
        r.raise_for_status()
        return r.json()

    async def info(self, name: str) -> dict | None:
        hit = self._cache.get(name)
        if hit and time.monotonic() - hit[0] < self.cache_seconds:
            return hit[1]
        try:
            value = await self._info(name)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("MLflow lookup for %s failed: %s", name, exc)
            return None  # not cached: try again next time
        self._cache[name] = (time.monotonic(), value)
        return value

    async def _info(self, name: str) -> dict[str, Any]:
        alias = await self._get("registered-models/alias", name=name, alias="champion")
        if alias is None:
            return {"registered": False}
        mv = alias["model_version"]
        out: dict[str, Any] = {
            "registered": True,
            "champion_version": mv["version"],
            "run_id": mv.get("run_id"),
            "description": mv.get("description"),
            "tags": {t["key"]: t["value"] for t in mv.get("tags", [])},
        }
        if mv.get("run_id"):
            run = await self._get("runs/get", run_id=mv["run_id"])
            if run:
                out["run"] = _run_summary(run["run"])
        # The latest finished run in the experiment of the same name (e.g. an online evaluation).
        exp = await self._get("experiments/get-by-name", experiment_name=name)
        if exp:
            r = await self.client.post(
                "/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [exp["experiment"]["experiment_id"]],
                    "filter": "attributes.status = 'FINISHED'",
                    "order_by": ["attributes.start_time DESC"],
                    "max_results": 1,
                },
            )
            r.raise_for_status()
            runs = r.json().get("runs", [])
            if runs:
                out["latest_run"] = _run_summary(runs[0])
        return out


def _run_summary(run: dict) -> dict[str, Any]:
    info, data = run["info"], run.get("data", {})
    return {
        "run_id": info["run_id"],
        "run_name": info.get("run_name"),
        "end_time": info.get("end_time"),
        "metrics": {m["key"]: m["value"] for m in data.get("metrics", [])},
        "tags": {
            t["key"]: t["value"] for t in data.get("tags", []) if not t["key"].startswith("mlflow.")
        },
    }


@router.get("", response_model=list[ModelInfo])
async def list_models(
    request: Request, conn: AsyncConnection = Depends(connection)
) -> list[ModelInfo]:
    """The version of each model currently in service (stage production in model_versions)."""
    rows = (
        (
            await conn.execute(
                text(
                    "SELECT DISTINCT ON (model_name) model_name, version, stage, mlflow_run_id, "
                    "metrics, registered_at FROM model_versions WHERE stage = 'production' "
                    "ORDER BY model_name, registered_at DESC"
                )
            )
        )
        .mappings()
        .all()
    )
    reader: MlflowReader = request.app.state.mlflow
    infos = await asyncio.gather(*(reader.info(r["model_name"]) for r in rows))
    return [
        ModelInfo(
            model_name=r["model_name"],
            version=r["version"],
            stage=r["stage"],
            mlflow_run_id=r["mlflow_run_id"],
            registered_at=r["registered_at"],
            details=r["metrics"] or {},
            mlflow=info,
        )
        for r, info in zip(rows, infos, strict=True)
    ]
