"""Drift monitoring (ml-monitor's results in drift_reports), retraining recommendations,
promotion decisions, and the demo drift mode switch."""

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service.db import connection
from api_service.responses import (
    DismissRequest,
    DriftHistoryPoint,
    DriftModeInfo,
    DriftModeRequest,
    DriftStatus,
    PromotionDecision,
    RetrainingRecommendation,
)
from api_service.routers.common import not_found
from traffic_common.driftmode import (
    EVENT_SQL,
    SELECT_SQL,
    UPSERT_SQL,
    DriftModeSettings,
    DriftModeState,
    state_from_row,
    write_params,
)
from traffic_common.schemas import DriftAlert

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])

# The models ml-monitor checks (ml.monitoring.config.MONITORED_MODELS).
MONITORED_MODELS = (
    "vehicle-detector",
    "congestion-classifier",
    "traffic-anomaly-detector",
    "incident-text-classifier",
)

_LATEST_SQL = """
SELECT DISTINCT ON (model_name) model_name, model_version, status, drift_score, threshold,
       check_id, created_at, window_start, window_end, report
FROM drift_reports WHERE feature IS NULL AND check_id IS NOT NULL
ORDER BY model_name, created_at DESC
"""
_RETRAINING_SQL = """
SELECT DISTINCT ON (model_name) model_name, model_version, status, reason, alert_count,
       drifted_features, first_alert_at, last_alert_at, resolved_at, resolution
FROM retraining_recommendations
ORDER BY model_name, (status IN ('recommended', 'retraining')) DESC, created_at DESC
"""
_PROMOTION_SQL = """
SELECT DISTINCT ON (model_name) model_name, version, promotion
FROM model_versions WHERE promotion <> '{}'::jsonb
ORDER BY model_name, promotion->>'decided_at' DESC NULLS LAST, registered_at DESC
"""


def _status(row, retraining, promotion) -> DriftStatus:
    report = row["report"] or {}
    mlflow = report.get("mlflow") or {}
    return DriftStatus(
        model_name=row["model_name"],
        model_version=row["model_version"],
        status=row["status"] or "unknown",
        drift_share=row["drift_score"],
        share_threshold=row["threshold"],
        check_id=row["check_id"],
        checked_at=row["created_at"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        current_rows=report.get("current_rows"),
        reference_rows=report.get("reference_rows"),
        features=report.get("features", []),
        drifted_features=report.get("drifted_features", []),
        quality=report.get("quality", []),
        report_url=report.get("report_url"),
        mlflow_run_id=mlflow.get("run_id"),
        mlflow_experiment_id=mlflow.get("experiment_id"),
        reference=report.get("reference", {}),
        note=report.get("note"),
        retraining=retraining,
        last_promotion=promotion,
    )


@router.get("/status", response_model=list[DriftStatus])
async def drift_status(conn: AsyncConnection = Depends(connection)) -> list[DriftStatus]:
    """The latest drift check of every monitored model (status ok, warning, drifted or no_data;
    "unknown" if never checked), with per-feature scores, data quality issues, the HTML report,
    the open (or last) retraining recommendation and the last promotion decision."""
    latest = {r["model_name"]: r for r in (await conn.execute(text(_LATEST_SQL))).mappings()}
    retraining = {
        r["model_name"]: RetrainingRecommendation(
            **{k: v for k, v in r.items() if k != "model_name"}
        )
        for r in (await conn.execute(text(_RETRAINING_SQL))).mappings()
    }
    promotions = {}
    for r in (await conn.execute(text(_PROMOTION_SQL))).mappings():
        p = r["promotion"]
        promotions[r["model_name"]] = PromotionDecision(
            **{k: p.get(k) for k in PromotionDecision.model_fields if k in p}
        )
    out = []
    for name in (*MONITORED_MODELS, *sorted(set(latest) - set(MONITORED_MODELS))):
        if name in latest:
            out.append(_status(latest[name], retraining.get(name), promotions.get(name)))
        else:
            out.append(
                DriftStatus(
                    model_name=name,
                    status="unknown",
                    retraining=retraining.get(name),
                    last_promotion=promotions.get(name),
                )
            )
    return out


@router.get("/history", response_model=list[DriftHistoryPoint])
async def drift_history(
    model_name: str | None = None,
    hours: float = Query(24, gt=0, le=24 * 30),
    conn: AsyncConnection = Depends(connection),
) -> list[DriftHistoryPoint]:
    """Every check in the last ``hours`` (oldest first): drift share (the drift score charted
    over time), status and drifted features."""
    where = "AND model_name = :m" if model_name else ""
    rows = await conn.execute(
        text(
            "SELECT check_id, model_name, model_version, created_at AS checked_at, window_end, "
            "drift_score AS drift_share, status, "
            "COALESCE(report->'drifted_features', '[]'::jsonb) AS drifted_features "
            "FROM drift_reports WHERE feature IS NULL AND check_id IS NOT NULL "
            f"AND created_at >= :since {where} ORDER BY created_at"
        ),
        {"since": datetime.now(UTC) - timedelta(hours=hours), "m": model_name},
    )
    return [DriftHistoryPoint(**r) for r in rows.mappings()]


@router.get("/alerts", response_model=list[DriftAlert])
async def drift_alerts(
    hours: float = Query(24, gt=0, le=24 * 30),
    limit: int = Query(50, ge=1, le=500),
    conn: AsyncConnection = Depends(connection),
) -> list[DriftAlert]:
    """Recent DriftAlerts (newest first), as published on drift.alerts."""
    rows = await conn.execute(
        text(
            "SELECT report->'alert' AS alert FROM drift_reports "
            "WHERE feature IS NULL AND report->'alert' IS NOT NULL "
            "AND jsonb_typeof(report->'alert') = 'object' AND created_at >= :since "
            "ORDER BY created_at DESC LIMIT :limit"
        ),
        {"since": datetime.now(UTC) - timedelta(hours=hours), "limit": limit},
    )
    return [DriftAlert.model_validate(r["alert"]) for r in rows.mappings()]


@router.get(
    "/reports/{check_id}",
    response_class=FileResponse,
    responses={200: {"content": {"text/html": {}}, "description": "Evidently HTML report"}},
)
async def drift_report(
    check_id: UUID, request: Request, conn: AsyncConnection = Depends(connection)
) -> FileResponse:
    """A check's Evidently HTML report. ml-monitor keeps the newest DRIFT_REPORTS_KEEP per model
    on disk; every report is also an artifact of the check's MLflow run."""
    row = (
        await conn.execute(
            text(
                "SELECT model_name, report->'mlflow'->>'run_id' AS run_id FROM drift_reports "
                "WHERE check_id = :c AND feature IS NULL"
            ),
            {"c": check_id},
        )
    ).first()
    if row is None:
        raise not_found("drift check")
    path = request.app.state.config.drift_reports_dir / row.model_name / f"{check_id}.html"
    if not path.is_file():
        detail = "report no longer kept on disk"
        if row.run_id:
            detail += f"; it is in MLflow run {row.run_id}"
        raise HTTPException(404, detail)
    return FileResponse(path, media_type="text/html")


_DISMISS_SQL = """
UPDATE retraining_recommendations
SET status = 'dismissed', resolved_at = now(), resolution = CAST(:resolution AS jsonb)
WHERE model_name = :m AND status IN ('recommended', 'retraining')
RETURNING model_version, status, reason, alert_count, drifted_features, first_alert_at,
          last_alert_at, resolved_at, resolution
"""


@router.post(
    "/recommendations/{model_name}/dismiss",
    response_model=RetrainingRecommendation,
    responses={404: {"description": "no open retraining recommendation for the model"}},
)
async def dismiss_recommendation(
    model_name: str, body: DismissRequest, request: Request
) -> RetrainingRecommendation:
    """Close a model's open "retraining recommended" record without retraining (status
    "dismissed", with the reason in ``resolution``). A later drift alert opens a new one."""
    resolution = {"decision": "dismissed", "reason": body.reason, "by": body.dismissed_by}
    async with request.app.state.engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text(_DISMISS_SQL), {"m": model_name, "resolution": json.dumps(resolution)}
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(404, f"no open retraining recommendation for {model_name!r}")
    return RetrainingRecommendation(**row)


def _mode_info(state: DriftModeState) -> DriftModeInfo:
    return DriftModeInfo(
        active=state.active(),
        enabled=state.enabled,
        settings=state.settings.model_dump(),
        visual_params=state.settings.visual_params(),
        expires_at=state.expires_at,
        updated_by=state.updated_by,
        updated_at=state.updated_at,
    )


@router.get("/mode", response_model=DriftModeInfo)
async def get_drift_mode(conn: AsyncConnection = Depends(connection)) -> DriftModeInfo:
    """Whether drift mode is on (``active``: enabled and not expired) and its settings."""
    return _mode_info(state_from_row((await conn.execute(text(SELECT_SQL))).mappings().first()))


@router.post("/mode", response_model=DriftModeInfo)
async def set_drift_mode(body: DriftModeRequest, request: Request) -> DriftModeInfo:
    """Switch drift mode on (night or fog frames in video_ingest, slang reports in
    event_simulator) or off. Services pick the change up within DRIFT_MODE_POLL_SECONDS.
    ``duration_minutes`` switches it off by itself."""
    settings = DriftModeSettings(
        **body.model_dump(exclude={"enabled", "duration_minutes"}, exclude_none=True)
    )
    params = write_params(body.enabled, settings, body.duration_minutes, "api")
    async with request.app.state.engine.begin() as conn:
        row = (await conn.execute(text(UPSERT_SQL), params)).mappings().first()
        await conn.execute(text(EVENT_SQL), params)
    return _mode_info(state_from_row(row))
