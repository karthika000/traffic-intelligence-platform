"""Incidents: filtered list (newest first) and full detail with evidence and summaries."""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service import records
from api_service.db import connection
from api_service.responses import Evidence, IncidentDetail, Page, ReportRecord
from api_service.routers.common import Paging, not_found, paging, time_range
from traffic_common.enums import IncidentStatus, IncidentType, Severity
from traffic_common.schemas import Incident

router = APIRouter(prefix="/api/v1/incidents", tags=["incidents"])


@router.get("", response_model=Page[Incident])
async def list_incidents(
    status: IncidentStatus | None = None,
    camera_id: str | None = None,
    incident_type: IncidentType | None = None,
    severity: Severity | None = None,
    since: datetime | None = Query(None, description="suspected at or after (ISO 8601)"),
    until: datetime | None = Query(None, description="suspected before (ISO 8601)"),
    conn: AsyncConnection = Depends(connection),
    p: Paging = Depends(paging),
) -> Page[Incident]:
    since, until = time_range(since, until)
    where, params = [], {}
    for column, value in (
        ("status", status),
        ("camera_id", camera_id),
        ("incident_type", incident_type),
        ("severity", severity),
    ):
        if value is not None:
            where.append(f"{column} = :{column}")
            params[column] = str(value)
    if since:
        where.append("coalesce(suspected_at, start_time) >= :since")
        params["since"] = since
    if until:
        where.append("coalesce(suspected_at, start_time) < :until")
        params["until"] = until
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = await conn.scalar(text(f"SELECT count(*) FROM incidents {clause}"), params)
    rows = await conn.execute(
        text(
            f"SELECT * FROM incidents {clause} "
            "ORDER BY coalesce(suspected_at, start_time) DESC, id LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": p.page_size, "offset": p.offset},
    )
    return Page(
        items=[records.incident(r) for r in rows], total=total, page=p.page, page_size=p.page_size
    )


@router.get("/{incident_id}", response_model=IncidentDetail)
async def incident_detail(
    incident_id: UUID, conn: AsyncConnection = Depends(connection)
) -> IncidentDetail:
    row = (
        await conn.execute(text("SELECT * FROM incidents WHERE id = :id"), {"id": incident_id})
    ).first()
    if row is None:
        raise not_found(f"incident {incident_id}")
    details = row._mapping["details"] or {}
    state = details.get("state") or {}
    # Evidence ids come from the incident's own state (complete even while a link row is still
    # waiting for its target); weights from incident_evidence.
    anomaly_ids = list(state.get("anomalies", {}))
    report_ids = list(state.get("reports", {}))
    metric_ids = state.get("metric_ids", [])
    weights = {
        f"{r.evidence_type}:{r.target}": r.weight
        for r in await conn.execute(
            text(
                "SELECT evidence_type, weight, coalesce(anomaly_id::text, report_id::text, "
                "traffic_metric_id::text) AS target FROM incident_evidence WHERE incident_id = :id"
            ),
            {"id": incident_id},
        )
    }
    anomalies = await conn.execute(
        text("SELECT * FROM anomalies WHERE id = ANY(CAST(:ids AS uuid[])) ORDER BY detected_at"),
        {"ids": anomaly_ids},
    )
    reports = await conn.execute(
        text(
            f"SELECT {records.REPORT_SELECT} FROM reports r {records.LATEST_NLP_JOIN} "
            "WHERE r.id = ANY(CAST(:ids AS uuid[])) ORDER BY r.reported_at"
        ),
        {"ids": report_ids},
    )
    metrics = await conn.execute(
        text(
            "SELECT * FROM traffic_metrics WHERE message_id = ANY(CAST(:ids AS uuid[])) "
            "ORDER BY window_end"
        ),
        {"ids": metric_ids},
    )
    summaries = await conn.execute(
        text("SELECT * FROM summaries WHERE incident_id = :id ORDER BY created_at"),
        {"id": incident_id},
    )
    return IncidentDetail(
        incident=records.incident(row),
        resolution=details.get("resolution"),
        at_confirmation=details.get("at_confirmation"),
        versions=details.get("history") or [],
        evidence=Evidence(
            anomalies=[a for a in (records.anomaly(r) for r in anomalies) if a is not None],
            reports=[
                ReportRecord(
                    report=records.report(r),
                    nlp=(n := records.nlp(r)),
                    unlocated=bool(n and n.camera_id is None),
                )
                for r in reports
            ],
            metrics=[records.metrics(r) for r in metrics],
            weights=weights,
        ),
        summaries=[records.summary(r) for r in summaries],
    )
