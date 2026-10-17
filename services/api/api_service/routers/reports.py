"""Reports with their (latest) NLP result, including unlocated ones."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service import records
from api_service.db import connection
from api_service.responses import Page, ReportRecord
from api_service.routers.common import Paging, paging, time_range
from traffic_common.enums import IncidentType

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("", response_model=Page[ReportRecord])
async def list_reports(
    camera_id: str | None = Query(None, description="camera the report was matched to"),
    incident_type: IncidentType | None = None,
    unlocated: bool | None = Query(None, description="true: only reports matched to no camera"),
    since: datetime | None = None,
    until: datetime | None = None,
    conn: AsyncConnection = Depends(connection),
    p: Paging = Depends(paging),
) -> Page[ReportRecord]:
    since, until = time_range(since, until, default_hours=24)
    where, params = ["r.reported_at >= :since"], {"since": since}
    if until:
        where.append("r.reported_at < :until")
        params["until"] = until
    if camera_id is not None:
        where.append("n.camera_id = :camera_id")
        params["camera_id"] = camera_id
    if incident_type is not None:
        where.append("n.incident_type = :incident_type")
        params["incident_type"] = str(incident_type)
    if unlocated is True:
        where.append("n.report_id IS NOT NULL AND n.camera_id IS NULL")
    elif unlocated is False:
        where.append("n.camera_id IS NOT NULL")
    clause = " AND ".join(where)
    base = f"FROM reports r {records.LATEST_NLP_JOIN} WHERE {clause}"
    total = await conn.scalar(text(f"SELECT count(*) {base}"), params)
    rows = await conn.execute(
        text(
            f"SELECT {records.REPORT_SELECT} {base} "
            "ORDER BY r.reported_at DESC, r.id LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": p.page_size, "offset": p.offset},
    )
    items = []
    for r in rows:
        n = records.nlp(r)
        items.append(
            ReportRecord(report=records.report(r), nlp=n, unlocated=bool(n and n.camera_id is None))
        )
    return Page(items=items, total=total, page=p.page, page_size=p.page_size)
