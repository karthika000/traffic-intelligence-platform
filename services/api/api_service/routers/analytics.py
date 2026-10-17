"""Aggregates for charts, computed in SQL."""

from collections import defaultdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service.db import connection
from api_service.responses import CongestionShare, HourlyVolume, IncidentCount
from traffic_common.enums import CongestionLevel

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
MAX_HOURS = 24 * 31


@router.get("/volume", response_model=list[HourlyVolume])
async def hourly_volume(
    hours: int = Query(24, ge=1, le=MAX_HOURS),
    camera_id: str | None = None,
    conn: AsyncConnection = Depends(connection),
) -> list[HourlyVolume]:
    """Vehicles (sum of per-window unique tracks) and mean speed per camera and hour."""
    rows = await conn.execute(
        text(
            "SELECT date_trunc('hour', window_end) AS hour, camera_id, "
            "sum(vehicle_count)::bigint AS vehicles, count(*) AS windows, "
            "avg(avg_speed_kmh)::float AS avg_speed_kmh FROM traffic_metrics "
            "WHERE window_end > now() - make_interval(hours => :h) "
            "AND (CAST(:c AS text) IS NULL OR camera_id = :c) GROUP BY 1, 2 ORDER BY 1, 2"
        ),
        {"h": hours, "c": camera_id},
    )
    return [HourlyVolume(**r) for r in rows.mappings()]


@router.get("/congestion", response_model=list[CongestionShare])
async def congestion_distribution(
    hours: int = Query(24, ge=1, le=MAX_HOURS),
    bucket_minutes: int = Query(60, ge=5, le=24 * 60),
    camera_id: str | None = None,
    conn: AsyncConnection = Depends(connection),
) -> list[CongestionShare]:
    """How the congestion classifier's levels were distributed per camera and time bucket."""
    rows = await conn.execute(
        text(
            "SELECT date_bin(make_interval(mins => :b), window_end, TIMESTAMPTZ '2000-01-01') "
            "AS bucket_start, camera_id, level, count(*) AS n FROM congestion_history "
            "WHERE window_end > now() - make_interval(hours => :h) "
            "AND (CAST(:c AS text) IS NULL OR camera_id = :c) GROUP BY 1, 2, 3 ORDER BY 1, 2"
        ),
        {"h": hours, "b": bucket_minutes, "c": camera_id},
    )
    buckets: dict[tuple, dict[str, int]] = defaultdict(dict)
    for r in rows:
        buckets[(r.bucket_start, r.camera_id)][r.level] = r.n
    out = []
    for (start, cam), counts in buckets.items():
        total = sum(counts.values())
        full = {level.value: counts.get(level.value, 0) for level in CongestionLevel}
        out.append(
            CongestionShare(
                bucket_start=start,
                camera_id=cam,
                counts=full,
                shares={k: round(v / total, 4) for k, v in full.items()},
            )
        )
    return out


@router.get("/incidents", response_model=list[IncidentCount])
async def incident_counts(
    hours: int = Query(24 * 7, ge=1, le=MAX_HOURS),
    camera_id: str | None = None,
    conn: AsyncConnection = Depends(connection),
) -> list[IncidentCount]:
    rows = await conn.execute(
        text(
            "SELECT incident_type, severity, status, count(*) AS count FROM incidents "
            "WHERE coalesce(suspected_at, start_time) > now() - make_interval(hours => :h) "
            "AND (CAST(:c AS text) IS NULL OR camera_id = :c) GROUP BY 1, 2, 3 "
            "ORDER BY count DESC"
        ),
        {"h": hours, "c": camera_id},
    )
    return [IncidentCount(**r) for r in rows.mappings()]
