"""Cameras: status list and detail (metrics history aggregated in SQL, recent anomalies)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service import records
from api_service.db import connection
from api_service.responses import CameraDetail, CameraStatus, Congestion, MetricsBucket, Page
from api_service.routers.common import Paging, not_found, paging

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])

CAMERAS_SQL = """
SELECT c.id, c.name, c.location_name, c.latitude, c.longitude, c.zone_config,
       cc.level, cc.confidence, cc.window_end AS congestion_window_end,
       (SELECT count(*) FROM incidents i
         WHERE i.camera_id = c.id AND i.status <> 'resolved') AS open_incidents,
       EXISTS (SELECT 1 FROM incidents i WHERE i.camera_id = c.id AND i.status <> 'resolved'
               AND i.vision_degraded) AS incident_degraded,
       EXISTS (SELECT 1 FROM anomalies a WHERE a.status = 'active' AND a.scope = 'system'
               AND a.affected_cameras @> jsonb_build_array(c.id)) AS system_degraded,
       EXISTS (SELECT 1 FROM anomalies a WHERE a.status = 'active' AND a.scope = 'camera'
               AND a.camera_id = c.id AND a.hints @> '["camera_degraded"]') AS camera_degraded,
       m.*
FROM cameras c
LEFT JOIN camera_congestion cc ON cc.camera_id = c.id
LEFT JOIN LATERAL (
    SELECT t.message_id, t.camera_id AS m_camera_id, t.window_start, t.window_end,
           t.vehicle_count, t.avg_speed_kmh, t.p85_speed_kmh, t.occupancy, t.density,
           t.stopped_vehicles, t.pedestrians_in_roadway, t.mean_confidence, t.frames,
           t.model_version, t.counts_by_class
    FROM traffic_metrics t WHERE t.camera_id = c.id ORDER BY t.window_end DESC LIMIT 1
) m ON true
{where}
ORDER BY c.id
"""


def camera_status(row) -> CameraStatus:
    m = row._mapping
    reasons = [
        name
        for name, flag in (
            ("system-wide degradation", m["system_degraded"]),
            ("camera reports itself degraded", m["camera_degraded"]),
            ("open incident with degraded vision", m["incident_degraded"]),
        )
        if flag
    ]
    latest = None
    if m["window_end"] is not None:
        latest = records.metrics(_Row({**m, "camera_id": m["m_camera_id"]}))
    return CameraStatus(
        id=m["id"],
        name=m["name"],
        location_name=m["location_name"],
        latitude=m["latitude"],
        longitude=m["longitude"],
        congestion=Congestion(
            level=m["level"], confidence=m["confidence"], window_end=m["congestion_window_end"]
        )
        if m["level"]
        else None,
        latest_metrics=latest,
        open_incidents=m["open_incidents"],
        vision_degraded=bool(reasons),
        vision_degraded_reasons=reasons,
        frame_url=f"/api/v1/frames/{m['id']}/latest.jpg",
    )


class _Row:
    """Adapter so records.metrics can read a plain dict."""

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping


@router.get("", response_model=Page[CameraStatus])
async def list_cameras(
    conn: AsyncConnection = Depends(connection), p: Paging = Depends(paging)
) -> Page[CameraStatus]:
    total = await conn.scalar(text("SELECT count(*) FROM cameras"))
    rows = await conn.execute(
        text(CAMERAS_SQL.format(where="") + " LIMIT :limit OFFSET :offset"),
        {"limit": p.page_size, "offset": p.offset},
    )
    return Page(
        items=[camera_status(r) for r in rows], total=total, page=p.page, page_size=p.page_size
    )


METRICS_SQL = """
SELECT date_bin(make_interval(secs => :res), window_end, TIMESTAMPTZ '2000-01-01') AS bucket_start,
       count(*) AS windows,
       avg(vehicle_count)::float AS vehicle_count_mean,
       avg(avg_speed_kmh)::float AS avg_speed_kmh,
       avg(p85_speed_kmh)::float AS p85_speed_kmh,
       avg(density)::float AS density_mean,
       max(stopped_vehicles) AS stopped_vehicles_max,
       max(pedestrians_in_roadway) AS pedestrians_in_roadway_max,
       avg(mean_confidence)::float AS mean_confidence,
       avg(frames)::float AS frames_mean
FROM traffic_metrics
WHERE camera_id = :c AND window_end > now() - make_interval(mins => :minutes)
GROUP BY 1 ORDER BY 1
"""

ANOMALIES_SQL = """
SELECT * FROM anomalies
WHERE (camera_id = :c OR (scope = 'system' AND affected_cameras @> jsonb_build_array(:c)))
  AND (detected_at > now() - make_interval(mins => :minutes) OR status = 'active')
ORDER BY detected_at DESC LIMIT :n
"""


@router.get("/{camera_id}", response_model=CameraDetail)
async def camera_detail(
    camera_id: str,
    minutes: int = Query(60, ge=1, le=60 * 24 * 7, description="how far back"),
    resolution: int = Query(60, ge=5, le=3600, description="bucket size in seconds"),
    anomalies: int = Query(50, ge=0, le=500, description="max recent anomalies"),
    conn: AsyncConnection = Depends(connection),
) -> CameraDetail:
    rows = (
        await conn.execute(text(CAMERAS_SQL.format(where="WHERE c.id = :c")), {"c": camera_id})
    ).all()
    if not rows:
        raise not_found(f"camera {camera_id}")
    # Keep the response bounded: at most ~2000 buckets.
    resolution = max(resolution, (minutes * 60) // 2000)
    params = {"c": camera_id, "minutes": minutes, "res": resolution}
    buckets = (await conn.execute(text(METRICS_SQL), params)).mappings().all()
    found = await conn.execute(text(ANOMALIES_SQL), {**params, "n": anomalies})
    return CameraDetail(
        camera=camera_status(rows[0]),
        zone_config=rows[0]._mapping["zone_config"] or {},
        range_minutes=minutes,
        resolution_seconds=resolution,
        metrics=[MetricsBucket(**b) for b in buckets],
        anomalies=[a for a in (records.anomaly(r) for r in found) if a is not None],
    )
