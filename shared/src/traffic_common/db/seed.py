"""Demo data: four cameras at busy London junctions. Idempotent (upserts by camera id).

Re-running the seed only refreshes descriptive fields (name, location, coordinates).
``stream_url`` and ``zone_config`` are set once on insert and then owned by other tools
(scripts/download_sample_videos.py and scripts/write_zone_configs.py), so a re-seed never
undoes them.
"""

from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from traffic_common.db.models import Camera

DEMO_CAMERAS: list[dict[str, Any]] = [
    {
        "id": "cam-hyde-park-corner",
        "name": "Hyde Park Corner - Piccadilly approach",
        "location_name": "Hyde Park Corner, London W1J",
        "latitude": 51.502700,
        "longitude": -0.152800,
        "stream_url": "/data/videos/cam-hyde-park-corner.webm",
    },
    {
        "id": "cam-trafalgar-square",
        "name": "Trafalgar Square - Strand junction",
        "location_name": "Trafalgar Square, London WC2N",
        "latitude": 51.508000,
        "longitude": -0.128100,
        "stream_url": "/data/videos/cam-trafalgar-square.webm",
    },
    {
        "id": "cam-elephant-castle",
        "name": "Elephant & Castle - Northern Roundabout",
        "location_name": "Elephant & Castle, London SE1",
        "latitude": 51.494900,
        "longitude": -0.100500,
        "stream_url": "/data/videos/cam-elephant-castle.webm",
    },
    {
        "id": "cam-old-street",
        "name": "Old Street Roundabout - City Road",
        "location_name": "Old Street, London EC1V",
        "latitude": 51.525700,
        "longitude": -0.087700,
        "stream_url": "/data/videos/cam-old-street.webm",
    },
]

_REFRESHED_ON_RESEED = ("name", "location_name", "latitude", "longitude")


def seed_cameras(engine: Engine) -> int:
    """Insert missing demo cameras and refresh their descriptive fields.

    Returns the number of demo cameras now present.
    """
    stmt = insert(Camera).values(DEMO_CAMERAS)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Camera.id],
        set_={col: stmt.excluded[col] for col in _REFRESHED_ON_RESEED} | {"updated_at": func.now()},
    )
    with engine.begin() as conn:
        conn.execute(stmt)
        ids = [c["id"] for c in DEMO_CAMERAS]
        return conn.scalar(select(func.count()).select_from(Camera).where(Camera.id.in_(ids))) or 0
