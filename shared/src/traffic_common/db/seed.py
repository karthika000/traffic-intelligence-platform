"""Demo data: four cameras at busy London junctions. Idempotent (upserts by camera id).

Zone polygons and count lines use normalised [0, 1] image coordinates.
"""

from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from traffic_common.db.models import Camera


def _zone_config(
    zones: list[tuple[str, str, list[list[float]]]], count_line: list[list[float]]
) -> dict[str, Any]:
    return {
        "frame_size": [1920, 1080],
        "zones": [
            {"name": name, "direction": direction, "polygon": polygon}
            for name, direction, polygon in zones
        ],
        "count_lines": [{"name": "main", "points": count_line}],
        "speed_calibration_m_per_px": 0.045,
    }


DEMO_CAMERAS: list[dict[str, Any]] = [
    {
        "id": "cam-hyde-park-corner",
        "name": "Hyde Park Corner - Piccadilly approach",
        "location_name": "Hyde Park Corner, London W1J",
        "latitude": 51.502700,
        "longitude": -0.152800,
        "stream_url": "file:///data/videos/cam-hyde-park-corner.mp4",
        "zone_config": _zone_config(
            [
                (
                    "piccadilly_westbound",
                    "W",
                    [[0.05, 0.55], [0.48, 0.45], [0.50, 0.95], [0.02, 0.98]],
                ),
                (
                    "knightsbridge_eastbound",
                    "E",
                    [[0.52, 0.45], [0.95, 0.52], [0.98, 0.98], [0.52, 0.95]],
                ),
            ],
            [[0.05, 0.70], [0.95, 0.70]],
        ),
    },
    {
        "id": "cam-trafalgar-square",
        "name": "Trafalgar Square - Strand junction",
        "location_name": "Trafalgar Square, London WC2N",
        "latitude": 51.508000,
        "longitude": -0.128100,
        "stream_url": "file:///data/videos/cam-trafalgar-square.mp4",
        "zone_config": _zone_config(
            [
                ("strand_inbound", "SW", [[0.10, 0.40], [0.55, 0.38], [0.60, 0.90], [0.05, 0.92]]),
                (
                    "whitehall_southbound",
                    "S",
                    [[0.60, 0.40], [0.90, 0.42], [0.95, 0.95], [0.62, 0.95]],
                ),
            ],
            [[0.08, 0.65], [0.92, 0.65]],
        ),
    },
    {
        "id": "cam-elephant-castle",
        "name": "Elephant & Castle - Northern Roundabout",
        "location_name": "Elephant & Castle, London SE1",
        "latitude": 51.494900,
        "longitude": -0.100500,
        "stream_url": "file:///data/videos/cam-elephant-castle.mp4",
        "zone_config": _zone_config(
            [
                (
                    "newington_causeway_north",
                    "N",
                    [[0.20, 0.35], [0.50, 0.33], [0.55, 0.95], [0.12, 0.95]],
                ),
                (
                    "london_road_south",
                    "S",
                    [[0.52, 0.33], [0.82, 0.36], [0.92, 0.95], [0.57, 0.95]],
                ),
            ],
            [[0.15, 0.60], [0.88, 0.60]],
        ),
    },
    {
        "id": "cam-old-street",
        "name": "Old Street Roundabout - City Road",
        "location_name": "Old Street, London EC1V",
        "latitude": 51.525700,
        "longitude": -0.087700,
        "stream_url": "file:///data/videos/cam-old-street.mp4",
        "zone_config": _zone_config(
            [
                (
                    "city_road_northbound",
                    "N",
                    [[0.08, 0.50], [0.46, 0.42], [0.48, 0.96], [0.04, 0.96]],
                ),
                (
                    "old_street_eastbound",
                    "E",
                    [[0.50, 0.42], [0.92, 0.48], [0.96, 0.96], [0.50, 0.96]],
                ),
            ],
            [[0.06, 0.72], [0.94, 0.72]],
        ),
    },
]


def seed_cameras(engine: Engine) -> int:
    """Insert or update the demo cameras. Returns the number of demo cameras now present."""
    stmt = insert(Camera).values(DEMO_CAMERAS)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Camera.id],
        set_={
            col: stmt.excluded[col]
            for col in (
                "name",
                "location_name",
                "latitude",
                "longitude",
                "stream_url",
                "zone_config",
            )
        }
        | {"updated_at": func.now()},
    )
    with engine.begin() as conn:
        conn.execute(stmt)
        ids = [c["id"] for c in DEMO_CAMERAS]
        return conn.scalar(select(func.count()).select_from(Camera).where(Camera.id.in_(ids))) or 0
