"""Write default zone configs (roadway polygon, scale, anomaly settings) for the 4 demo cameras.

The polygons were traced on the sample videos from scripts/download_sample_videos.py
(not the London junctions the cameras are named after). Scales come from median detected
car sizes: about 4.5 m long for side-on traffic, 1.8 m wide for oncoming traffic, with a
smaller vertical scale where vehicles move towards the camera (depth is foreshortened).

Usage (repo root, .venv active, stack up):
    python scripts/write_zone_configs.py                 # write all 4
    python scripts/write_zone_configs.py --only-missing  # keep existing roadways, add missing keys
    python scripts/write_zone_configs.py --dry-run
    python scripts/write_zone_configs.py --preview data/zone_previews   # draw them on frames

cv_inference picks up changes within CAMERA_REFRESH_SECONDS (default 60), no restart needed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "cv_inference"))

from sqlalchemy import select, update  # noqa: E402

from cv_inference.zones import ZoneConfig  # noqa: E402
from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.models import Camera  # noqa: E402

REFERENCE_WIDTH = 854  # the 480p sample videos

DEMO_ZONE_CONFIGS: dict[str, dict] = {
    # Street-level view of a mini roundabout; traffic crosses left-right just above mid-frame.
    "cam-hyde-park-corner": {
        "roadway": [[0.0, 0.40], [0.45, 0.42], [1.0, 0.47], [1.0, 0.96], [0.0, 0.96]],
        "pixels_per_metre": 30.0,  # between the far parked cars and the near lane
        "reference_width": REFERENCE_WIDTH,
    },
    # Wide signalised intersection; vehicles are small and cross the far side.
    "cam-trafalgar-square": {
        "roadway": [[0.18, 0.55], [0.6, 0.52], [1.0, 0.51], [1.0, 1.0], [0.0, 1.0], [0.0, 0.62]],
        "pixels_per_metre": 12.0,
        "reference_width": REFERENCE_WIDTH,
    },
    # Cobbled square with slow vans in the lower third; the left foreground is pavement.
    "cam-elephant-castle": {
        "roadway": [[0.15, 0.80], [0.55, 0.79], [1.0, 0.83], [1.0, 1.0], [0.35, 1.0], [0.15, 0.90]],
        "pixels_per_metre": 40.0,  # ~5.4 m vans span ~220 px
        "reference_width": REFERENCE_WIDTH,
        # The footage pans along the street: the camera's motion makes parked vans "move" and
        # moving ones "stop", and the polygon covers pavement in part of the pan. Its speeds and
        # stopped vehicles are camera motion, not traffic, so anomaly_service ignores them here.
        "anomaly": {"ignore": ["camera_speed", "stopped_vehicles"]},
    },
    # Multi-lane road with oncoming traffic: wide horizontally, strongly foreshortened in depth.
    "cam-old-street": {
        "roadway": [
            [0.30, 0.50],
            [0.70, 0.50],
            [1.0, 0.56],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.68],
            [0.25, 0.60],
        ],
        "pixels_per_metre": 45.0,
        "pixels_per_metre_y": 8.0,
        "reference_width": REFERENCE_WIDTH,
    },
}


def preview(out_dir: Path) -> None:
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, raw in DEMO_ZONE_CONFIGS.items():
        video = ROOT / "data" / "videos" / f"{camera_id}.webm"
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  {camera_id}: no video at {video}, skipped")
            continue
        h, w = frame.shape[:2]
        zone = ZoneConfig.model_validate(raw).resolve(w, h, 30.0)
        pts = np.array(zone.polygon, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
        cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
        # 5 m scale bars at the bottom-left of the roadway
        x0, y0 = int(w * 0.05), int(h * 0.93)
        cv2.line(frame, (x0, y0), (x0 + int(5 * zone.ppm_x), y0), (0, 0, 255), 3)
        cv2.putText(frame, "5 m", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        path = out_dir / f"{camera_id}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"  {camera_id}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="keep a roadway polygon already set; only add the keys a config lacks",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preview", type=Path, metavar="DIR", help="draw the polygons on video frames"
    )
    parser.add_argument("--db-url", help="database URL (default: POSTGRES_URL)")
    args = parser.parse_args()

    for raw in DEMO_ZONE_CONFIGS.values():
        ZoneConfig.model_validate(raw)  # fail before touching the database

    if args.preview:
        print("Previews:")
        preview(args.preview)
        return 0

    engine = create_db_engine(args.db_url)
    try:
        with engine.begin() as conn:
            current = dict(conn.execute(select(Camera.id, Camera.zone_config)).all())
            for camera_id, config in DEMO_ZONE_CONFIGS.items():
                if camera_id not in current:
                    print(
                        f"  {camera_id:<22} not in the cameras table "
                        "(run: python -m traffic_common.db seed)"
                    )
                    continue
                stored = current[camera_id] or {}
                if args.only_missing and stored.get("roadway"):
                    missing = {k: v for k, v in config.items() if k not in stored}
                    if not missing:
                        print(f"  {camera_id:<22} kept existing config")
                        continue
                    config = {**stored, **missing}  # e.g. an "anomaly" section added later
                    print(f"  {camera_id:<22} kept existing roadway, adding {sorted(missing)}")
                action = "would write" if args.dry_run else "wrote"
                if not args.dry_run:
                    conn.execute(
                        update(Camera).where(Camera.id == camera_id).values(zone_config=config)
                    )
                print(f"  {camera_id:<22} {action} {json.dumps(config)}")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
