"""Annotated frames: boxes, track ids, speeds and the roadway polygon."""

import os
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from cv_inference.analytics import ObjectAnalysis
from cv_inference.zones import ResolvedZone

# The frame's clock is written in the zone the dashboard and summaries use ("10:17:44 BST").
DISPLAY_TZ = ZoneInfo(os.getenv("DISPLAY_TIMEZONE", "").strip() or "Europe/London")

_COLOURS = {  # BGR
    "car": (255, 170, 0),
    "truck": (255, 90, 160),
    "bus": (0, 200, 255),
    "motorcycle": (180, 255, 0),
    "bicycle": (0, 255, 200),
    "person": (60, 230, 60),
}
_ALERT = (0, 0, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(
    frame: np.ndarray,
    analyses: list[ObjectAnalysis],
    zone: ResolvedZone,
    header: str,
    captured_at: datetime,
) -> np.ndarray:
    out = frame.copy()
    polygon = np.array(zone.polygon, dtype=np.int32)
    if not zone.whole_frame:
        overlay = out.copy()
        cv2.fillPoly(overlay, [polygon], (0, 200, 255))
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, dst=out)
        cv2.polylines(out, [polygon], isClosed=True, color=(0, 200, 255), thickness=2)

    for a in analyses:
        x1, y1, x2, y2 = (int(v) for v in a.obj.bbox)
        alert = a.stopped or a.is_pedestrian_in_roadway
        colour = _ALERT if alert else _COLOURS.get(a.obj.label, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        parts = [a.obj.label]
        if a.obj.track_id is not None:
            parts.append(f"#{a.obj.track_id}")
        if a.speed_kmh is not None and a.obj.label != "person":
            parts.append(f"{a.speed_kmh:.0f} km/h")
        if a.stopped:
            parts.append("STOPPED")
        if a.is_pedestrian_in_roadway:
            parts.append("IN ROAD")
        elif a.rider:
            parts.append("rider")
        _label(out, " ".join(parts), x1, y1, colour)

    _label(
        out,
        f"{header}  {frame_clock(captured_at)}",
        4,
        18,
        (40, 40, 40),
        scale=0.5,
        text_colour=(255, 255, 255),
    )
    return out


def frame_clock(captured_at: datetime, tz: ZoneInfo = DISPLAY_TZ) -> str:
    """ "10:17:44 BST": capture time in the display zone (naive times are UTC)."""
    at = captured_at if captured_at.tzinfo else captured_at.replace(tzinfo=UTC)
    return at.astimezone(tz).strftime("%H:%M:%S %Z")


def _label(img, text, x, y, colour, scale=0.42, text_colour=(0, 0, 0)) -> None:
    (w, h), base = cv2.getTextSize(text, _FONT, scale, 1)
    y = max(y, h + base + 2)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y), colour, -1)
    cv2.putText(img, text, (x + 2, y - base - 1), _FONT, scale, text_colour, 1, cv2.LINE_AA)


def write_latest(directory: Path, camera_id: str, image: np.ndarray, quality: int = 80) -> Path:
    """Atomically replace <directory>/<camera_id>/latest.jpg (readers never see a partial file)."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    target = directory / camera_id / "latest.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".latest.{os.getpid()}.tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, target)
    return target
