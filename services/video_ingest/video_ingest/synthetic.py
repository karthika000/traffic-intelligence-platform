"""Synthetic traffic video: coloured 'vehicles' moving along a road, for tests and offline demos."""

from pathlib import Path

import cv2
import numpy as np

_ROAD = (70, 70, 70)
_VERGE = (60, 110, 60)
_MARKING = (230, 230, 230)
_VEHICLE_COLOURS = [(40, 40, 220), (220, 120, 30), (40, 200, 230), (200, 200, 200), (160, 60, 160)]


def _background(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), _VERGE, dtype=np.uint8)
    top, bottom = int(height * 0.25), int(height * 0.85)
    frame[top:bottom] = _ROAD
    lane_height = (bottom - top) / 4
    for i in range(1, 4):
        y = int(top + i * lane_height)
        thickness = 4 if i == 2 else 2  # centre line
        for x in range(0, width, 60):
            cv2.line(frame, (x, y), (x + 35 if i != 2 else x + 60, y), _MARKING, thickness)
    return frame


def generate_traffic_video(
    path: Path,
    *,
    seconds: float = 20.0,
    fps: float = 25.0,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
) -> Path:
    """Write an MP4 (mp4v) of vehicles moving in four lanes, two in each direction."""
    rng = np.random.default_rng(seed)
    background = _background(width, height)
    top, bottom = int(height * 0.25), int(height * 0.85)
    lane_height = (bottom - top) / 4

    vehicles = []
    for lane in range(4):
        direction = 1 if lane >= 2 else -1
        for _ in range(3):
            length = int(rng.integers(70, 160))
            vehicles.append({
                "lane": lane,
                "x": float(rng.uniform(0, width)),
                "speed": direction * float(rng.uniform(120, 320)),  # pixels per second
                "length": length,
                "colour": _VEHICLE_COLOURS[int(rng.integers(len(_VEHICLE_COLOURS)))],
            })  # fmt: skip

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for i in range(int(seconds * fps)):
            frame = background.copy()
            for v in vehicles:
                v["x"] = (v["x"] + v["speed"] / fps) % (width + v["length"])
                x = int(v["x"]) - v["length"]
                y_mid = int(top + (v["lane"] + 0.5) * lane_height)
                half = int(lane_height * 0.3)
                cv2.rectangle(frame, (x, y_mid - half), (x + v["length"], y_mid + half),
                              v["colour"], thickness=-1)
            cv2.putText(frame, f"SYNTHETIC  t={i / fps:5.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
    return path
