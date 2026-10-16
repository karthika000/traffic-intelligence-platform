"""Per-camera rolling state: recent metric windows joined with loop readings, and the
alarm tracker. The join mirrors ml.features.attach_sensor_readings (latest reading at or
before the window end, within the tolerance), so serving sees the same rows as training."""

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ml.alarm import AlarmTracker
from ml.features import (
    BASELINE_MIN_WINDOWS,
    BASELINE_PREFIX,
    BASELINE_WINDOWS,
    METRIC_COLUMNS,
    RAW_COLUMNS,
    SENSOR_COLUMNS,
    SENSOR_TOLERANCE_S,
    SEQUENCE_LENGTH,
)
from traffic_common.schemas import RawEvent, TrafficMetrics

HEALTH_COLUMNS = ("frames", "mean_confidence")  # what a pipeline problem shows up in
HEALTH_RECENT_WINDOWS = 3


@dataclass
class SensorPoint:
    at: datetime
    values: dict[str, float | None]


def sensor_point(event: RawEvent) -> SensorPoint | None:
    if event.event_type != "loop_detector":
        return None
    p = event.payload
    return SensorPoint(
        event.timestamp,
        {
            "loop_vehicle_count": p.get("vehicle_count"),
            "loop_speed_kmh": p.get("avg_speed_kmh"),
            "loop_occupancy": p.get("occupancy"),
        },
    )


@dataclass
class ActiveAnomaly:
    anomaly_id: uuid.UUID
    started_at: datetime
    anomaly_type: str
    hints: list[str]
    metric_name: str
    observed: float
    expected: float | None
    contributions: dict[str, float]
    congestion_level: str | None
    severity: str = "low"
    fired_by: str | None = None
    score_parts: dict[str, float] = field(default_factory=dict)
    seen_hints: set[str] = field(default_factory=set)
    last_published: datetime | None = None
    updates: int = 0
    hold_reasons: list[str] = field(default_factory=list)
    threshold: float = 0.0  # latest, for ending the alert from outside the scoring path
    versions: dict[str, str] = field(default_factory=dict)


@dataclass
class Health:
    """A camera's pipeline health at ``ts``: recent frames / detection confidence relative to
    its baseline (None without enough history), and which of them dropped."""

    ts: datetime
    ratios: dict[str, float | None]
    dropped: list[str]


@dataclass
class SystemEpisode:
    """An active system-wide degradation (one scope="system" anomaly)."""

    anomaly_id: uuid.UUID
    started_at: datetime
    cameras: set[str]  # every camera affected so far
    ratios: dict[str, dict[str, float | None]]  # latest health ratios per affected camera
    metric_name: str
    severity: str
    peak_cameras: int
    linked: dict[str, uuid.UUID] = field(default_factory=dict)  # camera -> traffic alert id
    suppressed_checks: int = 0
    clear_checks: int = 0
    last_published: datetime | None = None
    updates: int = 0


@dataclass
class CameraState:
    camera_id: str
    history_windows: int
    tracker: AlarmTracker
    rows: deque = field(init=False)
    sensors: deque = field(default_factory=lambda: deque(maxlen=20))
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_scored: datetime | None = None
    congestion_level: str | None = None
    active: ActiveAnomaly | None = None
    # Baselines frozen when the active alert started (ml.features.current_baselines); passed to
    # the model as baseline_<col> columns and used by the hold rule until the alert ends.
    frozen_baseline: dict[str, float | None] | None = None
    health: Health | None = None

    def __post_init__(self) -> None:
        self.rows = deque(maxlen=self.history_windows)

    def add_sensor(self, point: SensorPoint) -> None:
        self.sensors.append(point)

    def sensor_at(self, ts: datetime) -> dict[str, float | None]:
        """Latest reading at or before ``ts`` within the tolerance, else all-missing."""
        best = None
        for p in self.sensors:
            if p.at <= ts and ts - p.at <= timedelta(seconds=SENSOR_TOLERANCE_S):
                if best is None or p.at > best.at:
                    best = p
        return dict(best.values) if best else dict.fromkeys(SENSOR_COLUMNS)

    def add_metrics(self, m: TrafficMetrics) -> None:
        row = {"camera_id": m.camera_id, "ts": m.window_end}
        row.update({c: getattr(m, c) for c in METRIC_COLUMNS})
        row.update(self.sensor_at(m.window_end))
        if self.rows and self.rows[-1]["ts"] >= m.window_end:
            return  # duplicate or out of order
        self.rows.append(row)

    @property
    def latest_ts(self) -> datetime | None:
        return self.rows[-1]["ts"] if self.rows else None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows), columns=RAW_COLUMNS)

    def model_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` plus baseline_<col> columns holding the frozen baseline for the windows
        since the active alert started (earlier windows keep their rolling baseline)."""
        if self.frozen_baseline is None or self.active is None:
            return frame
        out = frame.copy()
        since = out["ts"] >= self.active.started_at
        for col, value in self.frozen_baseline.items():
            if value is not None:
                out[f"{BASELINE_PREFIX}{col}"] = None
                out.loc[since, f"{BASELINE_PREFIX}{col}"] = value
        return out

    def health_check(self, frames_ratio: float, confidence_ratio: float) -> Health | None:
        """Mean of the last few windows of frames / mean_confidence against the camera's
        baseline (median of the windows BASELINE_WINDOWS..SEQUENCE_LENGTH back, like the
        anomaly features). A sustained change becomes the new normal after ~5 minutes."""
        if not self.rows:
            return None
        frame = pd.DataFrame(list(self.rows)[-(BASELINE_WINDOWS + SEQUENCE_LENGTH) :])
        recent, history = frame.tail(HEALTH_RECENT_WINDOWS), frame.iloc[:-SEQUENCE_LENGTH]
        limits = {"frames": frames_ratio, "mean_confidence": confidence_ratio}
        ratios: dict[str, float | None] = {}
        dropped = []
        for col in HEALTH_COLUMNS:
            base = pd.to_numeric(history[col], errors="coerce").dropna()
            now = pd.to_numeric(recent[col], errors="coerce").mean()
            if len(base) < BASELINE_MIN_WINDOWS or not base.median() > 0 or pd.isna(now):
                ratios[col] = None
                continue
            ratios[col] = round(float(now / base.median()), 3)
            if ratios[col] < limits[col]:
                dropped.append(col)
        return Health(self.rows[-1]["ts"], ratios, dropped)
