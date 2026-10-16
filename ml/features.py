"""Feature engineering shared by training and serving.

The MLflow models take *raw rows* (one per camera per 5 s metrics window, see RAW_COLUMNS)
and compute features themselves with this module, so training and serving can't drift.

Raw row = one TrafficMetrics window (cv_inference) joined with the latest loop-detector
reading for the same camera at or before the window end (within SENSOR_TOLERANCE_S).
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

WINDOW_S = 5.0
SEQUENCE_LENGTH = 12  # 12 x 5 s = one minute of context
MAX_GAP_S = 7.5  # consecutive windows further apart than this start a new sequence
SENSOR_TOLERANCE_S = 30.0  # loop readings arrive every 15 s; older ones count as missing
ROLLING = 12

METRIC_COLUMNS = [
    "vehicle_count",
    "avg_speed_kmh",
    "p85_speed_kmh",
    "density",
    "stopped_vehicles",
    "pedestrians_in_roadway",
    "mean_confidence",
    "frames",
]
SENSOR_COLUMNS = ["loop_vehicle_count", "loop_speed_kmh", "loop_occupancy"]
RAW_COLUMNS = ["camera_id", "ts", *METRIC_COLUMNS, *SENSOR_COLUMNS]

# Short-term baseline for the anomaly features: the camera's rolling median over the
# previous BASELINE_WINDOWS windows, ending one sequence before the current window (so an
# unfolding incident doesn't drag its own baseline down). Needs BASELINE_MIN_WINDOWS.
BASELINE_WINDOWS = 120  # 10 minutes
BASELINE_MIN_WINDOWS = 24
RELATIVE_COLUMNS = {
    # column -> smoothing added to value and baseline (counts get +1 so 0/0 is "normal")
    "avg_speed_kmh": 0.0,
    "p85_speed_kmh": 0.0,
    "loop_speed_kmh": 0.0,
    "density": 1.0,
    "vehicle_count": 1.0,
    "loop_vehicle_count": 1.0,
    "loop_occupancy": 0.05,
    "frames": 1.0,
}
REL = "_rel"
HISTORY_WINDOWS_NEEDED = SEQUENCE_LENGTH + BASELINE_WINDOWS  # serving history for full baselines

# Inputs to the anomaly autoencoder (after per-camera normalisation). Levels are relative to
# the camera's own recent baseline, so the daily pattern and slow drift (or a short real
# history that doesn't cover every hour) don't look anomalous; sudden changes do.
AE_FEATURES = [
    *(f"{c}{REL}" for c in RELATIVE_COLUMNS),
    "stopped_vehicles",
    "pedestrians_in_roadway",
    "mean_confidence",
    "camera_speed_present",
    "sensor_present",
]

# Every column with a baseline. A raw input column "baseline_<col>" (e.g. a baseline the
# service froze when an alert started) overrides the rolling baseline wherever it is set.
BASELINE_COLUMNS = [*RELATIVE_COLUMNS, "stopped_vehicles", "pedestrians_in_roadway"]
BASELINE_PREFIX = "baseline_"
# Camera-health columns measure the pipeline (frames processed per window), not traffic. Their
# baseline always rolls (never frozen) and they get no level term, so a lasting change in
# processing throughput is at most a brief reconstruction blip, not a held traffic alert.
HEALTH_COLUMNS = ("frames",)
FREEZABLE_COLUMNS = [c for c in BASELINE_COLUMNS if c not in HEALTH_COLUMNS]
LEVEL_RELATIVE_COLUMNS = [c for c in RELATIVE_COLUMNS if c not in HEALTH_COLUMNS]

# Distance-from-normal ("level") features: |<col>_rel - 1| for the traffic relative columns,
# and how far stopped vehicles / pedestrians exceed their baseline.
LEVEL_FEATURES = [
    *(f"{c}{REL}" for c in LEVEL_RELATIVE_COLUMNS),
    "stopped_excess",
    "pedestrians_excess",
]


@dataclass(frozen=True)
class CameraReference:
    """Per-camera reference values learned from training data: medians fill missing
    values, free-flow speeds turn speeds into ratios, median density scales density."""

    medians: dict[str, float]
    free_flow_kmh: float
    loop_free_flow_kmh: float
    median_density: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraReference":
        return cls(**d)


GLOBAL = "__global__"
DEFAULT_REFERENCE = CameraReference(
    medians={
        "vehicle_count": 3.0,
        "avg_speed_kmh": 25.0,
        "p85_speed_kmh": 35.0,
        "density": 2.0,
        "stopped_vehicles": 0.0,
        "pedestrians_in_roadway": 0.0,
        "mean_confidence": 0.6,
        "frames": 24.0,
        "loop_vehicle_count": 4.0,
        "loop_speed_kmh": 35.0,
        "loop_occupancy": 0.15,
    },
    free_flow_kmh=40.0,
    loop_free_flow_kmh=48.0,
    median_density=2.0,
)


def compute_references(raw: pd.DataFrame) -> dict[str, CameraReference]:
    """Reference values per camera, plus GLOBAL for cameras never seen in training."""

    def ref(df: pd.DataFrame) -> CameraReference:
        medians = {
            c: float(df[c].median()) if df[c].notna().any() else DEFAULT_REFERENCE.medians[c]
            for c in METRIC_COLUMNS + SENSOR_COLUMNS
        }
        speed = df["avg_speed_kmh"].dropna()
        loop = df["loop_speed_kmh"].dropna()
        return CameraReference(
            medians=medians,
            free_flow_kmh=max(5.0, float(speed.quantile(0.95)))
            if len(speed)
            else DEFAULT_REFERENCE.free_flow_kmh,
            loop_free_flow_kmh=max(10.0, float(loop.quantile(0.95)))
            if len(loop)
            else DEFAULT_REFERENCE.loop_free_flow_kmh,
            median_density=max(0.5, medians["density"]),
        )

    refs = {str(cam): ref(df) for cam, df in raw.groupby("camera_id")}
    refs[GLOBAL] = ref(raw)
    return refs


def reference_for(refs: dict[str, CameraReference], camera_id: str) -> CameraReference:
    return refs.get(camera_id) or refs.get(GLOBAL) or DEFAULT_REFERENCE


def attach_sensor_readings(
    metrics: pd.DataFrame, readings: pd.DataFrame, tolerance_s: float = SENSOR_TOLERANCE_S
) -> pd.DataFrame:
    """Join each metrics window with the latest loop reading of the same camera at or before
    its end (``ts``), within ``tolerance_s``. ``readings`` has camera_id, reading_at,
    vehicle_count, avg_speed_kmh, occupancy (as stored in sensor_readings)."""
    left = metrics.sort_values("ts").copy()
    left["ts"] = pd.to_datetime(left["ts"], utc=True)
    if readings.empty:
        for c in SENSOR_COLUMNS:
            left[c] = np.nan
        return left.sort_values(["camera_id", "ts"]).reset_index(drop=True)
    right = readings.rename(
        columns={
            "vehicle_count": "loop_vehicle_count",
            "avg_speed_kmh": "loop_speed_kmh",
            "occupancy": "loop_occupancy",
            "reading_at": "ts",
        }
    )[["camera_id", "ts", *SENSOR_COLUMNS]].copy()
    right["ts"] = pd.to_datetime(right["ts"], utc=True)
    right = right.sort_values("ts")
    out = pd.merge_asof(
        left.drop(columns=[c for c in SENSOR_COLUMNS if c in left.columns]),
        right,
        on="ts",
        by="camera_id",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_s),
    )
    return out.sort_values(["camera_id", "ts"]).reset_index(drop=True)


def base_features(raw: pd.DataFrame, refs: dict[str, CameraReference]) -> pd.DataFrame:
    """Missing-value flags, filled values, ratios to free flow and time of day."""
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["camera_id", "ts"]).reset_index(drop=True)
    for c in METRIC_COLUMNS + SENSOR_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["camera_speed_present"] = df["avg_speed_kmh"].notna().astype(float)
    df["sensor_present"] = df["loop_speed_kmh"].notna().astype(float)

    cam_refs = {cam: reference_for(refs, cam) for cam in df["camera_id"].unique()}
    for c in METRIC_COLUMNS + SENSOR_COLUMNS:
        fill = df["camera_id"].map(lambda cam, c=c: cam_refs[cam].medians[c])
        df[c] = df[c].fillna(fill)
    free_flow = df["camera_id"].map(lambda cam: cam_refs[cam].free_flow_kmh)
    loop_free_flow = df["camera_id"].map(lambda cam: cam_refs[cam].loop_free_flow_kmh)
    median_density = df["camera_id"].map(lambda cam: cam_refs[cam].median_density)
    df["speed_ratio"] = (df["avg_speed_kmh"] / free_flow).clip(0, 2)
    df["loop_speed_ratio"] = (df["loop_speed_kmh"] / loop_free_flow).clip(0, 2)
    df["density_ratio"] = (df["density"] / median_density).clip(0, 10)
    hours = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return add_relative_features(df)


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """baseline_<col> (rolling, unless overridden by an input column of that name) and
    <col>_rel = (value + s) / (baseline + s); 1.0 (no change) while a camera has too little
    history for a baseline. Also stopped_excess / pedestrians_excess over their baselines."""
    grouped = df.groupby("camera_id", sort=False)
    for col in BASELINE_COLUMNS:
        rolling = grouped[col].transform(
            lambda s: (
                s.shift(SEQUENCE_LENGTH)
                .rolling(BASELINE_WINDOWS, min_periods=BASELINE_MIN_WINDOWS)
                .median()
            )
        )
        name = f"{BASELINE_PREFIX}{col}"
        if name in df.columns and col in FREEZABLE_COLUMNS:  # frozen baseline from the caller
            rolling = pd.to_numeric(df[name], errors="coerce").astype(float).combine_first(rolling)
        df[name] = rolling
    for col, smooth in RELATIVE_COLUMNS.items():
        baseline = df[f"{BASELINE_PREFIX}{col}"]
        rel = (df[col] + smooth) / (baseline + smooth)
        df[f"{col}{REL}"] = rel.replace([np.inf, -np.inf], np.nan).clip(0, 5).fillna(1.0)
    for col, name in (
        ("stopped_vehicles", "stopped_excess"),
        ("pedestrians_in_roadway", "pedestrians_excess"),
    ):
        df[name] = (df[col] - df[f"{BASELINE_PREFIX}{col}"].fillna(df[col])).clip(lower=0)
    return df


def level_distances(base: pd.DataFrame) -> np.ndarray:
    """Distance from normal per LEVEL_FEATURES column: [rows, len(LEVEL_FEATURES)]."""
    rel = np.abs(base[[f"{c}{REL}" for c in LEVEL_RELATIVE_COLUMNS]].to_numpy(float) - 1.0)
    excess = base[["stopped_excess", "pedestrians_excess"]].to_numpy(float)
    return np.nan_to_num(np.hstack([rel, excess]), nan=0.0).astype(np.float32)


def current_baselines(raw: pd.DataFrame) -> dict[str, float | None]:
    """The camera's baseline right now (median of the windows BASELINE_WINDOWS..SEQUENCE_LENGTH
    back, missing values skipped), from raw rows of one camera; None without enough history.
    What anomaly_service freezes when an alert starts."""
    window = raw.sort_values("ts").iloc[-(BASELINE_WINDOWS + SEQUENCE_LENGTH) : -SEQUENCE_LENGTH]
    out: dict[str, float | None] = {}
    for col in FREEZABLE_COLUMNS:
        values = pd.to_numeric(window[col], errors="coerce").dropna()
        out[col] = float(values.median()) if len(values) >= BASELINE_MIN_WINDOWS else None
    return out


_ROLLING_SPECS = {
    "speed_ratio": ("mean", "std", "min"),
    "loop_speed_ratio": ("mean", "std", "min"),
    "density_ratio": ("mean", "max"),
    "stopped_vehicles": ("mean", "max"),
    "pedestrians_in_roadway": ("max",),
    "vehicle_count": ("mean",),
    "sensor_present": ("mean",),
    "mean_confidence": ("mean",),
}
CONGESTION_FEATURES = [
    "speed_ratio",
    "loop_speed_ratio",
    "density_ratio",
    "stopped_vehicles",
    "vehicle_count",
    "loop_vehicle_count",
    "loop_occupancy",
    "mean_confidence",
    "camera_speed_present",
    "sensor_present",
    "hour_sin",
    "hour_cos",
    *[f"{col}_{agg}_{ROLLING}" for col, aggs in _ROLLING_SPECS.items() for agg in aggs],
    f"loop_speed_ratio_slope_{ROLLING}",
    f"speed_ratio_slope_{ROLLING}",
]


def congestion_features(base: pd.DataFrame) -> pd.DataFrame:
    """Current values plus rolling statistics over the last ROLLING windows per camera."""
    df = base.copy()
    grouped = df.groupby("camera_id", sort=False)
    for col, aggs in _ROLLING_SPECS.items():
        roll = grouped[col].rolling(ROLLING, min_periods=1)
        for agg in aggs:
            values = getattr(roll, agg)().reset_index(level=0, drop=True)
            df[f"{col}_{agg}_{ROLLING}"] = values.fillna(0.0) if agg == "std" else values
    for col in ("loop_speed_ratio", "speed_ratio"):
        first = grouped[col].shift(ROLLING - 1)
        df[f"{col}_slope_{ROLLING}"] = ((df[col] - first) / ROLLING).fillna(0.0)
    return df


def sequence_segments(base: pd.DataFrame) -> np.ndarray:
    """Segment id per row: a new segment starts at a new camera or a gap > MAX_GAP_S."""
    gap = base.groupby("camera_id", sort=False)["ts"].diff().dt.total_seconds()
    new = gap.isna() | (gap > MAX_GAP_S)
    return new.cumsum().to_numpy()


def build_sequences(
    values: np.ndarray, segments: np.ndarray, length: int = SEQUENCE_LENGTH, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of ``length`` consecutive rows that never cross a segment boundary.

    Returns (sequences [N, length, F], index of each sequence's last row [N]).
    """
    seqs, ends = [], []
    boundaries = np.flatnonzero(np.diff(segments)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(segments)], strict=True):
        if stop - start < length:
            continue
        window = np.lib.stride_tricks.sliding_window_view(values[start:stop], length, axis=0)
        window = window.transpose(0, 2, 1)[::stride]  # [n, length, F]
        seqs.append(window)
        ends.append(np.arange(start + length - 1, stop)[::stride])
    if not seqs:
        return np.empty((0, length, values.shape[1]), dtype=np.float32), np.empty(0, dtype=int)
    return np.concatenate(seqs).astype(np.float32), np.concatenate(ends)
