"""Synthetic per-camera metric series (5 s windows) with labelled incident episodes.

Normal traffic follows a daily pattern (weekday peaks ~08:00 and ~17:30 UTC, flatter at
weekends). Injected episodes, each with ground truth (is_anomaly, anomaly_type, episode_id):

- incident            sudden speed drop + stopped vehicles + density spike, then recovery
- congestion_buildup  gradual slowdown off-peak, peaking mid-episode, then dissolving
- sensor_dropout      the loop detector stops reporting
- camera_failure      detection confidence collapses, few frames, no vehicles seen
- pedestrian_in_road  people in the carriageway, traffic slows a little

The congestion label comes from the noise-free state via ml.labels.congestion_level, the
same rule used to weak-label real data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.features import WINDOW_S
from ml.labels import CONGESTION_LEVELS, congestion_level

EPISODE_TYPES = {
    "incident": (0.30, (6, 15)),
    "congestion_buildup": (0.20, (15, 30)),
    "sensor_dropout": (0.20, (5, 15)),
    "camera_failure": (0.15, (5, 15)),
    "pedestrian_in_road": (0.15, (1, 3)),
}  # type -> (share, duration range in minutes)
SENSOR_EVERY = 3  # loop reading every 15 s = every 3rd window


def _bump(hour: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((hour - centre) / width) ** 2)


def flow_fraction(ts: pd.DatetimeIndex) -> np.ndarray:
    hour = np.asarray(ts.hour + ts.minute / 60 + ts.second / 3600, dtype=float)
    weekend = np.asarray(ts.weekday >= 5)
    weekday = (
        0.06
        + 0.80 * _bump(hour, 8.0, 1.1)
        + 0.40 * _bump(hour, 12.5, 2.5)
        + 0.85 * _bump(hour, 17.5, 1.5)
    )
    weekend_f = 0.06 + 0.55 * _bump(hour, 13.5, 3.5)
    return np.clip(np.where(weekend, weekend_f, weekday), 0, 1)


def speed_ratio_for_flow(f: np.ndarray) -> np.ndarray:
    slow = np.where(f < 0.25, 0.15 * np.sqrt(f / 0.25), 0.15 + 0.75 * ((f - 0.25) / 0.75) ** 2)
    return 1 - np.minimum(slow, 0.92)


@dataclass(frozen=True)
class CameraProfile:
    camera_id: str
    count_scale: float
    camera_free_flow: float
    loop_free_flow: float
    loop_capacity: float
    density_scale: float
    pedestrian_rate: float


def _profiles(n: int, rng: np.random.Generator) -> list[CameraProfile]:
    return [
        CameraProfile(
            camera_id=f"syn-cam-{i + 1}",
            count_scale=rng.uniform(0.7, 1.4),
            camera_free_flow=rng.uniform(22, 40),
            loop_free_flow=rng.uniform(42, 56),
            loop_capacity=rng.uniform(1400, 2600),
            density_scale=rng.uniform(0.6, 1.6),
            pedestrian_rate=rng.uniform(0.05, 1.2),
        )
        for i in range(n)
    ]


def _episodes(n_rows: int, ts: pd.DatetimeIndex, per_day: float, rng: np.random.Generator):
    """Non-overlapping (start, stop, type) row ranges."""
    days = n_rows * WINDOW_S / 86400
    count = rng.poisson(per_day * days)
    types = list(EPISODE_TYPES)
    shares = np.array([EPISODE_TYPES[t][0] for t in types])
    chosen, taken = [], np.zeros(n_rows, dtype=bool)
    for _ in range(count * 4):  # rejection sampling for non-overlap
        if len(chosen) >= count:
            break
        kind = rng.choice(types, p=shares / shares.sum())
        lo, hi = EPISODE_TYPES[kind][1]
        length = int(rng.uniform(lo, hi) * 60 / WINDOW_S)
        start = int(rng.integers(120, max(121, n_rows - length - 120)))
        if kind == "congestion_buildup" and ts[start].hour in (6, 7, 8, 9, 16, 17, 18, 19):
            continue  # build-ups are only anomalous outside the normal peaks
        margin = int(10 * 60 / WINDOW_S)
        if taken[max(0, start - margin) : start + length + margin].any():
            continue
        taken[start : start + length] = True
        chosen.append((start, start + length, kind))
    return sorted(chosen)


def _camera_series(
    profile: CameraProfile,
    ts: pd.DatetimeIndex,
    per_day: float,
    rng: np.random.Generator,
    episode_offset: int,
) -> pd.DataFrame:
    n = len(ts)
    f = flow_fraction(ts)
    ratio = speed_ratio_for_flow(f)
    count_mean = profile.count_scale * (2 + 12 * f)
    density_mean = profile.density_scale * (0.4 + 4 * f)
    stopped_rate = np.full(n, 0.03)
    ped_rate = np.full(n, profile.pedestrian_rate * (0.3 + f))
    conf_mean, frames_mean = np.full(n, 0.6), np.full(n, 24.0)
    sensor_ok = np.ones(n, dtype=bool)
    is_anomaly = np.zeros(n, dtype=int)
    anomaly_type = np.full(n, "", dtype=object)
    episode_id = np.full(n, -1)

    for k, (start, stop, kind) in enumerate(_episodes(n, ts, per_day, rng)):
        length = stop - start
        sl = slice(start, stop)
        is_anomaly[sl], anomaly_type[sl], episode_id[sl] = 1, kind, episode_offset + k
        if kind == "incident":
            floor = rng.uniform(0.06, 0.2)
            onset = np.clip(np.arange(length) / 6, 0, 1)  # ~30 s to the bottom
            ratio[sl] = ratio[sl] * (1 - onset) + floor * onset
            stopped_rate[sl] = rng.uniform(3, 6)
            density_mean[sl] *= rng.uniform(2.2, 3.0)
            count_mean[sl] *= 0.6
            rec = slice(stop, min(n, stop + 36))  # 3 min recovery (not labelled)
            ramp = np.linspace(0, 1, rec.stop - rec.start)
            ratio[rec] = ratio[stop - 1] * (1 - ramp) + ratio[rec] * ramp
        elif kind == "congestion_buildup":
            target = rng.uniform(0.2, 0.35)
            shape = 1 - np.abs(np.linspace(-1, 1, length))  # triangle 0 -> 1 -> 0
            ratio[sl] = ratio[sl] * (1 - shape) + target * shape
            density_mean[sl] *= 1 + 1.5 * shape
        elif kind == "sensor_dropout":
            sensor_ok[sl] = False
        elif kind == "camera_failure":
            conf_mean[sl], frames_mean[sl] = rng.uniform(0.05, 0.15), rng.uniform(3, 6)
            count_mean[sl] = 0.2
            density_mean[sl] = 0.05
            ped_rate[sl] = 0.0
        elif kind == "pedestrian_in_road":
            ped_rate[sl] += rng.uniform(2, 4)
            ratio[sl] *= 0.7

    vehicle_count = rng.poisson(count_mean)
    density = np.maximum(0, rng.normal(density_mean, 0.25 * density_mean + 0.1))
    stopped = rng.poisson(stopped_rate)
    pedestrians = rng.poisson(ped_rate)
    avg_speed = profile.camera_free_flow * ratio * rng.uniform(0.85, 1.15, n)
    avg_speed = np.where(vehicle_count > 0, avg_speed, np.nan)
    p85 = avg_speed * rng.uniform(1.15, 1.4, n)
    conf = np.clip(rng.normal(conf_mean, 0.03), 0.01, 0.99)
    conf = np.where(vehicle_count + pedestrians > 0, conf, np.nan)
    frames = np.clip(np.round(rng.normal(frames_mean, 1.0)), 1, 30)

    # Loop detector: a reading every 15 s, carried to the next windows like the as-of join.
    idx = (np.arange(n) // SENSOR_EVERY) * SENSOR_EVERY
    loop_speed = (profile.loop_free_flow * ratio * rng.uniform(0.92, 1.08, n))[idx]
    loop_count = rng.poisson(profile.loop_capacity * f * 15 / 3600)[idx].astype(float)
    loop_occ = np.clip(f * (1 / np.maximum(ratio, 0.1)) * 0.35, 0, 0.98)[idx]
    for arr in (loop_speed, loop_count, loop_occ):
        arr[~sensor_ok] = np.nan

    density_ratio_true = density_mean / np.median(profile.density_scale * (0.4 + 4 * f))
    level = congestion_level(ratio, density_ratio_true, stopped_rate)
    return pd.DataFrame(
        {
            "camera_id": profile.camera_id,
            "ts": ts,
            "vehicle_count": vehicle_count,
            "avg_speed_kmh": avg_speed,
            "p85_speed_kmh": p85,
            "density": density,
            "stopped_vehicles": stopped,
            "pedestrians_in_roadway": pedestrians,
            "mean_confidence": conf,
            "frames": frames,
            "loop_vehicle_count": loop_count,
            "loop_speed_kmh": loop_speed,
            "loop_occupancy": loop_occ,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "episode_id": episode_id,
            "congestion": np.asarray(CONGESTION_LEVELS)[level],
        }
    )


def generate(
    cameras: int = 4,
    days: float = 2.0,
    seed: int = 0,
    start: str = "2026-10-05T00:00:00Z",
    episodes_per_day: float = 10.0,
) -> pd.DataFrame:
    """Synthetic raw rows for ``cameras`` cameras over ``days`` days, with labels."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(
        start, periods=int(days * 86400 / WINDOW_S), freq=f"{int(WINDOW_S)}s", tz="UTC"
    )
    frames, offset = [], 0
    for profile in _profiles(cameras, rng):
        df = _camera_series(profile, ts, episodes_per_day, rng, offset)
        offset = int(df["episode_id"].max()) + 1 if (df["episode_id"] >= 0).any() else offset
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["origin"] = "synthetic"
    out["label_source"] = "synthetic"
    out["scenario_id"] = None
    return out
