"""Real data from Postgres: traffic_metrics joined with sensor_readings, and scenario ground truth.

Windows inside a scenario run's *active* phases (anything before the final "cleared"
phase) are labelled is_anomaly=1 with anomaly_type "scenario:<name>"; the rest of a run
(its resolution) is excluded from training as neither clearly normal nor anomalous
(is_anomaly=-1). Everything else is assumed normal.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from ml.features import METRIC_COLUMNS, attach_sensor_readings, base_features, compute_references
from ml.labels import weak_congestion_labels

_METRICS_SQL = f"""
SELECT camera_id, window_end AS ts, {", ".join(METRIC_COLUMNS)}
FROM traffic_metrics WHERE window_end >= :since AND window_end < :until
"""
_READINGS_SQL = """
SELECT camera_id, reading_at, vehicle_count, avg_speed_kmh, occupancy, scenario_id
FROM sensor_readings WHERE reading_at >= :since AND reading_at < :until
"""
_SCENARIOS_SQL = """
SELECT id, scenario_name, camera_id, status, time_scale, started_at,
       COALESCE(ended_at, expected_end_at) AS ended_at, timeline
FROM scenario_runs WHERE started_at < :until AND COALESCE(ended_at, expected_end_at) >= :since
"""


def scenario_windows(scenarios: pd.DataFrame) -> pd.DataFrame:
    """One row per run: [start, active_end) is anomalous, [active_end, end) is its resolution."""
    rows = []
    for s in scenarios.itertuples():
        timeline = s.timeline or []
        active = [p for p in timeline if p.get("stage") != "cleared"]
        start = pd.Timestamp(s.started_at)
        active_end = (
            pd.Timestamp(active[-1]["ends_at"])
            if active and "ends_at" in active[-1]
            else pd.Timestamp(s.ended_at)
        )
        rows.append(
            {
                "scenario_id": str(s.id),
                "scenario_name": s.scenario_name,
                "camera_id": s.camera_id,
                "status": s.status,
                "time_scale": s.time_scale,
                "start": start,
                "active_end": min(active_end, pd.Timestamp(s.ended_at)),
                "end": pd.Timestamp(s.ended_at),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "scenario_id",
            "scenario_name",
            "camera_id",
            "status",
            "time_scale",
            "start",
            "active_end",
            "end",
        ],
    )


def label_scenarios(raw: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["is_anomaly"], raw["anomaly_type"], raw["scenario_id"] = 0, "", None
    raw["episode_id"] = -1
    for k, w in enumerate(windows.itertuples()):
        cam = raw["camera_id"] == w.camera_id
        active = cam & (raw["ts"] >= w.start) & (raw["ts"] < w.active_end)
        resolution = cam & (raw["ts"] >= w.active_end) & (raw["ts"] < w.end)
        raw.loc[active, ["is_anomaly", "anomaly_type", "scenario_id", "episode_id"]] = (
            1,
            f"scenario:{w.scenario_name}",
            w.scenario_id,
            1_000_000 + k,
        )
        raw.loc[resolution, ["is_anomaly", "scenario_id"]] = (-1, w.scenario_id)
    return raw


def load_real(engine: Engine, hours: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real raw rows (labelled) for the last ``hours`` (None = everything), and scenario windows."""
    until = datetime.now(UTC)
    since = until - timedelta(hours=hours) if hours else datetime(2000, 1, 1, tzinfo=UTC)
    params = {"since": since, "until": until}
    with engine.connect() as conn:
        metrics = pd.read_sql(text(_METRICS_SQL), conn, params=params)
        readings = pd.read_sql(
            text(_READINGS_SQL),
            conn,
            params={"since": since - timedelta(minutes=1), "until": until},
        )
        scenarios = pd.read_sql(text(_SCENARIOS_SQL), conn, params=params)
    if metrics.empty:
        return metrics, scenario_windows(scenarios)
    raw = attach_sensor_readings(metrics, readings.drop(columns=["scenario_id"]))
    windows = scenario_windows(scenarios)
    raw = label_scenarios(raw, windows)
    base = base_features(raw, compute_references(raw))
    raw["congestion"] = weak_congestion_labels(base).to_numpy()
    raw["origin"], raw["label_source"] = "real", "weak"
    raw["frames"] = raw["frames"].astype(float)
    return raw, windows


def summarise(raw: pd.DataFrame) -> dict:
    if raw.empty:
        return {"rows": 0}
    return {
        "rows": len(raw),
        "cameras": sorted(raw["camera_id"].unique().tolist()),
        "from": str(raw["ts"].min()),
        "to": str(raw["ts"].max()),
        "sensor_coverage": round(float(raw["loop_speed_kmh"].notna().mean()), 3),
        "scenario_windows": int((raw["is_anomaly"] == 1).sum()),
        "congestion": raw["congestion"].value_counts().to_dict(),
        "nan_speed_share": round(float(np.isnan(raw["avg_speed_kmh"].astype(float)).mean()), 3),
    }
