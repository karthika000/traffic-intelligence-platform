"""Evaluate anomaly scores the way anomaly_service would act on them.

Scores are replayed per camera through ml.alarm (same threshold, cooldown and clearing
rules as the service), and the resulting alarm episodes are matched against labelled
episodes: an episode is detected if an alarm starts between its start and
``grace_s`` after its end; detection delay is alarm start - episode start (>= 0).
"""

import numpy as np
import pandas as pd

from ml.alarm import replay
from ml.features import WINDOW_S

GRACE_S = 60.0


def alarms(scored: pd.DataFrame, threshold: float, **tracker_kwargs) -> pd.DataFrame:
    """scored: camera_id, ts, score (NaN rows are skipped) -> alarm episodes per camera."""
    rows = []
    for cam, df in scored.dropna(subset=["score"]).sort_values("ts").groupby("camera_id"):
        times = df["ts"].map(pd.Timestamp.timestamp).to_numpy()
        for ep in replay(times, df["score"].to_numpy(), threshold, **tracker_kwargs):
            rows.append(
                {
                    "camera_id": cam,
                    "start": pd.Timestamp(ep.start, unit="s", tz="UTC"),
                    "end": pd.Timestamp(ep.end, unit="s", tz="UTC") if ep.end else pd.NaT,
                    "peak": ep.peak,
                }
            )
    return pd.DataFrame(rows, columns=["camera_id", "start", "end", "peak"])


def truth_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Labelled episodes from rows with episode_id >= 0."""
    eps = df[df["episode_id"] >= 0]
    if eps.empty:
        return pd.DataFrame(columns=["episode_id", "camera_id", "type", "start", "end"])
    return (
        eps.groupby("episode_id")
        .agg(
            camera_id=("camera_id", "first"),
            type=("anomaly_type", "first"),
            start=("ts", "min"),
            end=("ts", "max"),
        )
        .reset_index()
    )


def match(alarm_df: pd.DataFrame, episodes: pd.DataFrame, grace_s: float = GRACE_S):
    """Per-episode detection/delay, and which alarms hit an episode."""
    hit = np.zeros(len(alarm_df), dtype=bool)
    rows = []
    for ep in episodes.itertuples():
        cam = alarm_df["camera_id"] == ep.camera_id
        window = (
            cam
            & (alarm_df["start"] >= ep.start - pd.Timedelta(seconds=WINDOW_S * 12))
            & (alarm_df["start"] <= ep.end + pd.Timedelta(seconds=grace_s))
        )
        hit |= window.to_numpy()
        first = alarm_df.loc[window, "start"].min()
        detected = pd.notna(first)
        delay = max(0.0, (first - ep.start).total_seconds()) if detected else np.nan
        rows.append(
            {
                "episode_id": ep.episode_id,
                "camera_id": ep.camera_id,
                "type": ep.type,
                "start": ep.start,
                "detected": detected,
                "delay_s": delay,
            }
        )
    return pd.DataFrame(rows), hit


def episode_metrics(
    scored: pd.DataFrame, labels: pd.DataFrame, threshold: float, **tracker_kwargs
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Recall, precision, delay and false alarms per hour on labelled data.

    ``labels`` holds the rows (camera_id, ts, is_anomaly, episode_id, anomaly_type).
    """
    alarm_df = alarms(scored, threshold, **tracker_kwargs)
    episodes = truth_episodes(labels)
    per_episode, hit = match(alarm_df, episodes)
    normal_hours = float((labels["is_anomaly"] == 0).sum()) * WINDOW_S / 3600
    false_alarms = int((~hit).sum())
    metrics = {
        "episodes": len(episodes),
        "alarms": len(alarm_df),
        "recall": float(per_episode["detected"].mean()) if len(per_episode) else np.nan,
        "precision": float(hit.mean()) if len(alarm_df) else np.nan,
        "delay_median_s": float(per_episode["delay_s"].median()) if len(per_episode) else np.nan,
        "delay_mean_s": float(per_episode["delay_s"].mean()) if len(per_episode) else np.nan,
        "false_alarms": false_alarms,
        "false_alarms_per_hour": false_alarms / normal_hours if normal_hours else np.nan,
    }
    by_type = (
        per_episode.groupby("type")
        .agg(
            episodes=("detected", "size"),
            recall=("detected", "mean"),
            delay_median_s=("delay_s", "median"),
        )
        .reset_index()
        if len(per_episode)
        else pd.DataFrame()
    )
    return metrics, by_type, per_episode


def scenario_report(
    scored: pd.DataFrame, windows: pd.DataFrame, threshold: float, **tracker_kwargs
) -> pd.DataFrame:
    """Per scenario run: did an alarm fire during its active phases, and how fast.

    ``delay_s`` is wall-clock seconds from the run's start; ``delay_scenario_s`` multiplies
    by the run's time_scale (the delay in the scenario's own timeline).
    """
    alarm_df = alarms(scored, threshold, **tracker_kwargs)
    rows = []
    for w in windows.itertuples():
        covered = scored[
            (scored["camera_id"] == w.camera_id)
            & (scored["ts"] >= w.start)
            & (scored["ts"] < w.active_end)
        ]
        cam = alarm_df["camera_id"] == w.camera_id
        hits = alarm_df[
            cam
            & (alarm_df["start"] >= w.start - pd.Timedelta(seconds=WINDOW_S * 12))
            & (alarm_df["start"] <= w.active_end)
        ]
        first = hits["start"].min() if len(hits) else pd.NaT
        delay = max(0.0, (first - w.start).total_seconds()) if pd.notna(first) else np.nan
        rows.append(
            {
                "scenario_id": w.scenario_id,
                "scenario": w.scenario_name,
                "camera_id": w.camera_id,
                "time_scale": w.time_scale,
                "start": w.start,
                "active_end": w.active_end,
                "scored_windows": int(covered["score"].notna().sum()),
                "fired": bool(len(hits)),
                "alarms": len(hits),
                "delay_s": delay,
                "delay_scenario_s": delay * w.time_scale if pd.notna(delay) else np.nan,
            }
        )
    return pd.DataFrame(rows)
